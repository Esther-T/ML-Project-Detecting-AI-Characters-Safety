"""
Neural Network Training Pipeline: ml_df.feather -> nn_results_summary.csv + nn_best_model.pkl
Purpose: Train an MLP on sentence embeddings to predict character unsafety (Safer=0 / Unsafer=1)
Steps:
  1. Load ml_df.feather and encode description, scenario, tags per character via all-MiniLM-L6-v2 -> 384-dim normalized embedding
  2. Concatenate NSFW flag -> 385-dim input vector
  3. Stratified 80/20 train/test split and further split train -> 87.5/12.5 train/val
  4. Compute pos_weight (n_neg / n_pos) -> weighted BCEWithLogitsLoss (no undersampling)
  5. Train MLP original version [385-> 256 ->64-> 1] and (fine-tuned version) [385 ->128 -> 32 -> 1] with BatchNorm + Dropout:
       -AdamW optimizer + cosine LR schedule
       -Early stopping on val F1 (patience=5); restore best weights
  6. Evaluate on held-out test set: F1, precision, recall, confusion matrix
  7. Save results -> nn_results_summary.csv, model -> nn_best_model.pkl

Usage:
Install dependencies: pip install sentence-transformers torch scikit-learn pandas pyarrow
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sentence_transformers import SentenceTransformer

#Configurations
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
BATCH_SIZE      = 32
MAX_EPOCHS      = 100
# initially PATIENCE is 10
PATIENCE        = 5 
# initially LR is 1e-3
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

#Paths
DATA_PATH    = './data'
ML_DF_PATH   = os.path.join(DATA_PATH, 'ml_df.feather')
RESULTS_PATH = os.path.join(DATA_PATH, 'nn_results_summary.csv')
MODEL_PATH   = os.path.join(DATA_PATH, 'nn_best_model.pkl')

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


#Generate Text Embeddings

def generate_embeddings(ml_df):
    print("Generating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    def build_text(row):
        parts = []
        desc = str(row.get('description', '') or '').strip()
        if desc:
            parts.append(desc)
        scen = str(row.get('scenario', '') or '').strip()
        if scen:
            parts.append(scen)
        tags = row.get('tags', [])
        if isinstance(tags, list) and tags:
            parts.append('Tags: ' + ', '.join(str(t) for t in tags))
        elif isinstance(tags, str) and tags.strip():
            parts.append('Tags: ' + tags.strip())
        return ' [SEP] '.join(parts) if parts else 'No description available'

    texts = ml_df.apply(build_text, axis=1).tolist()
    print(f"  Sample text: {texts[0][:200]}...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


class CharacterDataset(Dataset):
    def __init__(self, embeddings, nsfw, y):
        nsfw_col = np.array(nsfw).reshape(-1, 1).astype(np.float32)
        X = np.concatenate([embeddings, nsfw_col], axis=1)
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(np.array(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#Build Model

class MLP(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


#Weighted Loss
def compute_pos_weight(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
    print(f"  Class counts -> safer(0): {n_neg}, unsafer(1): {n_pos}")
    print(f"  pos_weight: {pos_weight.item():.3f}")
    return pos_weight


#Training Loop
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.long().cpu().numpy())
    return total_loss / len(loader), f1_score(all_labels, all_preds, zero_division=0)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.long().cpu().numpy())
    return (
        total_loss / len(loader),
        f1_score(all_labels, all_preds, zero_division=0),
        np.array(all_preds),
        np.array(all_labels)
    )


def train_model(model, train_loader, val_loader, criterion):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_f1  = 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_f1     = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1, _, _   = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"train_loss={train_loss:.4f}  train_f1={train_f1:.4f} | "
                  f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val_f1



if __name__ == '__main__':

    print("Loading data...")
    ml_df = pd.read_feather(ML_DF_PATH).reset_index(drop=True)
    print(f"  Shape: {ml_df.shape}")
    print(f"  Target dist: {dict(ml_df['y'].value_counts())}")

    embeddings = generate_embeddings(ml_df)

    assert len(embeddings) == len(ml_df), (
        f"Size mismatch: embeddings={len(embeddings)}, ml_df={len(ml_df)}"
    )

    y    = ml_df['y'].values
    nsfw = ml_df['NSFW'].astype(int).values

    print(f"\n  Embedding dim:   {embeddings.shape[1]}")
    print(f"  + NSFW (1 dim)")
    print(f"  Total input dim: {embeddings.shape[1] + 1}")

    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_STATE,
    )

    emb_train,  emb_test  = embeddings[idx_train], embeddings[idx_test]
    nsfw_train, nsfw_test = nsfw[idx_train],        nsfw[idx_test]

    print(f"\n  Train: {len(y_train)} samples | Test: {len(y_test)} samples")
    print(f"  Train class dist: {dict(pd.Series(y_train).value_counts())}")

    idx_tr, idx_val, y_tr, y_val = train_test_split(
        np.arange(len(y_train)), y_train,
        test_size    = 0.125,
        stratify     = y_train,
        random_state = RANDOM_STATE,
    )

    train_ds = CharacterDataset(emb_train[idx_tr],  nsfw_train[idx_tr],  y_tr)
    val_ds   = CharacterDataset(emb_train[idx_val], nsfw_train[idx_val], y_val)
    test_ds  = CharacterDataset(emb_test,           nsfw_test,           y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print("\n  Computing class weights...")
    pos_weight = compute_pos_weight(y_train)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    input_dim = embeddings.shape[1] + 1 
    model     = MLP(input_dim=input_dim).to(device)
    print(f"\n  Model architecture:")
    print(f"    Input:  {input_dim} dims  (embeddings + NSFW)")
    print(f"    Layer1(original): 256 units  (BatchNorm + ReLU + Dropout)")
    print(f"    Layer2(original):  64 units  (BatchNorm + ReLU + Dropout)")
    print(f"    Output(original):   1 unit   (logit)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n  Training (max {MAX_EPOCHS} epochs, early stopping patience={PATIENCE})...")
    model, best_val_f1 = train_model(model, train_loader, val_loader, criterion)
    print(f"\n  Best validation F1: {best_val_f1:.4f}")

    _, test_f1, y_pred, y_true = eval_epoch(model, test_loader, criterion)
    test_precision = precision_score(y_true, y_pred, zero_division=0)
    test_recall    = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Safer', 'Unsafer'])}")
    print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")

    print(f"\n Baseline Reference")
    print(f"  Random Forest (tree baseline): F1 = 0.6091")
    print(f"  Paper:       F1 = 0.81")

    results = pd.DataFrame([{
        'model':          'MLP (embeddings + NSFW)',
        'architecture':   '385 -> 256 -> 64 -> 1',
        'best_val_f1':    round(best_val_f1,    4),
        'test_f1':        round(test_f1,         4),
        'test_precision': round(test_precision,  4),
        'test_recall':    round(test_recall,     4),
        'train_size':     len(y_train),
        'test_size':      len(y_test),
    }])
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved -> {RESULTS_PATH}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved   -> {MODEL_PATH}")
