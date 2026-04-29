"""
Neural Network Training Pipeline
==================================
Trains neural network classifiers for unsafe AI character detection.
Compares against baseline ML models (F1=0.69 with Random Forest).

Three approaches:
  1. Text Embeddings Only    — sentence-transformers on description+scenario+tags
  2. Binary Features Only    — same encoded features as ML baseline
  3. Hybrid                  — concatenated embeddings + binary features

Architecture:
  - Embedding model: all-MiniLM-L6-v2 (384-dim, fast, good quality)
  - NN: 2-3 layer MLP with BatchNorm, Dropout, ReLU
  - Training: AdamW + cosine LR schedule + early stopping
  - Balancing: class_weight (weighted BCE loss)

Usage (Google Colab):
  !pip install sentence-transformers torch torchmetrics
  Then run this file.

Expected F1 scores:
  Text embeddings only:  ~0.60-0.65
  Binary features only:  ~0.65-0.70
  Hybrid:                ~0.68-0.72
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# hyperparamaters
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
N_CV_FOLDS     = 5
BATCH_SIZE     = 32
MAX_EPOCHS     = 100
PATIENCE       = 10          # early stopping patience
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'   # 384-dim; fast and accurate

DATA_PATH       = './data'
ML_DF_PATH      = os.path.join(DATA_PATH, 'ml_df.feather')
ENCODED_PATH    = os.path.join(DATA_PATH, 'encoded_with_nulls.feather')   # or encoded_without_nulls
RESULTS_PATH    = os.path.join(DATA_PATH, 'nn_results_summary.csv')
MODELS_PATH     = os.path.join(DATA_PATH, 'nn_best_models.pkl')
EMBEDDINGS_CACHE = os.path.join(DATA_PATH, 'text_embeddings_cache.npy')   # cache to avoid re-encoding

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── STEP 1: TEXT EMBEDDING GENERATION ───────────────────────────────────────

def load_or_generate_embeddings(ml_df, cache_path=EMBEDDINGS_CACHE):
    """
    Generate sentence embeddings from description + scenario + tags.
    Caches to disk so you don't re-encode on every run (encoding ~1000
    samples takes ~30s on CPU, ~5s on GPU).

    Text construction strategy:
      - description: character bio (most informative)
      - scenario:    roleplay context (reveals unsafe intent)
      - tags:        categorical labels joined as text
    All three concatenated with [SEP] token as delimiter.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
        print(f"  Embeddings shape: {embeddings.shape}")
        return embeddings

    print("Generating text embeddings (this may take a few minutes)...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    def build_text(row):
        parts = []
        # Description: character bio
        desc = str(row.get('description', '') or '')
        if desc.strip():
            parts.append(desc.strip())
        # Scenario: roleplay context
        scen = str(row.get('scenario', '') or '')
        if scen.strip():
            parts.append(scen.strip())
        # Tags: join list as comma-separated text
        tags = row.get('tags', [])
        if isinstance(tags, list) and tags:
            parts.append('Tags: ' + ', '.join(str(t) for t in tags))
        elif isinstance(tags, str) and tags.strip():
            parts.append('Tags: ' + tags.strip())
        return ' [SEP] '.join(parts) if parts else 'No description available'

    texts = ml_df.apply(build_text, axis=1).tolist()
    print(f"  Built {len(texts)} text inputs. Sample:\n  {texts[0][:200]}...")

    # Encode in batches with progress bar
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize for cosine similarity
    )
    print(f"  Embeddings shape: {embeddings.shape}")  # (n_chars, 384)

    np.save(cache_path, embeddings)
    print(f"  Saved cache -> {cache_path}")
    return embeddings


# ─── STEP 2: DATASET CLASS ───────────────────────────────────────────────────

class CharacterDataset(Dataset):
    """PyTorch Dataset for character safety classification."""

    def __init__(self, X, y):
        """
        X: np.ndarray or pd.DataFrame of features
        y: array-like of binary labels (0=safer, 1=unsafer)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(np.array(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_weighted_sampler(y_train):
    """
    WeightedRandomSampler to oversample minority class during training.
    Alternative to modifying the loss function — ensures each batch
    has roughly balanced classes.
    """
    class_counts = np.bincount(y_train.astype(int))
    weights = 1.0 / class_counts
    sample_weights = weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ─── STEP 3: MODEL ARCHITECTURES ─────────────────────────────────────────────

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification.

    Architecture choices:
    - BatchNorm before activation: stabilises training on small datasets
    - Dropout (0.3-0.5): regularisation against overfitting (we have ~800 train samples)
    - ReLU: standard, avoids vanishing gradients
    - Sigmoid output: binary classification probability
    """

    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        # No sigmoid here — using BCEWithLogitsLoss which is more numerically stable
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class HybridMLP(nn.Module):
    """
    Hybrid model: separate encoders for text embeddings and binary features,
    then fused for classification.

    Why separate encoders?
    Text embeddings (dense, continuous) and binary features (sparse, categorical)
    have very different statistical properties. Processing them separately before
    fusion lets each encoder specialise before the combined decision layer.
    """

    def __init__(self, emb_dim, bin_dim, emb_hidden=(256, 128),
                 bin_hidden=(64, 32), fusion_hidden=(128, 64), dropout=0.4):
        super().__init__()

        # Text embedding encoder
        emb_layers = []
        prev = emb_dim
        for h in emb_hidden:
            emb_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.emb_encoder = nn.Sequential(*emb_layers)

        # Binary feature encoder
        bin_layers = []
        prev = bin_dim
        for h in bin_hidden:
            bin_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.bin_encoder = nn.Sequential(*bin_layers)

        # Fusion head
        fusion_input = emb_hidden[-1] + bin_hidden[-1]
        fusion_layers = []
        prev = fusion_input
        for h in fusion_hidden:
            fusion_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        fusion_layers.append(nn.Linear(prev, 1))
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, emb, binary):
        e = self.emb_encoder(emb)
        b = self.bin_encoder(binary)
        combined = torch.cat([e, b], dim=1)
        return self.fusion(combined).squeeze(1)


# ─── STEP 4: TRAINING LOOP ───────────────────────────────────────────────────

def compute_class_weights(y_train):
    """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
    print(f"  Class weights — neg: {n_neg}, pos: {n_pos}, pos_weight: {pos_weight.item():.2f}")
    return pos_weight


def train_epoch(model, loader, optimizer, criterion, is_hybrid=False):
    model.train()
    total_loss, preds, labels = 0, [], []
    for batch in loader:
        if is_hybrid:
            X_emb, X_bin, y = batch
            X_emb, X_bin, y = X_emb.to(device), X_bin.to(device), y.to(device)
            logits = model(X_emb, X_bin)
        else:
            X, y = batch
            X, y = X.to(device), y.to(device)
            logits = model(X)

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        preds.extend(pred)
        labels.extend(y.long().cpu().numpy())

    f1 = f1_score(labels, preds, zero_division=0)
    return total_loss / len(loader), f1


def eval_epoch(model, loader, criterion, is_hybrid=False):
    model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            if is_hybrid:
                X_emb, X_bin, y = batch
                X_emb, X_bin, y = X_emb.to(device), X_bin.to(device), y.to(device)
                logits = model(X_emb, X_bin)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                logits = model(X)

            loss = criterion(logits, y)
            total_loss += loss.item()
            pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            preds.extend(pred)
            labels.extend(y.long().cpu().numpy())

    f1 = f1_score(labels, preds, zero_division=0)
    return total_loss / len(loader), f1, np.array(preds), np.array(labels)


def train_model(model, train_loader, val_loader, criterion, epochs=MAX_EPOCHS,
                patience=PATIENCE, is_hybrid=False, model_name="model"):
    """Full training loop with early stopping and cosine LR schedule."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1   = 0.0
    best_state    = None
    patience_ctr  = 0
    history       = []

    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, is_hybrid)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader, criterion, is_hybrid)
        scheduler.step()

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_f1': train_f1,
                        'val_loss': val_loss, 'val_f1': val_f1})

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
                  f"| val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val_f1, history


# ─── STEP 5: DATASET WRAPPER FOR HYBRID MODEL ────────────────────────────────

class HybridDataset(Dataset):
    def __init__(self, X_emb, X_bin, y):
        self.X_emb = torch.FloatTensor(X_emb)
        self.X_bin = torch.FloatTensor(X_bin if not isinstance(X_bin, pd.DataFrame) else X_bin.values)
        self.y     = torch.FloatTensor(np.array(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_emb[idx], self.X_bin[idx], self.y[idx]


# ─── STEP 6: FULL EXPERIMENT RUNNER ──────────────────────────────────────────

def run_experiment(name, X_train, X_test, y_train, y_test,
                   model_factory, is_hybrid=False,
                   X_train_aux=None, X_test_aux=None):
    """
    Run one full experiment: train NN, evaluate on test set.

    Parameters
    ----------
    name         : display name for this experiment
    X_train/test : primary features (embeddings OR binary features)
    y_train/test : labels
    model_factory: callable that returns an untrained model
    is_hybrid    : whether to use HybridDataset + HybridMLP
    X_train/test_aux: secondary features for hybrid model (binary features)
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Train class dist: {dict(pd.Series(y_train).value_counts())}")

    pos_weight = compute_class_weights(np.array(y_train))
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train/val split from training data (use 10% for early stopping)
    if is_hybrid:
        X_tr_emb, X_val_emb, X_tr_bin, X_val_bin, y_tr, y_val = train_test_split(
            X_train, X_train_aux, y_train,
            test_size=0.125, random_state=RANDOM_STATE, stratify=y_train
        )
        train_ds = HybridDataset(X_tr_emb, X_tr_bin, y_tr)
        val_ds   = HybridDataset(X_val_emb, X_val_bin, y_val)
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.125, random_state=RANDOM_STATE, stratify=y_train
        )
        train_ds = CharacterDataset(X_tr, y_tr)
        val_ds   = CharacterDataset(X_val, y_val)

    # Use WeightedRandomSampler for training to handle class imbalance
    sampler     = make_weighted_sampler(np.array(y_tr))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = model_factory().to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, best_val_f1, history = train_model(
        model, train_loader, val_loader, criterion,
        is_hybrid=is_hybrid, model_name=name
    )
    print(f"  Best validation F1: {best_val_f1:.4f}")

    # Final evaluation on test set
    if is_hybrid:
        test_ds = HybridDataset(X_test, X_test_aux, y_test)
    else:
        test_ds = CharacterDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    _, test_f1, y_pred, y_true = eval_epoch(model, test_loader, criterion, is_hybrid)
    test_precision = precision_score(y_true, y_pred, zero_division=0)
    test_recall    = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n  ── Test Results ──")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Safer', 'Unsafer'])}")

    return {
        'experiment':      name,
        'best_val_f1':     round(best_val_f1,    4),
        'test_f1':         round(test_f1,         4),
        'test_precision':  round(test_precision,  4),
        'test_recall':     round(test_recall,     4),
        'train_size':      len(y_train),
        'test_size':       len(y_test),
    }, model, history


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    ml_df      = pd.read_feather(ML_DF_PATH)
    encoded_df = pd.read_feather(ENCODED_PATH)
    print(f"  ml_df shape:      {ml_df.shape}")
    print(f"  encoded_df shape: {encoded_df.shape}")
    print(f"  Target dist: {dict(encoded_df['y'].value_counts())}")

    # Align ml_df to encoded_df by bot ID (encoded_df may have fewer rows)
    ml_df_aligned = ml_df.set_index('bot').loc[encoded_df['bot'].values].reset_index()
    print(f"  Aligned ml_df:    {ml_df_aligned.shape}")

    # ── Generate text embeddings ───────────────────────────────────────────────
    embeddings = load_or_generate_embeddings(ml_df_aligned, cache_path=EMBEDDINGS_CACHE)
    # embeddings shape: (n_chars, 384)

    # ── Prepare features ──────────────────────────────────────────────────────
    y           = encoded_df['y'].values
    X_binary    = encoded_df.drop(columns=['bot', 'y']).values.astype(np.float32)
    X_embeddings = embeddings.astype(np.float32)

    # Hybrid: normalise embeddings (already L2-normed), scale binary features
    # Binary features are already 0/1 so scaling is minor but helps NN
    scaler = StandardScaler()
    X_binary_scaled = scaler.fit_transform(X_binary)   # will re-fit on train only below

    print(f"\n  Embedding dim:     {X_embeddings.shape[1]}")
    print(f"  Binary feature dim: {X_binary.shape[1]}")
    print(f"  Hybrid input dim:   {X_embeddings.shape[1] + X_binary.shape[1]}")

    # ── Train/test split (same 80/20 stratified as ML baseline) ───────────────
    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_emb_train, X_emb_test = X_embeddings[idx_train], X_embeddings[idx_test]
    X_bin_train_raw, X_bin_test_raw = X_binary[idx_train], X_binary[idx_test]

    # Fit scaler ONLY on training data (prevent data leakage)
    scaler_fit = StandardScaler()
    X_bin_train = scaler_fit.fit_transform(X_bin_train_raw)
    X_bin_test  = scaler_fit.transform(X_bin_test_raw)

    # ── Experiment 1: Text Embeddings Only ────────────────────────────────────
    EMB_DIM = X_emb_train.shape[1]   # 384

    results, best_models = [], {}

    res, model, hist = run_experiment(
        name         = "Text Embeddings Only",
        X_train      = X_emb_train,
        X_test       = X_emb_test,
        y_train      = y_train,
        y_test       = y_test,
        model_factory= lambda: MLP(
            input_dim   = EMB_DIM,
            hidden_dims = (256, 128, 64),
            dropout     = 0.4
        ),
    )
    results.append(res)
    best_models['embeddings_only'] = model

    # ── Experiment 2: Binary Features Only ────────────────────────────────────
    BIN_DIM = X_bin_train.shape[1]

    res, model, hist = run_experiment(
        name         = "Binary Features Only (NN)",
        X_train      = X_bin_train,
        X_test       = X_bin_test,
        y_train      = y_train,
        y_test       = y_test,
        model_factory= lambda: MLP(
            input_dim   = BIN_DIM,
            hidden_dims = (128, 64),
            dropout     = 0.3
        ),
    )
    results.append(res)
    best_models['binary_only'] = model

    # ── Experiment 3: Hybrid (Embeddings + Binary Features) ───────────────────
    res, model, hist = run_experiment(
        name          = "Hybrid (Embeddings + Binary)",
        X_train       = X_emb_train,
        X_test        = X_emb_test,
        y_train       = y_train,
        y_test        = y_test,
        model_factory = lambda: HybridMLP(
            emb_dim      = EMB_DIM,
            bin_dim      = BIN_DIM,
            emb_hidden   = (256, 128),
            bin_hidden   = (64, 32),
            fusion_hidden= (128, 64),
            dropout      = 0.4
        ),
        is_hybrid     = True,
        X_train_aux   = X_bin_train,
        X_test_aux    = X_bin_test,
    )
    results.append(res)
    best_models['hybrid'] = model

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY — Neural Networks vs Baseline")
    print(f"{'='*60}")
    print(results_df[['experiment', 'test_f1', 'test_precision', 'test_recall']].to_string(index=False))

    print(f"\n  ── Baseline Reference ──")
    print(f"  Random Forest (ML baseline): F1 = 0.6891")
    print(f"  Paper (Wei et al. 2025):     F1 = 0.81 (different data)")

    best_row = results_df.loc[results_df['test_f1'].idxmax()]
    print(f"\n  Best NN experiment: {best_row['experiment']} | Test F1={best_row['test_f1']}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved -> {RESULTS_PATH}")

    with open(MODELS_PATH, 'wb') as f:
        pickle.dump(best_models, f)
    print(f"Models saved  -> {MODELS_PATH}")

    print("\nDone.")