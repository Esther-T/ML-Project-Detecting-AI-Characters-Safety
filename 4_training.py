"""
ML Training Pipeline: encoded_with/without_nulls.feather -> results_summary.csv + best_models.pkl
Purpose: Train and evaluate classifiers to predict character unsafety (Safer=0 / Unsafer=1)
Steps:
  1. Load both encoded feather files; stratified 80/20 train/test split (test set never rebalanced)
  2. Balance training set via chosen strategy (default: SMOTE):
       - weights    : class_weight='balanced' on supported models; undersample fallback for KNN/GNB/GB
       - smote      : oversample minority class synthetically (imbalanced-learn)
       - undersample: downsample majority class to match minority size
  3. For each dataset x each of 6 models (RF, GB, LR, SVM, KNN, GNB):
       - GridSearchCV with 5-fold stratified CV, optimizing F1
       - Refit best params on full (balanced) training set
       - Evaluate on held-out test set: F1, precision, recall, confusion matrix
  4. Collect all results into summary table; identify best model by test F1
  5. Save results -> results_summary.csv, best estimators -> best_models.pkl
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Note: imbalanced-learn not installed. SMOTE strategy will be skipped.")
    print("      Install with: pip install imbalanced-learn")

warnings.filterwarnings('ignore')

#Paths
DATA_PATH = os.path.join('.', 'data')

INPUT_FILES = {
    'with_nulls':    os.path.join(DATA_PATH, 'encoded_with_nulls.feather'),
    'without_nulls': os.path.join(DATA_PATH, 'encoded_without_nulls.feather'),
}
RESULTS_PATH     = os.path.join(DATA_PATH, 'results_summary.csv')
BEST_MODELS_PATH = os.path.join(DATA_PATH, 'best_models.pkl')

RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_CV_FOLDS   = 5

BALANCE_STRATEGY = 'smote'

#Model Definitions
def get_models(strategy):
    """Return list of (name, estimator, param_grid) tuples."""

    cw = 'balanced' if strategy == 'weights' else None

    models = [
        (
            'Random Forest',
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight=cw, n_jobs=-1),
            {
                'n_estimators': [100, 200],
                'max_depth':    [None, 10, 20],
                'min_samples_split': [2, 5],
            }
        ),
        (
            'Gradient Boosting',
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                'n_estimators':  [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth':     [3, 5],
            }
        ),
        (
            'Logistic Regression',
            LogisticRegression(random_state=RANDOM_STATE, class_weight=cw,
                               max_iter=1000, solver='lbfgs'),
            {
                'C': [0.01, 0.1, 1, 10],
            }
        ),
        (
            'SVM',
            SVC(random_state=RANDOM_STATE, class_weight=cw, probability=True),
            {
                'C':      [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
            }
        ),
        (
            'KNN',
            KNeighborsClassifier(n_jobs=-1),
            {
                'n_neighbors': [3, 5, 7, 11],
                'weights':     ['uniform', 'distance'],
            }
        ),
        (
            'Gaussian Naive Bayes',
            GaussianNB(),
            {
                'var_smoothing': [1e-9, 1e-8, 1e-7],
            }
        ),
    ]
    return models


#Balancing Features

def balance_undersample(X_train, y_train):
    """Undersample majority class to match minority class size."""
    df = pd.concat([X_train, y_train], axis=1)
    majority = df[y_train == y_train.value_counts().idxmax()]
    minority = df[y_train == y_train.value_counts().idxmin()]
    majority_down = resample(majority, n_samples=len(minority),
                             random_state=RANDOM_STATE, replace=False)
    balanced = pd.concat([majority_down, minority]).sample(
        frac=1, random_state=RANDOM_STATE)
    y_col = y_train.name
    return balanced.drop(columns=[y_col]), balanced[y_col]


def balance_smote(X_train, y_train):
    """Oversample minority class using SMOTE."""
    if not SMOTE_AVAILABLE:
        print("    SMOTE unavailable, falling back to undersample.")
        return balance_undersample(X_train, y_train)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)


def get_sample_weight(y_train):
    """Compute per-sample weights for models that don't support class_weight."""
    counts   = y_train.value_counts()
    n_total  = len(y_train)
    n_classes = len(counts)
    weight_map = {cls: n_total / (n_classes * cnt) for cls, cnt in counts.items()}
    return y_train.map(weight_map).values


#Training
def train_and_evaluate(dataset_label, X_train_raw, X_test, y_train_raw, y_test, strategy):
    print(f"\n{'─'*60}")
    print(f"Dataset: {dataset_label}  |  Strategy: {strategy}")
    print(f"Train: {len(y_train_raw)} samples  |  Test: {len(y_test)} samples")
    print(f"Train class dist: {dict(y_train_raw.value_counts())}")
    print(f"Test  class dist: {dict(y_test.value_counts())}")

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models(strategy)

    all_results  = []
    best_estimators = {}

    for name, estimator, param_grid in models:
        print(f"\n  [{name}]")

        #Balance training data
        no_weight_models = {'KNN', 'Gaussian Naive Bayes', 'Gradient Boosting'}

        if strategy == 'undersample':
            X_train, y_train = balance_undersample(X_train_raw.copy(), y_train_raw.copy())
            fit_params = {}

        elif strategy == 'smote':
            X_train, y_train = balance_smote(X_train_raw.copy(), y_train_raw.copy())
            fit_params = {}

        else: 
            X_train, y_train = X_train_raw.copy(), y_train_raw.copy()
            if name in no_weight_models:
                sw = get_sample_weight(y_train)
                X_train, y_train = balance_undersample(X_train, y_train)
                fit_params = {}
                print(f"    Note: {name} does not support class_weight — used undersample fallback")
            else:
                fit_params = {}

        print(f"    Balanced train size: {len(y_train)}  dist: {dict(pd.Series(y_train).value_counts())}")

        #Grid Search + Cross Validation
        grid_search = GridSearchCV(
            estimator  = estimator,
            param_grid = param_grid,
            cv         = cv,
            scoring    = 'f1',          
            n_jobs     = -1,
            refit      = True,          
            verbose    = 0,
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_cv_f1  = grid_search.best_score_
        print(f"    Best CV F1:  {best_cv_f1:.4f}")
        print(f"    Best params: {best_params}")

        #Evaluate on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        test_f1        = f1_score(y_test, y_pred, zero_division=0)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall    = recall_score(y_test, y_pred, zero_division=0)

        print(f"    Test  F1:        {test_f1:.4f}")
        print(f"    Test  Precision: {test_precision:.4f}")
        print(f"    Test  Recall:    {test_recall:.4f}")
        print(f"    Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

        all_results.append({
            'dataset':       dataset_label,
            'strategy':      strategy,
            'model':         name,
            'best_cv_f1':    round(best_cv_f1,  4),
            'test_f1':       round(test_f1,       4),
            'test_precision':round(test_precision,4),
            'test_recall':   round(test_recall,   4),
            'best_params':   str(best_params),
            'train_size':    len(y_train),
            'test_size':     len(y_test),
        })

        best_estimators[f"{dataset_label}__{name}"] = best_model

    return all_results, best_estimators


if __name__ == '__main__':

    all_results     = []
    all_best_models = {}

    for dataset_key, feather_path in INPUT_FILES.items():

        print(f"\n{'='*60}")
        print(f"Loading: {feather_path}")

        df = pd.read_feather(feather_path)
        print(f"  Shape: {df.shape}  |  Target dist: {dict(df['y'].value_counts())}")

        X = df.drop(columns=['bot', 'y'])
        y = df['y']

        #80/20 stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = TEST_SIZE,
            stratify     = y,
            random_state = RANDOM_STATE,
        )

        results, best_models = train_and_evaluate(
            dataset_label = dataset_key,
            X_train_raw   = X_train,
            X_test        = X_test,
            y_train_raw   = y_train,
            y_test        = y_test,
            strategy      = BALANCE_STRATEGY,
        )

        all_results.extend(results)
        all_best_models.update(best_models)

    #Output
    results_df = pd.DataFrame(all_results)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(results_df[['dataset', 'model', 'best_cv_f1', 'test_f1',
                       'test_precision', 'test_recall']].to_string(index=False))

    print(f"\nBest model overall:")
    best_row = results_df.loc[results_df['test_f1'].idxmax()]
    print(f"  {best_row['model']} on {best_row['dataset']}  |  "
          f"Test F1={best_row['test_f1']}  CV F1={best_row['best_cv_f1']}")
    print(f"  Params: {best_row['best_params']}")

    #Save results
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved -> {RESULTS_PATH}")

    with open(BEST_MODELS_PATH, 'wb') as f:
        pickle.dump(all_best_models, f)
    print(f"Best models saved -> {BEST_MODELS_PATH}")
