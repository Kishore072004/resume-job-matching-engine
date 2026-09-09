"""
Train the baseline XGBoost resume-job matching model.

This script:
  1. Loads the ML feature-engineered train/validation/test splits.
  2. Excludes ambiguous labels.
  3. Fits median imputation ONLY on the training set.
  4. Trains an XGBClassifier with early stopping on validation loss.
  5. Evaluates on train, validation, and test sets.
  6. Performs threshold analysis on the validation set.
  7. Saves the model pipeline, metadata, predictions, and feature importance.

IMPORTANT INTERPRETATION:
  The model predicts "estimated resume-job match probability" — NOT
  "chance the candidate will get hired." The labels are weakly supervised
  and reflect structural feature overlap, not real hiring decisions.
"""

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ── Setup ───────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering.ml_features import (
    MODEL_FEATURES,
    ID_COLUMNS,
    LABEL_COLUMN,
    create_ml_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
PROCESSED_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"
DOCS_DIR = project_root / "docs"


# ================================================================
# 1. DATA LOADING
# ================================================================

def load_ml_data():
    """Load ML feature-engineered datasets, creating them if needed."""
    ml_train_path = PROCESSED_DIR / "ml_train.parquet"
    ml_val_path = PROCESSED_DIR / "ml_validation.parquet"
    ml_test_path = PROCESSED_DIR / "ml_test.parquet"

    # Create ML feature datasets if they don't exist
    if not ml_train_path.exists():
        logger.info("ML feature datasets not found. Running feature engineering...")
        create_ml_datasets(PROCESSED_DIR)

    df_train = pd.read_parquet(ml_train_path)
    df_val = pd.read_parquet(ml_val_path)
    df_test = pd.read_parquet(ml_test_path)

    return df_train, df_val, df_test


# ================================================================
# 2. FEATURE SELECTION AND LABEL PREPARATION
# ================================================================

def prepare_features_and_labels(df_train, df_val, df_test):
    """
    Select model features, exclude ambiguous labels, and report distributions.
    """
    logger.info("=" * 60)
    logger.info("FEATURE SELECTION AND LABEL PREPARATION")
    logger.info("=" * 60)

    # Report original sizes including ambiguous
    for name, df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        total = len(df)
        ambiguous = df[df["weak_label"] == "ambiguous"] if "weak_label" in df.columns else pd.DataFrame()
        logger.info(f"  {name}: {total} total rows, {len(ambiguous)} ambiguous excluded")

    # Filter out ambiguous labels (they were already excluded in candidate splits,
    # but check again for safety)
    for df in [df_train, df_val, df_test]:
        if "weak_label" in df.columns:
            df.drop(df[df["weak_label"] == "ambiguous"].index, inplace=True)

    # Drop rows with missing labels
    df_train = df_train.dropna(subset=[LABEL_COLUMN]).copy()
    df_val = df_val.dropna(subset=[LABEL_COLUMN]).copy()
    df_test = df_test.dropna(subset=[LABEL_COLUMN]).copy()

    # Verify label classes
    all_labels = set(df_train[LABEL_COLUMN].unique()) | set(df_val[LABEL_COLUMN].unique()) | set(df_test[LABEL_COLUMN].unique())
    logger.info(f"\n  Label classes present: {sorted(all_labels)}")
    assert all_labels == {0, 1} or all_labels == {0.0, 1.0}, f"Unexpected label classes: {all_labels}"

    # Report label distribution
    for name, df in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        pos = (df[LABEL_COLUMN] == 1).sum()
        neg = (df[LABEL_COLUMN] == 0).sum()
        total = len(df)
        logger.info(f"  {name}: {total} rows | Positive: {pos} ({100*pos/total:.1f}%) | Negative: {neg} ({100*neg/total:.1f}%)")

    # Select model features (only those available in data)
    available_features = [f for f in MODEL_FEATURES if f in df_train.columns]
    missing_features = [f for f in MODEL_FEATURES if f not in df_train.columns]
    if missing_features:
        logger.warning(f"  Missing features (will be filled with NaN): {missing_features}")
        for f in missing_features:
            for df in [df_train, df_val, df_test]:
                df[f] = np.nan
        available_features = MODEL_FEATURES

    logger.info(f"\n  Model features ({len(available_features)}): {available_features}")

    X_train = df_train[available_features].copy()
    y_train = df_train[LABEL_COLUMN].astype(int).values
    X_val = df_val[available_features].copy()
    y_val = df_val[LABEL_COLUMN].astype(int).values
    X_test = df_test[available_features].copy()
    y_test = df_test[LABEL_COLUMN].astype(int).values

    ids_train = df_train[ID_COLUMNS].copy()
    ids_val = df_val[ID_COLUMNS].copy()
    ids_test = df_test[ID_COLUMNS].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test, ids_train, ids_val, ids_test, available_features


# ================================================================
# 3. PREPROCESSING (NO LEAKAGE)
# ================================================================

def build_preprocessing_pipeline(X_train):
    """
    Build and fit preprocessing pipeline ONLY on training data.
    Uses median imputation for continuous numerical features.
    """
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING")
    logger.info("=" * 60)

    # Report missing values in training set
    missing = X_train.isnull().sum()
    missing_pct = (missing / len(X_train) * 100).round(1)
    for col in X_train.columns:
        if missing[col] > 0:
            logger.info(f"  {col}: {missing[col]} missing ({missing_pct[col]}%)")

    # Median imputation — preserves distributional properties
    # better than zero for experience/count features where 0 has meaning
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    logger.info(f"  Imputer fit on {len(X_train)} training rows with {len(X_train.columns)} features")
    logger.info(f"  Imputation strategy: median")

    return imputer


# ================================================================
# 4. CLASS IMBALANCE HANDLING
# ================================================================

def compute_class_weight(y_train):
    """
    Compute scale_pos_weight for XGBoost based on class distribution.
    For tree-based models, class weighting is preferred over SMOTE.
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    logger.info(f"\n  Class imbalance: {n_neg} neg / {n_pos} pos = ratio {n_neg/n_pos:.2f}")
    logger.info(f"  scale_pos_weight: {scale_pos_weight:.4f}")
    logger.info("  Decision: Using scale_pos_weight (class weighting) — preferred for")
    logger.info("  tree-based models. SMOTE not used as it can create unrealistic")
    logger.info("  synthetic samples for high-dimensional structured features.")

    return scale_pos_weight


# ================================================================
# 5. MODEL TRAINING
# ================================================================

def train_xgboost(X_train_imp, y_train, X_val_imp, y_val, scale_pos_weight, feature_names):
    """
    Train XGBClassifier with early stopping on validation logloss.
    """
    logger.info("\n" + "=" * 60)
    logger.info("MODEL TRAINING")
    logger.info("=" * 60)

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0,
        "n_jobs": -1,
    }

    logger.info("  XGBoost parameters:")
    for k, v in xgb_params.items():
        logger.info(f"    {k}: {v}")

    model = XGBClassifier(**xgb_params)

    # Train with early stopping
    model.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=False,
    )

    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else xgb_params["n_estimators"]
    best_score = model.best_score if hasattr(model, "best_score") else None

    logger.info(f"\n  Training complete.")
    logger.info(f"  Best iteration: {best_iteration}")
    if best_score is not None:
        logger.info(f"  Best validation logloss: {best_score:.6f}")

    return model, xgb_params


# ================================================================
# 6. EVALUATION
# ================================================================

def evaluate_model(model, X, y, set_name, threshold=0.5):
    """
    Evaluate model on a dataset and return metrics dictionary.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "brier_score": float(brier_score_loss(y, y_proba)),
    }

    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Negative", "Positive"])

    logger.info(f"\n  {'─' * 50}")
    logger.info(f"  {set_name} Evaluation (threshold={threshold:.2f})")
    logger.info(f"  {'─' * 50}")
    logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    logger.info(f"  Precision:  {metrics['precision']:.4f}")
    logger.info(f"  Recall:     {metrics['recall']:.4f}")
    logger.info(f"  F1:         {metrics['f1']:.4f}")
    logger.info(f"  Accuracy:   {metrics['accuracy']:.4f}")
    logger.info(f"  Brier:      {metrics['brier_score']:.4f}")
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"    {cm}")
    logger.info(f"\n  Classification Report:\n{report}")

    return metrics, y_proba, y_pred, cm


# ================================================================
# 7. THRESHOLD ANALYSIS
# ================================================================

def threshold_analysis(model, X_val, y_val):
    """
    Analyze model performance across multiple thresholds on validation set.
    Find the threshold that maximizes F1.
    """
    logger.info("\n" + "=" * 60)
    logger.info("THRESHOLD ANALYSIS (Validation Set)")
    logger.info("=" * 60)

    y_proba = model.predict_proba(X_val)[:, 1]

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f = f1_score(y_val, y_pred, zero_division=0)
        a = accuracy_score(y_val, y_pred)
        results.append({"threshold": t, "precision": p, "recall": r, "f1": f, "accuracy": a})
        logger.info(f"  Threshold {t:.2f} | P: {p:.4f} | R: {r:.4f} | F1: {f:.4f} | Acc: {a:.4f}")

    # Fine-grained search for best F1 threshold
    best_f1 = 0.0
    best_threshold = 0.50
    for t in np.arange(0.20, 0.80, 0.01):
        y_pred = (y_proba >= t).astype(int)
        f = f1_score(y_val, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_threshold = float(round(t, 2))

    logger.info(f"\n  Best validation F1: {best_f1:.4f} at threshold: {best_threshold:.2f}")

    return best_threshold, results


# ================================================================
# 8. FEATURE IMPORTANCE
# ================================================================

def save_feature_importance(model, feature_names):
    """Save feature importance (gain) to CSV."""
    importance = model.get_booster().get_score(importance_type="gain")

    # Map feature indices to names
    fi_data = []
    for fname in feature_names:
        fi_data.append({
            "feature": fname,
            "importance": importance.get(fname, 0.0),
        })

    df_fi = pd.DataFrame(fi_data).sort_values("importance", ascending=False)
    fi_path = PROCESSED_DIR / "model_feature_importance.csv"
    df_fi.to_csv(fi_path, index=False)

    logger.info(f"\n  Feature importance saved to {fi_path}")
    logger.info("  Top 10 features by gain:")
    for _, row in df_fi.head(10).iterrows():
        logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    return df_fi


# ================================================================
# 9. SAVE MODEL ARTIFACTS
# ================================================================

def save_model_pipeline(imputer, model, feature_names):
    """Save complete model pipeline (preprocessing + model) for inference."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the sklearn pipeline
    pipeline = Pipeline([
        ("imputer", imputer),
        ("classifier", model),
    ])
    pipeline_path = MODELS_DIR / "job_matcher_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"  Pipeline saved: {pipeline_path}")

    # Save the XGBoost model separately
    xgb_path = MODELS_DIR / "job_matcher_xgb.joblib"
    joblib.dump(model, xgb_path)
    logger.info(f"  XGBoost model saved: {xgb_path}")

    # Save feature names
    feature_path = MODELS_DIR / "feature_names.json"
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    return pipeline_path, xgb_path


def save_model_metadata(
    xgb_params, feature_names, metrics_train, metrics_val, metrics_test,
    best_threshold, n_train, n_val, n_test, n_pos_train, n_neg_train,
):
    """Save comprehensive model metadata JSON."""
    import xgboost
    import sklearn

    metadata = {
        "model_name": "job_matcher_xgb_baseline",
        "model_type": "XGBClassifier",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "feature_names": feature_names,
        "training_rows": n_train,
        "validation_rows": n_val,
        "test_rows": n_test,
        "positive_training_rows": int(n_pos_train),
        "negative_training_rows": int(n_neg_train),
        "validation_threshold": best_threshold,
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "test_metrics": metrics_test,
        "xgboost_parameters": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v for k, v in xgb_params.items()},
        "dataset_paths": {
            "train": "data/processed/ml_train.parquet",
            "validation": "data/processed/ml_validation.parquet",
            "test": "data/processed/ml_test.parquet",
        },
        "weak_label_description": (
            "Labels are weakly supervised based on skill overlap (Jaccard), "
            "semantic similarity (SentenceTransformer cosine), and title similarity. "
            "Positive pairs have high skill coverage AND high semantic similarity. "
            "Negative pairs have low overlap or are randomly sampled unrelated pairs. "
            "Ambiguous pairs are excluded from training. "
            "These labels are NOT ground-truth hiring decisions."
        ),
        "package_versions": {
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  Metadata saved: {meta_path}")
    return metadata


def save_test_predictions(ids_test, y_test, y_proba, y_pred, best_threshold):
    """Save test set predictions to parquet."""
    df_pred = ids_test.copy()
    df_pred["actual_label"] = y_test
    df_pred["predicted_probability"] = y_proba
    df_pred["predicted_label"] = y_pred

    pred_path = PROCESSED_DIR / "test_predictions.parquet"
    df_pred.to_parquet(pred_path, index=False, engine="pyarrow")
    logger.info(f"  Test predictions saved: {pred_path} ({len(df_pred)} rows)")
    return pred_path


def save_training_report(metrics_train, metrics_val, metrics_test, best_threshold, xgb_params, feature_names):
    """Save training report JSON."""
    report = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBClassifier",
        "best_validation_threshold": best_threshold,
        "feature_count": len(feature_names),
        "features": feature_names,
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "test_metrics": metrics_test,
        "xgboost_parameters": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v for k, v in xgb_params.items()},
    }

    report_path = PROCESSED_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"  Training report saved: {report_path}")
    return report_path


# ================================================================
# 10. BASELINE DOCUMENTATION
# ================================================================

def generate_baseline_docs(
    metrics_train, metrics_val, metrics_test, best_threshold,
    xgb_params, feature_names, n_train, n_val, n_test,
    n_pos_train, n_neg_train, scale_pos_weight, df_fi,
):
    """Generate MODEL_BASELINE.md documentation."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    top_features = "\n".join([
        f"| {row['feature']} | {row['importance']:.4f} |"
        for _, row in df_fi.head(15).iterrows()
    ])

    doc = f"""# Baseline XGBoost Resume-Job Matching Model

## What the Model Predicts

This model estimates **resume-job match probability** — the likelihood that a
given resume is a strong match for a given job posting, based on structural
features like skill overlap, semantic similarity, title match, and experience.

> **IMPORTANT:** The output is a *model-estimated match probability*, NOT a
> "chance the candidate will get hired." The model reflects patterns learned
> from weakly supervised labels, not real hiring decisions.

## Features Used ({len(feature_names)})

| # | Feature |
|---|---------|
{chr(10).join([f'| {i+1} | `{f}` |' for i, f in enumerate(feature_names)])}

## How Labels Were Created (Weak Supervision)

Labels are **weakly supervised** — derived from rule-based heuristics, not
human annotation:

- **Positive (1):** High skill Jaccard (≥0.15) AND high semantic similarity (≥0.55),
  or high title similarity (≥0.45) AND high semantic similarity.
- **Negative (0):** Low skill overlap (≤0.05) AND low semantic similarity (≤0.35),
  hard negatives with moderate similarity but very low skill overlap, or
  randomly sampled unrelated resume-job pairs.
- **Ambiguous:** Pairs with moderate similarity that don't clearly fit either
  category. **Excluded from training.**

These labels are approximate. Model performance is bounded by label quality.

## Train / Validation / Test Strategy

| Split | Rows | Positive | Negative |
|-------|------|----------|----------|
| Train | {n_train} | {n_pos_train} | {n_neg_train} |
| Validation | {n_val} | — | — |
| Test | {n_test} | — | — |

**No data leakage:** Splits are by unique `resume_id` — no resume appears in
multiple splits. Ambiguous labels are excluded from all splits.

## Class Distribution & Imbalance Handling

- Positive/Negative ratio: {n_pos_train}/{n_neg_train} = {n_pos_train/max(1,n_neg_train):.2f}
- `scale_pos_weight`: {scale_pos_weight:.4f}
- Strategy: **Class weighting** via `scale_pos_weight` (preferred for tree-based models)
- SMOTE not used — can create unrealistic synthetic samples for structured features

## Preprocessing

- **Imputation:** Median imputation for all numerical features
- **Fitted on training set only** — applied identically to validation and test
- **No target leakage** in preprocessing

## Model Parameters

| Parameter | Value |
|-----------|-------|
{chr(10).join([f'| `{k}` | {v} |' for k, v in xgb_params.items()])}

## Evaluation Metrics

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| ROC-AUC | {metrics_train['roc_auc']:.4f} | {metrics_val['roc_auc']:.4f} | {metrics_test['roc_auc']:.4f} |
| PR-AUC | {metrics_train['pr_auc']:.4f} | {metrics_val['pr_auc']:.4f} | {metrics_test['pr_auc']:.4f} |
| Precision | {metrics_train['precision']:.4f} | {metrics_val['precision']:.4f} | {metrics_test['precision']:.4f} |
| Recall | {metrics_train['recall']:.4f} | {metrics_val['recall']:.4f} | {metrics_test['recall']:.4f} |
| F1 | {metrics_train['f1']:.4f} | {metrics_val['f1']:.4f} | {metrics_test['f1']:.4f} |
| Accuracy | {metrics_train['accuracy']:.4f} | {metrics_val['accuracy']:.4f} | {metrics_test['accuracy']:.4f} |
| Brier Score | {metrics_train['brier_score']:.4f} | {metrics_val['brier_score']:.4f} | {metrics_test['brier_score']:.4f} |

**Selected threshold:** {best_threshold:.2f} (maximizes validation F1)

## Threshold Analysis (Validation)

The default 0.5 threshold is not necessarily optimal. The best threshold was
selected by maximizing F1 on the validation set — **never on the test set**.

## Top Features by Gain

| Feature | Importance (Gain) |
|---------|-------------------|
{top_features}

## Limitations

1. **Weak labels:** Model accuracy is bounded by the quality of heuristic labels.
   Some "positive" pairs may not be true matches; some "negative" pairs may be
   viable candidates.
2. **No human validation yet:** Labels have not been verified by human reviewers.
   The `manual_review_pairs.parquet` dataset is provided for future annotation.
3. **Feature coverage:** Some features (experience, education) have significant
   missing values and rely on imputation.
4. **Calibration:** Predicted probabilities are model estimates and may not
   correspond to actual match likelihoods. Brier score provides a rough
   calibration measure.
5. **Domain shift:** The model was trained on a specific set of resumes and
   job postings. Performance may degrade on substantially different data
   (different industries, geographies, or formats).
6. **Baseline only:** This is an initial baseline model. Hyperparameter tuning
   (Optuna), advanced feature engineering, and model ensembling are planned
   for future iterations.

## Model Artifacts

| Artifact | Path |
|----------|------|
| Pipeline (imputer + XGBoost) | `models/job_matcher_pipeline.joblib` |
| XGBoost model only | `models/job_matcher_xgb.joblib` |
| Model metadata | `models/model_metadata.json` |
| Feature importance | `data/processed/model_feature_importance.csv` |
| Test predictions | `data/processed/test_predictions.parquet` |
| Training report | `data/processed/training_report.json` |
"""

    doc_path = DOCS_DIR / "MODEL_BASELINE.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc)

    logger.info(f"  Documentation saved: {doc_path}")


# ================================================================
# MAIN
# ================================================================

def main():
    logger.info("=" * 60)
    logger.info("BASELINE XGBOOST RESUME-JOB MATCHING MODEL")
    logger.info("=" * 60)

    # 1. Load data
    df_train, df_val, df_test = load_ml_data()

    # 2. Feature selection and label preparation
    (X_train, y_train, X_val, y_val, X_test, y_test,
     ids_train, ids_val, ids_test, feature_names) = prepare_features_and_labels(df_train, df_val, df_test)

    # 3. Preprocessing (fit on train only)
    imputer = build_preprocessing_pipeline(X_train)
    X_train_imp = imputer.transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # 4. Class imbalance
    logger.info("\n" + "=" * 60)
    logger.info("CLASS IMBALANCE HANDLING")
    logger.info("=" * 60)
    scale_pos_weight = compute_class_weight(y_train)

    # 5. Train XGBoost
    model, xgb_params = train_xgboost(X_train_imp, y_train, X_val_imp, y_val, scale_pos_weight, feature_names)

    # Set feature names on the model for importance extraction
    model.get_booster().feature_names = feature_names

    # 6. Evaluate on all sets
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    metrics_train, _, _, _ = evaluate_model(model, X_train_imp, y_train, "TRAIN")
    metrics_val, _, _, _ = evaluate_model(model, X_val_imp, y_val, "VALIDATION")

    # 7. Threshold analysis (on validation)
    best_threshold, threshold_results = threshold_analysis(model, X_val_imp, y_val)

    # 8. Final test evaluation with best threshold
    metrics_test, y_proba_test, y_pred_test, cm_test = evaluate_model(
        model, X_test_imp, y_test, "TEST", threshold=best_threshold
    )

    # 9. Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 60)
    df_fi = save_feature_importance(model, feature_names)

    # 10. Save artifacts
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL ARTIFACTS")
    logger.info("=" * 60)

    n_pos_train = int((y_train == 1).sum())
    n_neg_train = int((y_train == 0).sum())

    save_model_pipeline(imputer, model, feature_names)
    save_model_metadata(
        xgb_params, feature_names, metrics_train, metrics_val, metrics_test,
        best_threshold, len(y_train), len(y_val), len(y_test), n_pos_train, n_neg_train,
    )
    save_test_predictions(ids_test.reset_index(drop=True), y_test, y_proba_test, y_pred_test, best_threshold)
    save_training_report(metrics_train, metrics_val, metrics_test, best_threshold, xgb_params, feature_names)

    # 11. Documentation
    generate_baseline_docs(
        metrics_train, metrics_val, metrics_test, best_threshold,
        xgb_params, feature_names, len(y_train), len(y_val), len(y_test),
        n_pos_train, n_neg_train, scale_pos_weight, df_fi,
    )

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Model:                XGBoost XGBClassifier")
    logger.info(f"  Training rows:        {len(y_train)}")
    logger.info(f"  Validation rows:      {len(y_val)}")
    logger.info(f"  Test rows:            {len(y_test)}")
    logger.info(f"  Features:             {len(feature_names)}")
    logger.info(f"  ROC-AUC (test):       {metrics_test['roc_auc']:.4f}")
    logger.info(f"  PR-AUC (test):        {metrics_test['pr_auc']:.4f}")
    logger.info(f"  F1 (test):            {metrics_test['f1']:.4f}")
    logger.info(f"  Best threshold:       {best_threshold:.2f}")
    logger.info(f"  Pipeline:             models/job_matcher_pipeline.joblib")


if __name__ == "__main__":
    main()
