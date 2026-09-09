"""
Train V2 XGBoost Model with Corrected Application Skill Layer.

This script:
  1. Loads V2 human-labeled train/validation/test splits.
  2. Trains a baseline XGBoost (matching V1 config for fair comparison).
  3. Compares V1 vs V2 baseline metrics.
  4. Runs Optuna (50 trials) to optimize PR-AUC on validation.
  5. Performs threshold analysis on validation (0.10..0.90).
  6. Evaluates calibration (Platt scaling).
  7. Final holdout evaluation on test set (ONCE).
  8. SHAP analysis and feature importance.
  9. Saves all model artifacts with version 3.0.0.
"""

import hashlib
import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering.ml_features import MODEL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_TRIALS = 50
MODEL_VERSION = "3.0.0"


# ================================================================
# HELPERS
# ================================================================

def evaluate(y_true, y_probs, threshold=0.50):
    y_pred = (y_probs >= threshold).astype(int)
    return {
        "roc_auc":    round(float(roc_auc_score(y_true, y_probs)), 4) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc":     round(float(average_precision_score(y_true, y_probs)), 4) if len(np.unique(y_true)) > 1 else 0.0,
        "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":   round(float(accuracy_score(y_true, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_probs)), 4),
        "threshold":  round(threshold, 2),
    }


def main():
    logger.info("=" * 70)
    logger.info("V2 MODEL TRAINING — CORRECTED APPLICATION SKILL LAYER")
    logger.info(f"MODEL VERSION: {MODEL_VERSION}")
    logger.info("=" * 70)

    # ── 1. LOAD DATA ─────────────────────────────────────────────
    train_path = PROCESSED_DIR / "human_train_v2.parquet"
    val_path   = PROCESSED_DIR / "human_validation_v2.parquet"
    test_path  = PROCESSED_DIR / "human_test_v2.parquet"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            logger.error(f"Missing: {p}")
            return

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Use human_label as target (joined as 'label' by prep script)
    label_col = "label" if "label" in train_df.columns else "human_label"

    avail_features = [f for f in MODEL_FEATURES if f in train_df.columns]
    missing = [f for f in MODEL_FEATURES if f not in train_df.columns]
    if missing:
        logger.warning(f"Missing features (filled with NaN): {missing}")
        for f in missing:
            for df in [train_df, val_df, test_df]:
                df[f] = np.nan
        avail_features = MODEL_FEATURES

    logger.info(f"Features ({len(avail_features)}): {avail_features}")

    # Drop rows with NaN labels
    train_df = train_df.dropna(subset=[label_col]).copy()
    val_df   = val_df.dropna(subset=[label_col]).copy()
    test_df  = test_df.dropna(subset=[label_col]).copy()
    logger.info(f"After dropping NaN labels — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train_raw = train_df[avail_features].copy()
    y_train     = train_df[label_col].astype(int).values
    X_val_raw   = val_df[avail_features].copy()
    y_val       = val_df[label_col].astype(int).values
    X_test_raw  = test_df[avail_features].copy()
    y_test      = test_df[label_col].astype(int).values

    n_pos_train = int((y_train == 1).sum())
    n_neg_train = int((y_train == 0).sum())
    logger.info(f"Train labels — Positive: {n_pos_train}, Negative: {n_neg_train}")

    # ── 2. PREPROCESSING ─────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_val   = imputer.transform(X_val_raw)
    X_test  = imputer.transform(X_test_raw)

    # ── 3. V2 BASELINE — matching V1 config for fair comparison ──
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: V2 BASELINE MODEL (matching V1 config)")
    logger.info("=" * 70)

    scale_pw = n_neg_train / max(1, n_pos_train)

    baseline_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "scale_pos_weight": scale_pw,
        "random_state": RANDOM_SEED,
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": -1,
    }

    baseline_xgb = XGBClassifier(**baseline_params)
    baseline_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    base_val_probs  = baseline_xgb.predict_proba(X_val)[:, 1]
    base_test_probs = baseline_xgb.predict_proba(X_test)[:, 1]

    base_val_metrics  = evaluate(y_val,  base_val_probs,  0.50)
    base_test_metrics = evaluate(y_test, base_test_probs, 0.50)

    logger.info(f"V2 Baseline Val  — ROC-AUC: {base_val_metrics['roc_auc']}, PR-AUC: {base_val_metrics['pr_auc']}, F1: {base_val_metrics['f1']}")
    logger.info(f"V2 Baseline Test — ROC-AUC: {base_test_metrics['roc_auc']}, PR-AUC: {base_test_metrics['pr_auc']}, F1: {base_test_metrics['f1']}")

    # ── 4. V1 vs V2 COMPARISON (Step 9) ──────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 9: V1 vs V2 BASELINE COMPARISON")
    logger.info("=" * 70)

    # Load V1 metadata for comparison
    v1_meta_path = MODELS_DIR / "final_model_metadata.json"
    if v1_meta_path.exists():
        with open(v1_meta_path) as f:
            v1_meta = json.load(f)
        v1_metrics = {
            "roc_auc": v1_meta.get("final_roc_auc", 0),
            "pr_auc":  v1_meta.get("final_pr_auc", 0),
            "f1":      v1_meta.get("final_f1", 0),
            "precision": v1_meta.get("final_precision", 0),
            "recall":  v1_meta.get("final_recall", 0),
            "accuracy": v1_meta.get("final_accuracy", 0),
            "brier_score": v1_meta.get("final_brier_score", 0),
        }
    else:
        v1_metrics = {k: 0.0 for k in ["roc_auc","pr_auc","f1","precision","recall","accuracy","brier_score"]}

    comp_rows = []
    for metric_name in ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "brier_score"]:
        comp_rows.append({
            "metric":   metric_name,
            "v1_value": v1_metrics[metric_name],
            "v2_baseline_val": base_val_metrics[metric_name],
            "v2_baseline_test": base_test_metrics[metric_name],
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_path = PROCESSED_DIR / "model_v1_v2_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"V1 vs V2 baseline comparison saved to {comp_path}")
    print(comp_df.to_string(index=False))

    # ── 5. OPTUNA (Step 10) ───────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"STEP 10: OPTUNA HYPERPARAMETER SEARCH ({N_TRIALS} trials)")
    logger.info("=" * 70)

    db_path = MODELS_DIR / "optuna_study_v2.db"
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="v2_resume_matcher",
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 400, step=25),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":           trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pw,
            "random_state": RANDOM_SEED,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "verbosity": 0,
        }
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        probs = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, probs)

    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    logger.info(f"Best Optuna PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {json.dumps(best_params, indent=2)}")

    # Save Optuna report
    trials_df = study.trials_dataframe()
    trials_df.to_parquet(PROCESSED_DIR / "optuna_trials_v2.parquet", index=False)

    optuna_report = {
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_pr_auc": round(float(study.best_value), 4),
        "best_params": best_params,
    }
    with open(PROCESSED_DIR / "optuna_report_v2.json", "w") as f:
        json.dump(optuna_report, f, indent=2)

    # Retrain best model
    best_full_params = {
        **best_params,
        "scale_pos_weight": scale_pw,
        "random_state": RANDOM_SEED,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "verbosity": 0,
    }
    optuna_xgb = XGBClassifier(**best_full_params)
    optuna_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ── 6. THRESHOLD ANALYSIS (Step 11) ───────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 11: THRESHOLD ANALYSIS (Validation)")
    logger.info("=" * 70)

    opt_val_probs = optuna_xgb.predict_proba(X_val)[:, 1]

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]
    best_f1 = -1.0
    best_th = 0.50
    best_recall_th = 0.50
    best_balanced_th = 0.50
    best_recall_val = 0.0

    thresh_results = []
    for th in thresholds:
        m = evaluate(y_val, opt_val_probs, th)
        thresh_results.append(m)
        logger.info(f"  T={th:.2f} | P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
        if m["recall"] > best_recall_val:
            best_recall_val = m["recall"]
            best_recall_th = th
        # balanced = closest to equal precision and recall
        if abs(m["precision"] - m["recall"]) < abs(
            evaluate(y_val, opt_val_probs, best_balanced_th)["precision"] -
            evaluate(y_val, opt_val_probs, best_balanced_th)["recall"]
        ):
            best_balanced_th = th

    logger.info(f"\nBest F1 threshold:       {best_th:.2f} (F1={best_f1:.4f})")
    logger.info(f"High-recall threshold:   {best_recall_th:.2f}")
    logger.info(f"Balanced threshold:      {best_balanced_th:.2f}")

    pd.DataFrame(thresh_results).to_csv(PROCESSED_DIR / "threshold_analysis_v2.csv", index=False)

    # ── 7. CALIBRATION (Step 12) ──────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 12: PROBABILITY CALIBRATION")
    logger.info("=" * 70)

    raw_brier = brier_score_loss(y_val, opt_val_probs)

    eps = 1e-15
    clipped = np.clip(opt_val_probs, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(logits, y_val)

    def calibrate(probs):
        c = np.clip(probs, eps, 1 - eps)
        l = np.log(c / (1 - c)).reshape(-1, 1)
        return platt.predict_proba(l)[:, 1]

    calib_val_probs = calibrate(opt_val_probs)
    calib_brier = brier_score_loss(y_val, calib_val_probs)

    logger.info(f"Uncalibrated Brier: {raw_brier:.4f}")
    logger.info(f"Calibrated   Brier: {calib_brier:.4f}")

    use_calibration = calib_brier < raw_brier
    calibrator = platt if use_calibration else None
    logger.info(f"Calibration {'ACCEPTED' if use_calibration else 'REJECTED'}")

    # ── 8. FINAL HOLDOUT (Step 13) ────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 13: FINAL HOLDOUT TEST EVALUATION")
    logger.info("=" * 70)

    test_probs = optuna_xgb.predict_proba(X_test)[:, 1]
    final_metrics = evaluate(y_test, test_probs, best_th)

    logger.info(f"FINAL HOLDOUT (threshold={best_th:.2f}):")
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v}")

    # ── 9. SHAP V2 (Step 14) ─────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 14: SHAP ANALYSIS")
    logger.info("=" * 70)

    explainer = shap.TreeExplainer(optuna_xgb)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": avail_features,
        "mean_abs_shap": [round(float(v), 6) for v in mean_abs_shap],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_path = PROCESSED_DIR / "shap_v2_importance.csv"
    shap_df.to_csv(shap_path, index=False)
    logger.info(f"SHAP V2 saved to {shap_path}")
    print(shap_df.head(10).to_string(index=False))

    # V1 vs V2 SHAP comparison
    v1_shap_path = PROCESSED_DIR / "optuna_shap_importance.csv"
    if v1_shap_path.exists():
        v1_shap = pd.read_csv(v1_shap_path)
        shap_comp = pd.merge(
            v1_shap.rename(columns={"mean_abs_shap": "v1_shap"}),
            shap_df.rename(columns={"mean_abs_shap": "v2_shap"}),
            on="feature", how="outer"
        ).fillna(0)
        shap_comp["shap_change"] = shap_comp["v2_shap"] - shap_comp["v1_shap"]
        shap_comp = shap_comp.sort_values("v2_shap", ascending=False)
        shap_comp_path = PROCESSED_DIR / "shap_v1_v2_comparison.csv"
        shap_comp.to_csv(shap_comp_path, index=False)
        logger.info(f"SHAP V1 vs V2 comparison saved to {shap_comp_path}")

    # ── 10. FEATURE IMPORTANCE (Step 15) ──────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 15: FEATURE IMPORTANCE (XGBoost Gain)")
    logger.info("=" * 70)

    optuna_xgb.get_booster().feature_names = avail_features
    gain = optuna_xgb.get_booster().get_score(importance_type="gain")
    fi_df = pd.DataFrame([
        {"feature": f, "gain_importance": round(float(gain.get(f, 0.0)), 4)}
        for f in avail_features
    ]).sort_values("gain_importance", ascending=False).reset_index(drop=True)

    fi_path = PROCESSED_DIR / "feature_importance_v2.csv"
    fi_df.to_csv(fi_path, index=False)
    logger.info(f"Feature importance saved to {fi_path}")
    print(fi_df.head(10).to_string(index=False))

    # ── 11. SAVE MODEL ARTIFACTS (Steps 16-18) ───────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEPS 16-18: SAVING MODEL ARTIFACTS")
    logger.info("=" * 70)

    # Save XGBoost model
    joblib.dump(optuna_xgb, MODELS_DIR / "job_matcher_v2.joblib")

    # Save pipeline
    v2_pipeline = Pipeline([("imputer", imputer), ("classifier", optuna_xgb)])
    
    # Save full production artifact dictionary compatible with JobMatchEngine
    production_artifact = {
        "pipeline": v2_pipeline,
        "imputer": imputer,
        "model": optuna_xgb,
        "calibrator": calibrator if use_calibration else None,
        "best_threshold": float(best_th),
        "feature_names": avail_features,
        "metadata": {
            "model_name": "Optuna_Trained_XGBoost_Resume_Matcher_V2",
            "version": MODEL_VERSION,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "optimal_threshold": float(best_th),
            "final_test_roc_auc": float(final_metrics["roc_auc"]),
            "final_test_pr_auc": float(final_metrics["pr_auc"]),
            "final_test_f1": float(final_metrics["f1"]),
        }
    }
    joblib.dump(production_artifact, MODELS_DIR / "job_matcher_final.joblib")
    joblib.dump(production_artifact, MODELS_DIR / "job_matcher_v2_pipeline.joblib")

    if use_calibration:
        joblib.dump(calibrator, MODELS_DIR / "calibration_v2.joblib")

    # Metadata
    metadata = {
        "model_version": MODEL_VERSION,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "features": avail_features,
        "dataset_version": "v2_corrected_application_skills",
        "human_label_counts": {
            "train_pos": n_pos_train, "train_neg": n_neg_train,
            "val_total": len(val_df), "test_total": len(test_df),
        },
        "best_optuna_parameters": best_params,
        "threshold": float(best_th),
        "high_recall_threshold": float(best_recall_th),
        "balanced_threshold": float(best_balanced_th),
        "calibration": "Platt/Sigmoid" if use_calibration else "None",
        "calibration_brier_raw": round(raw_brier, 4),
        "calibration_brier_calibrated": round(calib_brier, 4),
        "validation_metrics": evaluate(y_val, opt_val_probs, best_th),
        "holdout_test_metrics": final_metrics,
        "random_seed": RANDOM_SEED,
    }

    with open(MODELS_DIR / "model_v2_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to models/model_v2_metadata.json")

    # Update V1 vs V2 comparison with Optuna results
    comp_rows_final = []
    for metric_name in ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "brier_score"]:
        comp_rows_final.append({
            "metric":   metric_name,
            "v1_noisy_skills": v1_metrics[metric_name],
            "v2_baseline": base_test_metrics[metric_name],
            "v2_optuna":   final_metrics[metric_name],
        })
    pd.DataFrame(comp_rows_final).to_csv(comp_path, index=False)

    # Dataset fingerprint
    def hash_df(df):
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    fingerprint = {
        "ml_features_v2_hash": hash_df(pd.read_parquet(PROCESSED_DIR / "ml_features_v2.parquet")),
        "human_train_v2_hash": hash_df(train_df),
        "human_val_v2_hash": hash_df(val_df),
        "human_test_v2_hash": hash_df(test_df),
    }
    with open(PROCESSED_DIR / "dataset_v2_fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)

    # ── SUMMARY ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("V2 TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Model Version:    {MODEL_VERSION}")
    logger.info(f"  Train rows:       {len(y_train)}")
    logger.info(f"  Val rows:         {len(y_val)}")
    logger.info(f"  Test rows:        {len(y_test)}")
    logger.info(f"  Features:         {len(avail_features)}")
    logger.info(f"  Threshold:        {best_th}")
    logger.info(f"  Holdout ROC-AUC:  {final_metrics['roc_auc']}")
    logger.info(f"  Holdout PR-AUC:   {final_metrics['pr_auc']}")
    logger.info(f"  Holdout F1:       {final_metrics['f1']}")
    logger.info(f"  Pipeline:         models/job_matcher_v2_pipeline.joblib")


if __name__ == "__main__":
    main()
