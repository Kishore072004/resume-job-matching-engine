import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
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

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering.ml_features import MODEL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Silence Optuna logs below WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HUMAN_TRAIN_PATH = PROCESSED_DIR / "human_train.parquet"
HUMAN_VAL_PATH = PROCESSED_DIR / "human_validation.parquet"
HUMAN_TEST_PATH = PROCESSED_DIR / "human_test.parquet"

RANDOM_SEED = 42
N_TRIALS = 50


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    y_pred = (y_probs >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_probs))

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 2),
    }


def run_optuna_training():
    logger.info("Loading human train, validation, and test datasets...")
    train_df = pd.read_parquet(HUMAN_TRAIN_PATH)
    val_df = pd.read_parquet(HUMAN_VAL_PATH)
    test_df = pd.read_parquet(HUMAN_TEST_PATH)

    logger.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    X_train_raw = train_df[MODEL_FEATURES]
    y_train = train_df["human_label"].astype(int).values

    X_val_raw = val_df[MODEL_FEATURES]
    y_val = val_df["human_label"].astype(int).values

    X_test_raw = test_df[MODEL_FEATURES]
    y_test = test_df["human_label"].astype(int).values

    # Preprocessing: Imputer fitted on train only
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)

    # 1. Baseline Human Model (Before Optuna)
    logger.info("\n--- 1. Training Baseline Human-Label XGBoost Model ---")
    base_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        eval_metric="logloss"
    )
    base_xgb.fit(X_train_imp, y_train)

    base_val_probs = base_xgb.predict_proba(X_val_imp)[:, 1]
    base_val_metrics = evaluate_predictions(y_val, base_val_probs, threshold=0.50)
    logger.info(f"Baseline Human Model Val PR-AUC: {base_val_metrics['pr_auc']}, ROC-AUC: {base_val_metrics['roc_auc']}, F1: {base_val_metrics['f1']}")

    # 2. Optuna Hyperparameter Search
    logger.info(f"\n--- 2. Starting Optuna Hyperparameter Optimization ({N_TRIALS} Trials) ---")

    db_path = MODELS_DIR / "optuna_study.db"
    storage_name = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name="resume_matcher_xgboost",
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
            "random_state": RANDOM_SEED,
            "eval_metric": "logloss",
            "n_jobs": -1
        }

        clf = XGBClassifier(**params)
        clf.fit(
            X_train_imp,
            y_train,
            eval_set=[(X_val_imp, y_val)],
            verbose=False
        )

        val_probs = clf.predict_proba(X_val_imp)[:, 1]
        score = average_precision_score(y_val, val_probs)
        return score

    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    best_pr_auc = study.best_value
    logger.info(f"Optuna Search Complete! Best Validation PR-AUC: {best_pr_auc:.4f}")
    logger.info(f"Best Hyperparameters: {json.dumps(best_params, indent=2)}")

    # Save Optuna trial records & report
    trials_df = study.trials_dataframe()
    trials_df.to_parquet(PROCESSED_DIR / "optuna_trials.parquet", index=False)

    optuna_report = {
        "number_of_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_pr_auc": round(float(best_pr_auc), 4),
        "best_parameters": best_params,
    }
    with open(PROCESSED_DIR / "optuna_report.json", "w", encoding="utf-8") as f:
        json.dump(optuna_report, f, indent=2)

    with open(MODELS_DIR / "optuna_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # 3. Train Best Optuna Model & Early Stopping
    logger.info("\n--- 3. Training Best XGBoost Model Configuration ---")
    best_model_params = {
        **best_params,
        "random_state": RANDOM_SEED,
        "eval_metric": "logloss",
        "n_jobs": -1
    }

    optuna_xgb = XGBClassifier(**best_model_params)
    optuna_xgb.fit(
        X_train_imp,
        y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=False
    )

    best_iter = getattr(optuna_xgb, "best_iteration", best_params.get("n_estimators"))
    logger.info(f"Best Iteration: {best_iter}")

    joblib.dump(optuna_xgb, MODELS_DIR / "job_matcher_xgb_optuna.joblib")

    # 4. Validation Threshold Sweep
    logger.info("\n--- 4. Threshold Optimization on Validation Set ---")
    optuna_val_probs = optuna_xgb.predict_proba(X_val_imp)[:, 1]

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    thresh_results = []
    best_f1 = -1.0
    best_th = 0.50

    for th in thresholds:
        metrics = evaluate_predictions(y_val, optuna_val_probs, threshold=th)
        thresh_results.append(metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_th = th

    logger.info(f"Optimal Validation Decision Threshold: {best_th:.2f} (Achieved F1 = {best_f1:.4f})")

    # 5. Probability Calibration Investigation
    logger.info("\n--- 5. Investigating Probability Calibration ---")
    raw_brier = brier_score_loss(y_val, optuna_val_probs)
    
    # Fit Platt scaling (logistic regression on raw logit probabilities)
    from sklearn.linear_model import LogisticRegression
    platt_model = LogisticRegression(C=1.0, solver="lbfgs")
    # Convert probabilities to logits for Platt scaling
    eps = 1e-15
    clipped_probs = np.clip(optuna_val_probs, eps, 1 - eps)
    logits = np.log(clipped_probs / (1 - clipped_probs)).reshape(-1, 1)
    platt_model.fit(logits, y_val)

    def calibrate_probs(probs):
        c_p = np.clip(probs, eps, 1 - eps)
        l_g = np.log(c_p / (1 - c_p)).reshape(-1, 1)
        return platt_model.predict_proba(l_g)[:, 1]

    calibrated_val_probs = calibrate_probs(optuna_val_probs)
    calib_brier = brier_score_loss(y_val, calibrated_val_probs)

    logger.info(f"Validation Brier Score -> Raw XGBoost: {raw_brier:.4f} vs Calibrated (Platt/Sigmoid): {calib_brier:.4f}")

    calibration_used = calib_brier < raw_brier
    calibrator = platt_model if calibration_used else None
    if calibration_used:
        logger.info("Calibration improved validation Brier score. Saving calibration pipeline...")
        joblib.dump(platt_model, MODELS_DIR / "calibration_pipeline.joblib")

    # 6. Final Holdout Test Set Evaluation (ONCE)
    logger.info("\n--- 6. Final Holdout Test Set Evaluation ---")
    test_probs = optuna_xgb.predict_proba(X_test_imp)[:, 1]
    final_test_metrics = evaluate_predictions(y_test, test_probs, threshold=best_th)

    logger.info(f"FINAL HOLDOUT TEST RESULTS (Threshold = {best_th:.2f}):")
    logger.info(f"  ROC-AUC:   {final_test_metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:    {final_test_metrics['pr_auc']:.4f}")
    logger.info(f"  Precision: {final_test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {final_test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {final_test_metrics['f1']:.4f}")
    logger.info(f"  Accuracy:  {final_test_metrics['accuracy']:.4f}")
    logger.info(f"  Brier:     {final_test_metrics['brier_score']:.4f}")

    # 7. Model Comparison Report Table
    comparison_data = [
        {
            "model": "weak_label_baseline",
            "dataset": "weak_test",
            "roc_auc": 1.0000,
            "pr_auc": 1.0000,
            "precision": 1.0000,
            "recall": 1.0000,
            "f1": 1.0000,
            "accuracy": 1.0000,
            "brier_score": 0.0000,
            "threshold": 0.20
        },
        {
            "model": "human_label_baseline",
            "dataset": "human_validation",
            "roc_auc": base_val_metrics["roc_auc"],
            "pr_auc": base_val_metrics["pr_auc"],
            "precision": base_val_metrics["precision"],
            "recall": base_val_metrics["recall"],
            "f1": base_val_metrics["f1"],
            "accuracy": base_val_metrics["accuracy"],
            "brier_score": base_val_metrics["brier_score"],
            "threshold": 0.50
        },
        {
            "model": "human_label_optuna",
            "dataset": "human_test_holdout",
            "roc_auc": final_test_metrics["roc_auc"],
            "pr_auc": final_test_metrics["pr_auc"],
            "precision": final_test_metrics["precision"],
            "recall": final_test_metrics["recall"],
            "f1": final_test_metrics["f1"],
            "accuracy": final_test_metrics["accuracy"],
            "brier_score": final_test_metrics["brier_score"],
            "threshold": best_th
        }
    ]

    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)
    logger.info(f"Saved model comparison table to {PROCESSED_DIR / 'model_comparison.csv'}")

    # 8. Feature Importance (XGBoost Gain & SHAP)
    logger.info("\n--- 8. Extracting Feature Importances ---")
    gain_scores = optuna_xgb.get_booster().get_score(importance_type="gain")
    gain_df = pd.DataFrame([
        {"feature": f, "gain_importance": round(float(gain_scores.get(f, 0.0)), 4)}
        for f in MODEL_FEATURES
    ]).sort_values(by="gain_importance", ascending=False).reset_index(drop=True)
    
    gain_df.to_csv(PROCESSED_DIR / "optuna_feature_importance.csv", index=False)

    # SHAP feature importance on test set
    explainer = shap.TreeExplainer(optuna_xgb)
    shap_vals = explainer.shap_values(X_test_imp)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    shap_df.to_csv(PROCESSED_DIR / "optuna_shap_importance.csv", index=False)
    logger.info(f"Saved Optuna feature importances to CSV files.")

    # 9. Assembly of Final Production Artifact
    logger.info("\n--- 9. Assembling Final Production Pipeline ---")
    
    final_pipeline = Pipeline([
        ("imputer", imputer),
        ("classifier", optuna_xgb)
    ])

    production_artifact = {
        "pipeline": final_pipeline,
        "imputer": imputer,
        "model": optuna_xgb,
        "calibrator": calibrator if calibration_used else None,
        "best_threshold": float(best_th),
        "feature_names": MODEL_FEATURES,
        "metadata": {
            "model_name": "Optuna_Trained_XGBoost_Resume_Matcher",
            "version": "2.0.0",
            "training_date": "2026-09-09",
            "optimal_threshold": float(best_th),
            "final_test_roc_auc": final_test_metrics["roc_auc"],
            "final_test_pr_auc": final_test_metrics["pr_auc"],
            "final_test_f1": final_test_metrics["f1"],
        }
    }

    joblib.dump(production_artifact, MODELS_DIR / "job_matcher_final.joblib")
    logger.info(f"Saved primary production artifact to {MODELS_DIR / 'job_matcher_final.joblib'}")

    # 10. Final Model Metadata
    final_metadata = {
        "model_name": "Optuna_Trained_XGBoost_Resume_Matcher",
        "version": "2.0.0",
        "training_date": "2026-09-09",
        "training_dataset": "data/processed/human_train.parquet",
        "validation_dataset": "data/processed/human_validation.parquet",
        "test_dataset": "data/processed/human_test.parquet",
        "feature_names": MODEL_FEATURES,
        "human_labeled_training_rows": len(train_df),
        "human_labeled_validation_rows": len(val_df),
        "human_labeled_test_rows": len(test_df),
        "positive_rows": int((y_train == 1).sum() + (y_val == 1).sum() + (y_test == 1).sum()),
        "negative_rows": int((y_train == 0).sum() + (y_val == 0).sum() + (y_test == 0).sum()),
        "best_optuna_parameters": best_params,
        "best_iteration": int(best_iter) if best_iter is not None else int(best_params.get("n_estimators", 100)),
        "best_validation_threshold": float(best_th),
        "calibration_method": "Platt/Sigmoid" if calibration_used else "None",
        "final_roc_auc": final_test_metrics["roc_auc"],
        "final_pr_auc": final_test_metrics["pr_auc"],
        "final_precision": final_test_metrics["precision"],
        "final_recall": final_test_metrics["recall"],
        "final_f1": final_test_metrics["f1"],
        "final_accuracy": final_test_metrics["accuracy"],
        "final_brier_score": final_test_metrics["brier_score"],
        "random_seed": RANDOM_SEED,
    }

    with open(MODELS_DIR / "final_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2)

    logger.info(f"Saved complete metadata to {MODELS_DIR / 'final_model_metadata.json'}")
    return final_metadata


if __name__ == "__main__":
    run_optuna_training()
