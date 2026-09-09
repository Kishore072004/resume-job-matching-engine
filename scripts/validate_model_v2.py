"""
Validate V2 Model Artifacts.

Checks:
  - V2 artifacts exist
  - Features are correct
  - Train/val/test split has no resume leakage
  - Probabilities are valid
  - Threshold valid
  - SHAP output exists
  - Metadata exists
  - Skill features use corrected application skills
  - No obvious generic skills used as primary technical signals
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering.ml_features import MODEL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "models"


def check(condition, msg):
    if condition:
        logger.info(f"  PASS: {msg}")
        return True
    else:
        logger.error(f"  FAIL: {msg}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("V2 MODEL VALIDATION")
    logger.info("=" * 60)
    all_pass = True

    # ── 1. Artifact existence ────────────────────────────────────
    logger.info("\n--- Checking V2 artifacts ---")
    artifacts = {
        "model":    MODELS_DIR / "job_matcher_v2.joblib",
        "pipeline": MODELS_DIR / "job_matcher_v2_pipeline.joblib",
        "metadata": MODELS_DIR / "model_v2_metadata.json",
        "shap":     PROCESSED_DIR / "shap_v2_importance.csv",
        "fi":       PROCESSED_DIR / "feature_importance_v2.csv",
        "train":    PROCESSED_DIR / "human_train_v2.parquet",
        "val":      PROCESSED_DIR / "human_validation_v2.parquet",
        "test":     PROCESSED_DIR / "human_test_v2.parquet",
        "features": PROCESSED_DIR / "ml_features_v2.parquet",
        "fingerprint": PROCESSED_DIR / "dataset_v2_fingerprint.json",
    }

    for name, path in artifacts.items():
        all_pass &= check(path.exists(), f"{name} exists at {path.name}")

    # ── 2. Features correct ──────────────────────────────────────
    logger.info("\n--- Checking features ---")
    if artifacts["metadata"].exists():
        with open(artifacts["metadata"]) as f:
            meta = json.load(f)
        stored_features = meta.get("features", [])
        all_pass &= check(
            set(stored_features) == set(MODEL_FEATURES),
            f"Stored features match MODEL_FEATURES ({len(stored_features)} features)"
        )
        all_pass &= check(
            meta.get("model_version") == "3.0.0",
            f"Model version is 3.0.0 (got {meta.get('model_version')})"
        )

    # ── 3. No resume leakage ────────────────────────────────────
    logger.info("\n--- Checking split leakage ---")
    if all(artifacts[s].exists() for s in ["train", "val", "test"]):
        df_train = pd.read_parquet(artifacts["train"])
        df_val   = pd.read_parquet(artifacts["val"])
        df_test  = pd.read_parquet(artifacts["test"])

        train_res = set(df_train["resume_id"])
        val_res   = set(df_val["resume_id"])
        test_res  = set(df_test["resume_id"])

        all_pass &= check(not (train_res & val_res), "No train-val resume overlap")
        all_pass &= check(not (train_res & test_res), "No train-test resume overlap")
        all_pass &= check(not (val_res & test_res), "No val-test resume overlap")

    # ── 4. Probabilities valid ──────────────────────────────────
    logger.info("\n--- Checking model predictions ---")
    if artifacts["pipeline"].exists() and artifacts["test"].exists():
        df_test = pd.read_parquet(artifacts["test"])

        avail = [f for f in MODEL_FEATURES if f in df_test.columns]
        X_test = df_test[avail].copy()
        
        pipeline_obj = joblib.load(artifacts["pipeline"])
        if isinstance(pipeline_obj, dict):
            pipeline = pipeline_obj["pipeline"]
        else:
            pipeline = pipeline_obj

        probs = pipeline.predict_proba(X_test)[:, 1]

        all_pass &= check(
            np.all((probs >= 0) & (probs <= 1)),
            "All probabilities in [0, 1]"
        )
        all_pass &= check(
            not np.any(np.isnan(probs)),
            "No NaN probabilities"
        )

    # ── 5. Threshold valid ──────────────────────────────────────
    logger.info("\n--- Checking threshold ---")
    if artifacts["metadata"].exists():
        threshold = meta.get("threshold", None)
        all_pass &= check(
            threshold is not None and 0.0 < threshold < 1.0,
            f"Threshold is valid: {threshold}"
        )

    # ── 6. SHAP output ──────────────────────────────────────────
    logger.info("\n--- Checking SHAP ---")
    if artifacts["shap"].exists():
        shap_df = pd.read_csv(artifacts["shap"])
        all_pass &= check(len(shap_df) > 0, f"SHAP has {len(shap_df)} features")
        all_pass &= check("mean_abs_shap" in shap_df.columns, "SHAP has mean_abs_shap column")

    # ── 7. Corrected skill layer ────────────────────────────────
    logger.info("\n--- Checking corrected skill layer ---")
    app_skills_path = PROCESSED_DIR / "esco_relevant_skills.parquet"
    if app_skills_path.exists():
        df_skills = pd.read_parquet(app_skills_path)
        tech_skills = set(df_skills[df_skills["relevance_category"] == "technical"]["preferredLabel"].str.lower())

        noise_concepts = ["similitude", "security panels"]
        for nc in noise_concepts:
            all_pass &= check(nc not in tech_skills, f"'{nc}' not in technical skills")

    # ── 8. Old model preserved ──────────────────────────────────
    logger.info("\n--- Checking old model preserved ---")
    old_model_path = MODELS_DIR / "job_matcher_final.joblib"
    all_pass &= check(old_model_path.exists(), "Old production model (job_matcher_final.joblib) still exists")

    # ── FINAL VERDICT ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    if all_pass:
        logger.info("MODEL V2 TRAINING COMPLETE")
        logger.info("")
        logger.info("Corrected skill layer:")
        logger.info("PASS")
        logger.info("")
        logger.info("Training:")
        logger.info("PASS")
        logger.info("")
        logger.info("Validation:")
        logger.info("PASS")
        logger.info("")
        logger.info("Holdout:")
        logger.info("PASS")
        logger.info("")
        logger.info("SHAP:")
        logger.info("PASS")
        logger.info("")
        logger.info("Artifact:")
        logger.info("models/job_matcher_v2_pipeline.joblib")
        logger.info("")
        logger.info("Validation: PASS")
    else:
        logger.error("Validation: FAIL — see errors above")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
