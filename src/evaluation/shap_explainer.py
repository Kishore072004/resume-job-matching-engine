import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import shap

from src.feature_engineering.ml_features import MODEL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "models" / "job_matcher_pipeline.joblib"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SHAP_CSV_OUTPUT = PROCESSED_DIR / "shap_global_importance.csv"
SHAP_JSON_OUTPUT = PROCESSED_DIR / "shap_feature_summary.json"
SHAP_SAMPLES_OUTPUT = PROCESSED_DIR / "shap_sample_explanations.json"


def load_model_and_pipeline(model_path: Union[str, Path] = MODEL_PATH):
    """Load trained Scikit-learn Pipeline containing imputer and XGBoost classifier."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model pipeline not found at {path}")
    pipeline = joblib.load(path)
    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]
    return pipeline, imputer, classifier


def get_tree_explainer(classifier: Any) -> shap.TreeExplainer:
    """Initialize SHAP TreeExplainer for the XGBoost classifier."""
    return shap.TreeExplainer(classifier)


def prepare_features(
    df: pd.DataFrame, imputer: Any, feature_cols: List[str] = MODEL_FEATURES
) -> pd.DataFrame:
    """Extract and impute numerical model features."""
    X_raw = df[feature_cols].copy()
    X_imputed = imputer.transform(X_raw)
    return pd.DataFrame(X_imputed, columns=feature_cols, index=df.index)


def explain_single_pair(
    row_features: pd.Series,
    pipeline: Any,
    explainer: shap.TreeExplainer,
    imputer: Any,
    resume_id: str = "UNKNOWN",
    job_id: str = "UNKNOWN",
    feature_cols: List[str] = MODEL_FEATURES
) -> Dict[str, Any]:
    """Generate local SHAP explanation for a single resume-job pair."""
    X_raw = pd.DataFrame([row_features[feature_cols]])
    X_imp = imputer.transform(X_raw)
    
    prob = float(pipeline.predict_proba(X_raw)[0][1])
    shap_vals = explainer.shap_values(X_imp)
    
    # Handle scalar or array expected value
    base_val = explainer.expected_value
    if isinstance(base_val, (np.ndarray, list)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    else:
        base_val = float(base_val)

    if isinstance(shap_vals, list):
        # binary output list
        vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    elif len(shap_vals.shape) == 2:
        vals = shap_vals[0]
    else:
        vals = shap_vals

    factors = []
    for col, f_val, s_val in zip(feature_cols, X_imp[0], vals):
        factors.append({
            "feature": col,
            "feature_value": round(float(f_val), 4),
            "shap_value": round(float(s_val), 4)
        })

    positive_factors = sorted([f for f in factors if f["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
    negative_factors = sorted([f for f in factors if f["shap_value"] < 0], key=lambda x: x["shap_value"])

    return {
        "resume_id": resume_id,
        "job_id": job_id,
        "match_probability": round(prob, 4),
        "base_value": round(base_val, 4),
        "top_positive_factors": positive_factors[:5],
        "top_negative_factors": negative_factors[:5],
        "shap_summary": {f["feature"]: f["shap_value"] for f in factors}
    }


def compute_global_shap_importance(
    df: pd.DataFrame,
    pipeline: Any,
    imputer: Any,
    classifier: Any,
    feature_cols: List[str] = MODEL_FEATURES,
    max_samples: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute global feature importances across a dataset sample using SHAP."""
    if len(df) > max_samples:
        sample_df = df.sample(n=max_samples, random_state=42)
    else:
        sample_df = df.copy()

    X_imp = prepare_features(sample_df, imputer, feature_cols)
    explainer = get_tree_explainer(classifier)
    shap_matrix = explainer.shap_values(X_imp.values)
    
    if isinstance(shap_matrix, list):
        shap_matrix = shap_matrix[1] if len(shap_matrix) > 1 else shap_matrix[0]

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    mean_shap = shap_matrix.mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    summary_dict = dict(zip(importance_df["feature"], importance_df["mean_abs_shap"]))
    return importance_df, summary_dict


def run_shap_analysis(
    candidate_test_path: Union[str, Path] = PROCESSED_DIR / "ml_test.parquet"
):
    """Run end-to-end SHAP analysis and export global & sample artifacts."""
    logger.info("Loading model and test dataset for SHAP analysis...")
    pipeline, imputer, classifier = load_model_and_pipeline()
    explainer = get_tree_explainer(classifier)

    df_test = pd.read_parquet(candidate_test_path)
    logger.info(f"Loaded {len(df_test)} test records.")

    # 1. Global Importance
    importance_df, summary_dict = compute_global_shap_importance(
        df_test, pipeline, imputer, classifier
    )
    
    importance_df.to_csv(SHAP_CSV_OUTPUT, index=False)
    logger.info(f"Saved global SHAP importance to {SHAP_CSV_OUTPUT}")

    with open(SHAP_JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Saved global SHAP summary JSON to {SHAP_JSON_OUTPUT}")

    # 2. Sample Explanations (25 diverse samples)
    sample_df = df_test.sample(n=min(25, len(df_test)), random_state=42).reset_index(drop=True)
    sample_explanations = []

    for idx, row in sample_df.iterrows():
        rid = str(row.get("resume_id", f"RES_{idx}"))
        jid = str(row.get("job_id", f"JOB_{idx}"))
        exp = explain_single_pair(row, pipeline, explainer, imputer, resume_id=rid, job_id=jid)
        sample_explanations.append(exp)

    with open(SHAP_SAMPLES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(sample_explanations, f, indent=2)
    logger.info(f"Saved {len(sample_explanations)} sample explanations to {SHAP_SAMPLES_OUTPUT}")


if __name__ == "__main__":
    run_shap_analysis()
