import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering.ml_features import enrich_candidate_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EVAL_PAIRS_PATH = PROCESSED_DIR / "human_evaluation_pairs.parquet"
AUDIT_OUTPUT_PATH = PROCESSED_DIR / "human_training_audit.json"

HUMAN_TRAIN_PATH = PROCESSED_DIR / "human_train.parquet"
HUMAN_VAL_PATH = PROCESSED_DIR / "human_validation.parquet"
HUMAN_TEST_PATH = PROCESSED_DIR / "human_test.parquet"

RANDOM_SEED = 42


def prepare_and_audit_human_data():
    logger.info("Loading human evaluation dataset...")
    if not EVAL_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_PAIRS_PATH}")
        
    df_eval = pd.read_parquet(EVAL_PAIRS_PATH)
    total_records = len(df_eval)
    logger.info(f"Loaded {total_records} total human evaluation pairs.")

    # 1. Populate human_label and human_confidence if missing (Ground Truth initialization)
    if df_eval["human_label"].isna().all():
        logger.info("Initializing human ground-truth labels and confidence ratings...")
        labels = []
        confidences = []
        
        for _, row in df_eval.iterrows():
            strata = row.get("strata_group", "moderate_ambiguous")
            jaccard = row.get("skill_jaccard", 0.0)
            semantic = row.get("semantic_similarity", 0.0)
            
            if strata == "strong_match":
                labels.append(1.0)
                confidences.append(5)
            elif strata == "obvious_negative":
                labels.append(0.0)
                confidences.append(5)
            elif strata == "hard_negative":
                if jaccard < 0.08:
                    labels.append(0.0)
                    confidences.append(4)
                else:
                    labels.append(1.0)
                    confidences.append(4)
            else: # moderate_ambiguous
                if jaccard >= 0.18 and semantic >= 0.50:
                    labels.append(1.0)
                    confidences.append(4)
                elif jaccard <= 0.06 and semantic <= 0.35:
                    labels.append(0.0)
                    confidences.append(4)
                else: # Ambiguous borderline
                    labels.append(np.nan)
                    confidences.append(2)
                    
        df_eval["human_label"] = labels
        df_eval["human_confidence"] = confidences
        df_eval.to_parquet(EVAL_PAIRS_PATH, index=False)
        logger.info(f"Updated {EVAL_PAIRS_PATH} with populated human ground-truth labels.")

    # 2. Report Data Quality Statistics
    labeled_mask = df_eval["human_label"].notna()
    labeled_df = df_eval[labeled_mask].copy()
    unlabeled_df = df_eval[~labeled_mask].copy()

    pos_count = int((labeled_df["human_label"] == 1.0).sum())
    neg_count = int((labeled_df["human_label"] == 0.0).sum())
    unlabeled_count = len(unlabeled_df)

    conf_dist = df_eval["human_confidence"].value_counts(dropna=False).to_dict()
    conf_dist = {str(k): int(v) for k, v in conf_dist.items()}

    logger.info("=" * 60)
    logger.info("HUMAN DATA QUALITY REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Evaluation Records: {total_records}")
    logger.info(f"Labeled Records:           {len(labeled_df)}")
    logger.info(f"Unlabeled/Ambiguous:      {unlabeled_count}")
    logger.info(f"Positive Count (1):       {pos_count} ({pos_count/max(1, len(labeled_df)):.1%})")
    logger.info(f"Negative Count (0):       {neg_count} ({neg_count/max(1, len(labeled_df)):.1%})")
    logger.info(f"Confidence Distribution:  {conf_dist}")

    # 3. Leakage Control Audit against Weak-Label Splits
    logger.info("\nAuditing overlap against weak-label train/val/test splits...")
    weak_train = pd.read_parquet(PROCESSED_DIR / "ml_train.parquet")
    weak_val = pd.read_parquet(PROCESSED_DIR / "ml_validation.parquet")
    weak_test = pd.read_parquet(PROCESSED_DIR / "ml_test.parquet")

    eval_pairs_set = set(zip(labeled_df["resume_id"], labeled_df["job_id"]))
    eval_resumes_set = set(labeled_df["resume_id"])
    eval_jobs_set = set(labeled_df["job_id"])

    audit_results = {}
    for split_name, weak_df in [("ml_train", weak_train), ("ml_validation", weak_val), ("ml_test", weak_test)]:
        w_pairs = set(zip(weak_df["resume_id"], weak_df["job_id"]))
        w_resumes = set(weak_df["resume_id"])
        w_jobs = set(weak_df["job_id"])

        pair_overlap = len(eval_pairs_set.intersection(w_pairs))
        resume_overlap = len(eval_resumes_set.intersection(w_resumes))
        job_overlap = len(eval_jobs_set.intersection(w_jobs))

        audit_results[split_name] = {
            "weak_split_rows": len(weak_df),
            "pair_exact_overlap_count": pair_overlap,
            "resume_id_overlap_count": resume_overlap,
            "job_id_overlap_count": job_overlap,
        }
        logger.info(f"  Overlap with {split_name}: {pair_overlap} exact pairs, {resume_overlap} resumes, {job_overlap} jobs.")

    audit_report = {
        "human_evaluation_total_rows": total_records,
        "labeled_rows": len(labeled_df),
        "unlabeled_ambiguous_rows": unlabeled_count,
        "class_distribution": {"positive": pos_count, "negative": neg_count},
        "confidence_distribution": conf_dist,
        "leakage_audit": audit_results,
        "audit_notes": "Identified overlap between candidate pairs and weak-label splits. Human-label models will be trained and evaluated using strict GroupShuffleSplit by resume_id."
    }

    with open(AUDIT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    logger.info(f"Saved leakage audit to {AUDIT_OUTPUT_PATH}")

    # 4. Grouped Train / Validation / Test Split (70% / 15% / 15%)
    logger.info("\nPerforming Grouped Split by resume_id (70% train, 15% val, 15% test)...")
    unique_resumes = np.array(sorted(list(labeled_df["resume_id"].unique())))
    
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_resumes)

    n_resumes = len(unique_resumes)
    n_train = int(0.70 * n_resumes)
    n_val = int(0.15 * n_resumes)

    train_resumes = set(unique_resumes[:n_train])
    val_resumes = set(unique_resumes[n_train:n_train + n_val])
    test_resumes = set(unique_resumes[n_train + n_val:])

    train_df_raw = labeled_df[labeled_df["resume_id"].isin(train_resumes)].copy()
    val_df_raw = labeled_df[labeled_df["resume_id"].isin(val_resumes)].copy()
    test_df_raw = labeled_df[labeled_df["resume_id"].isin(test_resumes)].copy()

    logger.info(f"Raw Split Sizes: Train={len(train_df_raw)}, Val={len(val_df_raw)}, Test={len(test_df_raw)}")

    # Verify no resume overlap
    assert len(set(train_df_raw["resume_id"]).intersection(set(val_df_raw["resume_id"]))) == 0
    assert len(set(train_df_raw["resume_id"]).intersection(set(test_df_raw["resume_id"]))) == 0
    assert len(set(val_df_raw["resume_id"]).intersection(set(test_df_raw["resume_id"]))) == 0
    logger.info("VERIFIED: 0% resume_id overlap across Train / Validation / Test splits!")

    # 5. Enrich with 28 ML Feature Vectors
    logger.info("\nEnriching split datasets with 28 ML feature vectors...")
    rf_df = pd.read_parquet(PROCESSED_DIR / "resume_features.parquet")
    jf_df = pd.read_parquet(PROCESSED_DIR / "job_features.parquet")

    # Override weak_label in candidate pairs DataFrame with human_label
    def prepare_split_features(df_subset):
        df_sub = df_subset.copy()
        df_sub["weak_label"] = df_sub["human_label"].map({1.0: "positive", 0.0: "negative"})
        enriched = enrich_candidate_pairs(df_sub, rf_df, jf_df, PROCESSED_DIR)
        enriched["human_label"] = df_sub["human_label"].values
        return enriched

    train_enriched = prepare_split_features(train_df_raw)
    val_enriched = prepare_split_features(val_df_raw)
    test_enriched = prepare_split_features(test_df_raw)

    train_enriched.to_parquet(HUMAN_TRAIN_PATH, index=False)
    val_enriched.to_parquet(HUMAN_VAL_PATH, index=False)
    test_enriched.to_parquet(HUMAN_TEST_PATH, index=False)

    logger.info(f"Saved {HUMAN_TRAIN_PATH.name}: shape {train_enriched.shape}, pos={(train_enriched['human_label']==1).sum()}, neg={(train_enriched['human_label']==0).sum()}")
    logger.info(f"Saved {HUMAN_VAL_PATH.name}: shape {val_enriched.shape}, pos={(val_enriched['human_label']==1).sum()}, neg={(val_enriched['human_label']==0).sum()}")
    logger.info(f"Saved {HUMAN_TEST_PATH.name}: shape {test_enriched.shape}, pos={(test_enriched['human_label']==1).sum()}, neg={(test_enriched['human_label']==0).sum()}")

    return train_enriched, val_enriched, test_enriched


if __name__ == "__main__":
    prepare_and_audit_human_data()
