import json
import logging
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("PREPARING HUMAN EVALUATION DATASET V2")
    logger.info("=" * 60)

    processed_dir = project_root / "data" / "processed"
    
    # 1. Load V2 features and human labels
    features_path = processed_dir / "ml_features_v2.parquet"
    labels_path = processed_dir / "human_evaluation_pairs.parquet"
    
    if not features_path.exists():
        logger.error(f"Cannot find {features_path}")
        return
        
    if not labels_path.exists():
        logger.error(f"Cannot find {labels_path}")
        return
        
    logger.info(f"Loading V2 features from {features_path}...")
    df_features = pd.read_parquet(features_path)
    
    logger.info(f"Loading human labels from {labels_path}...")
    df_labels = pd.read_parquet(labels_path)
    
    # 2. Join
    # human_evaluation_pairs.parquet should have resume_id, job_id, human_label
    df_joined = pd.merge(
        df_labels[["resume_id", "job_id", "human_label"]],
        df_features,
        on=["resume_id", "job_id"],
        how="inner"
    )
    
    logger.info(f"Joined dataset size: {len(df_joined)}")
    
    # Ensure label column exists
    df_joined["label"] = df_joined["human_label"]
    
    # 3. Grouped Train/Val/Test Split (70/15/15)
    logger.info("Splitting dataset grouped by resume_id to prevent leakage...")
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    
    train_idx, temp_idx = next(gss1.split(df_joined, groups=df_joined['resume_id']))
    df_train = df_joined.iloc[train_idx].copy()
    df_temp = df_joined.iloc[temp_idx].copy()
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['resume_id']))
    
    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()
    
    # 4. Save and report
    train_path = processed_dir / "human_train_v2.parquet"
    val_path = processed_dir / "human_validation_v2.parquet"
    test_path = processed_dir / "human_test_v2.parquet"
    
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)
    
    logger.info(f"Train size: {len(df_train)}")
    logger.info(f"Validation size: {len(df_val)}")
    logger.info(f"Test size: {len(df_test)}")
    
    # Check for leakage
    train_res = set(df_train["resume_id"])
    val_res = set(df_val["resume_id"])
    test_res = set(df_test["resume_id"])
    
    leak1 = train_res & val_res
    leak2 = train_res & test_res
    leak3 = val_res & test_res
    
    if leak1 or leak2 or leak3:
        logger.error("DATA LEAKAGE DETECTED across splits!")
    else:
        logger.info("No resume ID leakage detected across splits.")
        
    # Generate dataset fingerprint (Hash)
    import hashlib
    def hash_df(df):
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
        
    fingerprint = {
        "human_train_v2_hash": hash_df(df_train),
        "human_validation_v2_hash": hash_df(df_val),
        "human_test_v2_hash": hash_df(df_test),
        "total_rows": len(df_joined),
        "train_rows": len(df_train),
        "validation_rows": len(df_val),
        "test_rows": len(df_test)
    }
    
    fingerprint_path = processed_dir / "dataset_v2_fingerprint.json"
    with open(fingerprint_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    logger.info(f"Dataset fingerprint saved to {fingerprint_path}")

if __name__ == "__main__":
    main()
