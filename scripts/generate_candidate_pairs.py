import json
import sys
from pathlib import Path
import pandas as pd

# Add src package to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.matching.candidate_generator import (
    load_config,
    retrieve_title_candidates,
    retrieve_skill_candidates,
    retrieve_semantic_candidates,
    combine_and_featurize_candidates,
    stratify_and_label_pairs,
    split_candidates_no_leakage,
    create_manual_review_sample,
)
from scripts.validate_candidate_pairs import validate_candidate_pairs


def main():
    config_path = project_root / "configs" / "candidate_generation.yaml"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)

    res_features_path = processed_dir / "resume_features.parquet"
    job_features_path = processed_dir / "job_features.parquet"

    print(f"Loading resume features from {res_features_path}...")
    df_resumes = pd.read_parquet(res_features_path)

    print(f"Loading job features from {job_features_path}...")
    df_jobs = pd.read_parquet(job_features_path)

    # 1. Strategy A: Title/Category matching
    df_title_cands = retrieve_title_candidates(
        df_resumes, df_jobs, max_jobs_per_resume=config.get("max_jobs_per_resume_per_strategy", 20)
    )

    # 2. Strategy B/C: Skill Overlap matching
    df_skill_cands = retrieve_skill_candidates(
        df_resumes, df_jobs, max_jobs_per_resume=config.get("max_jobs_per_resume_per_strategy", 20)
    )

    # 3. Strategy D: Semantic SentenceTransformers retrieval
    df_sem_cands = retrieve_semantic_candidates(
        df_resumes,
        df_jobs,
        top_k=config.get("semantic_top_k", 15),
        model_name=config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        batch_size=config.get("embedding_batch_size", 64),
    )

    # 4. Combine & Featurize Candidates
    df_pairs = combine_and_featurize_candidates(
        df_resumes, df_jobs, [df_title_cands, df_skill_cands, df_sem_cands], config
    )

    # 5. Stratify & Assign Weak Labels
    df_pairs = stratify_and_label_pairs(df_pairs, config)

    # Save master candidate_pairs.parquet
    pairs_parquet = processed_dir / "candidate_pairs.parquet"
    df_pairs.to_parquet(pairs_parquet, index=False, engine="pyarrow")

    # 6. Grouped Train/Val/Test Split (No Leakage)
    df_train, df_val, df_test = split_candidates_no_leakage(df_pairs, config)

    train_parquet = processed_dir / "candidate_train.parquet"
    val_parquet = processed_dir / "candidate_validation.parquet"
    test_parquet = processed_dir / "candidate_test.parquet"

    df_train.to_parquet(train_parquet, index=False, engine="pyarrow")
    df_val.to_parquet(val_parquet, index=False, engine="pyarrow")
    df_test.to_parquet(test_parquet, index=False, engine="pyarrow")

    # 7. Create Manual Review Dataset Sample
    df_review = create_manual_review_sample(df_pairs, df_resumes, df_jobs, sample_size=300)
    review_parquet = processed_dir / "manual_review_pairs.parquet"
    df_review.to_parquet(review_parquet, index=False, engine="pyarrow")

    # 8. Compute Statistics & Export JSON
    label_counts = df_pairs["weak_label"].value_counts().to_dict()
    method_dist = df_pairs["retrieval_method"].value_counts().to_dict()

    train_res_cnt = int(df_train["resume_id"].nunique())
    val_res_cnt = int(df_val["resume_id"].nunique())
    test_res_cnt = int(df_test["resume_id"].nunique())

    stats_data = {
        "total_candidate_pairs": int(len(df_pairs)),
        "positive_pairs": int(label_counts.get("positive", 0)),
        "negative_pairs": int(label_counts.get("negative", 0)),
        "ambiguous_pairs": int(label_counts.get("ambiguous", 0)),
        "unique_resumes": int(df_pairs["resume_id"].nunique()),
        "unique_jobs": int(df_pairs["job_id"].nunique()),
        "average_shared_skills": float(df_pairs["shared_skill_count"].mean()),
        "median_shared_skills": float(df_pairs["shared_skill_count"].median()),
        "average_semantic_similarity": float(df_pairs["semantic_similarity"].mean()),
        "median_semantic_similarity": float(df_pairs["semantic_similarity"].median()),
        "average_skill_coverage": float(df_pairs["skill_coverage_resume"].mean()),
        "retrieval_method_distribution": method_dist,
        "train_pairs": int(len(df_train)),
        "validation_pairs": int(len(df_val)),
        "test_pairs": int(len(df_test)),
        "train_resume_count": train_res_cnt,
        "validation_resume_count": val_res_cnt,
        "test_resume_count": test_res_cnt,
    }

    stats_json_path = processed_dir / "candidate_pair_statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    # 9. Run Validation
    validate_candidate_pairs()


if __name__ == "__main__":
    main()
