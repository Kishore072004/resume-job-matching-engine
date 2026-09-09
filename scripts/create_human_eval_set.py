import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CANDIDATE_PAIRS_PATH = PROCESSED_DIR / "candidate_pairs.parquet"
RESUMES_CLEAN_PATH = PROCESSED_DIR / "resumes_clean.parquet"
JOBS_CLEAN_PATH = PROCESSED_DIR / "jobs_clean.parquet"
RESUME_FEATURES_PATH = PROCESSED_DIR / "resume_features.parquet"
JOB_FEATURES_PATH = PROCESSED_DIR / "job_features.parquet"

OUTPUT_EVAL_PAIRS_PATH = PROCESSED_DIR / "human_evaluation_pairs.parquet"
OUTPUT_REPORT_PATH = PROCESSED_DIR / "human_eval_sampling_report.json"

RANDOM_SEED = 42

def create_human_evaluation_dataset(target_total: int = 750) -> pd.DataFrame:
    logger.info("Loading candidate pairs and text feature sources...")
    
    pairs_df = pd.read_parquet(CANDIDATE_PAIRS_PATH)
    resumes_df = pd.read_parquet(RESUMES_CLEAN_PATH)
    jobs_df = pd.read_parquet(JOBS_CLEAN_PATH)
    rf_df = pd.read_parquet(RESUME_FEATURES_PATH)
    jf_df = pd.read_parquet(JOB_FEATURES_PATH)
    
    logger.info(f"Loaded {len(pairs_df)} candidate pairs.")
    
    # Map resume skills and job skills
    resume_skills_map = dict(zip(rf_df["resume_id"], rf_df["skills"]))
    job_skills_map = dict(zip(jf_df["job_id"], jf_df["combined_skills"]))
    
    # Define strata logic
    # 1. Strong Match: weak_label == 'positive' (or 1.0)
    cond_strong = (pairs_df["weak_label"].isin(["positive", 1.0, "1.0"])) & (pairs_df["skill_jaccard"] >= 0.15)
    
    # 2. Hard Negative: High title or semantic similarity (>=0.40) but weak_label is negative or skill_jaccard < 0.08
    cond_hard_neg = (
        ((pairs_df["semantic_similarity"] >= 0.40) | (pairs_df["title_similarity"] >= 0.40)) & 
        ((pairs_df["weak_label"].isin(["negative", 0.0, "0.0"])) | (pairs_df["skill_jaccard"] < 0.08)) &
        (~cond_strong)
    )
    
    # 3. Obvious Negative: weak_label == 'negative', low semantic sim (<0.25), low skill jaccard (<0.05)
    cond_obvious_neg = (
        (pairs_df["weak_label"].isin(["negative", 0.0, "0.0"])) & 
        (pairs_df["semantic_similarity"] < 0.25) & 
        (pairs_df["skill_jaccard"] < 0.05) &
        (~cond_strong) & (~cond_hard_neg)
    )
    
    # 4. Moderate / Ambiguous: everything else
    cond_moderate = ~(cond_strong | cond_hard_neg | cond_obvious_neg)
    
    pairs_df["strata_group"] = "moderate_ambiguous"
    pairs_df.loc[cond_strong, "strata_group"] = "strong_match"
    pairs_df.loc[cond_hard_neg, "strata_group"] = "hard_negative"
    pairs_df.loc[cond_obvious_neg, "strata_group"] = "obvious_negative"
    
    # Define targets per strata
    strata_targets = {
        "strong_match": min(200, (pairs_df["strata_group"] == "strong_match").sum()),
        "moderate_ambiguous": min(250, (pairs_df["strata_group"] == "moderate_ambiguous").sum()),
        "hard_negative": min(150, (pairs_df["strata_group"] == "hard_negative").sum()),
        "obvious_negative": min(150, (pairs_df["strata_group"] == "obvious_negative").sum()),
    }
    
    sampled_frames = []
    for strata_name, target_count in strata_targets.items():
        subset = pairs_df[pairs_df["strata_group"] == strata_name]
        if len(subset) > target_count:
            sampled_subset = subset.sample(n=target_count, random_state=RANDOM_SEED)
        else:
            sampled_subset = subset.copy()
        logger.info(f"Sampled {len(sampled_subset)} pairs for strata '{strata_name}' (available: {len(subset)}).")
        sampled_frames.append(sampled_subset)
        
    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    
    # Shuffle final set
    sampled_df = sampled_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Add unique review_id
    sampled_df["review_id"] = [f"EVAL_PAIR_{i+1:04d}" for i in range(len(sampled_df))]
    
    # Merge textual details
    resumes_text_map = dict(zip(resumes_df["resume_id"], resumes_df["clean_text"]))
    resumes_cat_map = dict(zip(resumes_df["resume_id"], resumes_df["category"]))
    
    jobs_title_map = dict(zip(jobs_df["job_id"], jobs_df["title"]))
    jobs_desc_map = dict(zip(jobs_df["job_id"], jobs_df["clean_description"]))
    jobs_comp_map = dict(zip(jobs_df["job_id"], jobs_df["company_name"]))
    jobs_loc_map = dict(zip(jobs_df["job_id"], jobs_df["location"]))
    
    sampled_df["resume_category"] = sampled_df["resume_id"].map(resumes_cat_map)
    sampled_df["job_title"] = sampled_df["job_id"].map(jobs_title_map)
    sampled_df["company_name"] = sampled_df["job_id"].map(jobs_comp_map).fillna("Unknown")
    sampled_df["location"] = sampled_df["job_id"].map(jobs_loc_map).fillna("Unknown")
    
    sampled_df["resume_text_snippet"] = sampled_df["resume_id"].map(lambda rid: str(resumes_text_map.get(rid, ""))[:1500])
    sampled_df["job_description_snippet"] = sampled_df["job_id"].map(lambda jid: str(jobs_desc_map.get(jid, ""))[:1500])
    
    # Format skills as JSON string or comma-separated for clean storage
    def get_skill_list(skills_obj):
        if isinstance(skills_obj, (list, np.ndarray)):
            return [str(s) for s in skills_obj]
        return []
    
    res_skills_list = [get_skill_list(resume_skills_map.get(rid, [])) for rid in sampled_df["resume_id"]]
    job_skills_list = [get_skill_list(job_skills_map.get(jid, [])) for jid in sampled_df["job_id"]]
    
    shared_skills_list = []
    for r_s, j_s in zip(res_skills_list, job_skills_list):
        r_set = set(r_s)
        j_set = set(j_s)
        shared_skills_list.append(sorted(list(r_set.intersection(j_set))))
        
    sampled_df["resume_skills"] = [json.dumps(s) for s in res_skills_list]
    sampled_df["job_skills"] = [json.dumps(s) for s in job_skills_list]
    sampled_df["shared_skills"] = [json.dumps(s) for s in shared_skills_list]
    
    # Initialize human annotation columns
    sampled_df["human_label"] = None
    sampled_df["human_label_1"] = None
    sampled_df["human_label_2"] = None
    sampled_df["human_confidence"] = None
    sampled_df["human_notes"] = None
    
    # Save output dataset
    sampled_df.to_parquet(OUTPUT_EVAL_PAIRS_PATH, index=False)
    logger.info(f"Saved {len(sampled_df)} evaluation pairs to {OUTPUT_EVAL_PAIRS_PATH}")
    
    # Generate sampling report
    report = {
        "total_candidate_pairs": len(pairs_df),
        "sampled_evaluation_pairs": len(sampled_df),
        "random_seed": RANDOM_SEED,
        "strata_breakdown": {
            k: int((sampled_df["strata_group"] == k).sum()) for k in strata_targets.keys()
        },
        "weak_label_distribution": {
            "positive_1.0": int((sampled_df["weak_label"] == 1.0).sum()),
            "negative_0.0": int((sampled_df["weak_label"] == 0.0).sum()),
            "unlabeled_null": int(sampled_df["weak_label"].isna().sum())
        },
        "resume_category_diversity": int(sampled_df["resume_category"].nunique()),
        "columns": sampled_df.columns.tolist()
    }
    
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Saved sampling report to {OUTPUT_REPORT_PATH}")
    return sampled_df

if __name__ == "__main__":
    create_human_evaluation_dataset()
