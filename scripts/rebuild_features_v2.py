import json
import logging
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering.ml_features import enrich_candidate_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def jaccard_similarity(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def rebuild_features_v2():
    processed_dir = project_root / "data" / "processed"
    
    # Load canonical application skills
    app_skills_path = processed_dir / "esco_relevant_skills.parquet"
    if not app_skills_path.exists():
        logger.error(f"Cannot find {app_skills_path}")
        return
        
    df_app_skills = pd.read_parquet(app_skills_path)
    # The valid skills are the preferredLabels that are NOT generic (i.e. is_application_skill == True)
    if 'is_application_skill' in df_app_skills.columns:
        df_valid = df_app_skills[df_app_skills['is_application_skill'] == True]
    else:
        df_valid = df_app_skills[df_app_skills['relevance_category'] != 'generic']
        
    valid_skills = set(str(s).lower().strip() for s in df_valid["preferredLabel"].unique())
    logger.info(f"Loaded {len(valid_skills)} valid application skills from ESCO relevant skills layer.")
    
    # Count breakdown
    technical = (df_app_skills['relevance_category'] == 'technical').sum()
    professional = (df_app_skills['relevance_category'] == 'professional').sum()
    domain = (df_app_skills['relevance_category'] == 'domain').sum()
    logger.info(f"Application skill count: {len(df_app_skills)}")
    logger.info(f"Technical skill count: {technical}")
    logger.info(f"Professional skill count: {professional}")
    logger.info(f"Domain skill count: {domain}")

    # Check for obvious low-value concepts in primary technical skills
    bad_concepts = ["similitude", "security panels", "generic behavioral"]
    tech_skills = set(df_app_skills[df_app_skills['relevance_category'] == 'technical']['preferredLabel'].str.lower())
    for bc in bad_concepts:
        if bc in tech_skills:
            logger.warning(f"Low-value concept '{bc}' found in primary technical skills!")
        else:
            logger.info(f"Low-value concept '{bc}' correctly excluded from technical skills.")

    # Load original candidate pairs, resumes, and jobs
    logger.info("Loading base candidate pairs, resumes, and jobs...")
    df_pairs_v1 = pd.read_parquet(processed_dir / "candidate_pairs.parquet")
    df_resumes = pd.read_parquet(processed_dir / "resume_features.parquet")
    df_jobs = pd.read_parquet(processed_dir / "job_features.parquet")
    
    # Create filtered copies of resumes and jobs
    logger.info("Filtering resume and job skills to only include valid application skills...")
    
    def filter_skills(skills_array):
        if not hasattr(skills_array, "__iter__") or pd.isna(skills_array).all() if isinstance(skills_array, pd.Series) else False:
            return np.array([])
        s_list = [str(s) for s in skills_array]
        valid = [s for s in s_list if s.lower().strip() in valid_skills]
        return valid

    import numpy as np
    
    df_resumes_v2 = df_resumes.copy()
    df_jobs_v2 = df_jobs.copy()
    
    df_resumes_v2["skills"] = df_resumes_v2["skills"].apply(filter_skills)
    df_jobs_v2["combined_skills"] = df_jobs_v2["combined_skills"].apply(filter_skills)
    
    # Recalculate base skill counts & jaccard on the pairs BEFORE enrichment, 
    # because enrich_candidate_pairs relies on some of them being present in df_pairs
    logger.info("Recalculating base candidate pair skill overlap metrics...")
    
    res_dict = df_resumes_v2.set_index("resume_id")["skills"].to_dict()
    job_dict = df_jobs_v2.set_index("job_id")["combined_skills"].to_dict()
    
    shared_counts = []
    res_counts = []
    job_counts = []
    skill_jaccards = []
    skill_cov_res = []
    skill_cov_job = []
    
    for _, row in df_pairs_v1.iterrows():
        r_skills = set(str(s).lower().strip() for s in res_dict.get(row["resume_id"], []))
        j_skills = set(str(s).lower().strip() for s in job_dict.get(row["job_id"], []))
        
        shared = r_skills & j_skills
        sc = len(shared)
        rc = len(r_skills)
        jc = len(j_skills)
        
        shared_counts.append(sc)
        res_counts.append(rc)
        job_counts.append(jc)
        skill_jaccards.append(jaccard_similarity(r_skills, j_skills))
        skill_cov_res.append(sc / max(1, rc))
        skill_cov_job.append(sc / max(1, jc))
        
    df_pairs_v2 = df_pairs_v1.copy()
    df_pairs_v2["shared_skill_count"] = shared_counts
    df_pairs_v2["resume_skill_count"] = res_counts
    df_pairs_v2["job_skill_count"] = job_counts
    df_pairs_v2["skill_jaccard"] = skill_jaccards
    df_pairs_v2["skill_coverage_resume"] = skill_cov_res
    df_pairs_v2["skill_coverage_job"] = skill_cov_job
    
    # Enrich features (this computes essential/optional coverage, word counts, semantic matching etc using the filtered df_resumes_v2 and df_jobs_v2)
    logger.info("Enriching features (computing essential/optional coverage, match indicators)...")
    df_ml_features_v2 = enrich_candidate_pairs(df_pairs_v2, df_resumes_v2, df_jobs_v2, processed_dir)
    
    # Ensure ID columns remain
    df_ml_features_v2["resume_id"] = df_pairs_v1["resume_id"]
    df_ml_features_v2["job_id"] = df_pairs_v1["job_id"]
    
    output_path = processed_dir / "ml_features_v2.parquet"
    df_ml_features_v2.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df_ml_features_v2)} enriched V2 feature rows to {output_path}")
    
    # ---------------------------------------------------------
    # 5. FEATURE COMPARISON
    # ---------------------------------------------------------
    logger.info("Generating feature comparison report...")
    df_pairs_v1_enriched = pd.read_parquet(processed_dir / "candidate_pairs.parquet") # We'll just compare base pair skill metrics for simplicity, as we want to see distribution shifts
    
    comparison_data = []
    
    metrics_to_compare = ["shared_skill_count", "skill_jaccard", "skill_coverage_resume", "skill_coverage_job"]
    
    for metric in metrics_to_compare:
        mean_v1 = df_pairs_v1[metric].mean()
        mean_v2 = df_pairs_v2[metric].mean()
        diff = mean_v2 - mean_v1
        
        comparison_data.append({
            "feature": metric,
            "v1_mean": mean_v1,
            "v2_mean": mean_v2,
            "diff": diff,
            "reason": "Filtered noisy skills via application skill layer classification"
        })
        
    df_comp = pd.DataFrame(comparison_data)
    comp_path = processed_dir / "feature_comparison_v1_v2.csv"
    df_comp.to_csv(comp_path, index=False)
    logger.info(f"Saved comparison to {comp_path}")
    print(df_comp)

if __name__ == "__main__":
    rebuild_features_v2()
