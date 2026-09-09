import json
import sys
from pathlib import Path
import pandas as pd

# Add src package to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.job_cleaner import (
    clean_job_postings,
    check_job_quality,
    clean_job_skills,
    clean_company_info,
    extract_job_esco_skills,
    build_job_skill_profile,
    extract_job_experience,
    extract_job_education,
    extract_job_titles,
    build_master_job_features,
)
from scripts.validate_jobs import validate_jobs


def process_jobs():
    raw_dir = project_root / "data" / "raw" / "jobs"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    postings_csv = raw_dir / "postings.csv"
    job_skills_csv = raw_dir / "jobs" / "job_skills.csv"
    skills_map_csv = raw_dir / "mappings" / "skills.csv"
    companies_csv = raw_dir / "companies" / "companies.csv"
    company_ind_csv = raw_dir / "companies" / "company_industries.csv"

    # 1. Clean Job Postings
    df_clean = clean_job_postings(postings_csv)
    clean_parquet = processed_dir / "jobs_clean.parquet"
    df_clean.to_parquet(clean_parquet, index=False, engine="pyarrow")

    # 2. Quality Report
    df_quality = check_job_quality(df_clean)
    quality_parquet = processed_dir / "job_quality_report.parquet"
    df_quality.to_parquet(quality_parquet, index=False, engine="pyarrow")

    # 3. Explicit Job Skills & Unknown Skills
    df_explicit, df_unknown = clean_job_skills(job_skills_csv, skills_map_csv, df_clean)
    explicit_parquet = processed_dir / "job_skills_clean.parquet"
    unknown_parquet = processed_dir / "job_unknown_skills.parquet"
    df_explicit.to_parquet(explicit_parquet, index=False, engine="pyarrow")
    df_unknown.to_parquet(unknown_parquet, index=False, engine="pyarrow")

    # 4. Clean Company Information
    df_companies = clean_company_info(companies_csv, company_ind_csv)
    companies_parquet = processed_dir / "job_companies_clean.parquet"
    df_companies.to_parquet(companies_parquet, index=False, engine="pyarrow")

    # 5. Extract ESCO Skills from Text
    esco_skills_parquet = processed_dir / "esco_skills.parquet"
    esco_aliases_parquet = processed_dir / "esco_skill_aliases.parquet"

    df_esco_skills = pd.read_parquet(esco_skills_parquet)
    df_esco_aliases = pd.read_parquet(esco_aliases_parquet)

    df_job_esco = extract_job_esco_skills(df_clean, df_esco_skills, df_esco_aliases)
    job_esco_parquet = processed_dir / "job_esco_skills.parquet"
    df_job_esco.to_parquet(job_esco_parquet, index=False, engine="pyarrow")

    # 6. Combined Job Skill Profile
    df_profile = build_job_skill_profile(df_clean, df_explicit, df_job_esco)
    profile_parquet = processed_dir / "job_skill_profile.parquet"
    df_profile.to_parquet(profile_parquet, index=False, engine="pyarrow")

    # 7. Experience Extraction
    df_exp = extract_job_experience(df_clean)
    exp_parquet = processed_dir / "job_experience.parquet"
    df_exp.to_parquet(exp_parquet, index=False, engine="pyarrow")

    # 8. Education Extraction
    df_edu = extract_job_education(df_clean)
    edu_parquet = processed_dir / "job_education.parquet"
    df_edu.to_parquet(edu_parquet, index=False, engine="pyarrow")

    # 9. Job Titles
    df_titles = extract_job_titles(df_clean)
    titles_parquet = processed_dir / "job_titles.parquet"
    df_titles.to_parquet(titles_parquet, index=False, engine="pyarrow")

    # 10. Master Job Features Table
    df_features = build_master_job_features(df_clean, df_quality, df_profile, df_exp, df_edu)
    features_parquet = processed_dir / "job_features.parquet"
    df_features.to_parquet(features_parquet, index=False, engine="pyarrow")

    # 11. Dataset Statistics
    desc_lengths = df_quality["description_length"]
    comb_counts = df_profile["combined_skill_count"]

    work_type_dist = df_clean["work_type"].value_counts().to_dict()
    exp_lvl_dist = df_clean["experience_level"].value_counts().to_dict()

    jobs_with_exp = int(df_exp["min_years_experience"].notna().sum())
    jobs_with_edu = int((df_edu["education_requirement"] != "Not Specified").sum())
    jobs_with_exp_skills = int(df_explicit["job_id"].nunique())
    jobs_with_esco = int(df_job_esco["job_id"].nunique())

    dup_jobs = int(df_clean["job_id"].duplicated().sum())

    stats_data = {
        "total_jobs": int(len(df_clean)),
        "valid_jobs": int((df_quality["quality_flag"] != "missing_description").sum()),
        "missing_description_jobs": int((df_quality["quality_flag"] == "missing_description").sum()),
        "duplicate_jobs": dup_jobs,
        "average_description_length": float(desc_lengths.mean()),
        "median_description_length": float(desc_lengths.median()),
        "average_skill_count": float(comb_counts.mean()),
        "median_skill_count": float(comb_counts.median()),
        "jobs_with_experience": jobs_with_exp,
        "jobs_with_education": jobs_with_edu,
        "jobs_with_explicit_skills": jobs_with_exp_skills,
        "jobs_with_esco_skills": jobs_with_esco,
        "unique_job_titles": int(df_clean["clean_title"].nunique()),
        "unique_companies": int(df_clean["company_name"].nunique()),
        "work_type_distribution": work_type_dist,
        "experience_level_distribution": exp_lvl_dist,
    }

    stats_json_path = processed_dir / "job_statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    # 12. Run Validation
    validate_jobs()


if __name__ == "__main__":
    process_jobs()
