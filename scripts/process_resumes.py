import json
import sys
from pathlib import Path
import pandas as pd

# Add src package to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.resume_cleaner import (
    clean_resume_text,
    check_resume_quality,
    detect_sections,
    extract_resume_skills,
    extract_experience,
    extract_education,
    extract_job_titles,
    build_master_resume_features,
)
from scripts.validate_resumes import validate_resumes


def process_resumes():
    raw_csv = project_root / "data" / "raw" / "resumes" / "Resume" / "Resume.csv"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw resumes from {raw_csv}...")
    df_raw = pd.read_csv(raw_csv, low_memory=False)

    # 1. Clean Resumes Dataset
    clean_records = []
    for _, row in df_raw.iterrows():
        r_id = str(row["ID"]).strip()
        raw_t = str(row["Resume_str"])
        c_text = clean_resume_text(raw_t)
        cat = str(row["Category"]).strip()

        clean_records.append({
            "resume_id": r_id,
            "raw_text": raw_t,
            "clean_text": c_text,
            "category": cat,
        })

    df_clean = pd.DataFrame(clean_records)
    clean_parquet = processed_dir / "resumes_clean.parquet"
    df_clean.to_parquet(clean_parquet, index=False, engine="pyarrow")

    # 2. Text Quality Report
    df_quality = check_resume_quality(df_clean)
    quality_parquet = processed_dir / "resume_quality_report.parquet"
    df_quality.to_parquet(quality_parquet, index=False, engine="pyarrow")

    # 3. Section Detection
    df_sections = detect_sections(df_clean)
    sections_parquet = processed_dir / "resume_sections.parquet"
    df_sections.to_parquet(sections_parquet, index=False, engine="pyarrow")

    # 4 & 5. Skill Extraction & Normalization
    esco_skills_parquet = processed_dir / "esco_skills.parquet"
    esco_aliases_parquet = processed_dir / "esco_skill_aliases.parquet"

    df_esco_skills = pd.read_parquet(esco_skills_parquet)
    df_esco_aliases = pd.read_parquet(esco_aliases_parquet)

    df_res_skills = extract_resume_skills(df_clean, df_esco_skills, df_esco_aliases)
    skills_parquet = processed_dir / "resume_skills.parquet"
    df_res_skills.to_parquet(skills_parquet, index=False, engine="pyarrow")

    # 6. Experience Extraction
    df_exp = extract_experience(df_clean)
    exp_parquet = processed_dir / "resume_experience.parquet"
    df_exp.to_parquet(exp_parquet, index=False, engine="pyarrow")

    # 7. Education Extraction
    df_edu = extract_education(df_clean)
    edu_parquet = processed_dir / "resume_education.parquet"
    df_edu.to_parquet(edu_parquet, index=False, engine="pyarrow")

    # 8. Job Title Extraction
    esco_occ_parquet = processed_dir / "esco_occupations.parquet"
    df_esco_occ = pd.read_parquet(esco_occ_parquet)
    df_titles = extract_job_titles(df_clean, df_esco_occ)
    titles_parquet = processed_dir / "resume_job_titles.parquet"
    df_titles.to_parquet(titles_parquet, index=False, engine="pyarrow")

    # 9. Master Feature Table
    df_features = build_master_resume_features(
        df_clean, df_quality, df_res_skills, df_exp, df_edu, df_titles
    )
    features_parquet = processed_dir / "resume_features.parquet"
    df_features.to_parquet(features_parquet, index=False, engine="pyarrow")

    # 10. Dataset Statistics
    word_counts = df_quality["word_count"]
    skill_counts = df_features["skill_count"]
    cat_counts = df_clean["category"].value_counts().to_dict()

    res_with_exp = int(df_exp["years_experience"].notna().sum())
    res_with_edu = int((df_edu["education_level"] != "Not Specified").sum())
    res_with_skills = int((df_features["skill_count"] > 0).sum())

    stats_data = {
        "total_resumes": int(len(df_clean)),
        "valid_resumes": int((df_quality["quality_flag"] != "very_short").sum()),
        "empty_resumes": int(df_quality["is_empty"].sum()),
        "duplicate_resumes": int(df_quality["is_duplicate"].sum()),
        "average_word_count": float(word_counts.mean()),
        "median_word_count": float(word_counts.median()),
        "average_skill_count": float(skill_counts.mean()),
        "median_skill_count": float(skill_counts.median()),
        "categories_count": cat_counts,
        "unique_categories": int(df_clean["category"].nunique()),
        "resumes_with_experience": res_with_exp,
        "resumes_with_education": res_with_edu,
        "resumes_with_detected_skills": res_with_skills,
    }

    stats_json_path = processed_dir / "resume_statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    # 11. Run Resume Validation Script
    validate_resumes()


if __name__ == "__main__":
    process_resumes()
