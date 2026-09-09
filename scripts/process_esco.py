import json
import sys
from pathlib import Path
import pandas as pd

# Add src package to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.esco_cleaner import (
    clean_skills,
    clean_occupations,
    clean_relations,
    build_occupation_profiles,
)
from scripts.validate_esco import validate_esco


def process_esco():
    raw_dir = project_root / "data" / "raw" / "esco"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    skills_csv = raw_dir / "skills_en.csv"
    occupations_csv = raw_dir / "occupations_en.csv"
    relations_csv = raw_dir / "occupationSkillRelations_en.csv"

    # 1. Clean ESCO Skills & Aliases
    df_skills, df_aliases = clean_skills(skills_csv)
    skills_parquet = processed_dir / "esco_skills.parquet"
    aliases_parquet = processed_dir / "esco_skill_aliases.parquet"
    df_skills.to_parquet(skills_parquet, index=False, engine="pyarrow")
    df_aliases.to_parquet(aliases_parquet, index=False, engine="pyarrow")

    # 2. Clean ESCO Occupations
    df_occ = clean_occupations(occupations_csv)
    occ_parquet = processed_dir / "esco_occupations.parquet"
    df_occ.to_parquet(occ_parquet, index=False, engine="pyarrow")

    # 3. Clean Occupation-Skill Relations
    df_rels, missing_skill_type_cnt = clean_relations(relations_csv)
    rels_parquet = processed_dir / "esco_occupation_skill_relations.parquet"
    df_rels.to_parquet(rels_parquet, index=False, engine="pyarrow")

    # 4. Build Occupation Skill Profiles
    df_profiles = build_occupation_profiles(df_occ, df_skills, df_rels)
    profiles_parquet = processed_dir / "esco_occupation_skill_profile.parquet"
    df_profiles.to_parquet(profiles_parquet, index=False, engine="pyarrow")

    # 5. Calculate Metrics & Generate Quality Report
    skill_uris = set(df_skills["conceptUri"])
    occ_uris = set(df_occ["conceptUri"])

    invalid_occ_refs = int((~df_rels["occupationUri"].isin(occ_uris)).sum())
    invalid_skill_refs = int((~df_rels["skillUri"].isin(skill_uris)).sum())

    essential_cnt = int((df_rels["relationType"] == "essential").sum())
    optional_cnt = int((df_rels["relationType"] == "optional").sum())

    quality_data = {
        "skills_rows": len(df_skills),
        "occupations_rows": len(df_occ),
        "relation_rows": len(df_rels),
        "duplicate_counts": {
            "skills": int(df_skills.duplicated(subset=["conceptUri"]).sum()),
            "occupations": int(df_occ.duplicated(subset=["conceptUri"]).sum()),
            "relations": int(df_rels.duplicated(subset=["occupationUri", "skillUri", "relationType"]).sum()),
            "aliases": int(df_aliases.duplicated(subset=["skill_uri", "alias_normalized", "alias_type"]).sum()),
        },
        "missing_counts": {
            "skills_alt_labels": int((df_skills["altLabels"] == "").sum()),
            "skills_hidden_labels": int((df_skills["hiddenLabels"] == "").sum()),
            "skills_description": int((df_skills["description"] == "").sum()),
            "occupations_description": int((df_occ["description"] == "").sum()),
            "relation_skill_type": missing_skill_type_cnt,
        },
        "invalid_occupation_uri_references": invalid_occ_refs,
        "invalid_skill_uri_references": invalid_skill_refs,
        "number_of_unique_skills": int(df_skills["conceptUri"].nunique()),
        "number_of_unique_occupations": int(df_occ["conceptUri"].nunique()),
        "number_of_unique_occupation_skill_relationships": int(len(df_rels)),
        "number_of_essential_relationships": essential_cnt,
        "number_of_optional_relationships": optional_cnt,
    }

    quality_json_path = processed_dir / "esco_data_quality.json"
    with open(quality_json_path, "w", encoding="utf-8") as f:
        json.dump(quality_data, f, indent=2)

    # 6. Run Validation Script
    validate_esco()

    # 7. Print Final Summary
    print("\nESCO PROCESSING COMPLETE\n")
    print(f"Skills: {len(df_skills)}")
    print(f"Occupations: {len(df_occ)}")
    print(f"Relationships: {len(df_rels)}\n")
    print(f"Essential relationships: {essential_cnt}")
    print(f"Optional relationships: {optional_cnt}\n")
    print("Validation: PASS")


if __name__ == "__main__":
    process_esco()
