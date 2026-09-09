"""
ML Feature Engineering Module.

Enriches candidate pair data with additional computed features
for the XGBoost resume-job matching model.

Features are computed from the candidate pair base features plus
additional lookups into resume and job feature tables.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Feature list used by the model
# ────────────────────────────────────────────────────────────────

MODEL_FEATURES = [
    "semantic_similarity",
    "semantic_similarity_normalized",
    "shared_skill_count",
    "resume_skill_count",
    "job_skill_count",
    "skill_jaccard",
    "resume_skill_coverage",
    "job_skill_coverage",
    "required_skill_coverage",
    "essential_matched_count",
    "essential_total_count",
    "essential_skill_coverage",
    "optional_matched_count",
    "optional_total_count",
    "optional_skill_coverage",
    "missing_required_skill_count",
    "matched_required_skill_count",
    "matched_optional_skill_count",
    "title_similarity",
    "resume_years_experience",
    "job_min_years_experience",
    "job_max_years_experience",
    "experience_gap",
    "experience_match",
    "education_match",
    "occupation_match",
    "resume_word_count",
    "job_word_count",
]

# ID columns preserved for tracing
ID_COLUMNS = ["resume_id", "job_id"]

# Label column
LABEL_COLUMN = "label"


def build_esco_occupation_skill_lookup(processed_dir: Path) -> dict:
    """
    Build a lookup mapping occupationUri -> {essential: set, optional: set}
    from the ESCO occupation-skill relation data.
    """
    rel_path = processed_dir / "esco_occupation_skill_relations.parquet"
    if not rel_path.exists():
        logger.warning("ESCO occupation-skill relations not found; essential/optional breakdown unavailable.")
        return {}

    df_rel = pd.read_parquet(rel_path)
    occ_skills = {}
    for _, row in df_rel.iterrows():
        occ_uri = row["occupationUri"]
        rel_type = str(row["relationType"]).lower()
        skill_label = str(row["skillLabel"]).lower().strip()
        if occ_uri not in occ_skills:
            occ_skills[occ_uri] = {"essential": set(), "optional": set()}
        if rel_type == "essential":
            occ_skills[occ_uri]["essential"].add(skill_label)
        else:
            occ_skills[occ_uri]["optional"].add(skill_label)

    return occ_skills


def compute_essential_optional_coverage(
    resume_skills: set,
    job_esco_skills: set,
    occ_skill_lookup: dict,
    job_esco_skill_uris: set = None,
) -> dict:
    """
    Compute essential/optional skill match metrics.
    Falls back to treating all job skills as required if no ESCO occupation mapping.
    """
    # Default: treat all job skills as required
    essential_skills = job_esco_skills
    optional_skills = set()

    # If we have occupation-level skill breakdown, use it
    if occ_skill_lookup and job_esco_skill_uris:
        for occ_uri, skill_sets in occ_skill_lookup.items():
            if skill_sets["essential"] & job_esco_skills:
                essential_skills = skill_sets["essential"] & job_esco_skills
                optional_skills = skill_sets["optional"] & job_esco_skills
                break

    essential_matched = resume_skills & essential_skills
    optional_matched = resume_skills & optional_skills
    all_required = essential_skills | optional_skills if optional_skills else essential_skills

    return {
        "essential_matched_count": len(essential_matched),
        "essential_total_count": len(essential_skills),
        "essential_skill_coverage": float(len(essential_matched) / max(1, len(essential_skills))),
        "optional_matched_count": len(optional_matched),
        "optional_total_count": len(optional_skills),
        "optional_skill_coverage": float(len(optional_matched) / max(1, len(optional_skills))) if optional_skills else 0.0,
        "missing_required_skill_count": len(all_required - resume_skills),
        "matched_required_skill_count": len(resume_skills & all_required),
        "matched_optional_skill_count": len(optional_matched),
    }


def enrich_candidate_pairs(
    df_pairs: pd.DataFrame,
    df_resumes: pd.DataFrame,
    df_jobs: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Enrich candidate pair data with additional ML features.

    Adds: normalized similarity, essential/optional skill coverage,
    experience match indicator, occupation match, word counts, etc.
    """
    logger.info(f"Enriching {len(df_pairs)} candidate pairs with ML features...")

    df = df_pairs.copy()

    # ── Build lookup dicts ──────────────────────────────────────
    res_dict = df_resumes.set_index("resume_id").to_dict("index")
    job_dict = df_jobs.set_index("job_id").to_dict("index")

    # ── 1. Semantic similarity normalized ───────────────────────
    sem_min = df["semantic_similarity"].min()
    sem_max = df["semantic_similarity"].max()
    sem_range = sem_max - sem_min if sem_max != sem_min else 1.0
    df["semantic_similarity_normalized"] = (df["semantic_similarity"] - sem_min) / sem_range

    # ── 2. Rename existing coverage columns for consistency ─────
    df["resume_skill_coverage"] = df["skill_coverage_resume"]
    df["job_skill_coverage"] = df["skill_coverage_job"]

    # ── 3. Required skill coverage (same as job_skill_coverage baseline) ──
    df["required_skill_coverage"] = df["skill_coverage_job"]

    # ── 4. Essential / Optional skill breakdown ─────────────────
    logger.info("  Computing essential/optional skill coverage...")
    occ_skill_lookup = build_esco_occupation_skill_lookup(processed_dir)

    essential_matched = []
    essential_total = []
    essential_cov = []
    optional_matched = []
    optional_total = []
    optional_cov = []
    missing_required = []
    matched_required = []
    matched_optional = []

    for _, row in df.iterrows():
        r_info = res_dict.get(row["resume_id"], {})
        j_info = job_dict.get(row["job_id"], {})

        r_skills_raw = r_info.get("skills", [])
        j_skills_raw = j_info.get("combined_skills", [])

        r_skills = set(str(s).lower().strip() for s in r_skills_raw if str(s).strip()) if hasattr(r_skills_raw, '__iter__') else set()
        j_skills = set(str(s).lower().strip() for s in j_skills_raw if str(s).strip()) if hasattr(j_skills_raw, '__iter__') else set()

        metrics = compute_essential_optional_coverage(r_skills, j_skills, occ_skill_lookup)

        essential_matched.append(metrics["essential_matched_count"])
        essential_total.append(metrics["essential_total_count"])
        essential_cov.append(metrics["essential_skill_coverage"])
        optional_matched.append(metrics["optional_matched_count"])
        optional_total.append(metrics["optional_total_count"])
        optional_cov.append(metrics["optional_skill_coverage"])
        missing_required.append(metrics["missing_required_skill_count"])
        matched_required.append(metrics["matched_required_skill_count"])
        matched_optional.append(metrics["matched_optional_skill_count"])

    df["essential_matched_count"] = essential_matched
    df["essential_total_count"] = essential_total
    df["essential_skill_coverage"] = essential_cov
    df["optional_matched_count"] = optional_matched
    df["optional_total_count"] = optional_total
    df["optional_skill_coverage"] = optional_cov
    df["missing_required_skill_count"] = missing_required
    df["matched_required_skill_count"] = matched_required
    df["matched_optional_skill_count"] = matched_optional

    # ── 5. Experience match indicator ───────────────────────────
    def compute_experience_match(row):
        r_exp = row.get("resume_years_experience")
        j_min = row.get("job_min_years_experience")
        j_max = row.get("job_max_years_experience")
        if pd.isna(r_exp) or (pd.isna(j_min) and pd.isna(j_max)):
            return np.nan
        if not pd.isna(j_min) and not pd.isna(j_max):
            return 1.0 if j_min <= r_exp <= j_max else 0.0
        if not pd.isna(j_min):
            return 1.0 if r_exp >= j_min else 0.0
        return 1.0 if r_exp <= j_max else 0.0

    df["experience_match"] = df.apply(compute_experience_match, axis=1)

    # ── 6. Occupation match ─────────────────────────────────────
    logger.info("  Computing occupation match indicators...")
    occ_match = []
    for _, row in df.iterrows():
        r_info = res_dict.get(row["resume_id"], {})
        j_info = job_dict.get(row["job_id"], {})

        r_cat = str(r_info.get("category", "")).lower().strip()
        j_title = str(j_info.get("clean_title", "")).lower().strip()

        # Simple occupation match: check if resume category words appear in job title
        cat_words = set(r_cat.replace("-", " ").split())
        title_words = set(j_title.split())
        overlap = cat_words & title_words
        significant_overlap = len([w for w in overlap if len(w) >= 3])
        occ_match.append(1.0 if significant_overlap >= 1 else 0.0)

    df["occupation_match"] = occ_match

    # ── 7. Word counts ──────────────────────────────────────────
    logger.info("  Computing word counts...")
    resume_wc = []
    job_wc = []
    for _, row in df.iterrows():
        r_info = res_dict.get(row["resume_id"], {})
        j_info = job_dict.get(row["job_id"], {})
        r_text = str(r_info.get("clean_text", ""))
        j_text = str(j_info.get("clean_description", ""))
        resume_wc.append(len(r_text.split()))
        job_wc.append(len(j_text.split()))

    df["resume_word_count"] = resume_wc
    df["job_word_count"] = job_wc

    # ── 8. Create binary label from weak_label ──────────────────
    label_map = {"positive": 1, "negative": 0}
    df[LABEL_COLUMN] = df["weak_label"].map(label_map)

    # ── 9. Select final columns ─────────────────────────────────
    output_cols = ID_COLUMNS + MODEL_FEATURES + [LABEL_COLUMN, "weak_label", "label_confidence"]
    available_cols = [c for c in output_cols if c in df.columns]
    df_out = df[available_cols].copy()

    # Convert education_match from bool to int
    if "education_match" in df_out.columns:
        df_out["education_match"] = df_out["education_match"].astype(float)

    logger.info(f"  Enriched dataset shape: {df_out.shape}")
    return df_out


def create_ml_datasets(processed_dir: Path) -> None:
    """
    Create ml_train, ml_validation, ml_test parquet files from candidate splits.
    """
    logger.info("=" * 60)
    logger.info("ML FEATURE ENGINEERING")
    logger.info("=" * 60)

    df_resumes = pd.read_parquet(processed_dir / "resume_features.parquet")
    df_jobs = pd.read_parquet(processed_dir / "job_features.parquet")

    for split_name, input_file, output_file in [
        ("train", "candidate_train.parquet", "ml_train.parquet"),
        ("validation", "candidate_validation.parquet", "ml_validation.parquet"),
        ("test", "candidate_test.parquet", "ml_test.parquet"),
    ]:
        logger.info(f"\nProcessing {split_name} split...")
        df_split = pd.read_parquet(processed_dir / input_file)
        df_enriched = enrich_candidate_pairs(df_split, df_resumes, df_jobs, processed_dir)
        df_enriched.to_parquet(processed_dir / output_file, index=False, engine="pyarrow")
        logger.info(f"  Saved {output_file}: {df_enriched.shape}")

    logger.info("\nML feature engineering complete.")


if __name__ == "__main__":
    processed_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    create_ml_datasets(processed_dir)
