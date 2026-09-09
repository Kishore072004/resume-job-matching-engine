import logging
import re
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_text(text: str, lowercase: bool = False) -> str:
    """
    Normalize text by stripping whitespace, replacing newlines/tabs with spaces,
    and collapsing multiple spaces into a single space.
    """
    if pd.isna(text) or text is None:
        return ""
    text_str = str(text)
    # Replace line breaks and tabs with spaces
    text_clean = re.sub(r"[\r\n\t]+", " ", text_str)
    # Collapse multiple consecutive spaces
    text_clean = re.sub(r"\s+", " ", text_clean).strip()
    if lowercase:
        text_clean = text_clean.lower()
    return text_clean


def clean_skills(skills_csv_path: Path) -> (pd.DataFrame, pd.DataFrame):
    """
    Clean ESCO skills and build skill aliases table.
    """
    logger.info(f"Loading ESCO skills from {skills_csv_path}")
    df_raw = pd.read_csv(skills_csv_path, low_memory=False)

    keep_cols = [
        "conceptUri",
        "skillType",
        "reuseLevel",
        "preferredLabel",
        "altLabels",
        "hiddenLabels",
        "status",
        "description",
    ]
    df_skills = df_raw[keep_cols].copy()

    # Deduplicate on conceptUri to ensure unique primary key
    df_skills.drop_duplicates(subset=["conceptUri"], keep="first", inplace=True)

    # Fill NaNs with empty string for string columns
    str_cols = ["skillType", "reuseLevel", "preferredLabel", "altLabels", "hiddenLabels", "status", "description"]
    for col in str_cols:
        df_skills[col] = df_skills[col].fillna("").astype(str)

    # Normalize whitespace & linebreaks while preserving original casing
    for col in ["preferredLabel", "altLabels", "hiddenLabels", "description"]:
        df_skills[col] = df_skills[col].apply(lambda x: normalize_text(x, lowercase=False))

    # Add normalized_label
    df_skills["normalized_label"] = df_skills["preferredLabel"].apply(lambda x: normalize_text(x, lowercase=True))

    # Generate skill aliases table
    alias_records = []
    for _, row in df_raw.iterrows():
        uri = str(row["conceptUri"]).strip()
        pref_raw = str(row["preferredLabel"]).strip() if pd.notna(row["preferredLabel"]) else ""
        pref_clean = normalize_text(pref_raw, lowercase=False)
        pref_norm = normalize_text(pref_raw, lowercase=True)

        if pref_clean:
            alias_records.append({
                "skill_uri": uri,
                "canonical_skill": pref_clean,
                "alias": pref_clean,
                "alias_normalized": pref_norm,
                "alias_type": "preferred",
            })

        # altLabels (split by linebreaks in raw)
        if pd.notna(row["altLabels"]):
            raw_alts = str(row["altLabels"]).split("\n")
            for alt in raw_alts:
                alt_clean = normalize_text(alt, lowercase=False)
                alt_norm = normalize_text(alt, lowercase=True)
                if alt_clean:
                    alias_records.append({
                        "skill_uri": uri,
                        "canonical_skill": pref_clean,
                        "alias": alt_clean,
                        "alias_normalized": alt_norm,
                        "alias_type": "alternative",
                    })

        # hiddenLabels
        if pd.notna(row["hiddenLabels"]):
            raw_hids = str(row["hiddenLabels"]).split("\n")
            for hid in raw_hids:
                hid_clean = normalize_text(hid, lowercase=False)
                hid_norm = normalize_text(hid, lowercase=True)
                if hid_clean:
                    alias_records.append({
                        "skill_uri": uri,
                        "canonical_skill": pref_clean,
                        "alias": hid_clean,
                        "alias_normalized": hid_norm,
                        "alias_type": "hidden",
                    })

    df_aliases = pd.DataFrame(alias_records)
    df_aliases.drop_duplicates(subset=["skill_uri", "alias_normalized", "alias_type"], inplace=True)

    return df_skills, df_aliases


def clean_occupations(occupations_csv_path: Path) -> pd.DataFrame:
    """
    Clean ESCO occupations.
    """
    logger.info(f"Loading ESCO occupations from {occupations_csv_path}")
    df_raw = pd.read_csv(occupations_csv_path, low_memory=False)

    keep_cols = [
        "conceptUri",
        "iscoGroup",
        "preferredLabel",
        "altLabels",
        "hiddenLabels",
        "status",
        "modifiedDate",
        "scopeNote",
        "definition",
        "description",
        "code",
        "naceCode",
    ]
    df_occ = df_raw[keep_cols].copy()

    # Deduplicate on conceptUri
    df_occ.drop_duplicates(subset=["conceptUri"], keep="first", inplace=True)

    # Fill NaNs with empty string
    str_cols = [col for col in keep_cols if col != "conceptUri"]
    for col in str_cols:
        df_occ[col] = df_occ[col].fillna("").astype(str)

    # Normalize whitespace & linebreaks while preserving original casing
    text_cols = ["preferredLabel", "altLabels", "hiddenLabels", "scopeNote", "definition", "description"]
    for col in text_cols:
        df_occ[col] = df_occ[col].apply(lambda x: normalize_text(x, lowercase=False))

    # Add normalized_label
    df_occ["normalized_label"] = df_occ["preferredLabel"].apply(lambda x: normalize_text(x, lowercase=True))

    return df_occ


def clean_relations(relations_csv_path: Path) -> (pd.DataFrame, int):
    """
    Clean ESCO occupation-skill relations.
    Returns cleaned dataframe and missing skillType count.
    """
    logger.info(f"Loading ESCO occupation-skill relations from {relations_csv_path}")
    df_raw = pd.read_csv(relations_csv_path, low_memory=False)

    keep_cols = [
        "occupationUri",
        "occupationLabel",
        "relationType",
        "skillType",
        "skillUri",
        "skillLabel",
    ]
    df_rels = df_raw[keep_cols].copy()

    # Normalize text fields
    for col in ["occupationLabel", "relationType", "skillType", "skillLabel"]:
        df_rels[col] = df_rels[col].fillna("").astype(str).apply(lambda x: normalize_text(x, lowercase=False))

    # Report missing skillType count before filling
    missing_skill_type_cnt = (df_raw["skillType"].isna() | (df_raw["skillType"].str.strip() == "")).sum()

    # Filter invalid rows (must have non-empty occupationUri, skillUri, relationType)
    df_rels = df_rels[
        (df_rels["occupationUri"].str.strip() != "")
        & (df_rels["skillUri"].str.strip() != "")
        & (df_rels["relationType"].str.strip() != "")
    ].copy()

    # Drop duplicate relationship rows
    df_rels.drop_duplicates(subset=["occupationUri", "skillUri", "relationType"], inplace=True)

    return df_rels, int(missing_skill_type_cnt)


def build_occupation_profiles(
    df_occupations: pd.DataFrame,
    df_skills: pd.DataFrame,
    df_relations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ESCO occupation skill profiles.
    """
    logger.info("Building occupation skill profiles...")

    skill_norm_map = dict(zip(df_skills["conceptUri"], df_skills["normalized_label"]))
    rel_grouped = df_relations.groupby("occupationUri")

    profile_records = []

    for _, occ_row in df_occupations.iterrows():
        occ_uri = occ_row["conceptUri"]
        occ_label = occ_row["preferredLabel"]

        if occ_uri in rel_grouped.groups:
            group = rel_grouped.get_group(occ_uri)
            essential_cnt = int((group["relationType"] == "essential").sum())
            optional_cnt = int((group["relationType"] == "optional").sum())
            knowledge_cnt = int((group["skillType"] == "knowledge").sum())
            competence_cnt = int(
                ((group["skillType"] == "skill/competence") | (group["skillType"] == "competence")).sum()
            )
            total_cnt = len(group)

            # Construct canonical skill names list
            skill_list = []
            for _, r_row in group.iterrows():
                s_uri = r_row["skillUri"]
                s_name = skill_norm_map.get(s_uri, normalize_text(r_row["skillLabel"], lowercase=True))
                skill_list.append(s_name)
        else:
            essential_cnt = 0
            optional_cnt = 0
            knowledge_cnt = 0
            competence_cnt = 0
            total_cnt = 0
            skill_list = []

        profile_records.append({
            "occupation_uri": occ_uri,
            "occupation_label": occ_label,
            "essential_skill_count": essential_cnt,
            "optional_skill_count": optional_cnt,
            "knowledge_count": knowledge_cnt,
            "competence_count": competence_cnt,
            "total_skill_count": total_cnt,
            "skills": skill_list,
        })

    df_profiles = pd.DataFrame(profile_records)
    return df_profiles
