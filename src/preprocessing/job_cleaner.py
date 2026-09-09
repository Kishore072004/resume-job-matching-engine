import logging
import re
import unicodedata
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_job_text(text: str) -> str:
    """
    Clean raw job title/description text preserving technical terms and punctuation.
    """
    if pd.isna(text) or not text:
        return ""

    text_str = unicodedata.normalize("NFKC", str(text))
    text_str = text_str.replace("\uff0d", "-").replace("\xa0", " ")

    # Remove HTML tags/artifacts
    text_str = re.sub(r"<[^>]+>", " ", text_str)

    # Normalize whitespace & line breaks
    text_str = text_str.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text_str = re.sub(r"[ \t]+", " ", text_str)
    text_str = re.sub(r"\n{3,}", "\n\n", text_str)

    lines = [line.strip() for line in text_str.split("\n")]
    clean_text = "\n".join(lines).strip()

    return clean_text


def clean_job_postings(postings_csv_path: Path) -> pd.DataFrame:
    """
    Clean primary job postings dataset.
    """
    logger.info(f"Loading job postings from {postings_csv_path}")
    df_raw = pd.read_csv(postings_csv_path, low_memory=False)

    clean_records = []
    for _, row in df_raw.iterrows():
        j_id = str(row["job_id"]).strip()
        c_name = str(row["company_name"]).strip() if pd.notna(row["company_name"]) else ""
        raw_title = str(row["title"]) if pd.notna(row["title"]) else ""
        raw_desc = str(row["description"]) if pd.notna(row["description"]) else ""
        loc = str(row["location"]).strip() if pd.notna(row["location"]) else ""
        comp_id = str(row["company_id"]).strip() if pd.notna(row["company_id"]) else ""

        exp_lvl = str(row["formatted_experience_level"]).strip() if pd.notna(row["formatted_experience_level"]) else ""
        work_t = str(row["formatted_work_type"]).strip() if pd.notna(row["formatted_work_type"]) else (
            str(row["work_type"]).strip() if pd.notna(row["work_type"]) else ""
        )
        remote_a = float(row["remote_allowed"]) if pd.notna(row["remote_allowed"]) else 0.0
        sk_desc = str(row["skills_desc"]).strip() if pd.notna(row["skills_desc"]) else ""

        clean_t = clean_job_text(raw_title)
        clean_d = clean_job_text(raw_desc)

        clean_records.append({
            "job_id": j_id,
            "company_name": c_name,
            "title": raw_title,
            "description": raw_desc,
            "clean_title": clean_t,
            "clean_description": clean_d,
            "location": loc,
            "company_id": comp_id,
            "experience_level": exp_lvl,
            "work_type": work_t,
            "remote_allowed": remote_a,
            "skills_description": sk_desc,
        })

    df_clean = pd.DataFrame(clean_records)
    return df_clean


def check_job_quality(df_jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quality metrics and quality flags for job postings.
    """
    logger.info("Computing job posting quality report...")
    quality_records = []

    for _, row in df_jobs.iterrows():
        j_id = row["job_id"]
        c_title = row["clean_title"]
        c_desc = row["clean_description"]
        exp_lvl = row["experience_level"]
        sk_desc = row["skills_description"]

        t_len = len(c_title)
        d_len = len(c_desc)
        words = c_desc.split()
        w_cnt = len(words)

        has_desc = bool(c_desc.strip())
        has_exp = bool(exp_lvl.strip())
        has_sk_desc = bool(sk_desc.strip())

        if not has_desc:
            q_flag = "missing_description"
        elif w_cnt < 20 or d_len < 100:
            q_flag = "very_short"
        elif w_cnt > 3000 or d_len > 25000:
            q_flag = "very_long"
        else:
            q_flag = "normal"

        quality_records.append({
            "job_id": j_id,
            "title_length": t_len,
            "description_length": d_len,
            "word_count": w_cnt,
            "has_description": has_desc,
            "has_experience_level": has_exp,
            "has_skills_description": has_sk_desc,
            "quality_flag": q_flag,
        })

    return pd.DataFrame(quality_records)


def clean_job_skills(
    job_skills_csv_path: Path,
    skills_mapping_csv_path: Path,
    df_jobs: pd.DataFrame,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Clean and resolve explicit job skills from job_skills.csv and skills mapping.
    Filters out job_ids not present in postings.
    """
    logger.info("Processing explicit job skill relationships...")
    df_js = pd.read_csv(job_skills_csv_path, low_memory=False)
    df_map = pd.read_csv(skills_mapping_csv_path, low_memory=False)

    df_js["job_id"] = df_js["job_id"].astype(str).str.strip()
    df_js["skill_abr"] = df_js["skill_abr"].astype(str).str.strip()

    # Filter to valid job_ids present in postings.csv
    valid_job_ids = set(df_jobs["job_id"])
    df_js = df_js[df_js["job_id"].isin(valid_job_ids)].copy()

    df_map["skill_abr"] = df_map["skill_abr"].astype(str).str.strip()
    df_map["skill_name"] = df_map["skill_name"].astype(str).str.strip()

    mapping_dict = dict(zip(df_map["skill_abr"], df_map["skill_name"]))

    known_records = []
    unknown_records = []

    for _, row in df_js.iterrows():
        j_id = row["job_id"]
        s_abr = row["skill_abr"]

        if s_abr in mapping_dict:
            known_records.append({
                "job_id": j_id,
                "skill_abr": s_abr,
                "skill_name": mapping_dict[s_abr],
            })
        else:
            unknown_records.append({
                "job_id": j_id,
                "skill_abr": s_abr,
            })

    df_known = pd.DataFrame(known_records).drop_duplicates(subset=["job_id", "skill_abr"])
    df_unknown = pd.DataFrame(unknown_records)
    if df_unknown.empty:
        df_unknown = pd.DataFrame(columns=["job_id", "skill_abr"])

    return df_known, df_unknown


def clean_company_info(companies_csv_path: Path, company_ind_csv_path: Path) -> pd.DataFrame:
    """
    Clean and merge company metadata and industry sectors.
    """
    logger.info("Cleaning company metadata...")
    df_comp = pd.read_csv(companies_csv_path, low_memory=False)
    df_ind = pd.read_csv(company_ind_csv_path, low_memory=False)

    df_comp["company_id"] = df_comp["company_id"].astype(str).str.strip()
    df_ind["company_id"] = df_ind["company_id"].astype(str).str.strip()

    ind_grouped = df_ind.groupby("company_id")["industry"].apply(
        lambda s: ", ".join(dict.fromkeys([str(x).strip() for x in s if pd.notna(x)]))
    ).reset_index()

    df_merged = pd.merge(df_comp, ind_grouped, on="company_id", how="left")
    df_merged["industry"] = df_merged["industry"].fillna("")
    df_merged["company_name"] = df_merged["name"].fillna("").astype(str).str.strip()
    df_merged["company_size"] = df_merged["company_size"].fillna("").astype(str)

    keep_cols = ["company_id", "company_name", "industry", "company_size"]
    df_comp_clean = df_merged[keep_cols].copy()
    df_comp_clean.drop_duplicates(subset=["company_id"], inplace=True)
    return df_comp_clean


def extract_job_esco_skills(
    df_jobs: pd.DataFrame,
    df_esco_skills: pd.DataFrame,
    df_esco_aliases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract ESCO skill matches from job title, description, and skills_description.
    """
    logger.info("Extracting ESCO skills from job postings...")

    # Build alias map
    alias_map = {}
    for _, row in df_esco_aliases.iterrows():
        norm = str(row["alias_normalized"]).strip()
        if len(norm) < 2 and norm not in {"r", "c"}:
            continue
        if norm not in alias_map:
            alias_map[norm] = (row["skill_uri"], row["canonical_skill"], row["alias_type"])

    skill_records = []

    for _, row in df_jobs.iterrows():
        j_id = row["job_id"]
        c_title = row["clean_title"]
        c_desc = row["clean_description"]
        sk_desc = row["skills_description"]

        combined_text = f"{c_title} {c_desc} {sk_desc}".strip()
        if not combined_text:
            continue

        text_clean = re.sub(r"[\r\n\t]+", " ", combined_text)
        words = [w.strip(',;:()[]{}!"\'') for w in text_clean.split() if w.strip(',;:()[]{}!"\'')]
        words_norm = [w.lower() for w in words]

        detected = {}
        n = len(words_norm)
        max_n = 5

        for k in range(1, min(max_n + 1, n + 1)):
            for i in range(n - k + 1):
                phrase = " ".join(words_norm[i : i + k])
                if phrase in alias_map:
                    s_uri, canonical, alias_type = alias_map[phrase]

                    if alias_type == "preferred":
                        method = "preferred_label"
                        conf = 1.0
                    elif alias_type == "alternative":
                        method = "alternative_label"
                        conf = 0.9
                    else:
                        method = "normalized_match"
                        conf = 0.8

                    matched_orig = " ".join(words[i : i + k])

                    if s_uri not in detected or conf > detected[s_uri]["confidence"]:
                        detected[s_uri] = {
                            "job_id": j_id,
                            "skill_uri": s_uri,
                            "canonical_skill": canonical,
                            "matched_text": matched_orig,
                            "match_method": method,
                            "confidence": conf,
                        }

        skill_records.extend(list(detected.values()))

    df_job_esco = pd.DataFrame(skill_records)
    if df_job_esco.empty:
        df_job_esco = pd.DataFrame(columns=["job_id", "skill_uri", "canonical_skill", "matched_text", "match_method", "confidence"])
    return df_job_esco


def build_job_skill_profile(
    df_jobs: pd.DataFrame,
    df_explicit_skills: pd.DataFrame,
    df_esco_skills: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine explicit job skills and ESCO extracted skills into job skill profile.
    """
    logger.info("Combining explicit and ESCO job skill profiles...")

    explicit_grouped = df_explicit_skills.groupby("job_id")["skill_name"].apply(lambda s: list(dict.fromkeys(s))).to_dict() if not df_explicit_skills.empty else {}
    esco_grouped = df_esco_skills.groupby("job_id")["canonical_skill"].apply(lambda s: list(dict.fromkeys(s))).to_dict() if not df_esco_skills.empty else {}

    profile_records = []

    for _, row in df_jobs.iterrows():
        j_id = row["job_id"]

        exp_s = explicit_grouped.get(j_id, [])
        esco_s = esco_grouped.get(j_id, [])

        comb_s = list(dict.fromkeys(exp_s + esco_s))

        profile_records.append({
            "job_id": j_id,
            "explicit_skill_count": len(exp_s),
            "esco_skill_count": len(esco_s),
            "combined_skill_count": len(comb_s),
            "explicit_skills": exp_s,
            "esco_skills": esco_s,
            "combined_skills": comb_s,
        })

    return pd.DataFrame(profile_records)


def extract_job_experience(df_jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Extract min/max years of experience from experience level and description.
    """
    logger.info("Extracting job experience requirements...")

    exp_patterns = [
        r"(\d+)\s*(?:-|to)\s*(\d+)\s*\+?\s*(?:years?|yrs?)",
        r"(\d+)\s*\+\s*(?:years?|yrs?)",
        r"(\d+)\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)",
        r"(?:minimum|at\s+least)\s+(\d+)\s*(?:years?|yrs?)",
    ]

    exp_records = []

    for _, row in df_jobs.iterrows():
        j_id = row["job_id"]
        exp_lvl = row["experience_level"]
        c_desc = row["clean_description"]

        text_clean = re.sub(r"[\r\n\t]+", " ", f"{exp_lvl} {c_desc}")

        min_y = None
        max_y = None
        exp_text = exp_lvl
        conf = 0.70 if exp_lvl else None

        for pat in exp_patterns:
            m = re.search(pat, text_clean, flags=re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 2 and groups[1] is not None:
                    min_y = float(groups[0])
                    max_y = float(groups[1])
                else:
                    min_y = float(groups[0])
                    max_y = None
                exp_text = m.group(0).strip()
                conf = 0.85
                break

        exp_records.append({
            "job_id": j_id,
            "experience_level": exp_lvl if exp_lvl else "Not Specified",
            "min_years_experience": min_y,
            "max_years_experience": max_y,
            "experience_text": exp_text if exp_text else "",
            "confidence": conf,
        })

    return pd.DataFrame(exp_records)


def extract_job_education(df_jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Extract education requirements from job title and description.
    """
    logger.info("Extracting job education requirements...")

    edu_patterns = [
        (r"\b(?:PhD|Ph\.D\.|Doctorate|Doctor\s+of\s+Philosophy)\b", "PhD", 5),
        (r"\b(?:Master|Master's|Masters|M\.S\.|MS|M\.Sc|M\.Tech|M\.E\.|MBA|M\.B\.A\.)\b", "Master", 4),
        (r"\b(?:Bachelor|Bachelor's|Bachelors|B\.S\.|BS|B\.Sc|B\.Tech|B\.E\.|B\.A\.|BA|Degree)\b", "Bachelor", 3),
        (r"\b(?:Associate|A\.S\.|A\.A\.)\b", "Associate", 2),
        (r"\b(?:Diploma|Certification|Certificate)\b", "Diploma", 1),
    ]

    edu_records = []

    for _, row in df_jobs.iterrows():
        j_id = row["job_id"]
        c_title = row["clean_title"]
        c_desc = row["clean_description"]

        combined = f"{c_title} {c_desc}"
        matches = []

        for pat, level, rank in edu_patterns:
            for m in re.finditer(pat, combined, flags=re.IGNORECASE):
                matches.append((rank, level, m.group(0)))

        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            best_rank, best_level, best_text = matches[0]
            all_texts = list(dict.fromkeys([m[2] for m in matches]))
            edu_records.append({
                "job_id": j_id,
                "education_requirement": best_level,
                "education_text": ", ".join(all_texts[:5]),
                "confidence": 0.85,
            })
        else:
            edu_records.append({
                "job_id": j_id,
                "education_requirement": "Not Specified",
                "education_text": "",
                "confidence": 0.0,
            })

    return pd.DataFrame(edu_records)


def extract_job_titles(df_jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and normalize job titles.
    """
    logger.info("Extracting job title representations...")

    title_records = []
    for _, row in df_jobs.iterrows():
        j_id = row["job_id"]
        orig_title = row["title"]
        c_title = row["clean_title"]

        norm_title = re.sub(r"[^\w\s]", "", c_title.lower()).strip()
        norm_title = re.sub(r"\s+", " ", norm_title)

        title_records.append({
            "job_id": j_id,
            "title": orig_title,
            "normalized_title": norm_title,
        })

    return pd.DataFrame(title_records)


def build_master_job_features(
    df_clean: pd.DataFrame,
    df_quality: pd.DataFrame,
    df_profile: pd.DataFrame,
    df_exp: pd.DataFrame,
    df_edu: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build master job feature table joining features per job_id.
    """
    logger.info("Constructing master job feature table...")

    quality_map = dict(zip(df_quality["job_id"], df_quality["quality_flag"]))
    exp_min_map = dict(zip(df_exp["job_id"], df_exp["min_years_experience"]))
    exp_max_map = dict(zip(df_exp["job_id"], df_exp["max_years_experience"]))
    edu_map = dict(zip(df_edu["job_id"], df_edu["education_requirement"]))

    explicit_skills_map = dict(zip(df_profile["job_id"], df_profile["explicit_skills"]))
    esco_skills_map = dict(zip(df_profile["job_id"], df_profile["esco_skills"]))
    combined_skills_map = dict(zip(df_profile["job_id"], df_profile["combined_skills"]))

    master_records = []

    for _, row in df_clean.iterrows():
        j_id = row["job_id"]
        orig_t = row["title"]
        clean_t = row["clean_title"]
        clean_d = row["clean_description"]
        loc = row["location"]
        exp_lvl = row["experience_level"]

        q_flag = quality_map.get(j_id, "normal")
        min_y = exp_min_map.get(j_id, None)
        max_y = exp_max_map.get(j_id, None)
        edu_req = edu_map.get(j_id, "Not Specified")

        exp_s = explicit_skills_map.get(j_id, [])
        esco_s = esco_skills_map.get(j_id, [])
        comb_s = combined_skills_map.get(j_id, [])

        master_records.append({
            "job_id": j_id,
            "title": orig_t,
            "clean_title": clean_t,
            "clean_description": clean_d,
            "location": loc,
            "experience_level": exp_lvl if exp_lvl else "Not Specified",
            "min_years_experience": min_y,
            "max_years_experience": max_y,
            "education_requirement": edu_req,
            "explicit_skills": exp_s,
            "esco_skills": esco_s,
            "combined_skills": comb_s,
            "skill_count": len(comb_s),
            "quality_flag": q_flag,
        })

    return pd.DataFrame(master_records)
