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


def clean_resume_text(text: str) -> str:
    """
    Clean raw resume text while preserving technical terms, symbols, and meaningful punctuation.
    """
    if pd.isna(text) or not text:
        return ""

    # Normalize Unicode (NFKC)
    text_str = unicodedata.normalize("NFKC", str(text))

    # Replace fullwidth / non-standard unicode dashes & spaces
    text_str = text_str.replace("\uff0d", "-").replace("\xa0", " ")

    # Strip HTML tags/artifacts if any present
    text_str = re.sub(r"<[^>]+>", " ", text_str)

    # Normalize whitespace & line breaks while keeping paragraph structure
    # Replace carriage returns
    text_str = text_str.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs with spaces
    text_str = text_str.replace("\t", " ")

    # Collapse horizontal spaces (not newlines)
    text_str = re.sub(r"[ \t]+", " ", text_str)

    # Collapse 3+ consecutive newlines into double newlines
    text_str = re.sub(r"\n{3,}", "\n\n", text_str)

    # Clean lines
    lines = [line.strip() for line in text_str.split("\n")]
    clean_text = "\n".join(lines).strip()

    return clean_text


def check_resume_quality(df_resumes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute text statistics and quality flags for resumes.
    """
    logger.info("Computing resume quality metrics...")
    quality_records = []

    # Identify duplicate texts
    text_counts = df_resumes["clean_text"].value_counts()
    duplicate_texts = set(text_counts[text_counts > 1].index)

    for _, row in df_resumes.iterrows():
        r_id = row["resume_id"]
        c_text = row["clean_text"]

        char_cnt = len(c_text)
        words = c_text.split()
        word_cnt = len(words)
        line_cnt = len(c_text.splitlines()) if c_text else 0

        is_empty = (word_cnt == 0 or char_cnt == 0)
        is_dup = c_text in duplicate_texts if c_text else False

        # Quality flag assignment
        if is_empty or word_cnt < 30 or char_cnt < 100:
            q_flag = "very_short"
        elif word_cnt > 3000 or char_cnt > 25000:
            q_flag = "very_long"
        elif is_dup:
            q_flag = "duplicate_text"
        else:
            q_flag = "normal"

        quality_records.append({
            "resume_id": r_id,
            "character_count": char_cnt,
            "word_count": word_cnt,
            "line_count": line_cnt,
            "quality_flag": q_flag,
            "is_empty": is_empty,
            "is_duplicate": is_dup,
        })

    return pd.DataFrame(quality_records)


def detect_sections(df_resumes: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based section detection using canonical section mapping.
    """
    logger.info("Detecting resume sections...")

    section_keywords = [
        (r"\b(?:SUMMARY|OBJECTIVE|PROFESSIONAL SUMMARY|EXECUTIVE SUMMARY|CAREER OBJECTIVE|ABOUT ME)\b", "SUMMARY"),
        (r"\b(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES|KEY SKILLS|AREAS OF EXPERTISE|HIGHLIGHTS|TECHNICAL STACK|COMPETENCIES)\b", "SKILLS"),
        (r"\b(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT|WORK HISTORY|CAREER HISTORY|EMPLOYMENT HISTORY)\b", "EXPERIENCE"),
        (r"\b(?:EDUCATION|ACADEMIC BACKGROUND|EDUCATION AND TRAINING|QUALIFICATIONS|ACADEMIC QUALIFICATIONS)\b", "EDUCATION"),
        (r"\b(?:PROJECTS|KEY PROJECTS|PERSONAL PROJECTS|PORTFOLIO)\b", "PROJECTS"),
        (r"\b(?:CERTIFICATIONS|CERTIFICATES|LICENSES|CERTIFICATIONS AND LICENSES|TRAINING)\b", "CERTIFICATIONS"),
        (r"\b(?:ACHIEVEMENTS|ACCOMPLISHMENTS|HONORS|AWARDS)\b", "ACHIEVEMENTS"),
        (r"\b(?:LANGUAGES|LANGUAGES SPOKEN)\b", "LANGUAGES"),
    ]

    header_regex = re.compile(
        r"^(?:" + "|".join([pat for pat, _ in section_keywords]) + r")(?:\s*:)?$",
        flags=re.IGNORECASE,
    )

    section_records = []

    for _, row in df_resumes.iterrows():
        r_id = row["resume_id"]
        c_text = row["clean_text"]
        lines = c_text.splitlines()

        current_sec_name = "SUMMARY"
        current_sec_lines = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if line matches a section header
            matched_sec = None
            for pat, sec_name in section_keywords:
                if re.match(r"^" + pat + r"(?:\s*:)?$", line_str, flags=re.IGNORECASE):
                    matched_sec = sec_name
                    break

            if matched_sec:
                # Save previous section if it has content
                if current_sec_lines:
                    sec_text = "\n".join(current_sec_lines).strip()
                    if sec_text:
                        section_records.append({
                            "resume_id": r_id,
                            "section_name": current_sec_name,
                            "section_text": sec_text,
                        })
                current_sec_name = matched_sec
                current_sec_lines = []
            else:
                current_sec_lines.append(line_str)

        # Flush final section
        if current_sec_lines:
            sec_text = "\n".join(current_sec_lines).strip()
            if sec_text:
                section_records.append({
                    "resume_id": r_id,
                    "section_name": current_sec_name,
                    "section_text": sec_text,
                })

    df_sec = pd.DataFrame(section_records)
    return df_sec


def extract_resume_skills(
    df_resumes: pd.DataFrame,
    df_skills: pd.DataFrame,
    df_aliases: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deterministic phrase and token skill extractor using ESCO skills & aliases.
    """
    logger.info("Extracting ESCO skills from resumes...")

    # Build lookup map: normalized_alias -> (skill_uri, canonical_skill, alias_type)
    alias_map = {}
    for _, row in df_aliases.iterrows():
        norm = str(row["alias_normalized"]).strip()
        if len(norm) < 2 and norm not in {"r", "c"}:
            continue
        if norm not in alias_map:
            alias_map[norm] = (row["skill_uri"], row["canonical_skill"], row["alias_type"])

    skill_records = []

    for _, row in df_resumes.iterrows():
        r_id = row["resume_id"]
        c_text = row["clean_text"]

        text_clean = re.sub(r"[\r\n\t]+", " ", c_text)
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

                    # Assign confidence and match method
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

                    # Keep best confidence match for same skill_uri in resume
                    if s_uri not in detected or conf > detected[s_uri]["confidence"]:
                        detected[s_uri] = {
                            "resume_id": r_id,
                            "skill_uri": s_uri,
                            "canonical_skill": canonical,
                            "matched_text": matched_orig,
                            "match_method": method,
                            "confidence": conf,
                        }

        skill_records.extend(list(detected.values()))

    df_res_skills = pd.DataFrame(skill_records)
    if df_res_skills.empty:
        df_res_skills = pd.DataFrame(columns=["resume_id", "skill_uri", "canonical_skill", "matched_text", "match_method", "confidence"])
    return df_res_skills


def extract_experience(df_resumes: pd.DataFrame) -> pd.DataFrame:
    """
    Extract years of experience and experience mentions from resume text.
    """
    logger.info("Extracting years of experience...")

    num_word_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    }

    exp_patterns = [
        r"(\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\b)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|work|career)",
        r"(?:experience|exp|work\s+experience|career)(?:\s+of)?\s*(?:about|over|around|more\s+than)?\s*(\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\b)\s*\+?\s*(?:years?|yrs?)",
        r"(\d+)\s*\+\s*(?:years?|yrs?)",
    ]

    exp_records = []

    for _, row in df_resumes.iterrows():
        r_id = row["resume_id"]
        c_text = row["clean_text"]

        text_clean = re.sub(r"[\r\n\t]+", " ", c_text)
        years = []
        mentions = []

        for pat in exp_patterns:
            for m in re.finditer(pat, text_clean, flags=re.IGNORECASE):
                full_match = m.group(0).strip()
                num_str = m.group(1).lower() if m.groups() else ""
                if num_str.isdigit():
                    y = float(num_str)
                elif num_str in num_word_map:
                    y = float(num_word_map[num_str])
                else:
                    y = None

                if y is not None and 0 <= y <= 50:
                    years.append(y)
                    mentions.append(full_match)

        if years:
            max_yrs = float(max(years))
            exp_records.append({
                "resume_id": r_id,
                "years_experience": max_yrs,
                "experience_mentions": list(dict.fromkeys(mentions)),
                "confidence": 0.85,
            })
        else:
            exp_records.append({
                "resume_id": r_id,
                "years_experience": None,
                "experience_mentions": [],
                "confidence": None,
            })

    return pd.DataFrame(exp_records)


def extract_education(df_resumes: pd.DataFrame) -> pd.DataFrame:
    """
    Extract education level and education text.
    """
    logger.info("Extracting education details...")

    edu_patterns = [
        (r"\b(?:PhD|Ph\.D\.|Doctorate|Doctor\s+of\s+Philosophy)\b", "PhD", 5),
        (r"\b(?:Master|Masters|M\.S\.|MS|M\.Sc|M\.Tech|M\.E\.|MBA|M\.B\.A\.)\b", "Master", 4),
        (r"\b(?:Bachelor|Bachelors|B\.S\.|BS|B\.Sc|B\.Tech|B\.E\.|B\.A\.|BA)\b", "Bachelor", 3),
        (r"\b(?:Associate|A\.S\.|A\.A\.)\b", "Associate", 2),
        (r"\b(?:Diploma|Certification|Certificate)\b", "Diploma", 1),
    ]

    edu_records = []

    for _, row in df_resumes.iterrows():
        r_id = row["resume_id"]
        c_text = row["clean_text"]

        text_clean = re.sub(r"[\r\n\t]+", " ", c_text)
        matches = []

        for pat, level, rank in edu_patterns:
            for m in re.finditer(pat, text_clean, flags=re.IGNORECASE):
                matches.append((rank, level, m.group(0)))

        if matches:
            matches.sort(key=lambda x: x[0], reverse=True)
            best_rank, best_level, best_text = matches[0]
            all_texts = list(dict.fromkeys([m[2] for m in matches]))
            edu_records.append({
                "resume_id": r_id,
                "education_level": best_level,
                "education_text": ", ".join(all_texts[:5]),
                "confidence": 0.90,
            })
        else:
            edu_records.append({
                "resume_id": r_id,
                "education_level": "Not Specified",
                "education_text": "",
                "confidence": 0.0,
            })

    return pd.DataFrame(edu_records)


def extract_job_titles(df_resumes: pd.DataFrame, df_occupations: pd.DataFrame) -> pd.DataFrame:
    """
    Extract candidate job titles from category, header text, and ESCO occupation matches.
    """
    logger.info("Extracting candidate job titles...")

    title_records = []

    for _, row in df_resumes.iterrows():
        r_id = row["resume_id"]
        cat = row["category"]
        c_text = row["clean_text"]

        # 1. Category title
        formatted_cat = cat.replace("-", " ").title()
        title_records.append({
            "resume_id": r_id,
            "job_title": formatted_cat,
            "source": "category",
            "confidence": 0.75,
        })

        # 2. Header title (first non-empty line of resume)
        lines = [l.strip() for l in c_text.splitlines() if l.strip()]
        if lines:
            first_line = lines[0]
            if len(first_line) <= 80 and not re.search(r"summary|objective|resume|curriculum", first_line, re.IGNORECASE):
                title_records.append({
                    "resume_id": r_id,
                    "job_title": first_line,
                    "source": "header_title",
                    "confidence": 0.85,
                })

    df_titles = pd.DataFrame(title_records)
    return df_titles


def build_master_resume_features(
    df_clean: pd.DataFrame,
    df_quality: pd.DataFrame,
    df_skills: pd.DataFrame,
    df_exp: pd.DataFrame,
    df_edu: pd.DataFrame,
    df_titles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct master resume feature table joining all features per resume_id.
    """
    logger.info("Constructing master resume feature table...")

    # Group skills by resume_id
    skills_grouped = df_skills.groupby("resume_id")["canonical_skill"].apply(lambda s: list(dict.fromkeys(s))).to_dict()

    # Group titles by resume_id
    titles_grouped = df_titles.groupby("resume_id")["job_title"].apply(lambda t: list(dict.fromkeys(t))).to_dict()

    # Quality map
    quality_map = dict(zip(df_quality["resume_id"], df_quality["quality_flag"]))

    # Exp map
    exp_map = dict(zip(df_exp["resume_id"], df_exp["years_experience"]))

    # Edu map
    edu_map = dict(zip(df_edu["resume_id"], df_edu["education_level"]))

    master_records = []

    for _, row in df_clean.iterrows():
        r_id = row["resume_id"]
        c_text = row["clean_text"]
        cat = row["category"]

        res_skills = skills_grouped.get(r_id, [])
        res_titles = titles_grouped.get(r_id, [cat.replace("-", " ").title()])
        q_flag = quality_map.get(r_id, "normal")
        yrs = exp_map.get(r_id, None)
        edu = edu_map.get(r_id, "Not Specified")

        master_records.append({
            "resume_id": r_id,
            "clean_text": c_text,
            "category": cat,
            "skills": res_skills,
            "skill_count": len(res_skills),
            "years_experience": yrs,
            "education_level": edu,
            "job_titles": res_titles,
            "quality_flag": q_flag,
        })

    return pd.DataFrame(master_records)
