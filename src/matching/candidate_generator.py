import logging
import random
import re
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from difflib import SequenceMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load candidate generation YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_title_similarity(t1: str, t2: str) -> float:
    """Calculate normalized sequence matcher similarity between two job titles."""
    if not t1 or not t2:
        return 0.0
    clean1 = re.sub(r"[^\w\s]", "", str(t1).lower()).strip()
    clean2 = re.sub(r"[^\w\s]", "", str(t2).lower()).strip()
    if clean1 == clean2:
        return 1.0
    return SequenceMatcher(None, clean1, clean2).ratio()


def retrieve_title_candidates(
    df_resumes: pd.DataFrame,
    df_jobs: pd.DataFrame,
    max_jobs_per_resume: int = 20,
) -> pd.DataFrame:
    """
    Strategy A: Retrieve candidate jobs matching resume category / candidate job titles.
    Uses vectorized operations for building the inverted title word index.
    """
    logger.info("Strategy A: Retrieving title/category candidate pairs...")

    # Build title word index using vectorized operations
    logger.info("  Building job title word index (vectorized)...")
    title_words_df = df_jobs[["job_id", "clean_title"]].copy()
    title_words_df["clean_title"] = title_words_df["clean_title"].astype(str).str.lower()
    # Extract words as lists, then explode
    title_words_df["words"] = title_words_df["clean_title"].str.findall(r"\w{3,}")
    title_words_exploded = title_words_df[["job_id", "words"]].explode("words").dropna(subset=["words"])
    job_title_index = title_words_exploded.groupby("words")["job_id"].apply(list).to_dict()
    logger.info(f"  Title index built: {len(job_title_index)} unique words")

    title_records = []

    for _, r_row in df_resumes.iterrows():
        r_id = r_row["resume_id"]
        cat = str(r_row["category"]).replace("-", " ").lower()
        r_titles = r_row["job_titles"] if isinstance(r_row["job_titles"], (list, np.ndarray)) else [cat]
        r_query = " ".join([cat] + list(r_titles)).lower()
        query_words = set(re.findall(r"\w+", r_query))

        candidate_job_scores = {}
        for w in query_words:
            if w in job_title_index:
                for j_id in job_title_index[w]:
                    candidate_job_scores[j_id] = candidate_job_scores.get(j_id, 0) + 1

        top_job_ids = sorted(candidate_job_scores.keys(), key=lambda k: candidate_job_scores[k], reverse=True)[:max_jobs_per_resume]

        for j_id in top_job_ids:
            title_records.append({
                "resume_id": r_id,
                "job_id": j_id,
                "retrieval_method": "title",
                "retrieval_score": float(candidate_job_scores[j_id] / max(1, len(query_words))),
            })

    logger.info(f"  Title candidate pairs generated: {len(title_records)}")
    return pd.DataFrame(title_records)


def retrieve_skill_candidates(
    df_resumes: pd.DataFrame,
    df_jobs: pd.DataFrame,
    max_jobs_per_resume: int = 20,
) -> pd.DataFrame:
    """
    Strategy B & C: Retrieve candidate jobs sharing meaningful skills with resume.
    Uses vectorized pandas operations for building the inverted skill index.
    """
    logger.info("Strategy B & C: Retrieving skill overlap candidate pairs...")

    # Build inverted skill index using vectorized explode instead of iterrows
    logger.info("  Building job skill inverted index (vectorized)...")
    job_skills_series = df_jobs[["job_id", "combined_skills"]].copy()
    # Explode the combined_skills arrays into individual rows
    job_skills_exploded = job_skills_series.explode("combined_skills").dropna(subset=["combined_skills"])
    job_skills_exploded["combined_skills"] = job_skills_exploded["combined_skills"].astype(str).str.lower().str.strip()
    job_skills_exploded = job_skills_exploded[job_skills_exploded["combined_skills"] != ""]

    # Group by skill to create inverted index
    job_skill_index = job_skills_exploded.groupby("combined_skills")["job_id"].apply(list).to_dict()
    logger.info(f"  Skill index built: {len(job_skill_index)} unique skills across jobs")

    skill_records = []
    n_resumes = len(df_resumes)

    for idx, (_, r_row) in enumerate(df_resumes.iterrows()):
        if idx % 500 == 0:
            logger.info(f"  Processing resume {idx+1}/{n_resumes}...")
        r_id = r_row["resume_id"]
        r_skills = r_row["skills"] if isinstance(r_row["skills"], (list, np.ndarray)) else []
        r_skills_norm = [str(s).lower().strip() for s in r_skills if str(s).strip()]
        r_skill_set = set(r_skills_norm)

        if not r_skill_set:
            continue

        candidate_scores = {}
        for s in r_skill_set:
            if s in job_skill_index:
                for j_id in job_skill_index[s]:
                    candidate_scores[j_id] = candidate_scores.get(j_id, 0) + 1

        top_job_ids = sorted(candidate_scores.keys(), key=lambda k: candidate_scores[k], reverse=True)[:max_jobs_per_resume]

        for j_id in top_job_ids:
            shared = candidate_scores[j_id]
            score = float(shared / max(1, len(r_skill_set)))
            skill_records.append({
                "resume_id": r_id,
                "job_id": j_id,
                "retrieval_method": "skills",
                "retrieval_score": score,
            })

    logger.info(f"  Skill candidate pairs generated: {len(skill_records)}")
    return pd.DataFrame(skill_records)


def retrieve_semantic_candidates(
    df_resumes: pd.DataFrame,
    df_jobs: pd.DataFrame,
    top_k: int = 15,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    cache_dir: Path = None,
) -> pd.DataFrame:
    """
    Strategy D: Retrieve candidate jobs using SentenceTransformers semantic embedding similarity.
    Caches embeddings to disk for instant execution.
    """
    logger.info(f"Strategy D: Retrieving semantic candidate pairs using {model_name}...")
    from sentence_transformers import SentenceTransformer

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

    res_emb_file = cache_dir / "resume_embeddings.npy"
    job_emb_file = cache_dir / "job_embeddings.npy"
    job_ids_file = cache_dir / "job_embedding_ids.npy"

    df_jobs_subset = df_jobs.sort_values(by="skill_count", ascending=False).head(15000).copy()
    job_ids = list(df_jobs_subset["job_id"])

    if res_emb_file.exists() and job_emb_file.exists() and job_ids_file.exists():
        logger.info("Loading cached sentence embeddings from disk...")
        resume_embeddings = np.load(res_emb_file)
        job_embeddings = np.load(job_emb_file)
        job_ids = list(np.load(job_ids_file))
    else:
        logger.info("Computing sentence embeddings using SentenceTransformer...")
        model = SentenceTransformer(model_name)

        resume_texts = [
            (str(row["category"]) + " " + " ".join(list(row["skills"])[:15]) + " " + str(row["clean_text"])[:400]).strip()
            for _, row in df_resumes.iterrows()
        ]
        logger.info(f"  Encoding {len(resume_texts)} resume representations...")
        resume_embeddings = model.encode(
            resume_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
        ).astype(np.float32)
        logger.info(f"  Resume embeddings shape: {resume_embeddings.shape}")

        job_texts = [
            (str(row["clean_title"]) + " " + str(row["clean_description"])[:300]).strip()
            for _, row in df_jobs_subset.iterrows()
        ]
        logger.info(f"  Encoding {len(job_texts)} job representations...")
        job_embeddings = model.encode(
            job_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
        ).astype(np.float32)
        logger.info(f"  Job embeddings shape: {job_embeddings.shape}")

        np.save(res_emb_file, resume_embeddings)
        np.save(job_emb_file, job_embeddings)
        np.save(job_ids_file, np.array(job_ids))

    # Matrix similarity
    sim_matrix = np.dot(resume_embeddings, job_embeddings.T)

    semantic_records = []
    res_ids = list(df_resumes["resume_id"])

    for i, r_id in enumerate(res_ids):
        sims = sim_matrix[i]
        top_k_idx = np.argpartition(sims, -top_k)[-top_k:]
        sorted_top_k = sorted(top_k_idx, key=lambda idx: sims[idx], reverse=True)

        for idx in sorted_top_k:
            j_id = job_ids[idx]
            score = float(sims[idx])
            semantic_records.append({
                "resume_id": r_id,
                "job_id": j_id,
                "retrieval_method": "semantic",
                "retrieval_score": score,
                "semantic_sim": score,
            })

    logger.info(f"  Semantic candidate pairs generated: {len(semantic_records)}")
    return pd.DataFrame(semantic_records)


def combine_and_featurize_candidates(
    df_resumes: pd.DataFrame,
    df_jobs: pd.DataFrame,
    candidate_dfs: list,
    config: dict,
) -> pd.DataFrame:
    """
    Merge candidate pairs, add random negatives, and compute feature metrics.
    """
    logger.info("Merging candidate pairs from all strategies...")

    res_dict = df_resumes.set_index("resume_id").to_dict("index")
    job_dict = df_jobs.set_index("job_id").to_dict("index")

    all_job_ids = list(df_jobs["job_id"])

    pair_methods = {}
    pair_scores = {}
    pair_semantic = {}

    for df_c in candidate_dfs:
        if df_c.empty:
            continue
        for _, row in df_c.iterrows():
            key = (row["resume_id"], row["job_id"])
            method = row["retrieval_method"]
            score = float(row["retrieval_score"])

            if key not in pair_methods:
                pair_methods[key] = set()
                pair_scores[key] = score
            pair_methods[key].add(method)
            pair_scores[key] = max(pair_scores[key], score)

            if "semantic_sim" in row and pd.notna(row["semantic_sim"]):
                pair_semantic[key] = float(row["semantic_sim"])

    # Add Random Negatives per resume
    random.seed(config.get("random_seed", 42))
    for r_id in df_resumes["resume_id"]:
        for _ in range(3):
            j_id = random.choice(all_job_ids)
            key = (r_id, j_id)
            if key not in pair_methods:
                pair_methods[key] = {"random_negative"}
                pair_scores[key] = 0.0

    logger.info(f"Total unique candidate pairs generated: {len(pair_methods)}")

    edu_rank = {"PhD": 5, "Master": 4, "Bachelor": 3, "Associate": 2, "Diploma": 1, "Not Specified": 0}

    feature_records = []

    for (r_id, j_id), methods in pair_methods.items():
        r_info = res_dict.get(r_id)
        j_info = job_dict.get(j_id)

        if not r_info or not j_info:
            continue

        r_skills = set([str(s).lower().strip() for s in r_info["skills"]]) if isinstance(r_info["skills"], (list, np.ndarray)) else set()
        j_skills = set([str(s).lower().strip() for s in j_info["combined_skills"]]) if isinstance(j_info["combined_skills"], (list, np.ndarray)) else set()

        shared_set = r_skills & j_skills
        shared_cnt = len(shared_set)
        r_skill_cnt = len(r_skills)
        j_skill_cnt = len(j_skills)

        union_cnt = len(r_skills | j_skills)
        jaccard = float(shared_cnt / union_cnt) if union_cnt > 0 else 0.0

        cov_resume = float(shared_cnt / r_skill_cnt) if r_skill_cnt > 0 else 0.0
        cov_job = float(shared_cnt / j_skill_cnt) if j_skill_cnt > 0 else 0.0

        r_cat = str(r_info["category"]).replace("-", " ")
        r_titles = r_info["job_titles"] if isinstance(r_info["job_titles"], (list, np.ndarray)) else [r_cat]
        j_title = str(j_info["clean_title"])

        title_sim = max([calculate_title_similarity(t, j_title) for t in r_titles] + [calculate_title_similarity(r_cat, j_title)])

        semantic_sim = pair_semantic.get((r_id, j_id), title_sim * 0.7)

        r_exp = r_info.get("years_experience")
        j_min_exp = j_info.get("min_years_experience")
        j_max_exp = j_info.get("max_years_experience")

        if r_exp is not None and j_min_exp is not None:
            exp_gap = float(r_exp - j_min_exp)
        else:
            exp_gap = None

        r_edu = r_info.get("education_level", "Not Specified")
        j_edu = j_info.get("education_requirement", "Not Specified")
        edu_match = bool(edu_rank.get(r_edu, 0) >= edu_rank.get(j_edu, 0))

        method_str = "+".join(sorted(list(methods)))

        feature_records.append({
            "resume_id": r_id,
            "job_id": j_id,
            "retrieval_method": method_str,
            "retrieval_score": pair_scores[(r_id, j_id)],
            "shared_skill_count": shared_cnt,
            "resume_skill_count": r_skill_cnt,
            "job_skill_count": j_skill_cnt,
            "skill_jaccard": jaccard,
            "skill_coverage_resume": cov_resume,
            "skill_coverage_job": cov_job,
            "title_similarity": float(title_sim),
            "semantic_similarity": float(semantic_sim),
            "resume_years_experience": float(r_exp) if r_exp is not None else None,
            "job_min_years_experience": float(j_min_exp) if j_min_exp is not None else None,
            "job_max_years_experience": float(j_max_exp) if j_max_exp is not None else None,
            "experience_gap": exp_gap,
            "education_match": edu_match,
        })

    return pd.DataFrame(feature_records)


def stratify_and_label_pairs(df_pairs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Assign weak labels (positive, negative, ambiguous) and label_reason based on configurable rules.
    """
    logger.info("Assigning weak labels and stratification categories...")

    pos_cfg = config.get("positive_threshold", {})
    min_jaccard = pos_cfg.get("min_skill_jaccard", 0.15)
    min_sem = pos_cfg.get("min_semantic_similarity", 0.55)
    min_title = pos_cfg.get("min_title_similarity", 0.45)

    neg_cfg = config.get("negative_threshold", {})
    max_jaccard = neg_cfg.get("max_skill_jaccard", 0.05)
    max_sem = neg_cfg.get("max_semantic_similarity", 0.35)

    weak_labels = []
    reasons = []
    confidences = []

    for _, row in df_pairs.iterrows():
        jaccard = row["skill_jaccard"]
        sem_sim = row["semantic_similarity"]
        title_sim = row["title_similarity"]
        method = row["retrieval_method"]

        if "random_negative" in method:
            w_label = "negative"
            reason = "random_unrelated_negative"
            conf = 0.95
        elif jaccard >= min_jaccard and sem_sim >= min_sem:
            w_label = "positive"
            reason = "high_skill_coverage+high_semantic_similarity"
            conf = 0.90
        elif title_sim >= min_title and sem_sim >= min_sem:
            w_label = "positive"
            reason = "high_title_similarity+high_semantic_similarity"
            conf = 0.85
        elif jaccard <= max_jaccard and sem_sim <= max_sem:
            w_label = "negative"
            reason = "low_skill_overlap+low_semantic_similarity"
            conf = 0.90
        elif (sem_sim >= 0.40 or title_sim >= 0.35) and jaccard <= 0.05:
            w_label = "negative"
            reason = "hard_negative_moderate_sim_low_skill"
            conf = 0.85
        else:
            w_label = "ambiguous"
            reason = "moderate_similarity"
            conf = 0.50

        weak_labels.append(w_label)
        reasons.append(reason)
        confidences.append(conf)

    df_pairs["weak_label"] = weak_labels
    df_pairs["label_reason"] = reasons
    df_pairs["label_confidence"] = confidences
    return df_pairs


def split_candidates_no_leakage(df_pairs: pd.DataFrame, config: dict) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split candidate pairs by unique resume_id (70% train, 15% val, 15% test) preventing leakage.
    Excludes ambiguous pairs from initial training splits.
    """
    logger.info("Splitting candidate pairs by resume_id to prevent data leakage...")

    seed = config.get("random_seed", 42)
    train_ratio = config.get("train_ratio", 0.70)
    val_ratio = config.get("validation_ratio", 0.15)

    unique_resumes = sorted(list(df_pairs["resume_id"].unique()))
    random.seed(seed)
    random.shuffle(unique_resumes)

    n_total = len(unique_resumes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_resumes = set(unique_resumes[:n_train])
    val_resumes = set(unique_resumes[n_train : n_train + n_val])
    test_resumes = set(unique_resumes[n_train + n_val:])

    # Exclude ambiguous pairs from training sets
    df_clean_pairs = df_pairs[df_pairs["weak_label"] != "ambiguous"].copy()

    df_train = df_clean_pairs[df_clean_pairs["resume_id"].isin(train_resumes)].copy()
    df_val = df_clean_pairs[df_clean_pairs["resume_id"].isin(val_resumes)].copy()
    df_test = df_clean_pairs[df_clean_pairs["resume_id"].isin(test_resumes)].copy()

    return df_train, df_val, df_test


def create_manual_review_sample(
    df_pairs: pd.DataFrame,
    df_resumes: pd.DataFrame,
    df_jobs: pd.DataFrame,
    sample_size: int = 300,
) -> pd.DataFrame:
    """
    Sample representative candidate pairs across positive, hard negative, random negative, and ambiguous categories.
    """
    logger.info("Creating manual review dataset sample...")

    res_dict = df_resumes.set_index("resume_id").to_dict("index")
    job_dict = df_jobs.set_index("job_id").to_dict("index")

    review_samples = []

    grouped = df_pairs.groupby("weak_label")
    for label, group in grouped:
        n_sample = min(len(group), sample_size // max(1, len(grouped)))
        sample = group.sample(n=n_sample, random_state=42)

        for _, row in sample.iterrows():
            r_id = row["resume_id"]
            j_id = row["job_id"]
            r_info = res_dict.get(r_id, {})
            j_info = job_dict.get(j_id, {})

            r_text = str(r_info.get("clean_text", ""))[:400]
            j_title = str(j_info.get("title", ""))
            j_desc = str(j_info.get("clean_description", ""))[:400]
            r_skills = ", ".join(list(r_info.get("skills", []))[:10])
            j_skills = ", ".join(list(j_info.get("combined_skills", []))[:10])

            review_samples.append({
                "resume_id": r_id,
                "job_id": j_id,
                "resume_text": r_text,
                "job_title": j_title,
                "job_description": j_desc,
                "resume_skills": r_skills,
                "job_skills": j_skills,
                "weak_label": row["weak_label"],
                "label_reason": row["label_reason"],
                "human_label": None,
            })

    return pd.DataFrame(review_samples)
