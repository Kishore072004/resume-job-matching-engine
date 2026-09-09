"""
Production Inference Engine for AI Resume-to-Job Matching.

Provides a unified, reusable inference interface for evaluating resume-job compatibility.
Reuses exact preprocessing & feature engineering logic to eliminate training-serving skew.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
import shap
from sentence_transformers import SentenceTransformer

from src.feature_engineering.ml_features import (
    MODEL_FEATURES,
    enrich_candidate_pairs,
)
from src.inference.resume_processor import ResumeProcessor
from src.inference.job_processor import JobProcessor
from src.matching.candidate_generator import calculate_title_similarity

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "job_matcher_final.joblib"
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


class JobMatchEngine:
    """
    Production inference engine for single and batch resume-job match evaluation.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_path = Path(model_path)
        self.processed_dir = Path(processed_dir)
        self.embedding_model_name = embedding_model_name

        self._load_production_artifact()
        self._init_processors()
        self._load_embedding_model()
        self._init_shap_explainer()

    def _load_production_artifact(self):
        """Load trained XGBoost model, imputer, calibrator, and metadata artifact."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Production model artifact missing at {self.model_path}")

        logger.info(f"Loading production artifact from {self.model_path}")
        self.artifact = joblib.load(self.model_path)

        self.pipeline = self.artifact["pipeline"]
        self.imputer = self.artifact["imputer"]
        self.model = self.artifact["model"]
        self.calibrator = self.artifact.get("calibrator")
        self.best_threshold = float(self.artifact.get("best_threshold", 0.10))
        self.feature_names = self.artifact.get("feature_names", MODEL_FEATURES)
        self.metadata = self.artifact.get("metadata", {})

    def _init_processors(self):
        """Initialize resume and job processors."""
        self.resume_processor = ResumeProcessor(self.processed_dir)
        self.job_processor = JobProcessor(self.processed_dir)

    def _load_embedding_model(self):
        """Load sentence-transformers model for semantic similarity computation."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)

    def _init_shap_explainer(self):
        """Initialize SHAP TreeExplainer for XGBoost explainability."""
        logger.info("Initializing SHAP TreeExplainer...")
        self.shap_explainer = shap.TreeExplainer(self.model)

    def predict_match(
        self,
        resume_text: str,
        job_title: str,
        job_description: str,
        category: str = "General",
        experience_level: str = "",
        skills_description: str = "",
        company_name: str = "",
        location: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate match compatibility between a raw resume and job posting.
        """
        # 1. Process Resume & Job raw inputs
        r_info = self.resume_processor.process(raw_text=resume_text, category=category)
        j_info = self.job_processor.process(
            title=job_title,
            description=job_description,
            experience_level=experience_level,
            skills_description=skills_description,
            company_name=company_name,
            location=location,
        )

        r_id = r_info["resume_id"]
        j_id = j_info["job_id"]

        # 2. Compute Semantic Embeddings & Similarity
        r_rep = f"{r_info['category']} {' '.join(r_info['skills'][:15])} {r_info['clean_text'][:400]}".strip()
        j_rep = f"{j_info['clean_title']} {j_info['clean_description'][:300]}".strip()

        embeddings = self.embedder.encode([r_rep, j_rep], normalize_embeddings=True)
        semantic_sim = float(np.dot(embeddings[0], embeddings[1]))

        # 3. Base Candidate Pair Feature Construction
        r_skills = set(r_info["skills"])
        j_skills = set(j_info["combined_skills"])
        shared_skills = r_skills & j_skills
        union_skills = r_skills | j_skills

        shared_cnt = len(shared_skills)
        r_skill_cnt = len(r_skills)
        j_skill_cnt = len(j_skills)
        jaccard = float(shared_cnt / len(union_skills)) if union_skills else 0.0
        cov_resume = float(shared_cnt / r_skill_cnt) if r_skill_cnt > 0 else 0.0
        cov_job = float(shared_cnt / j_skill_cnt) if j_skill_cnt > 0 else 0.0

        r_cat = str(r_info["category"]).replace("-", " ")
        r_titles = r_info["job_titles"] if isinstance(r_info["job_titles"], list) else [r_cat]
        j_clean_title = str(j_info["clean_title"])
        title_sim = max([calculate_title_similarity(t, j_clean_title) for t in r_titles] + [calculate_title_similarity(r_cat, j_clean_title)])

        r_exp = r_info.get("years_experience")
        j_min_exp = j_info.get("min_years_experience")
        j_max_exp = j_info.get("max_years_experience")
        exp_gap = float(r_exp - j_min_exp) if r_exp is not None and j_min_exp is not None else None

        edu_rank = {"PhD": 5, "Master": 4, "Bachelor": 3, "Associate": 2, "Diploma": 1, "Not Specified": 0}
        r_edu = r_info.get("education_level", "Not Specified")
        j_edu = j_info.get("education_requirement", "Not Specified")
        edu_match = float(edu_rank.get(r_edu, 0) >= edu_rank.get(j_edu, 0))

        df_pair = pd.DataFrame([{
            "resume_id": r_id,
            "job_id": j_id,
            "retrieval_method": "inference",
            "retrieval_score": semantic_sim,
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
            "weak_label": "positive",
        }])

        df_resumes_single = pd.DataFrame([r_info])
        df_jobs_single = pd.DataFrame([j_info])

        # 4. Enrich Features
        df_enriched = enrich_candidate_pairs(
            df_pairs=df_pair,
            df_resumes=df_resumes_single,
            df_jobs=df_jobs_single,
            processed_dir=self.processed_dir,
        )

        # Single row semantic_similarity_normalized fallback
        if "semantic_similarity_normalized" in df_enriched.columns:
            df_enriched["semantic_similarity_normalized"] = semantic_sim

        X_raw = df_enriched[self.feature_names]

        # 5. Impute & Model Prediction
        X_imp = self.imputer.transform(X_raw)
        raw_prob = float(self.model.predict_proba(X_imp)[0, 1])

        # 6. Apply Calibration if Present
        if self.calibrator is not None:
            eps = 1e-15
            clipped = np.clip([raw_prob], eps, 1 - eps)
            logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            prob = float(self.calibrator.predict_proba(logit)[0, 1])
        else:
            prob = raw_prob

        # 7. Scale Match Score (0 - 100)
        match_score = round(float(prob * 100), 2)
        is_match = bool(prob >= self.best_threshold)

        # 8. SHAP Explainability
        shap_vals = self.shap_explainer.shap_values(X_imp)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        
        row_shap = shap_vals[0]
        feature_importance = [
            {
                "feature": f_name,
                "shap_value": round(float(s_val), 4),
                "feature_value": round(float(val), 4) if pd.notna(val) else None,
            }
            for f_name, s_val, val in zip(self.feature_names, row_shap, X_raw.iloc[0])
        ]
        feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_contributing_features = feature_importance[:7]

        # 9. Skill Breakdown
        matched_skills = sorted(list(shared_skills))
        missing_skills = sorted(list(j_skills - r_skills))

        # 10. Format Output Response
        return {
            "estimated_match_score": match_score,
            "is_match": is_match,
            "estimated_probability": round(prob, 4),
            "raw_model_probability": round(raw_prob, 4),
            "decision_threshold": self.best_threshold,
            "skills_analysis": {
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "shared_skill_count": shared_cnt,
                "resume_skill_count": r_skill_cnt,
                "job_skill_count": j_skill_cnt,
                "skill_jaccard": round(jaccard, 4),
                "job_skill_coverage": round(cov_job, 4),
            },
            "compatibility": {
                "experience_match": df_enriched.iloc[0].get("experience_match"),
                "education_match": bool(edu_match),
                "experience_gap_years": exp_gap,
                "resume_years_experience": r_exp,
                "job_min_years_experience": j_min_exp,
            },
            "top_contributing_features": top_contributing_features,
            "metadata": {
                "model_name": self.metadata.get("model_name", "Optuna_Trained_XGBoost_Resume_Matcher"),
                "model_version": self.metadata.get("version", "2.0.0"),
            },
        }
