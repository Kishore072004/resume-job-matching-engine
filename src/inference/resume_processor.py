"""
Resume Processor for Single-Instance Inference.

Preprocesses raw resume text using the exact training preprocessing logic,
ensuring zero training-serving skew.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

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

logger = logging.getLogger(__name__)


class ResumeProcessor:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.df_skills = None
        self.df_aliases = None
        self._load_reference_data()

    def _load_reference_data(self):
        """Load skill references needed for ESCO skill extraction."""
        skills_path = self.processed_dir / "esco_skills.parquet"
        aliases_path = self.processed_dir / "esco_skill_aliases.parquet"

        if not skills_path.exists() or not aliases_path.exists():
            raise FileNotFoundError(
                f"Skill reference files missing in {self.processed_dir}. "
                "Ensure step 2/3 outputs exist."
            )

        self.df_skills = pd.read_parquet(skills_path)
        self.df_aliases = pd.read_parquet(aliases_path)

    def process(self, raw_text: str, category: str = "General", resume_id: str = "inf_res_001") -> Dict[str, Any]:
        """
        Process a single raw resume string into a feature dictionary.
        
        Returns:
            Dict containing clean_text, category, skills, skill_count,
            years_experience, education_level, job_titles, quality_flag.
        """
        clean_text = clean_resume_text(raw_text)

        df_clean = pd.DataFrame([{
            "resume_id": resume_id,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "category": category,
        }])

        df_quality = check_resume_quality(df_clean)
        df_res_skills = extract_resume_skills(df_clean, self.df_skills, self.df_aliases)
        df_exp = extract_experience(df_clean)
        df_edu = extract_education(df_clean)
        
        # Load occupations if available for title matching, else empty dataframe
        occ_path = self.processed_dir / "esco_occupations.parquet"
        df_occ = pd.read_parquet(occ_path) if occ_path.exists() else pd.DataFrame()
        df_titles = extract_job_titles(df_clean, df_occ)

        df_master = build_master_resume_features(
            df_clean=df_clean,
            df_quality=df_quality,
            df_skills=df_res_skills,
            df_exp=df_exp,
            df_edu=df_edu,
            df_titles=df_titles,
        )

        record = df_master.iloc[0].to_dict()
        return record
