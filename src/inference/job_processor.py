"""
Job Processor for Single-Instance Inference.

Preprocesses raw job title & description using the exact training preprocessing logic,
ensuring zero training-serving skew.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.preprocessing.job_cleaner import (
    clean_job_text,
    check_job_quality,
    extract_job_esco_skills,
    build_job_skill_profile,
    extract_job_experience,
    extract_job_education,
    extract_job_titles,
    build_master_job_features,
)

logger = logging.getLogger(__name__)


class JobProcessor:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.df_esco_skills = None
        self.df_esco_aliases = None
        self._load_reference_data()

    def _load_reference_data(self):
        """Load ESCO reference data for skill extraction."""
        skills_path = self.processed_dir / "esco_skills.parquet"
        aliases_path = self.processed_dir / "esco_skill_aliases.parquet"

        if not skills_path.exists() or not aliases_path.exists():
            raise FileNotFoundError(
                f"Skill reference files missing in {self.processed_dir}."
            )

        self.df_esco_skills = pd.read_parquet(skills_path)
        self.df_esco_aliases = pd.read_parquet(aliases_path)

    def process(
        self,
        title: str,
        description: str,
        job_id: str = "inf_job_001",
        experience_level: str = "",
        skills_description: str = "",
        company_name: str = "",
        location: str = "",
    ) -> Dict[str, Any]:
        """
        Process a single raw job posting into a master job feature dictionary.
        """
        clean_t = clean_job_text(title)
        clean_d = clean_job_text(description)

        df_clean = pd.DataFrame([{
            "job_id": job_id,
            "company_name": company_name,
            "title": title,
            "description": description,
            "clean_title": clean_t,
            "clean_description": clean_d,
            "location": location,
            "company_id": "",
            "experience_level": experience_level,
            "work_type": "",
            "remote_allowed": 0.0,
            "skills_description": skills_description,
        }])

        df_quality = check_job_quality(df_clean)

        # Extract ESCO skills from text
        df_esco_extracted = extract_job_esco_skills(
            df_clean, self.df_esco_skills, self.df_esco_aliases
        )

        # Explicit skills empty for custom single job inference
        df_explicit = pd.DataFrame(columns=["job_id", "skill_name"])

        df_profile = build_job_skill_profile(df_clean, df_explicit, df_esco_extracted)
        df_exp = extract_job_experience(df_clean)
        df_edu = extract_job_education(df_clean)

        df_master = build_master_job_features(
            df_clean=df_clean,
            df_quality=df_quality,
            df_profile=df_profile,
            df_exp=df_exp,
            df_edu=df_edu,
        )

        record = df_master.iloc[0].to_dict()
        return record
