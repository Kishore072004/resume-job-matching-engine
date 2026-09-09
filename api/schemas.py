"""
Pydantic Request and Response Schemas for FastAPI Resume-to-Job Matcher.
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator, ValidationInfo

MAX_INPUT_CHARS = 50000
MAX_BATCH_SIZE = 50


class MatchRequest(BaseModel):
    resume_text: str = Field(..., description="Raw resume text")
    job_title: str = Field(..., description="Job posting title")
    job_description: str = Field(..., description="Raw job description")
    category: Optional[str] = Field("General", description="Resume category")
    experience_level: Optional[str] = Field("", description="Job experience level")
    resume_id: Optional[str] = Field(None, description="Optional resume identifier")
    job_id: Optional[str] = Field(None, description="Optional job identifier")

    @field_validator("resume_text", "job_title", "job_description", mode="before")
    @classmethod
    def validate_and_trim_string(cls, value: Any, info: ValidationInfo) -> str:
        if value is None:
            raise ValueError(f"Field '{info.field_name}' must not be null or missing.")
        
        val_str = str(value).strip()
        if not val_str:
            raise ValueError(f"Field '{info.field_name}' must not be empty or whitespace only.")
        
        if len(val_str) > MAX_INPUT_CHARS:
            raise ValueError(
                f"Field '{info.field_name}' exceeds maximum allowed size of {MAX_INPUT_CHARS} characters "
                f"(received {len(val_str)} characters)."
            )
        return val_str


class SkillsAnalysis(BaseModel):
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    shared_skill_count: int = 0
    resume_skill_count: int = 0
    job_skill_count: int = 0
    skill_jaccard: float = 0.0
    job_skill_coverage: float = 0.0


class Compatibility(BaseModel):
    experience_match: Optional[float] = None
    education_match: bool = False
    experience_gap_years: Optional[float] = None
    resume_years_experience: Optional[float] = None
    job_min_years_experience: Optional[float] = None


class ContributingFeature(BaseModel):
    feature: str
    shap_value: float
    feature_value: Optional[float] = None


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str


class MatchResponse(BaseModel):
    estimated_match_score: float = Field(..., description="Match score from 0.0 to 100.0")
    is_match: bool = Field(..., description="Match decision relative to threshold")
    estimated_probability: float = Field(..., description="Calibrated match probability (0.0 - 1.0)")
    raw_model_probability: float = Field(..., description="Raw XGBoost prediction probability")
    decision_threshold: float = Field(..., description="Threshold used for match decision")
    skills_analysis: SkillsAnalysis
    compatibility: Compatibility
    top_contributing_features: List[ContributingFeature] = Field(default_factory=list)
    metadata: ModelMetadata


class BatchMatchRequest(BaseModel):
    requests: List[MatchRequest] = Field(..., description="List of match requests to process in batch")

    @field_validator("requests")
    @classmethod
    def validate_batch_size(cls, requests: List[MatchRequest]) -> List[MatchRequest]:
        if not requests:
            raise ValueError("Batch request list must not be empty.")
        if len(requests) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size exceeds maximum limit of {MAX_BATCH_SIZE} requests (received {len(requests)})."
            )
        return requests


class BatchItemResult(BaseModel):
    index: int
    status: str = Field(..., description="'success' or 'failed'")
    result: Optional[MatchResponse] = None
    error: Optional[str] = None


class BatchMatchResponse(BaseModel):
    results: List[BatchItemResult]
    successful_count: int
    failed_count: int
    total_duration_seconds: float


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_version: str
    embedding_model: str


class ReadinessResponse(BaseModel):
    status: str = "ready"
    model_ready: bool
    artifacts_available: bool
    detail: Optional[str] = None
