"""
Health and Readiness Check Routes.
"""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from api.schemas import HealthResponse, ReadinessResponse
from api.dependencies import get_raw_engine, is_engine_ready, get_engine_error

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Application Health Check")
def health_check():
    """Basic health check endpoint reporting model status."""
    engine = get_raw_engine()
    is_loaded = engine is not None
    model_ver = engine.metadata.get("version", "2.0.0") if is_loaded else "unknown"
    emb_model = engine.embedding_model_name if is_loaded else "sentence-transformers/all-MiniLM-L6-v2"

    return HealthResponse(
        status="ok",
        model_loaded=is_loaded,
        model_version=model_ver,
        embedding_model=emb_model,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Application Readiness Check")
def readiness_check():
    """Readiness probe checking if ML artifacts and model service are fully operational."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    model_file = root_dir / "models" / "job_matcher_final.joblib"
    artifacts_exist = model_file.exists()
    is_ready = is_engine_ready() and artifacts_exist

    if not is_ready:
        err_detail = get_engine_error() or "Production model artifact not loaded in memory."
        logger.warning(f"Readiness check failed: {err_detail}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ReadinessResponse(
                status="not_ready",
                model_ready=False,
                artifacts_available=artifacts_exist,
                detail=err_detail,
            ).model_dump(),
        )

    return ReadinessResponse(
        status="ready",
        model_ready=True,
        artifacts_available=True,
        detail="All model artifacts and inference dependencies loaded.",
    )
