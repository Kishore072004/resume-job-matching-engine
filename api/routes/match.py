"""
Resume-to-Job Matching Routes.
"""

import time
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas import (
    MatchRequest,
    MatchResponse,
    BatchMatchRequest,
    BatchMatchResponse,
    BatchItemResult,
)
from api.dependencies import get_engine
from src.inference.engine import JobMatchEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Matching"])


@router.post("/match", response_model=MatchResponse, summary="Predict Resume-to-Job Match")
def predict_match_endpoint(
    request: MatchRequest,
    engine: JobMatchEngine = Depends(get_engine),
):
    """
    Evaluates compatibility between a resume and job posting.
    
    Returns estimated match score (0-100), decision, skill breakdown,
    experience/education compatibility, and top contributing SHAP features.
    """
    try:
        res_dict = engine.predict_match(
            resume_text=request.resume_text,
            job_title=request.job_title,
            job_description=request.job_description,
            category=request.category or "General",
            experience_level=request.experience_level or "",
        )
        return MatchResponse(**res_dict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error during match prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during inference processing.",
        )


@router.post("/match/batch", response_model=BatchMatchResponse, summary="Batch Resume-Job Pair Predictions")
def batch_predict_match_endpoint(
    batch_request: BatchMatchRequest,
    engine: JobMatchEngine = Depends(get_engine),
):
    """
    Evaluates compatibility for a bounded batch of resume-job pairs.
    """
    start_time = time.time()
    results: List[BatchItemResult] = []
    success_cnt = 0
    fail_cnt = 0

    for idx, item in enumerate(batch_request.requests):
        try:
            res_dict = engine.predict_match(
                resume_text=item.resume_text,
                job_title=item.job_title,
                job_description=item.job_description,
                category=item.category or "General",
                experience_level=item.experience_level or "",
            )
            match_res = MatchResponse(**res_dict)
            results.append(BatchItemResult(index=idx, status="success", result=match_res))
            success_cnt += 1
        except Exception as e:
            logger.error(f"Batch item {idx} failed: {e}", exc_info=True)
            results.append(BatchItemResult(index=idx, status="failed", error=str(e)))
            fail_cnt += 1

    duration = round(time.time() - start_time, 4)
    return BatchMatchResponse(
        results=results,
        successful_count=success_cnt,
        failed_count=fail_cnt,
        total_duration_seconds=duration,
    )
