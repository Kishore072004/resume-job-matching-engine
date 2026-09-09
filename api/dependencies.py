"""
Dependencies and Singleton Management for FastAPI App.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from fastapi import HTTPException, status

from src.inference.engine import JobMatchEngine

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "api.yaml"

_config: Optional[Dict[str, Any]] = None
_engine_instance: Optional[JobMatchEngine] = None
_engine_error: Optional[str] = None


def load_app_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config = yaml.safe_load(f)
        else:
            _config = {
                "host": "127.0.0.1",
                "port": 8000,
                "log_level": "INFO",
                "allowed_origins": ["*"],
                "max_input_characters": 50000,
                "max_batch_size": 50,
            }
    return _config


def init_engine():
    """Initialize JobMatchEngine once during application startup."""
    global _engine_instance, _engine_error
    try:
        logger.info("Initializing JobMatchEngine singleton at application startup...")
        _engine_instance = JobMatchEngine()
        _engine_error = None
        logger.info("JobMatchEngine singleton successfully initialized and ready in memory.")
    except Exception as e:
        logger.error(f"Failed to initialize JobMatchEngine: {e}", exc_info=True)
        _engine_instance = None
        _engine_error = str(e)


def get_raw_engine() -> Optional[JobMatchEngine]:
    """Return raw _engine_instance reference dynamically."""
    return _engine_instance


def get_engine_error() -> Optional[str]:
    """Return initialization error string if any."""
    return _engine_error


def is_engine_ready() -> bool:
    """Check if engine instance is loaded and operational."""
    return _engine_instance is not None


def get_engine() -> JobMatchEngine:
    """FastAPI Dependency for obtaining the JobMatchEngine instance."""
    if _engine_instance is None:
        err_msg = _engine_error or "JobMatchEngine artifact is not initialized or unavailable."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {err_msg}",
        )
    return _engine_instance


def get_config() -> Dict[str, Any]:
    return load_app_config()
