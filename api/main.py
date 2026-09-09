"""
FastAPI Main Application Entry Point.
"""

from contextlib import asynccontextmanager
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.dependencies import load_app_config, init_engine, get_raw_engine
from api.routes import health, match

# ────────────────────────────────────────────────────────────────
# Structured Logging Setup
# ────────────────────────────────────────────────────────────────
config = load_app_config()
log_level_str = config.get("log_level", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api.main")


# ────────────────────────────────────────────────────────────────
# Application Lifespan (Startup / Shutdown)
# ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-loads ML models and artifacts once during startup."""
    logger.info("Starting FastAPI Application Lifespan...")
    init_engine()
    yield
    logger.info("Shutting down FastAPI Application Lifespan...")


# ────────────────────────────────────────────────────────────────
# App Initialization
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Resume-to-Job Matching Engine API",
    description=(
        "Production REST API for AI Resume-to-Job Matching & Skill Gap Intelligence. "
        "Evaluates resume-job pair compatibility, extracts matched/missing skills, "
        "and provides SHAP feature attributions."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ────────────────────────────────────────────────────────────────
# CORS Middleware Configuration
# ────────────────────────────────────────────────────────────────
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = config.get("allowed_origins", ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────
# Request Tracing & Structured Logging Middleware
# ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    # Request ID tracking
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.time()
    response: Response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)

    # Attach tracking headers to response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = str(duration_ms)

    engine = get_raw_engine()
    model_ver = engine.metadata.get("version", "2.0.0") if engine else "not_loaded"

    # Sanitized log: strictly excludes resume text and job description
    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"duration_ms={duration_ms} "
        f"model_version={model_ver}"
    )

    return response


# ────────────────────────────────────────────────────────────────
# Custom Exception Handlers
# ────────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.warning(f"request_id={req_id} Validation error on {request.url.path}: {exc}")
    
    # Clean error details
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(x) for x in err.get("loc", [])])
        errors.append({"location": loc, "message": err.get("msg")})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Input validation failed.",
            "errors": errors,
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"request_id={req_id} Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "request_id": req_id,
        },
        headers={"X-Request-ID": req_id},
    )


# ────────────────────────────────────────────────────────────────
# Route Registration
# ────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(match.router)


if __name__ == "__main__":
    import uvicorn
    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8000)
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
