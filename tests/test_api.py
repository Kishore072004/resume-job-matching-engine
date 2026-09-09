"""
Unit Test Suite for FastAPI Backend.
Uses starlette.testclient.TestClient with lifespan fixture.
"""

from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app
from api import dependencies


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# Sample valid payloads
VALID_RESUME = "Software engineer with 5 years experience in Python, Django, PostgreSQL, Docker, and AWS."
VALID_JOB_TITLE = "Senior Python Developer"
VALID_JOB_DESC = "Looking for a Senior Python Engineer proficient in Django, PostgreSQL, and Docker."


def test_1_health_endpoint(client):
    """Test GET /health returns 200 OK with health status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_version" in data


def test_2_readiness_endpoint(client):
    """Test GET /ready returns 200 OK when model is initialized."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_ready"] is True


def test_3_valid_match_request(client):
    """Test POST /api/v1/match with valid resume and job details."""
    payload = {
        "resume_text": VALID_RESUME,
        "job_title": VALID_JOB_TITLE,
        "job_description": VALID_JOB_DESC,
        "category": "Information-Technology",
    }
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["estimated_match_score"] <= 100.0
    assert isinstance(data["is_match"], bool)
    assert "skills_analysis" in data
    assert "compatibility" in data
    assert "top_contributing_features" in data


def test_4_invalid_empty_resume(client):
    """Test POST /api/v1/match with empty resume text returns HTTP 422."""
    payload = {
        "resume_text": "   ",
        "job_title": VALID_JOB_TITLE,
        "job_description": VALID_JOB_DESC,
    }
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 422


def test_5_invalid_empty_job_title(client):
    """Test POST /api/v1/match with empty job title returns HTTP 422."""
    payload = {
        "resume_text": VALID_RESUME,
        "job_title": "",
        "job_description": VALID_JOB_DESC,
    }
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 422


def test_6_invalid_empty_job_description(client):
    """Test POST /api/v1/match with empty job description returns HTTP 422."""
    payload = {
        "resume_text": VALID_RESUME,
        "job_title": VALID_JOB_TITLE,
        "job_description": "  ",
    }
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 422


def test_7_batch_endpoint(client):
    """Test POST /api/v1/match/batch with valid batch list."""
    payload = {
        "requests": [
            {
                "resume_text": VALID_RESUME,
                "job_title": VALID_JOB_TITLE,
                "job_description": VALID_JOB_DESC,
            },
            {
                "resume_text": "Registered Nurse with 8 years acute care experience.",
                "job_title": VALID_JOB_TITLE,
                "job_description": VALID_JOB_DESC,
            },
        ]
    }
    response = client.post("/api/v1/match/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["successful_count"] == 2
    assert data["failed_count"] == 0
    assert len(data["results"]) == 2


def test_8_oversized_batch(client):
    """Test POST /api/v1/match/batch exceeding max batch size returns HTTP 422."""
    items = [
        {
            "resume_text": VALID_RESUME,
            "job_title": VALID_JOB_TITLE,
            "job_description": VALID_JOB_DESC,
        }
        for _ in range(55)  # Max allowed is 50
    ]
    response = client.post("/api/v1/match/batch", json={"requests": items})
    assert response.status_code == 422


def test_9_request_id_header(client):
    """Test X-Request-ID response header presence."""
    response = client.get("/health", headers={"X-Request-ID": "test-req-12345"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-req-12345"


def test_10_expected_response_schema(client):
    """Test detailed response structure matches schema requirements."""
    payload = {
        "resume_text": VALID_RESUME,
        "job_title": VALID_JOB_TITLE,
        "job_description": VALID_JOB_DESC,
    }
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Check top-level keys
    req_keys = [
        "estimated_match_score",
        "is_match",
        "estimated_probability",
        "raw_model_probability",
        "decision_threshold",
        "skills_analysis",
        "compatibility",
        "top_contributing_features",
        "metadata",
    ]
    for k in req_keys:
        assert k in data

    # Check skills_analysis keys
    skills_keys = [
        "matched_skills",
        "missing_skills",
        "shared_skill_count",
        "resume_skill_count",
        "job_skill_count",
        "skill_jaccard",
        "job_skill_coverage",
    ]
    for k in skills_keys:
        assert k in data["skills_analysis"]


def test_11_model_unavailable_behavior(monkeypatch, client):
    """Test HTTP 503 response when engine singleton is unavailable."""
    monkeypatch.setattr(dependencies, "_engine_instance", None)
    monkeypatch.setattr(dependencies, "_engine_error", "Simulated unreadiness failure")

    response = client.get("/ready")
    assert response.status_code == 503

    response_match = client.post(
        "/api/v1/match",
        json={
            "resume_text": VALID_RESUME,
            "job_title": VALID_JOB_TITLE,
            "job_description": VALID_JOB_DESC,
        },
    )
    assert response_match.status_code == 503
