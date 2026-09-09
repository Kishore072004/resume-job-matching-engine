# FastAPI Production Backend Documentation

The **FastAPI Backend** provides a high-performance REST API for the AI Resume-to-Job Matching Engine. It exposes single and batch matching endpoints, health/readiness checks, structured logging, request tracing, and input validation.

---

## 🚀 Quick Start & Startup Command

To run the development server with hot-reloading:

```bash
uvicorn api.main:app --reload
```

Server will start on `http://127.0.0.1:8000`.

Interactive Swagger UI documentation is available at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

---

## ⚙️ Configuration

Application settings are managed in `configs/api.yaml`:

```yaml
host: "127.0.0.1"
port: 8000
log_level: "INFO"
allowed_origins:
  - "http://localhost:3000"
  - "http://localhost:8501"
max_input_characters: 50000
max_batch_size: 50
```

---

## 📡 Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Application health probe returning model status. |
| `GET` | `/ready` | Application readiness probe checking ML artifacts. |
| `POST` | `/api/v1/match` | Predict compatibility for a single resume-job pair. |
| `POST` | `/api/v1/match/batch` | Predict compatibility for a bounded batch of pairs. |

---

## 📥 Example Requests & Responses

### 1. Single Prediction (`POST /api/v1/match`)

#### Request

```http
POST /api/v1/match HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "resume_text": "Software engineer with 5 years experience in Python, Django, PostgreSQL, Docker, and AWS microservices.",
  "job_title": "Senior Python Developer",
  "job_description": "Seeking a Senior Python Developer proficient in Django, PostgreSQL, Docker, and cloud deployments.",
  "category": "Information-Technology"
}
```

#### Response

```json
{
  "estimated_match_score": 99.36,
  "is_match": true,
  "estimated_probability": 0.9936,
  "raw_model_probability": 0.9984,
  "decision_threshold": 0.1,
  "skills_analysis": {
    "matched_skills": [
      "PostgreSQL",
      "Python (computer programming)",
      "apply knowledge of science, technology and engineering",
      "cloud technologies"
    ],
    "missing_skills": [
      "Django"
    ],
    "shared_skill_count": 4,
    "resume_skill_count": 5,
    "job_skill_count": 5,
    "skill_jaccard": 0.6667,
    "job_skill_coverage": 0.8
  },
  "compatibility": {
    "experience_match": 1.0,
    "education_match": true,
    "experience_gap_years": 0.0,
    "resume_years_experience": 5.0,
    "job_min_years_experience": 5.0
  },
  "top_contributing_features": [
    {
      "feature": "semantic_similarity",
      "shap_value": 1.482,
      "feature_value": 0.841
    },
    {
      "feature": "job_skill_coverage",
      "shap_value": 0.924,
      "feature_value": 0.8
    }
  ],
  "metadata": {
    "model_name": "Optuna_Trained_XGBoost_Resume_Matcher",
    "model_version": "2.0.0"
  }
}
```

---

### 2. Batch Prediction (`POST /api/v1/match/batch`)

#### Request

```http
POST /api/v1/match/batch HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "requests": [
    {
      "resume_text": "Python developer with Django and Postgres experience.",
      "job_title": "Python Dev",
      "job_description": "Seeking Python Dev with Django."
    },
    {
      "resume_text": "Registered Nurse with 8 years clinical experience.",
      "job_title": "Python Dev",
      "job_description": "Seeking Python Dev with Django."
    }
  ]
}
```

#### Response

```json
{
  "results": [
    {
      "index": 0,
      "status": "success",
      "result": {
        "estimated_match_score": 98.42,
        "is_match": true,
        "estimated_probability": 0.9842,
        "raw_model_probability": 0.989,
        "decision_threshold": 0.1,
        "skills_analysis": {
          "matched_skills": ["PostgreSQL", "Python (computer programming)"],
          "missing_skills": [],
          "shared_skill_count": 2,
          "resume_skill_count": 2,
          "job_skill_count": 2,
          "skill_jaccard": 1.0,
          "job_skill_coverage": 1.0
        },
        "compatibility": {
          "experience_match": 1.0,
          "education_match": true,
          "experience_gap_years": null,
          "resume_years_experience": null,
          "job_min_years_experience": null
        },
        "top_contributing_features": [],
        "metadata": {
          "model_name": "Optuna_Trained_XGBoost_Resume_Matcher",
          "model_version": "2.0.0"
        }
      },
      "error": null
    },
    {
      "index": 1,
      "status": "success",
      "result": {
        "estimated_match_score": 1.28,
        "is_match": false,
        "estimated_probability": 0.0128,
        "raw_model_probability": 0.015,
        "decision_threshold": 0.1,
        "skills_analysis": {
          "matched_skills": [],
          "missing_skills": ["PostgreSQL", "Python (computer programming)"],
          "shared_skill_count": 0,
          "resume_skill_count": 1,
          "job_skill_count": 2,
          "skill_jaccard": 0.0,
          "job_skill_coverage": 0.0
        },
        "compatibility": {
          "experience_match": null,
          "education_match": true,
          "experience_gap_years": null,
          "resume_years_experience": null,
          "job_min_years_experience": null
        },
        "top_contributing_features": [],
        "metadata": {
          "model_name": "Optuna_Trained_XGBoost_Resume_Matcher",
          "model_version": "2.0.0"
        }
      },
      "error": null
    }
  ],
  "successful_count": 2,
  "failed_count": 0,
  "total_duration_seconds": 0.4852
}
```

---

## 🛡 Error Codes & Headers

- **HTTP 422 Unprocessable Entity**: Returned when input validation fails (e.g. empty strings, string exceeding `max_input_characters`, or batch size > 50).
- **HTTP 503 Service Unavailable**: Returned when model artifacts fail to load or model service is unready.
- **HTTP 500 Internal Server Error**: Generic server error handler preventing stack traces from leaking to clients.

Every response includes tracing headers:
- `X-Request-ID`: Unique tracking ID per request.
- `X-Process-Time-ms`: Endpoint execution latency in milliseconds.
