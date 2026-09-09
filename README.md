# AI Resume-to-Job Matching & Skill Gap Intelligence Engine

An end-to-end, production-grade Machine Learning engine that matches resumes to job descriptions using ESCO skill extraction, semantic sentence embeddings, Optuna-tuned XGBoost classification, Platt probability calibration, and SHAP explainability.

Exposed via a production **FastAPI REST API backend** and an interactive **Streamlit Web Dashboard**, with **Docker Compose** multi-container orchestration.

```
                  Docker Compose
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌──────────────┐       ┌──────────────┐
    │   FastAPI    │       │  Streamlit   │
    │   :8000      │◄──────│    :8501     │
    └──────┬───────┘       └──────────────┘
           │
           ▼
    JobMatchEngine
           │
      ┌────┴─────┐
      ▼          ▼
   XGBoost     Sentence
    Model      Transformer
```

---

## 🚀 Quick Start - Docker Compose (Recommended)

Run the full multi-container application (FastAPI Backend + Streamlit UI) in one command:

```bash
docker compose up --build
```

Access the interfaces:
- **Streamlit Web Dashboard**: [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

To stop services:
```bash
docker compose down
```

---

## 💻 Local Development Setup

### 1. Installation

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

### 2. Run FastAPI Backend Server

```bash
uvicorn api.main:app --reload
```

The backend runs on `http://127.0.0.1:8000`.

### 3. Run Streamlit Web UI Dashboard

In a separate terminal window:

```bash
streamlit run ui/app.py
```

The frontend dashboard runs on `http://localhost:8501`.

---

## 📡 API Endpoints Summary

- `GET /health`: Liveness probe reporting model status and version.
- `GET /ready`: Readiness probe checking model artifact availability.
- `POST /api/v1/match`: Single resume-to-job prediction, skill gap analysis, and SHAP attributions.
- `POST /api/v1/match/batch`: Bounded batch predictions for datasets of pairs.

---

## 🧪 Testing

The system currently relies on the standard Python `unittest` module. Tests can be run from the root directory.
