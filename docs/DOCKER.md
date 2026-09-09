# Docker & Multi-Container Deployment Documentation

This directory contains instructions for building, running, and orchestrating the multi-container deployment of the **AI Resume-to-Job Matching Engine** using Docker and Docker Compose.

---

## 🏗 Architecture Overview

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

The system is containerized into two services:
1. **`api` (`Dockerfile.api`)**: FastAPI backend serving predictions on port `8000`. Warm-loads ML models and artifacts once during container initialization.
2. **`ui` (`Dockerfile.ui`)**: Streamlit web dashboard serving interactive UI on port `8501`. Connects to the backend via HTTP.

---

## 🚀 Quick Start - Running with Docker Compose

### 1. Build and Start All Containers

Run the following command from the project root directory:

```bash
docker compose up --build
```

Docker Compose will:
1. Build `resume_analyzer_api` image using `Dockerfile.api`.
2. Build `resume_analyzer_ui` image using `Dockerfile.ui`.
3. Start the `api` container and wait for its healthcheck (`GET /health`) to pass.
4. Start the `ui` container once the `api` service is healthy.

### 2. Access Web Dashboard & API

- **Streamlit Web UI Dashboard**: [http://localhost:8501](http://localhost:8501)
- **FastAPI OpenAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Probe**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🛑 Stopping Containers

To stop and remove containers:

```bash
docker compose down
```

To stop containers and remove volumes:

```bash
docker compose down -v
```

---

## ⚙️ Environment Variables

The container setup can be configured via environment variables:

| Variable | Container | Default Value | Description |
|---|---|---|---|
| `API_URL` | `ui` | `http://api:8000` | FastAPI backend URL accessed by Streamlit container. |
| `ALLOWED_ORIGINS` | `api` | `http://localhost:8501...` | Allowed CORS origins for FastAPI. |
| `LOG_LEVEL` | `api` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## 🔍 Troubleshooting & Logs

### View Logs for All Services

```bash
docker compose logs -f
```

### View Logs for Specific Service

```bash
docker compose logs -f api
docker compose logs -f ui
```

### Inspect Container Health

```bash
docker compose ps
```
