# Automated Predictive Modeling Platform

> Full-stack AutoML platform covering the complete ML lifecycle — data ingestion, automated preprocessing, model training across 5 algorithms, real-time explainability (SHAP/LIME), data drift monitoring, and experiment tracking.

[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178C6?style=flat-square&logo=typescript)](https://typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=flat-square&logo=postgresql)](https://postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-Celery-DC382D?style=flat-square&logo=redis)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Deployed](https://img.shields.io/badge/Deployed-Vercel%20%2B%20Render-black?style=flat-square&logo=vercel)](https://automated-predictive-modeling.vercel.app/)

**[Live Demo](https://automated-predictive-modeling.vercel.app/)** &nbsp;·&nbsp; **[API Docs](https://automated-predictive-modeling.vercel.app/docs)**

> **Demo note:** Hosted on free-tier Render — the backend may take 30–45 seconds to cold-start. Background features (async training, WebSocket updates) require paid infrastructure (Redis + Celery workers) not provisioned in the demo environment.

---

## Screenshots

<p align="center">
  <img src="images/dashboard.png" width="49%" alt="Dashboard Overview" />
  <img src="images/model-training.png" width="49%" alt="Model Training" />
</p>
<p align="center">
  <img src="images/explainability.png" width="49%" alt="SHAP / LIME Explainability" />
  <img src="images/drift-monitoring.png" width="49%" alt="Data Drift Monitoring" />
</p>
<p align="center">
  <img src="images/experiments.png" width="49%" alt="MLflow Experiment Tracking" />
  <img src="images/predictions.png" width="49%" alt="Real-time Predictions" />
</p>

<details>
<summary>More screenshots</summary>
<br>
<p align="center">
  <img src="images/data-ingestion.png" width="49%" alt="Data Ingestion" />
  <img src="images/model-management.png" width="49%" alt="Model Management" />
</p>
<p align="center">
  <img src="images/data-quality.png" width="49%" alt="Data Quality Profiling" />
  <img src="images/settings.png" width="49%" alt="Settings & API Keys" />
</p>
<p align="center">
  <img src="images/additional-view.png" width="49%" alt="Additional View" />
</p>
</details>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend  (Next.js 14 · TypeScript · Tailwind · Recharts)      │
│  App Router · React Context · WebSocket client · Framer Motion  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ REST + WebSocket
┌───────────────────────▼─────────────────────────────────────────┐
│  API Layer  (FastAPI · Pydantic · JWT auth · Rate limiting)      │
│  OpenAPI docs · API key management · Prometheus metrics          │
└──────┬─────────────────────────────┬────────────────────────────┘
       │ Sync queries                │ Async tasks
┌──────▼──────────┐        ┌─────────▼──────────────────────────┐
│  PostgreSQL 15  │        │  Celery Workers + Redis Broker      │
│  + TimescaleDB  │        │  Model training · Batch processing  │
│  SQLAlchemy ORM │        │  Drift checks · Retraining pipelines│
└─────────────────┘        └─────────────────────────────────────┘
       │                            │
┌──────▼────────────────────────────▼─────────────────────────────┐
│  ML Layer                                                        │
│  scikit-learn · XGBoost · CatBoost · LightGBM                   │
│  Optuna (HPO) · MLflow (experiment tracking)                     │
│  SHAP · LIME (explainability) · Evidently (drift detection)      │
└──────────────────────────────────┬──────────────────────────────┘
                                   │ Artifacts
                        ┌──────────▼──────────┐
                        │  Cloud Storage       │
                        │  AWS S3 / Azure Blob │
                        └─────────────────────┘
```

---

## Key Features

### ML Pipeline
- **5 algorithms** — Random Forest, XGBoost, CatBoost, LightGBM, Neural Networks
- **Automated preprocessing** — missing value imputation, encoding, scaling, feature engineering
- **Hyperparameter optimization** — Optuna with grid, random, and TPE search strategies
- **Async training jobs** — Celery workers with real-time WebSocket progress updates
- **Model versioning** — artifact storage on S3/Azure with full metadata tracking

### Observability & Explainability
- **SHAP** — waterfall charts, feature importance, global model explanations
- **LIME** — local instance-level explanations for any prediction
- **Data drift detection** — statistical tests (KS, PSI) with automated alerts
- **Data quality profiling** — lineage tracking, schema validation, distribution reports
- **Experiment tracking** — MLflow integration for comparing runs and hyperparameters

### Production-Grade Infrastructure
- **JWT authentication** — access + refresh token rotation
- **Rate limiting** — Redis-backed per-user request throttling
- **REST API** — 40+ endpoints with full OpenAPI documentation
- **Batch processing** — async CSV ingestion with job status polling and result download
- **A/B testing framework** — compare deployed model versions in production
- **Alert system** — configurable thresholds for model performance degradation
- **Docker Compose** — single-command local setup for the full stack

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Next.js 14 (App Router), TypeScript, Tailwind CSS, Recharts, Framer Motion |
| **Backend** | FastAPI, Pydantic v2, SQLAlchemy, Alembic, Uvicorn |
| **ML / AI** | scikit-learn, XGBoost, CatBoost, LightGBM, SHAP, LIME, Optuna, MLflow |
| **Database** | PostgreSQL 15 + TimescaleDB, Redis |
| **Async** | Celery, WebSockets |
| **Auth** | JWT (access + refresh tokens), API key management |
| **Storage** | AWS S3 / Azure Blob Storage |
| **DevOps** | Docker, Docker Compose, Vercel (frontend), Render (backend) |

---

## Local Development

**Prerequisites:** Docker + Docker Compose

```bash
git clone https://github.com/Phantomjkk/Predictive-Modeling-Automation.git
cd Predictive-Modeling-Automation

# Copy and configure environment variables
cp backend/.env.example backend/.env

# Start the full stack (API + DB + Redis + Celery + Frontend)
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## API Overview

40+ REST endpoints organized across 10 domains:

`Authentication` · `Data Ingestion` · `ML Models` · `Predictions` · `Batch Processing` · `Explainability` · `Experiments` · `Hyperparameter Optimization` · `Drift Detection` · `Data Quality`

Full interactive documentation available at `/docs` (Swagger UI) or `/redoc` when the backend is running.

---

## Deployment

| Component | Platform | Notes |
|---|---|---|
| Frontend | Vercel | Auto-deploy from `main` |
| Backend API | Render | Dockerized FastAPI |
| Database | Render PostgreSQL | Managed instance |
| Background workers | Self-hosted | Redis + Celery (not in free-tier demo) |
| Model artifacts | AWS S3 / Azure Blob | Configurable via env vars |
