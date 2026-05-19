# ChronoLTV

> **End-to-End Multimodal Customer Churn & Dynamic Lifetime Value Predictive Pipeline**  
> Survival Analysis · MLflow · FastAPI · Docker · Great Expectations · Evidently AI

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ChronoLTV Pipeline                         │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │   Data       │    │   Feature    │    │   Survival Model   │    │
│  │  Simulator   │───▶│  Engineering │───▶│  (Cox PH / XGB /  │    │
│  │  (Faker +    │    │  (sklearn    │    │   DeepSurv)        │    │
│  │   Noise)     │    │   pipeline)  │    │                    │    │
│  └──────────────┘    └──────────────┘    └────────┬───────────┘    │
│         │                  │                      │                │
│         ▼                  ▼                      ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │    Great     │    │    MLflow    │    │   FastAPI Serving  │    │
│  │ Expectations │    │  Tracking & │    │   (Docker)         │    │
│  │ (Validation) │    │  Registry   │    │                    │    │
│  └──────────────┘    └──────────────┘    └────────────────────┘    │
│                                                   │                │
│                             ┌─────────────────────▼──────────┐    │
│                             │  Evidently AI Drift Monitor     │    │
│                             └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Layout

```
chrono-ltv/
├── conf/                   # Hydra configs (data, model, mlflow, serving)
├── data/
│   ├── raw/                # Parquet output of the simulator
│   ├── processed/          # Feature-engineered data
│   ├── reference/          # Evidently reference datasets
│   └── expectations/       # Great Expectations suites
├── docker/                 # Dockerfiles + docker-compose.yml
├── src/chrono_ltv/
│   ├── data/               # simulator.py, schemas.py, validators.py
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Cox PH, XGBoost-Survival, DeepSurv
│   ├── training/           # MLflow-integrated trainer + evaluator
│   ├── serving/            # FastAPI app + predictor
│   ├── monitoring/         # Evidently drift detection
│   └── utils/              # logging.py, io.py
├── tests/
│   ├── unit/               # Schema, statistical, noise tests
│   ├── integration/        # Pipeline + API tests
│   └── behavioral/         # Invariance & directional model tests
├── scripts/                # CLI entry points
├── .github/workflows/      # CI/CD pipelines
├── pyproject.toml
└── Makefile
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install in editable mode with dev extras
pip install -e ".[dev]"

# 3. Copy environment template
cp .env.example .env

# 4. Generate synthetic data (10k customers)
make simulate
# or: python scripts/generate_data.py --n-customers 10000

# 5. Run all tests
make test
```

## Build Modules (Iterative Steps)

| Step | Module | Status |
|------|--------|--------|
| 1 | Data Stream Simulator | Complete |
| 2 | Great Expectations Validators | Next |
| 3 | Feature Engineering Pipeline | Pending |
| 4 | Survival Analysis Models | Pending |
| 5 | MLflow Trainer + Evaluator | Pending |
| 6 | FastAPI Serving Layer | Pending |
| 7 | Evidently Drift Monitor | Pending |
| 8 | Behavioral / Invariance Tests | Pending |

## Tech Stack

| Concern | Library |
|---------|---------|
| Simulation | `Faker`, `NumPy`, `Pandas` |
| Survival Analysis | `scikit-survival`, `XGBoost`, `PyTorch` |
| Feature Engineering | `scikit-learn` pipelines |
| Text Embeddings | `sentence-transformers` |
| Config Management | `Hydra` + `OmegaConf` |
| Experiment Tracking | `MLflow` |
| Data Validation | `Great Expectations` |
| Drift Monitoring | `Evidently AI` |
| Serving | `FastAPI` + `Uvicorn` |
| Containerisation | `Docker` + `docker-compose` |
| Testing | `Pytest` + `pytest-cov` |
| Linting / Formatting | `Ruff` |
| Type Checking | `Mypy` |
