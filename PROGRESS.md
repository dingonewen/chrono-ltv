# ChronoLTV — Project Progress

> Paste this file at the start of every conversation so the assistant has full context on what exists, what is tested, and what comes next.

---

## What This Project Is

An **end-to-end, production-grade ML pipeline** for a DTC e-commerce platform that predicts:
- **When** a customer will churn (survival analysis, not binary classification)
- **How much revenue** they will generate before churning (dynamic LTV)

It is multimodal: tabular transaction data + clickstream sessions + LLM-embedded support ticket text.

---

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         ChronoLTV Pipeline                            │
│                                                                       │
│  [Step 1]          [Step 2]          [Step 3]          [Step 4]      │
│  Data Simulator ──▶ GE Validator ──▶ Feature Eng. ──▶ Survival Model │
│  (Faker + noise)   (schema/range)   (sklearn pipe)   (Cox/XGB/Deep)  │
│       │                                                    │          │
│       ▼                                                    ▼          │
│  Parquet files                                        MLflow Run      │
│  data/raw/                                            (tracked)       │
│                                                            │          │
│  [Step 7]          [Step 6]          [Step 5]             │          │
│  Drift Monitor ◀── API Serving  ◀─── Model Registry ◀────┘          │
│  (Evidently AI)    (FastAPI)         (MLflow)                         │
│                                                                       │
│  [Step 8] Behavioral & Invariance Tests run in CI against all models  │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Build Progress

| Step | Module | Status | Key Files |
|------|--------|--------|-----------|
| 1 | Data Stream Simulator | **Done** | [simulator.py](src/chrono_ltv/data/simulator.py), [schemas.py](src/chrono_ltv/data/schemas.py) |
| 2 | Great Expectations Validators | **Next** | `src/chrono_ltv/data/validators.py` *(not yet written)* |
| 3 | Feature Engineering Pipeline | Pending | `src/chrono_ltv/features/pipeline.py` *(not yet written)* |
| 4 | Survival Analysis Models | Pending | `src/chrono_ltv/models/` *(stubs only)* |
| 5 | MLflow Trainer + Evaluator | Pending | `src/chrono_ltv/training/` *(stubs only)* |
| 6 | FastAPI Serving Layer | Pending | `src/chrono_ltv/serving/` *(stubs only)* |
| 7 | Evidently Drift Monitor | Pending | `src/chrono_ltv/monitoring/` *(stubs only)* |
| 8 | Behavioral / Invariance Tests | Pending | `tests/behavioral/` *(stubs only)* |

---

## Infrastructure Map

Every file that exists and what it does.

### Tooling & Config

| File | Role |
|------|------|
| [pyproject.toml](pyproject.toml) | Single source of truth: dependencies, Ruff, Mypy, Pytest config |
| [Makefile](Makefile) | `make simulate`, `make test`, `make lint`, `make docker-up` |
| [.env.example](.env.example) | All environment variable names with defaults; copy → `.env` |
| [.gitignore](.gitignore) | Excludes data files, mlruns, .env, caches |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | CI: Ruff → Mypy → Pytest on Python 3.10 + 3.11 |

### Hydra Configuration (`conf/`)

| File | Controls |
|------|---------|
| [conf/config.yaml](conf/config.yaml) | Root Hydra config; composes all sub-configs |
| [conf/data/simulator.yaml](conf/data/simulator.yaml) | `n_customers`, date window, noise rates, churn threshold, catalogue |
| [conf/model/cox_ph.yaml](conf/model/cox_ph.yaml) | Hyperparams for Cox PH, XGBoost-Survival, and DeepSurv |
| [conf/mlflow/tracking.yaml](conf/mlflow/tracking.yaml) | Tracking URI, experiment name, registry aliases |
| [conf/serving/api.yaml](conf/serving/api.yaml) | FastAPI host/port, model URI, CORS, SLA |

### Source Code (`src/chrono_ltv/`)

#### `data/` — Step 1 (Done)

| File | What it contains |
|------|-----------------|
| [simulator.py](src/chrono_ltv/data/simulator.py) | `EcommerceSimulator` façade + 5 private factories: `_CustomerFactory`, `_TransactionFactory`, `_ClickstreamFactory`, `_SupportTicketFactory`, `_SurvivalLabelBuilder`, `_NoiseInjector`. Generates 5 Parquet datasets. |
| [schemas.py](src/chrono_ltv/data/schemas.py) | Pydantic v2 models: `CustomerRecord`, `TransactionRecord`, `ClickstreamRecord`, `SupportTicketRecord`, `SurvivalLabel`. Used for row-level validation and OpenAPI docs. |
| `validators.py` | *(Step 2 — not yet written)* Great Expectations suite |

#### `features/` — Step 3 (Pending)

| File | What it will contain |
|------|---------------------|
| `pipeline.py` | sklearn `Pipeline` with imputation, encoding, survival-compatible transformers |
| `encoders.py` | Custom transformers: text embedding aggregator, RFM features, tenure features |

#### `models/` — Step 4 (Pending)

| File | What it will contain |
|------|---------------------|
| `base.py` | Abstract `SurvivalModel` interface |
| `cox_ph.py` | `CoxPHSurvivalAnalysis` wrapper (scikit-survival) |
| `xgb_survival.py` | XGBoost `objective=survival:cox` wrapper |
| `deepsurv.py` | PyTorch `DeepSurv` network (challenger model) |

#### `training/` — Step 5 (Pending)

| File | What it will contain |
|------|---------------------|
| `trainer.py` | MLflow-integrated training loop with k-fold CV |
| `evaluator.py` | Concordance index, time-dependent AUC, Brier score |

#### `serving/` — Step 6 (Pending)

| File | What it will contain |
|------|---------------------|
| `api.py` | FastAPI app with `/predict`, `/health`, `/metrics` endpoints |
| `predictor.py` | MLflow model loader + inference wrapper |
| `schemas.py` | Pydantic request/response models for the API |

#### `monitoring/` — Step 7 (Pending)

| File | What it will contain |
|------|---------------------|
| `drift.py` | Evidently `DataDriftPreset` + `DataQualityPreset` reports |
| `alerts.py` | PSI threshold checks, alert routing |

#### `utils/` — Done

| File | What it contains |
|------|-----------------|
| [utils/logging.py](src/chrono_ltv/utils/logging.py) | `configure_logging()` (Loguru), `get_logger(name)` |
| [utils/io.py](src/chrono_ltv/utils/io.py) | `save_parquet()`, `load_parquet()` |

### Scripts (`scripts/`)

| File | Command |
|------|---------|
| [scripts/generate_data.py](scripts/generate_data.py) | `python scripts/generate_data.py [--n-customers N ...]` or `make simulate` |
| `scripts/train.py` | *(Step 5 — not yet written)* |
| `scripts/monitor.py` | *(Step 7 — not yet written)* |

### Tests (`tests/`)

| File | Coverage |
|------|---------|
| [tests/conftest.py](tests/conftest.py) | Session-scoped fixtures: `clean_datasets`, `noisy_datasets`, `clean_sim_cfg`, `noisy_sim_cfg` |
| [tests/unit/test_simulator.py](tests/unit/test_simulator.py) | 25+ tests across 5 categories: schema, statistical, survival logic, noise injection, reproducibility |
| `tests/unit/test_validators.py` | *(Step 2 — not yet written)* |
| `tests/integration/` | *(Steps 5–6 — not yet written)* |
| `tests/behavioral/` | *(Step 8 — not yet written)* Invariance + directional model tests |

### Docker

| File | Role |
|------|------|
| [docker/Dockerfile.api](docker/Dockerfile.api) | Multi-stage build for the FastAPI service |
| [docker/docker-compose.yml](docker/docker-compose.yml) | `mlflow` tracking server + `api` service, both health-checked |

---

## Data Outputs (Step 1)

Running `make simulate` writes five Parquet files to `data/raw/`:

| File | Rows (10k customers) | Description |
|------|----------------------|-------------|
| `customers.parquet` | 10,000 | Master customer table — clean, no noise injected |
| `transactions.parquet` | ~120,000 | Purchase events; noise: missing values, duplicates, outliers, negative amounts |
| `clickstream.parquet` | ~900,000 | Page-view sessions; noise: missing values, future timestamps |
| `support_tickets.parquet` | ~1,500 | Raw ticket text with sentiment scores; noise injected |
| `survival_labels.parquet` | 10,000 | Ground-truth `(duration_days, event_observed)` per customer |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Survival analysis instead of binary churn | Predicts *when*, not just *if*; handles censored data correctly |
| Noise injected at simulator layer | Downstream validators and drift monitors have real faults to catch |
| `from __future__ import annotations` only in `simulator.py` | Pydantic v2 models in `schemas.py` need runtime type resolution — PEP 563 breaks it |
| `if TYPE_CHECKING:` block after all imports | isort requires TYPE_CHECKING blocks at the end of the import section |
| Session-scoped pytest fixtures | Simulator is slow; running it once per session keeps the test suite fast |
| Hydra for config management | Enables sweep experiments and CLI overrides without touching code |

---

## Environment Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install (editable + dev extras)
pip install -e ".[dev]"

# 3. Copy env template
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux

# 4. Generate synthetic data
make simulate
# python scripts/generate_data.py --n-customers 10000

# 5. Run tests
make test

# 6. Start services
make docker-up                  # MLflow UI → http://localhost:5000
make serve                      # FastAPI  → http://localhost:8000/docs
```

---

## CI Status

Pipeline: **Ruff lint → Ruff format check → Mypy type check → Pytest (Python 3.10 + 3.11)**

Known-resolved issues (do not re-introduce):
- `setuptools.build_meta` — not `setuptools.backends.legacy:build`
- `if TYPE_CHECKING:` block must go **after** all regular imports
- `from __future__ import annotations` must **not** be used in Pydantic model files
- `_NoiseInjector` / `_SurvivalLabelBuilder` are internal; do not import them in tests unless directly constructing them
