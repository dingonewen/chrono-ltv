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
| 1 | Data Stream Simulator | **Done ✓** | [simulator.py](src/chrono_ltv/data/simulator.py), [schemas.py](src/chrono_ltv/data/schemas.py) |
| 2 | Great Expectations Validators | **Done ✓** | [validators.py](src/chrono_ltv/data/validators.py) |
| 3 | Feature Engineering Pipeline | **Done ✓** | [pipeline.py](src/chrono_ltv/features/pipeline.py), [encoders.py](src/chrono_ltv/features/encoders.py) |
| 4 | Survival Analysis Models | **Done ✓** | [base.py](src/chrono_ltv/models/base.py), [cox_ph.py](src/chrono_ltv/models/cox_ph.py), [xgb_survival.py](src/chrono_ltv/models/xgb_survival.py), [deepsurv.py](src/chrono_ltv/models/deepsurv.py) |
| 5 | MLflow Trainer + Evaluator | **Done ✓** | [trainer.py](src/chrono_ltv/training/trainer.py), [evaluator.py](src/chrono_ltv/training/evaluator.py) |
| 6 | FastAPI Serving Layer | **Done ✓** | [api.py](src/chrono_ltv/serving/api.py), [predictor.py](src/chrono_ltv/serving/predictor.py), [schemas.py](src/chrono_ltv/serving/schemas.py) |
| 7 | Evidently Drift Monitor | **Done ✓** | [drift.py](src/chrono_ltv/monitoring/drift.py), [alerts.py](src/chrono_ltv/monitoring/alerts.py), [scripts/monitor.py](scripts/monitor.py) |
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

#### `data/` — Steps 1–2

| File | What it contains |
|------|-----------------|
| [simulator.py](src/chrono_ltv/data/simulator.py) | `EcommerceSimulator` façade + 5 private factories: `_CustomerFactory`, `_TransactionFactory`, `_ClickstreamFactory`, `_SupportTicketFactory`, `_SurvivalLabelBuilder`, `_NoiseInjector`. Generates 5 Parquet datasets. |
| [schemas.py](src/chrono_ltv/data/schemas.py) | Pydantic v2 models: `CustomerRecord`, `TransactionRecord`, `ClickstreamRecord`, `SupportTicketRecord`, `SurvivalLabel`. Used for row-level validation and OpenAPI docs. |
| `validators.py` | *(Step 2 — this session)* `DataValidator` + `ValidationSummary`: GX 1.x ephemeral suites for all 5 datasets |

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

#### `serving/` — Step 6 (Done ✓)

| File | What it contains |
|------|-----------------|
| [api.py](src/chrono_ltv/serving/api.py) | `create_app()` factory + `lifespan` + `/predict`, `/health`, `/metrics` endpoints; `app` module instance for `make serve` |
| [predictor.py](src/chrono_ltv/serving/predictor.py) | `ModelPredictor`: lazy MLflow load, single-row `predict()` → `PredictionOutput`; `_model` injectable for testing |
| [schemas.py](src/chrono_ltv/serving/schemas.py) | `PredictRequest`, `PredictResponse`, `SurvivalCurve`, `HealthResponse`, `MetricsResponse` |

#### `monitoring/` — Step 7 (Done ✓)

| File | What it contains |
|------|-----------------|
| [drift.py](src/chrono_ltv/monitoring/drift.py) | `DriftMonitor` wrapping Evidently 0.7 `Report([DataDriftPreset(...)])`. Returns `DriftReport` + `FeatureDriftStat` dataclasses; `_is_drifted` routes p-value vs distance-based tests. |
| [alerts.py](src/chrono_ltv/monitoring/alerts.py) | `DriftAlerter.check()` → `list[DriftAlert]`; PSI severity tiers (warning ≥ 0.1, critical ≥ 0.2); `summarise()` for log output |

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
| [scripts/monitor.py](scripts/monitor.py) | `python scripts/monitor.py REFERENCE CURRENT [OPTIONS]` or `make monitor` |

### Tests (`tests/`)

| File | Coverage |
|------|---------|
| [tests/conftest.py](tests/conftest.py) | Session-scoped fixtures: `clean_datasets`, `noisy_datasets`, `clean_sim_cfg`, `noisy_sim_cfg` |
| [tests/unit/test_simulator.py](tests/unit/test_simulator.py) | 33 tests: schema, statistical, survival logic, noise injection, reproducibility, persistence |
| [tests/unit/test_schemas.py](tests/unit/test_schemas.py) | 20 tests: valid + invalid Pydantic instantiation for all 5 schemas |
| [tests/unit/test_io.py](tests/unit/test_io.py) | 6 tests: save/load roundtrip, overwrite guard, missing-file error |
| `tests/unit/test_validators.py` | *(Step 2 — this session)* |
| `tests/integration/` | *(Steps 5–6 — not yet written)* |
| [tests/unit/test_serving.py](tests/unit/test_serving.py) | 30 tests: ModelPredictor unit tests (no MLflow), /health, /metrics, /predict endpoint tests via TestClient |
| [tests/unit/test_monitoring.py](tests/unit/test_monitoring.py) | 33 tests: `_is_drifted` helper, `DriftMonitor` (stable/drifted pairs, PSI method, HTML/JSON output), `DriftAlerter` (severity tiers, sorting, summarise) |
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
- `_inject_outliers`: always cast float outlier values to `int` when the target column dtype is `np.integer` — pandas 2.x raises `TypeError` on float-into-int64 assignment
- `TCH` ruff rule is suppressed for `tests/*` — test files legitimately import stdlib outside `TYPE_CHECKING`
- Coverage threshold is 65% (not 80%) — `utils/logging.py` is intentionally not tested in unit suite; empty stub packages are omitted from coverage
- `np.row_stack` is deprecated in newer numpy — always use `np.vstack` for stacking survival function arrays
- DeepSurv tests live in `tests/unit/test_deepsurv.py` (separate file) so that `pytest.importorskip("torch")` at module level does not skip the Cox/XGB tests in CI
- `survival` extras group (`scikit-survival`, `xgboost`) installed in CI; `torch` is not (too large) — DeepSurv tests are skipped in CI and run locally only
- MLflow tracking URIs on Windows: never use `file://` or bare `C:\...` paths — use `sqlite:///path/to/mlflow.db` in tests
- `cumulative_dynamic_auc` raises `ValueError` on small CV folds when the censoring survival function hits zero — caught and treated as empty `td_auc` dict
- Evidently 0.7.x has a completely new API — `evidently.report.Report` no longer exists; use `from evidently import Report` + `from evidently.presets import DataDriftPreset`. The `Report.run()` returns a `Snapshot`. Constraint updated to `evidently>=0.7.0`
- Evidently `DataDriftPreset` snapshot dict: `metrics[0]` is always `DriftedColumnsCount`, `metrics[1:]` are `ValueDrift` per column. PSI/distance methods: drift if value > threshold. p-value methods: drift if value < threshold.
- `scripts/monitor.py` must NOT use `from __future__ import annotations` — Typer inspects `Path` annotations at runtime (same rule as generate_data.py)
- FastAPI lifespan checks `hasattr(app.state, "predictor")` before loading — tests pre-set the predictor on `app.state` to bypass MLflow; `_model` can also be injected directly on `ModelPredictor` for unit tests
- `AsyncGenerator` from `collections.abc` triggers ruff TC003 in `api.py` — move to `TYPE_CHECKING` block (safe because `from __future__ import annotations` makes the return annotation a string at runtime)
- `loguru.Logger` may not be importable at runtime on older loguru installs — annotate `get_logger` return as `Any` and guard `from loguru import Logger` under `TYPE_CHECKING`
