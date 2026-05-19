"""FastAPI serving layer for ChronoLTV survival predictions.

Endpoints
---------
GET  /health   — liveness + model-load status
GET  /metrics  — request counters and average latency
POST /predict  — single-customer survival prediction

Usage (dev)
-----------
    uvicorn src.chrono_ltv.serving.api:app --reload

Usage (programmatic / tests)
-----------------------------
    from chrono_ltv.serving.api import create_app
    app = create_app(model_uri="models:/chrono-ltv-cox/Production")
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request

from chrono_ltv.serving.predictor import ModelPredictor
from chrono_ltv.serving.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    SurvivalCurve,
)
from chrono_ltv.utils.logging import get_logger

logger = get_logger(__name__)


# ── server-wide metrics ───────────────────────────────────────────────────────


class _ServerMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total: int = 0
        self.requests_failed: int = 0
        self._total_latency_ms: float = 0.0

    def record(self, latency_ms: float, *, failed: bool = False) -> None:
        with self._lock:
            self.requests_total += 1
            self._total_latency_ms += latency_ms
            if failed:
                self.requests_failed += 1

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self._total_latency_ms / self.requests_total


# ── lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Allow tests to inject a predictor before the lifespan runs.
    if not hasattr(app.state, "predictor"):
        predictor = ModelPredictor(
            model_uri=app.state.model_uri,
            time_points=getattr(app.state, "time_points", None),
        )
        try:
            predictor.load()
            logger.info(f"Model loaded from {predictor.model_uri}")
        except Exception as exc:
            logger.warning(f"Model could not be loaded at startup: {exc}")
        app.state.predictor = predictor
    yield


# ── app factory ───────────────────────────────────────────────────────────────


def create_app(
    model_uri: str = "models:/chrono-ltv-cox/Production",
    time_points: list[int] | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure a FastAPI application instance."""
    _metrics = _ServerMetrics()

    app = FastAPI(
        title="ChronoLTV Survival API",
        description="Predict customer churn timing and LTV via survival analysis.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.model_uri = model_uri
    app.state.time_points = time_points
    app.state.metrics = _metrics

    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    # ── routes ────────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> Any:
        pred: ModelPredictor | None = getattr(request.app.state, "predictor", None)
        loaded = pred is not None and pred.is_loaded
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            model_uri=pred.model_uri if pred is not None else model_uri,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics_endpoint(request: Request) -> Any:
        m: _ServerMetrics = request.app.state.metrics
        return MetricsResponse(
            requests_total=m.requests_total,
            requests_failed=m.requests_failed,
            avg_latency_ms=m.avg_latency_ms,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> Any:
        pred: ModelPredictor | None = getattr(request.app.state, "predictor", None)
        m: _ServerMetrics = request.app.state.metrics

        if pred is None or not pred.is_loaded:
            raise HTTPException(status_code=503, detail="Model not ready")

        t0 = time.perf_counter()
        try:
            output = pred.predict(body.features)
            latency = (time.perf_counter() - t0) * 1000.0
            m.record(latency)
            return PredictResponse(
                customer_id=body.customer_id,
                risk_score=output.risk_score,
                median_survival_days=output.median_survival_days,
                survival_curve=SurvivalCurve(
                    time_days=output.survival_times,
                    probabilities=output.survival_probs,
                ),
                model_version=pred.model_version,
            )
        except HTTPException:
            raise
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            m.record(latency, failed=True)
            logger.error(f"Prediction failed for {body.customer_id}: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# ── default app instance (used by `make serve` / uvicorn) ────────────────────

app = create_app()
