"""Pydantic request / response models for the ChronoLTV serving API."""

from pydantic import BaseModel, Field


class SurvivalCurve(BaseModel):
    """Discretised survival function S(t) at requested time horizons."""

    time_days: list[int]
    probabilities: list[float]


class PredictRequest(BaseModel):
    """Inference request — one customer, pre-engineered feature vector."""

    customer_id: str
    features: dict[str, float] = Field(..., description="Named feature values (post-pipeline).")


class PredictResponse(BaseModel):
    """Survival-analysis predictions for a single customer."""

    customer_id: str
    risk_score: float = Field(..., description="Higher = higher churn risk.")
    median_survival_days: float = Field(..., description="Expected days until churn.")
    survival_curve: SurvivalCurve
    model_version: str = "unknown"


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    model_loaded: bool
    model_uri: str


class MetricsResponse(BaseModel):
    requests_total: int
    requests_failed: int
    avg_latency_ms: float
