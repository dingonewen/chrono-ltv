"""MLflow model loader and inference wrapper for the serving layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from chrono_ltv.models.base import SurvivalModel


@dataclass
class PredictionOutput:
    risk_score: float
    median_survival_days: float
    survival_times: list[int]
    survival_probs: list[float]


class ModelPredictor:
    """Loads a SurvivalModel from an MLflow URI and serves single-row predictions.

    Parameters
    ----------
    model_uri : str
        MLflow model URI, e.g. ``"models:/chrono-ltv-cox/Production"``
        or ``"runs:/<run_id>/model"``.
    time_points : list[int] | None
        Day horizons at which to evaluate the survival curve.
    """

    def __init__(
        self,
        model_uri: str,
        time_points: list[int] | None = None,
    ) -> None:
        self.model_uri = model_uri
        self.time_points: list[int] = time_points or [30, 60, 90, 180, 365]
        self._model: SurvivalModel | None = None
        self._model_version: str = model_uri

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        return self._model_version

    def load(self) -> None:
        """Pull the model artifact from MLflow and set ``is_loaded = True``."""
        import mlflow.sklearn

        self._model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self, features: dict[str, float]) -> PredictionOutput:
        """Run inference for a single customer feature dict."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import pandas as pd

        X: pd.DataFrame = pd.DataFrame([features])
        times: Any = np.array(self.time_points, dtype=np.float64)

        risk = float(self._model.predict_risk_score(X)[0])
        median = float(self._model.predict_median_survival_time(X)[0])
        surv_matrix = self._model.predict_survival_function(X, times=times)
        probs: list[float] = [float(p) for p in surv_matrix[0]]

        return PredictionOutput(
            risk_score=risk,
            median_survival_days=median,
            survival_times=self.time_points,
            survival_probs=probs,
        )
