"""Abstract base class for all ChronoLTV survival models.

Every concrete model must implement fit, predict_survival_function,
predict_risk_score, and predict_median_survival_time so the training
loop and evaluator can treat them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


class SurvivalModel(ABC):
    """Common interface for Cox PH, XGBoost-Survival, and DeepSurv."""

    # ── fit ──────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[Any],
        **kwargs: Any,
    ) -> SurvivalModel:
        """Train on feature matrix *X* and structured survival array *y*.

        *y* must have dtype ``[("event", bool), ("duration", float)]``.
        Returns ``self`` for method chaining.
        """

    # ── predict ──────────────────────────────────────────────────────────

    @abstractmethod
    def predict_survival_function(
        self,
        X: pd.DataFrame,
        times: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Return survival probabilities S(t | X).

        Shape: ``(n_samples, n_times)``.  *times* defaults to the
        time points seen during training.
        """

    @abstractmethod
    def predict_risk_score(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Return a scalar risk score per sample (higher = higher risk).

        Shape: ``(n_samples,)``.  Used by concordance-index evaluators.
        """

    @abstractmethod
    def predict_median_survival_time(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Return median survival time in days per sample.

        Shape: ``(n_samples,)``.
        """

    # ── serialisation ────────────────────────────────────────────────────

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return hyperparameters as a JSON-serialisable dict."""

    # ── convenience ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{self.name}({params})"
