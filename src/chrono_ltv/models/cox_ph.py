"""Cox Proportional Hazards wrapper (scikit-survival).

Requires: pip install -e ".[ml]"   (scikit-survival >= 0.22)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chrono_ltv.models.base import SurvivalModel

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


class CoxPHModel(SurvivalModel):
    """Thin wrapper around ``sksurv.linear_model.CoxPHSurvivalAnalysis``.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength (Tikhonov).
    ties : str
        Method for handling tied event times — ``"breslow"`` or ``"efron"``.
    n_iter : int
        Maximum number of Newton-Raphson iterations.
    tol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        ties: str = "breslow",
        n_iter: int = 100,
        tol: float = 1e-9,
    ) -> None:
        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter
        self.tol = tol
        self._model: Any = None
        self._unique_times: npt.NDArray[Any] | None = None

    # ── SurvivalModel interface ───────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[Any],
        **kwargs: Any,
    ) -> CoxPHModel:
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        self._model = CoxPHSurvivalAnalysis(
            alpha=self.alpha,
            ties=self.ties,
            n_iter=self.n_iter,
            tol=self.tol,
        )
        self._model.fit(X, y)
        self._unique_times = self._model.unique_times_
        return self

    def predict_survival_function(
        self,
        X: pd.DataFrame,
        times: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        self._check_fitted()
        surv_fns = self._model.predict_survival_function(X)
        t = times if times is not None else self._unique_times
        return np.vstack([fn(t) for fn in surv_fns])

    def predict_risk_score(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        self._check_fitted()
        result: npt.NDArray[Any] = self._model.predict(X)
        return result

    def predict_median_survival_time(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        self._check_fitted()
        surv_fns = self._model.predict_survival_function(X)
        medians = np.empty(len(surv_fns))
        for i, fn in enumerate(surv_fns):
            # Median is the first time where S(t) <= 0.5
            below = fn.x[fn(fn.x) <= 0.5]
            medians[i] = below[0] if len(below) > 0 else np.inf
        return medians

    def get_params(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "ties": self.ties,
            "n_iter": self.n_iter,
            "tol": self.tol,
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Call fit() before predict.")

    @property
    def coef_(self) -> npt.NDArray[Any]:
        self._check_fitted()
        return self._model.coef_  # type: ignore[no-any-return]
