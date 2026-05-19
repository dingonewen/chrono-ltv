"""XGBoost Survival wrapper (objective=survival:cox).

Requires: pip install -e ".[ml]"   (xgboost >= 2.0, scikit-survival >= 0.22)

XGBoost outputs log-hazard ratios (positive = higher risk), which are
monotonically equivalent to risk scores needed by the concordance index.
Survival functions are estimated by pairing the XGBoost risk scores with
a baseline cumulative hazard computed via the Breslow estimator from
scikit-survival.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chrono_ltv.models.base import SurvivalModel

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


class XGBSurvivalModel(SurvivalModel):
    """XGBoost with ``objective=survival:cox``.

    Parameters
    ----------
    n_estimators : int
    max_depth : int
    learning_rate : float
    subsample : float
    colsample_bytree : float
    min_child_weight : int
    early_stopping_rounds : int | None
        Requires a validation set passed to ``fit(eval_set=...)``.
    random_state : int
    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        early_stopping_rounds: int | None = 30,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._model: Any = None
        self._baseline_hazard: Any = None
        self._train_y: npt.NDArray[Any] | None = None

    # ── SurvivalModel interface ───────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: npt.NDArray[Any],
        eval_set: list[tuple[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> XGBSurvivalModel:
        import xgboost as xgb
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        # XGBoost survival:cox expects labels as (sign * duration):
        # positive if event observed, negative if censored.
        labels = _to_xgb_labels(y)

        params: dict[str, Any] = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "random_state": self.random_state,
            "tree_method": "hist",
        }
        if self.early_stopping_rounds is not None and eval_set is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        self._model = xgb.XGBRegressor(**params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [(ev[0], _to_xgb_labels(ev[1])) for ev in eval_set]
            fit_kwargs["verbose"] = False
        self._model.fit(X, labels, **fit_kwargs)

        cox_baseline = CoxPHSurvivalAnalysis(alpha=0.0)
        cox_baseline.fit(X, y)
        self._baseline_hazard = cox_baseline
        self._train_y = y
        return self

    def predict_survival_function(
        self,
        X: pd.DataFrame,
        times: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        self._check_fitted()
        surv_fns = self._baseline_hazard.predict_survival_function(X)
        t = times if times is not None else self._baseline_hazard.unique_times_
        return np.vstack([fn(t) for fn in surv_fns])

    def predict_risk_score(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        self._check_fitted()
        result: npt.NDArray[Any] = self._model.predict(X)
        return result

    def predict_median_survival_time(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        self._check_fitted()
        surv_fns = self._baseline_hazard.predict_survival_function(X)
        medians = np.empty(len(surv_fns))
        for i, fn in enumerate(surv_fns):
            below = fn.x[fn(fn.x) <= 0.5]
            medians[i] = below[0] if len(below) > 0 else np.inf
        return medians

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Call fit() before predict.")


def _to_xgb_labels(y: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert structured array to XGBoost survival labels.

    XGBoost convention: label = duration if event, label = -duration if censored.
    """
    durations: npt.NDArray[Any] = y["duration"].astype(float)
    events: npt.NDArray[Any] = y["event"].astype(bool)
    return np.where(events, durations, -durations)
