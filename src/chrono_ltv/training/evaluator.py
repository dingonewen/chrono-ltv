"""Survival-model evaluation metrics.

Metrics
-------
- Concordance index (Harrell's C) — overall discrimination
- Time-dependent AUC — discrimination at specific horizons
- Integrated Brier score — calibration over the whole time axis
- Brier score at fixed time points — calibration snapshots

Requires: pip install -e ".[survival]"   (scikit-survival >= 0.22)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from chrono_ltv.models.base import SurvivalModel


@dataclass
class EvaluationResult:
    """Structured container for one model's evaluation on one fold."""

    model_name: str
    fold: int
    c_index: float
    ibs: float  # integrated Brier score
    brier_scores: dict[int, float] = field(default_factory=dict)  # time_days → score
    td_auc: dict[int, float] = field(default_factory=dict)  # time_days → AUC
    n_train: int = 0
    n_test: int = 0

    def summary(self) -> str:
        td_auc_str = ", ".join(f"{t}d={v:.3f}" for t, v in sorted(self.td_auc.items()))
        return (
            f"{self.model_name} fold={self.fold} "
            f"C={self.c_index:.4f} IBS={self.ibs:.4f} "
            f"tdAUC=[{td_auc_str}]"
        )


class SurvivalEvaluator:
    """Compute standard survival-analysis metrics for a fitted model.

    Parameters
    ----------
    time_points : list[int]
        Days at which to evaluate time-dependent AUC and Brier scores.
        Defaults to ``[30, 60, 90, 180, 365]``.
    """

    def __init__(self, time_points: list[int] | None = None) -> None:
        self.time_points = time_points if time_points is not None else [30, 60, 90, 180, 365]

    def evaluate(
        self,
        model: SurvivalModel,
        X_test: pd.DataFrame,
        y_test: npt.NDArray[Any],
        y_train: npt.NDArray[Any],
        fold: int = 0,
    ) -> EvaluationResult:
        from sksurv.metrics import (
            brier_score,
            concordance_index_censored,
            cumulative_dynamic_auc,
            integrated_brier_score,
        )

        events_test: npt.NDArray[Any] = y_test["event"].astype(bool)
        durations_test: npt.NDArray[Any] = y_test["duration"].astype(float)

        # ── C-index ────────────────────────────────────────────────────
        risk_scores = model.predict_risk_score(X_test)
        c_idx, _, _, _, _ = concordance_index_censored(events_test, durations_test, risk_scores)

        # ── Survival function matrix ────────────────────────────────────
        # Clip time points to the range observed in training data
        train_max = float(y_train["duration"].max())
        valid_times = np.array([t for t in self.time_points if t < train_max], dtype=np.float64)

        surv_matrix = model.predict_survival_function(X_test, times=valid_times)

        # ── Brier score at each time point ──────────────────────────────
        brier_times, brier_vals = brier_score(y_train, y_test, surv_matrix, valid_times)
        brier_at = {int(t): float(v) for t, v in zip(brier_times, brier_vals, strict=True)}

        # ── Integrated Brier score ──────────────────────────────────────
        # requires ≥ 2 time points
        ibs_val = (
            float(integrated_brier_score(y_train, y_test, surv_matrix, valid_times))
            if len(valid_times) >= 2
            else float("nan")
        )

        # ── Time-dependent AUC ─────────────────────────────────────────
        # ValueError occurs on small folds when censoring survival reaches zero
        td_auc_dict: dict[int, float] = {}
        if len(valid_times) >= 2:
            try:
                auc_vals, _ = cumulative_dynamic_auc(
                    y_train, y_test, risk_scores, valid_times
                )
                td_auc_dict = {
                    int(t): float(v) for t, v in zip(valid_times, auc_vals, strict=True)
                }
            except ValueError:
                pass

        return EvaluationResult(
            model_name=model.name,
            fold=fold,
            c_index=float(c_idx),
            ibs=ibs_val,
            brier_scores=brier_at,
            td_auc=td_auc_dict,
            n_train=len(y_train),
            n_test=len(y_test),
        )
