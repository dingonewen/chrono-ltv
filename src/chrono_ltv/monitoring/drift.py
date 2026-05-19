"""Evidently-based data drift detection for ChronoLTV.

Wraps Evidently 0.7.x's new-style Report / Snapshot API and returns
plain Python dataclasses so the rest of the pipeline stays decoupled
from Evidently internals.

Requires: pip install -e ".[monitoring]"  (evidently >= 0.7.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


# ── p-value vs distance method routing ───────────────────────────────────────

# These methods produce a *distance / divergence* score — drift when value > threshold.
# All others are assumed to produce a *p-value* — drift when value < threshold.
_DISTANCE_METHODS = frozenset(
    {"psi", "wasserstein", "ed", "es", "kl_div", "jensenshannon", "hellinger"}
)


def _is_drifted(method: str, value: float, threshold: float) -> bool:
    if method.lower() in _DISTANCE_METHODS:
        return value > threshold
    return value < threshold  # p-value based


# ── result dataclasses ────────────────────────────────────────────────────────


@dataclass
class FeatureDriftStat:
    """Drift statistics for one column."""

    drift_score: float
    drift_detected: bool
    stattest_name: str
    threshold: float


@dataclass
class DriftReport:
    """Aggregated drift report over a reference / current dataset pair."""

    n_drifted_features: int
    drift_share: float  # fraction of monitored columns that drifted
    dataset_drift: bool  # True when drift_share >= drift_share_threshold
    n_features: int
    feature_stats: dict[str, FeatureDriftStat] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        status = "DRIFTED" if self.dataset_drift else "STABLE"
        return (
            f"[{status}] {self.n_drifted_features}/{self.n_features} features drifted "
            f"({self.drift_share:.1%})"
        )


# ── monitor ───────────────────────────────────────────────────────────────────


class DriftMonitor:
    """Run Evidently drift detection and return a structured :class:`DriftReport`.

    Parameters
    ----------
    columns : list[str] | None
        Subset of columns to monitor.  ``None`` monitors all shared columns.
    num_method : str
        Statistical test for numerical columns.  ``"ks"`` (default) or ``"psi"``.
    cat_method : str
        Statistical test for categorical columns.  ``"chi2"`` (default) or ``"psi"``.
    drift_share : float
        Fraction threshold above which the dataset is considered drifted.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        num_method: str = "ks",
        cat_method: str = "chi2",
        drift_share: float = 0.5,
    ) -> None:
        self.columns = columns
        self.num_method = num_method
        self.cat_method = cat_method
        self.drift_share = drift_share

    def run(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        save_html: Path | str | None = None,
        save_json: Path | str | None = None,
    ) -> DriftReport:
        """Compute drift between *reference* and *current* DataFrames.

        Parameters
        ----------
        reference : pd.DataFrame
            Baseline dataset (training distribution).
        current : pd.DataFrame
            Dataset to check for drift.
        save_html : Path | str | None
            If given, write an Evidently HTML report to this path.
        save_json : Path | str | None
            If given, write the raw Evidently snapshot JSON to this path.
        """
        from evidently import Report
        from evidently.presets import DataDriftPreset

        preset = DataDriftPreset(
            columns=self.columns,
            num_method=self.num_method,
            cat_method=self.cat_method,
            drift_share=self.drift_share,
        )
        snapshot = Report([preset]).run(reference_data=reference, current_data=current)

        if save_html is not None:
            snapshot.save_html(str(save_html))
        if save_json is not None:
            snapshot.save_json(str(save_json))

        return self._parse(snapshot)

    # ── private ───────────────────────────────────────────────────────────────

    def _parse(self, snapshot: Any) -> DriftReport:
        metrics: list[dict[str, Any]] = snapshot.dict()["metrics"]

        # metrics[0] is always DriftedColumnsCount
        summary_val: dict[str, Any] = metrics[0]["value"]
        n_drifted = int(summary_val["count"])

        feature_stats: dict[str, FeatureDriftStat] = {}
        for m in metrics[1:]:
            cfg: dict[str, Any] = m["config"]
            col: str = cfg["column"]
            method: str = cfg["method"]
            threshold: float = float(cfg.get("threshold", 0.05))
            value: float = float(m["value"])
            feature_stats[col] = FeatureDriftStat(
                drift_score=value,
                drift_detected=_is_drifted(method, value, threshold),
                stattest_name=method,
                threshold=threshold,
            )

        n_features = len(feature_stats)
        drift_share = n_drifted / n_features if n_features > 0 else 0.0

        return DriftReport(
            n_drifted_features=n_drifted,
            drift_share=drift_share,
            dataset_drift=drift_share >= self.drift_share,
            n_features=n_features,
            feature_stats=feature_stats,
        )
