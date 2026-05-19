"""Unit tests for DriftMonitor, DriftReport, and DriftAlerter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

evidently = pytest.importorskip("evidently", reason="evidently not installed")

from chrono_ltv.monitoring.alerts import DriftAlert, DriftAlerter  # noqa: E402
from chrono_ltv.monitoring.drift import (  # noqa: E402
    DriftMonitor,
    DriftReport,
    FeatureDriftStat,
    _is_drifted,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_COLS = ["num_a", "num_b", "num_c"]


@pytest.fixture(scope="module")
def stable_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same distribution — should show little or no drift."""
    rng = np.random.default_rng(0)
    n = 300
    ref = pd.DataFrame({c: rng.normal(0, 1, n) for c in _COLS})
    cur = pd.DataFrame({c: rng.normal(0, 1, n) for c in _COLS})
    return ref, cur


@pytest.fixture(scope="module")
def drifted_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Large mean shift — should reliably trigger drift on all columns."""
    rng = np.random.default_rng(1)
    n = 300
    ref = pd.DataFrame({c: rng.normal(0, 1, n) for c in _COLS})
    cur = pd.DataFrame({c: rng.normal(10, 1, n) for c in _COLS})
    return ref, cur


@pytest.fixture(scope="module")
def monitor_default() -> DriftMonitor:
    return DriftMonitor()


@pytest.fixture(scope="module")
def report_stable(
    monitor_default: DriftMonitor,
    stable_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> DriftReport:
    ref, cur = stable_pair
    return monitor_default.run(ref, cur)


@pytest.fixture(scope="module")
def report_drifted(
    monitor_default: DriftMonitor,
    drifted_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> DriftReport:
    ref, cur = drifted_pair
    return monitor_default.run(ref, cur)


# ---------------------------------------------------------------------------
# _is_drifted helper
# ---------------------------------------------------------------------------


class TestIsDrifted:
    def test_psi_above_threshold_is_drifted(self) -> None:
        assert _is_drifted("psi", 0.25, 0.1) is True

    def test_psi_below_threshold_not_drifted(self) -> None:
        assert _is_drifted("psi", 0.05, 0.1) is False

    def test_pvalue_below_threshold_is_drifted(self) -> None:
        assert _is_drifted("K-S p_value", 0.01, 0.05) is True

    def test_pvalue_above_threshold_not_drifted(self) -> None:
        assert _is_drifted("K-S p_value", 0.5, 0.05) is False

    def test_wasserstein_is_distance_based(self) -> None:
        assert _is_drifted("wasserstein", 1.5, 0.1) is True

    def test_chi2_is_pvalue_based(self) -> None:
        assert _is_drifted("chi2", 0.001, 0.05) is True


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------


class TestDriftMonitor:
    def test_run_returns_drift_report(self, report_stable: DriftReport) -> None:
        assert isinstance(report_stable, DriftReport)

    def test_n_features_matches_columns(self, report_stable: DriftReport) -> None:
        assert report_stable.n_features == len(_COLS)

    def test_feature_stats_keys_match_columns(self, report_stable: DriftReport) -> None:
        assert set(report_stable.feature_stats.keys()) == set(_COLS)

    def test_drift_share_in_unit_interval(self, report_stable: DriftReport) -> None:
        assert 0.0 <= report_stable.drift_share <= 1.0

    def test_n_drifted_non_negative(self, report_stable: DriftReport) -> None:
        assert report_stable.n_drifted_features >= 0

    def test_n_drifted_at_most_n_features(self, report_stable: DriftReport) -> None:
        assert report_stable.n_drifted_features <= report_stable.n_features

    def test_drifted_pair_detects_all_features_drifted(self, report_drifted: DriftReport) -> None:
        assert report_drifted.n_drifted_features == len(_COLS)

    def test_drifted_pair_dataset_drift_true(self, report_drifted: DriftReport) -> None:
        assert report_drifted.dataset_drift is True

    def test_stable_pair_dataset_drift_false(self, report_stable: DriftReport) -> None:
        assert report_stable.dataset_drift is False

    def test_feature_stat_has_expected_fields(self, report_drifted: DriftReport) -> None:
        stat = next(iter(report_drifted.feature_stats.values()))
        assert isinstance(stat.drift_score, float)
        assert isinstance(stat.drift_detected, bool)
        assert isinstance(stat.stattest_name, str)
        assert isinstance(stat.threshold, float)

    def test_columns_subset_limits_features(
        self, stable_pair: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = stable_pair
        monitor = DriftMonitor(columns=["num_a", "num_b"])
        report = monitor.run(ref, cur)
        assert report.n_features == 2
        assert "num_c" not in report.feature_stats

    def test_psi_method_recorded_in_stats(
        self, drifted_pair: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = drifted_pair
        monitor = DriftMonitor(num_method="psi")
        report = monitor.run(ref, cur)
        for stat in report.feature_stats.values():
            assert "psi" in stat.stattest_name.lower()

    def test_save_json_creates_file(
        self,
        stable_pair: tuple[pd.DataFrame, pd.DataFrame],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        ref, cur = stable_pair
        out = tmp_path / "drift.json"  # type: ignore[operator]
        DriftMonitor().run(ref, cur, save_json=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_html_creates_file(
        self,
        stable_pair: tuple[pd.DataFrame, pd.DataFrame],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        ref, cur = stable_pair
        out = tmp_path / "drift.html"  # type: ignore[operator]
        DriftMonitor().run(ref, cur, save_html=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_summary_contains_status(self, report_drifted: DriftReport) -> None:
        assert "DRIFTED" in report_drifted.summary()

    def test_summary_stable_contains_stable(self, report_stable: DriftReport) -> None:
        assert "STABLE" in report_stable.summary()


# ---------------------------------------------------------------------------
# DriftAlerter
# ---------------------------------------------------------------------------


class TestDriftAlerter:
    def test_check_returns_list(self, report_drifted: DriftReport) -> None:
        alerts = DriftAlerter().check(report_drifted)
        assert isinstance(alerts, list)

    def test_alerts_on_drifted_data(self, report_drifted: DriftReport) -> None:
        alerts = DriftAlerter().check(report_drifted)
        assert len(alerts) > 0

    def test_no_alerts_on_stable_data(self, report_stable: DriftReport) -> None:
        alerts = DriftAlerter().check(report_stable)
        assert len(alerts) == 0

    def test_alert_has_all_fields(self, report_drifted: DriftReport) -> None:
        alert = DriftAlerter().check(report_drifted)[0]
        assert isinstance(alert, DriftAlert)
        assert alert.feature in _COLS
        assert alert.severity in ("warning", "critical")
        assert alert.drift_score > 0.0
        assert isinstance(alert.message, str)

    def test_severity_critical_for_pvalue_drift(self) -> None:
        stat = FeatureDriftStat(
            drift_score=0.001, drift_detected=True, stattest_name="K-S p_value", threshold=0.05
        )
        report = DriftReport(
            n_drifted_features=1,
            drift_share=1.0,
            dataset_drift=True,
            n_features=1,
            feature_stats={"x": stat},
        )
        alert = DriftAlerter().check(report)[0]
        assert alert.severity == "critical"

    def test_psi_warning_tier(self) -> None:
        stat = FeatureDriftStat(
            drift_score=0.15, drift_detected=True, stattest_name="psi", threshold=0.1
        )
        report = DriftReport(
            n_drifted_features=1,
            drift_share=1.0,
            dataset_drift=True,
            n_features=1,
            feature_stats={"x": stat},
        )
        alert = DriftAlerter(psi_warning=0.1, psi_critical=0.2).check(report)[0]
        assert alert.severity == "warning"

    def test_psi_critical_tier(self) -> None:
        stat = FeatureDriftStat(
            drift_score=0.25, drift_detected=True, stattest_name="psi", threshold=0.1
        )
        report = DriftReport(
            n_drifted_features=1,
            drift_share=1.0,
            dataset_drift=True,
            n_features=1,
            feature_stats={"x": stat},
        )
        alert = DriftAlerter(psi_warning=0.1, psi_critical=0.2).check(report)[0]
        assert alert.severity == "critical"

    def test_alerts_sorted_critical_first(self, report_drifted: DriftReport) -> None:
        alerts = DriftAlerter().check(report_drifted)
        severities = [a.severity for a in alerts]
        critical_indices = [i for i, s in enumerate(severities) if s == "critical"]
        warning_indices = [i for i, s in enumerate(severities) if s == "warning"]
        if critical_indices and warning_indices:
            assert max(critical_indices) < min(warning_indices)

    def test_summarise_no_alerts(self) -> None:
        assert DriftAlerter().summarise([]) == "No drift alerts."

    def test_summarise_contains_count(self, report_drifted: DriftReport) -> None:
        alerts = DriftAlerter().check(report_drifted)
        summary = DriftAlerter().summarise(alerts)
        assert str(len(alerts)) in summary

    def test_summarise_contains_feature_name(self, report_drifted: DriftReport) -> None:
        alerts = DriftAlerter().check(report_drifted)
        summary = DriftAlerter().summarise(alerts)
        assert alerts[0].feature in summary
