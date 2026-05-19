"""PSI / p-value threshold checks and alert routing for drift reports.

Usage
-----
    monitor = DriftMonitor(num_method="psi")
    report  = monitor.run(reference, current)

    alerter = DriftAlerter()
    alerts  = alerter.check(report)
    print(alerter.summarise(alerts))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chrono_ltv.monitoring.drift import DriftReport, FeatureDriftStat

_PSI_METHOD_NAMES = frozenset({"psi", "psi_threshold"})


@dataclass
class DriftAlert:
    feature: str
    drift_score: float
    threshold: float
    severity: str  # "warning" | "critical"
    stattest_name: str
    message: str


class DriftAlerter:
    """Apply threshold rules to a :class:`DriftReport` and produce alerts.

    Parameters
    ----------
    psi_warning : float
        PSI value above which a *warning* alert is raised.
    psi_critical : float
        PSI value above which a *critical* alert is raised.
    """

    def __init__(
        self,
        psi_warning: float = 0.1,
        psi_critical: float = 0.2,
    ) -> None:
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical

    def check(self, report: DriftReport) -> list[DriftAlert]:
        """Return one :class:`DriftAlert` per drifted feature."""
        alerts: list[DriftAlert] = []
        for feature, stat in report.feature_stats.items():
            if not stat.drift_detected:
                continue
            severity = self._severity(stat)
            alerts.append(
                DriftAlert(
                    feature=feature,
                    drift_score=stat.drift_score,
                    threshold=stat.threshold,
                    severity=severity,
                    stattest_name=stat.stattest_name,
                    message=(
                        f"{feature}: {stat.stattest_name}={stat.drift_score:.4f}"
                        f" (threshold={stat.threshold})"
                    ),
                )
            )
        # Sort critical first, then by feature name for deterministic output
        alerts.sort(key=lambda a: (0 if a.severity == "critical" else 1, a.feature))
        return alerts

    def summarise(self, alerts: list[DriftAlert]) -> str:
        """Return a human-readable summary string."""
        if not alerts:
            return "No drift alerts."
        critical = sum(1 for a in alerts if a.severity == "critical")
        warning = len(alerts) - critical
        header = f"Drift alerts: {len(alerts)} total ({critical} critical, {warning} warning)"
        lines = [header] + [f"  [{a.severity.upper()}] {a.message}" for a in alerts]
        return "\n".join(lines)

    # ── private ───────────────────────────────────────────────────────────────

    def _severity(self, stat: FeatureDriftStat) -> str:
        if stat.stattest_name.lower() in _PSI_METHOD_NAMES:
            return "critical" if stat.drift_score >= self.psi_critical else "warning"
        # Binary for p-value tests: drift_detected already confirmed
        return "critical"
