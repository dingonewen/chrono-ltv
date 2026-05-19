"""CLI entry point: run drift monitoring on a pair of Parquet files.

Usage
-----
    python scripts/monitor.py REFERENCE CURRENT [OPTIONS]

    # or via Make:
    make monitor
"""

from pathlib import Path

import typer
from loguru import logger

from chrono_ltv.monitoring.alerts import DriftAlerter
from chrono_ltv.monitoring.drift import DriftMonitor
from chrono_ltv.utils.io import load_parquet
from chrono_ltv.utils.logging import configure_logging

app = typer.Typer(name="chrono-monitor", add_completion=False)


@app.command()
def main(
    reference: Path = typer.Argument(..., help="Reference (baseline) Parquet file."),
    current: Path = typer.Argument(..., help="Current (production) Parquet file."),
    columns: str = typer.Option(
        "", help="Comma-separated column names to monitor. Empty = all columns."
    ),
    num_method: str = typer.Option("ks", help="Drift test for numerical columns (ks, psi, ...)."),
    cat_method: str = typer.Option(
        "chi2", help="Drift test for categorical columns (chi2, psi, ...)."
    ),
    drift_share: float = typer.Option(
        0.5, min=0.0, max=1.0, help="Dataset drift threshold (fraction of drifted columns)."
    ),
    psi_warning: float = typer.Option(0.1, help="PSI threshold for WARNING severity."),
    psi_critical: float = typer.Option(0.2, help="PSI threshold for CRITICAL severity."),
    output_html: Path = typer.Option(None, help="Save Evidently HTML report to this path."),
    output_json: Path = typer.Option(None, help="Save raw Evidently JSON to this path."),
    log_level: str = typer.Option("INFO", help="Logging verbosity."),
) -> None:
    """Run drift detection between REFERENCE and CURRENT datasets."""
    configure_logging(level=log_level)
    logger.info("ChronoLTV — Drift Monitor")

    col_list = [c.strip() for c in columns.split(",") if c.strip()] or None

    ref_df = load_parquet(reference)
    cur_df = load_parquet(current)
    logger.info(f"Reference: {len(ref_df):,} rows  |  Current: {len(cur_df):,} rows")

    monitor = DriftMonitor(
        columns=col_list,
        num_method=num_method,
        cat_method=cat_method,
        drift_share=drift_share,
    )
    report = monitor.run(ref_df, cur_df, save_html=output_html, save_json=output_json)

    alerter = DriftAlerter(psi_warning=psi_warning, psi_critical=psi_critical)
    alerts = alerter.check(report)

    typer.echo("\n── Drift Report ─────────────────────────────────────────")
    typer.echo(f"  {report.summary()}")
    typer.echo(f"\n{alerter.summarise(alerts)}")

    if output_html:
        typer.echo(f"\nHTML report saved to: {output_html}")
    if output_json:
        typer.echo(f"JSON snapshot saved to: {output_json}")

    if report.dataset_drift:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
