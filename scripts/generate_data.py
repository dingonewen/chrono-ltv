"""CLI entry point: simulate e-commerce data.

Usage
-----
    python scripts/generate_data.py [OPTIONS]

    # or via Make:
    make simulate
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from chrono_ltv.data.simulator import EcommerceSimulator, NoiseConfig, SimulatorConfig
from chrono_ltv.utils.logging import configure_logging

app = typer.Typer(name="chrono-simulate", add_completion=False)


@app.command()
def main(
    n_customers: int = typer.Option(10_000, help="Number of synthetic customers to generate."),
    start_date: str = typer.Option("2022-01-01", help="Observation window start (YYYY-MM-DD)."),
    end_date: str = typer.Option("2024-12-31", help="Observation window end (YYYY-MM-DD)."),
    churn_threshold: int = typer.Option(90, help="Days of inactivity that define churn."),
    output_dir: Path = typer.Option(Path("data/raw"), help="Directory for output Parquet files."),
    seed: int = typer.Option(42, help="Global random seed for reproducibility."),
    missing_rate: float = typer.Option(
        0.04, min=0.0, max=0.5, help="Rate of injected missing values."
    ),
    duplicate_rate: float = typer.Option(
        0.01, min=0.0, max=0.2, help="Rate of injected duplicate rows."
    ),
    no_noise: bool = typer.Option(
        False, "--no-noise", help="Disable all noise injection (clean data)."
    ),
    log_level: str = typer.Option("INFO", help="Logging verbosity."),
) -> None:
    """Generate synthetic e-commerce datasets for the ChronoLTV pipeline."""
    configure_logging(level=log_level)
    logger.info("ChronoLTV — Data Stream Simulator")

    noise_cfg = NoiseConfig(
        missing_rate=0.0 if no_noise else missing_rate,
        duplicate_rate=0.0 if no_noise else duplicate_rate,
        outlier_rate=0.0 if no_noise else 0.02,
        future_date_rate=0.0 if no_noise else 0.005,
        negative_amount_rate=0.0 if no_noise else 0.005,
    )

    cfg = SimulatorConfig(
        n_customers=n_customers,
        start_date=start_date,
        end_date=end_date,
        churn_threshold_days=churn_threshold,
        random_seed=seed,
        noise=noise_cfg,
        output_dir=output_dir,
    )

    sim = EcommerceSimulator(cfg)
    datasets = sim.run(persist=True)

    typer.echo("\n── Simulation Complete ──────────────────────────────────")
    for name, df in datasets.items():
        typer.echo(f"  {name:<20} {len(df):>10,} rows")
    typer.echo(f"\nFiles written to: {output_dir.resolve()}")


if __name__ == "__main__":
    app()
