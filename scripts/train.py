"""CLI entry point: train a survival model and register it in MLflow.

Reads the five Parquet files written by ``generate_data.py``, runs the full
feature engineering pipeline, and trains a CoxPH model with k-fold CV.
The best fold's model is then trained on the full dataset and registered in
the MLflow model registry so the dashboard can load it directly.

Usage
-----
    # Step 1 — generate data (if not already done)
    python scripts/generate_data.py

    # Step 2 — train and register
    python scripts/train.py

    # or with overrides
    python scripts/train.py --n-splits 3 --alpha 0.1 --tracking-uri sqlite:///mlflow.db

Requires
--------
    pip install -e ".[ml,mlops,features,survival]"
"""

from pathlib import Path

import typer
from loguru import logger

from chrono_ltv.utils.logging import configure_logging

app = typer.Typer(name="chrono-train", add_completion=False)


@app.command()
def main(
    data_dir: Path = typer.Option(
        Path("data/raw"), help="Directory containing the five simulator Parquet files."
    ),
    tracking_uri: str = typer.Option(
        "mlruns", help="MLflow tracking URI (file store or HTTP server)."
    ),
    experiment_name: str = typer.Option(
        "chrono-ltv-survival", help="MLflow experiment name."
    ),
    register_as: str = typer.Option(
        "chrono-ltv-cox", help="Name to register the final model under in MLflow."
    ),
    alpha: float = typer.Option(0.5, help="CoxPH L2 regularisation strength."),
    n_splits: int = typer.Option(5, help="Number of CV folds."),
    seed: int = typer.Option(42, help="Random seed for CV splits."),
    log_level: str = typer.Option("INFO", help="Logging verbosity."),
) -> None:
    """Train CoxPH on real simulator data and register the model in MLflow."""
    configure_logging(level=log_level)
    logger.info("ChronoLTV — Survival Model Trainer")

    # ── 1. Load parquet datasets ──
    from chrono_ltv.utils.io import load_parquet

    parquet_names = [
        "customers", "transactions", "clickstream", "support_tickets", "survival_labels"
    ]
    datasets = {}
    for name in parquet_names:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            logger.error(
                f"Missing: {path}  —  run `python scripts/generate_data.py` first."
            )
            raise typer.Exit(code=1)
        datasets[name] = load_parquet(path)

    logger.info(
        f"Loaded datasets: "
        + ", ".join(f"{k}={len(v):,}" for k, v in datasets.items())
    )

    # ── 2. Feature engineering ──
    from chrono_ltv.features.pipeline import FeaturePipeline

    logger.info("Running feature engineering pipeline…")
    pipe = FeaturePipeline(scale=True)
    X, y = pipe.fit_transform(
        customers=datasets["customers"],
        transactions=datasets["transactions"],
        clickstream=datasets["clickstream"],
        tickets=datasets["support_tickets"],
        labels=datasets["survival_labels"],
    )
    logger.info(f"Feature matrix: {X.shape[0]:,} customers × {X.shape[1]} features")

    # ── 3. Cross-validated training ──
    import mlflow

    from chrono_ltv.models.cox_ph import CoxPHModel
    from chrono_ltv.training.trainer import SurvivalTrainer

    mlflow.set_tracking_uri(tracking_uri)
    trainer = SurvivalTrainer(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
    model = CoxPHModel(alpha=alpha)

    logger.info(f"Running {n_splits}-fold CV…")
    cv_results = trainer.cross_validate(model, X, y, n_splits=n_splits, random_state=seed)

    mean_c = sum(r.c_index for r in cv_results) / len(cv_results)
    logger.info(f"CV complete — mean C-index: {mean_c:.4f}")

    # ── 4. Final model (full dataset) → register in MLflow ──
    logger.info("Training final model on full dataset and registering…")
    run_id = trainer.train_final(
        CoxPHModel(alpha=alpha),
        X,
        y,
        run_name=f"cox_final_alpha{alpha}",
        register_as=register_as,
    )

    # Tag the run so the dashboard's _try_mlflow_model() finds it immediately
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    client.set_tag(run_id, "training_mode", "final")

    typer.echo("\n── Training Complete ────────────────────────────────────")
    typer.echo(f"  CV mean C-index : {mean_c:.4f}")
    typer.echo(f"  MLflow run ID   : {run_id}")
    typer.echo(f"  Registered as   : {register_as}")
    typer.echo(f"  Tracking URI    : {tracking_uri}")
    typer.echo("\nDashboard will now load this model automatically.")


if __name__ == "__main__":
    app()
