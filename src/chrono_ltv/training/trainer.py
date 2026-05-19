"""MLflow-integrated survival model trainer with stratified k-fold CV.

Usage
-----
>>> from chrono_ltv.training.trainer import SurvivalTrainer
>>> trainer = SurvivalTrainer(experiment_name="chrono-ltv-survival")
>>> results = trainer.cross_validate(model, X, y, n_splits=5)

Each CV fold opens a child MLflow run that logs:
  - fold metrics (C-index, IBS, time-dependent AUC, Brier scores)
  - model artifact
  - feature names + hyperparams

The parent run logs mean ± std of each metric across folds.

Requires: pip install -e ".[mlops,survival]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chrono_ltv.training.evaluator import EvaluationResult, SurvivalEvaluator
from chrono_ltv.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from chrono_ltv.models.base import SurvivalModel

logger = get_logger(__name__)


class SurvivalTrainer:
    """Trains and evaluates a SurvivalModel with MLflow tracking.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment to log into.
    tracking_uri : str | None
        MLflow tracking server URI.  ``None`` uses the local ``mlruns/`` default.
    time_points : list[int] | None
        Evaluation horizons in days — passed to ``SurvivalEvaluator``.
    """

    def __init__(
        self,
        experiment_name: str = "chrono-ltv-survival",
        tracking_uri: str | None = None,
        time_points: list[int] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.time_points = time_points
        self._evaluator = SurvivalEvaluator(time_points=time_points)

    # ── public ───────────────────────────────────────────────────────────

    def cross_validate(
        self,
        model: SurvivalModel,
        X: pd.DataFrame,
        y: npt.NDArray[Any],
        n_splits: int = 5,
        random_state: int = 42,
        run_name: str | None = None,
    ) -> list[EvaluationResult]:
        """Run stratified k-fold CV and log everything to MLflow.

        Returns a list of ``EvaluationResult`` — one per fold.
        """
        import mlflow
        from sklearn.model_selection import StratifiedKFold

        self._setup_mlflow(mlflow)

        parent_run_name = run_name or f"{model.name}_cv{n_splits}"
        fold_results: list[EvaluationResult] = []

        events = y["event"].astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.set_tags(
                {
                    "model": model.name,
                    "n_splits": str(n_splits),
                    "n_samples": str(len(X)),
                    "n_features": str(X.shape[1]),
                }
            )
            mlflow.log_params(model.get_params())

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, events)):
                result = self._run_fold(
                    model=model,
                    X=X,
                    y=y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    fold_idx=fold_idx,
                    parent_run_id=parent_run.info.run_id,
                )
                fold_results.append(result)
                logger.info(result.summary())

            self._log_aggregate_metrics(fold_results)

        return fold_results

    def train_final(
        self,
        model: SurvivalModel,
        X: pd.DataFrame,
        y: npt.NDArray[Any],
        run_name: str | None = None,
        register_as: str | None = None,
    ) -> str:
        """Train on the full dataset and register the model in MLflow.

        Returns the MLflow run ID.
        """
        import mlflow
        import mlflow.sklearn

        self._setup_mlflow(mlflow)
        final_run_name = run_name or f"{model.name}_final"

        with mlflow.start_run(run_name=final_run_name) as run:
            mlflow.set_tag("training_mode", "final")
            mlflow.log_params(model.get_params())
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", X.shape[1])

            model.fit(X, y)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=register_as,
            )
            run_id: str = run.info.run_id

        logger.info(f"Final model logged. Run ID: {run_id}")
        return run_id

    # ── private ──────────────────────────────────────────────────────────

    def _setup_mlflow(self, mlflow: Any) -> None:
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def _run_fold(
        self,
        model: SurvivalModel,
        X: pd.DataFrame,
        y: npt.NDArray[Any],
        train_idx: npt.NDArray[Any],
        test_idx: npt.NDArray[Any],
        fold_idx: int,
        parent_run_id: str,
    ) -> EvaluationResult:
        import mlflow
        import mlflow.sklearn

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        with mlflow.start_run(
            run_name=f"{model.name}_fold{fold_idx}",
            nested=True,
            tags={"parent_run_id": parent_run_id, "fold": str(fold_idx)},
        ):
            model.fit(X_train, y_train)
            result = self._evaluator.evaluate(model, X_test, y_test, y_train, fold=fold_idx)

            mlflow.log_metrics(
                {
                    "c_index": result.c_index,
                    "ibs": result.ibs,
                    **{f"brier_{t}d": v for t, v in result.brier_scores.items()},
                    **{f"td_auc_{t}d": v for t, v in result.td_auc.items()},
                }
            )
            mlflow.log_param("fold", fold_idx)
            mlflow.sklearn.log_model(model, artifact_path=f"model_fold{fold_idx}")

        return result

    @staticmethod
    def _log_aggregate_metrics(results: list[EvaluationResult]) -> None:
        """Log mean ± std of each metric across folds to the parent run."""
        import mlflow

        c_indices = [r.c_index for r in results]
        ibs_vals = [r.ibs for r in results]

        mlflow.log_metrics(
            {
                "mean_c_index": float(np.mean(c_indices)),
                "std_c_index": float(np.std(c_indices)),
                "mean_ibs": float(np.mean(ibs_vals)),
                "std_ibs": float(np.std(ibs_vals)),
            }
        )

        if results and results[0].td_auc:
            for t in results[0].td_auc:
                vals = [r.td_auc[t] for r in results if t in r.td_auc]
                mlflow.log_metrics(
                    {
                        f"mean_td_auc_{t}d": float(np.mean(vals)),
                        f"std_td_auc_{t}d": float(np.std(vals)),
                    }
                )
