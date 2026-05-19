"""Unit tests for SurvivalEvaluator, EvaluationResult, and SurvivalTrainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")

from chrono_ltv.features.pipeline import SURVIVAL_DTYPE  # noqa: E402
from chrono_ltv.models.cox_ph import CoxPHModel  # noqa: E402
from chrono_ltv.training.evaluator import EvaluationResult, SurvivalEvaluator  # noqa: E402

mlflow_mod = pytest.importorskip("mlflow", reason="mlflow not installed")

from chrono_ltv.training.trainer import SurvivalTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def survival_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 120
    X = pd.DataFrame(
        {f"f{i}": rng.standard_normal(n) for i in range(5)},
    )
    durations = rng.exponential(scale=300, size=n).astype(np.float64)
    events = rng.random(n) > 0.25
    y = np.empty(n, dtype=SURVIVAL_DTYPE)
    y["event"] = events
    y["duration"] = durations
    return X, y


@pytest.fixture(scope="module")
def fitted_cox_for_eval(
    survival_dataset: tuple[pd.DataFrame, np.ndarray],
) -> tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray]:
    X, y = survival_dataset
    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    model = CoxPHModel(alpha=1.0).fit(X_train, y_train)
    return model, X_test, y_test, y_train


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_summary_contains_model_name(self) -> None:
        r = EvaluationResult(
            model_name="CoxPHModel",
            fold=0,
            c_index=0.72,
            ibs=0.15,
            td_auc={90: 0.74, 180: 0.71},
        )
        s = r.summary()
        assert "CoxPHModel" in s
        assert "0.7200" in s

    def test_summary_with_empty_td_auc(self) -> None:
        r = EvaluationResult(model_name="M", fold=1, c_index=0.6, ibs=0.2)
        assert "M" in r.summary()

    def test_dataclass_defaults(self) -> None:
        r = EvaluationResult(model_name="X", fold=0, c_index=0.5, ibs=0.3)
        assert r.brier_scores == {}
        assert r.td_auc == {}
        assert r.n_train == 0
        assert r.n_test == 0


# ---------------------------------------------------------------------------
# SurvivalEvaluator
# ---------------------------------------------------------------------------


class TestSurvivalEvaluator:
    def test_returns_evaluation_result(
        self,
        fitted_cox_for_eval: tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        model, X_test, y_test, y_train = fitted_cox_for_eval
        result = SurvivalEvaluator().evaluate(model, X_test, y_test, y_train)
        assert isinstance(result, EvaluationResult)

    def test_c_index_in_unit_interval(
        self,
        fitted_cox_for_eval: tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        model, X_test, y_test, y_train = fitted_cox_for_eval
        result = SurvivalEvaluator().evaluate(model, X_test, y_test, y_train)
        assert 0.0 <= result.c_index <= 1.0

    def test_ibs_positive(
        self,
        fitted_cox_for_eval: tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        model, X_test, y_test, y_train = fitted_cox_for_eval
        result = SurvivalEvaluator().evaluate(model, X_test, y_test, y_train)
        assert result.ibs > 0.0

    def test_brier_scores_keys_are_valid_times(
        self,
        fitted_cox_for_eval: tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        model, X_test, y_test, y_train = fitted_cox_for_eval
        evaluator = SurvivalEvaluator(time_points=[30, 90, 180])
        result = evaluator.evaluate(model, X_test, y_test, y_train)
        for t in result.brier_scores:
            assert t in [30, 90, 180]

    def test_brier_scores_in_range(
        self,
        fitted_cox_for_eval: tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        model, X_test, y_test, y_train = fitted_cox_for_eval
        result = SurvivalEvaluator().evaluate(model, X_test, y_test, y_train)
        for v in result.brier_scores.values():
            assert 0.0 <= v <= 1.0

    def test_td_auc_in_unit_interval(
        self,
        fitted_cox_for_eval: tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        model, X_test, y_test, y_train = fitted_cox_for_eval
        result = SurvivalEvaluator().evaluate(model, X_test, y_test, y_train)
        for v in result.td_auc.values():
            assert 0.0 <= v <= 1.0

    def test_n_train_n_test_populated(
        self,
        fitted_cox_for_eval: tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        model, X_test, y_test, y_train = fitted_cox_for_eval
        result = SurvivalEvaluator().evaluate(model, X_test, y_test, y_train, fold=2)
        assert result.n_train == len(y_train)
        assert result.n_test == len(y_test)
        assert result.fold == 2

    def test_time_points_beyond_training_data_are_dropped(
        self,
        fitted_cox_for_eval: tuple[CoxPHModel, pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        model, X_test, y_test, y_train = fitted_cox_for_eval
        # 1_000_000 days is well beyond any training duration
        evaluator = SurvivalEvaluator(time_points=[30, 1_000_000])
        result = evaluator.evaluate(model, X_test, y_test, y_train)
        assert 1_000_000 not in result.brier_scores


# ---------------------------------------------------------------------------
# SurvivalTrainer (requires mlflow)
# ---------------------------------------------------------------------------


class TestSurvivalTrainer:
    def test_cross_validate_returns_correct_n_folds(
        self,
        survival_dataset: tuple[pd.DataFrame, np.ndarray],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        X, y = survival_dataset
        trainer = SurvivalTrainer(
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            time_points=[90, 180],
        )
        results = trainer.cross_validate(CoxPHModel(alpha=1.0), X, y, n_splits=3)
        assert len(results) == 3

    def test_cross_validate_all_results_are_evaluation_results(
        self,
        survival_dataset: tuple[pd.DataFrame, np.ndarray],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        X, y = survival_dataset
        trainer = SurvivalTrainer(
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            time_points=[90, 180],
        )
        results = trainer.cross_validate(CoxPHModel(alpha=1.0), X, y, n_splits=3)
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_cross_validate_c_indices_plausible(
        self,
        survival_dataset: tuple[pd.DataFrame, np.ndarray],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        X, y = survival_dataset
        trainer = SurvivalTrainer(
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            time_points=[90, 180],
        )
        results = trainer.cross_validate(CoxPHModel(alpha=1.0), X, y, n_splits=3)
        for r in results:
            assert 0.0 <= r.c_index <= 1.0

    def test_train_final_returns_run_id_string(
        self,
        survival_dataset: tuple[pd.DataFrame, np.ndarray],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        X, y = survival_dataset
        trainer = SurvivalTrainer(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
        run_id = trainer.train_final(CoxPHModel(alpha=1.0), X, y)
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_mlflow_run_created_on_disk(
        self,
        survival_dataset: tuple[pd.DataFrame, np.ndarray],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        X, y = survival_dataset
        db_path = tmp_path / "mlflow.db"
        trainer = SurvivalTrainer(tracking_uri=f"sqlite:///{db_path}")
        trainer.train_final(CoxPHModel(alpha=1.0), X, y)
        assert db_path.exists()
