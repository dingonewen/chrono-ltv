"""Unit tests for SurvivalModel implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chrono_ltv.features.pipeline import SURVIVAL_DTYPE
from chrono_ltv.models.base import SurvivalModel

sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")
xgb = pytest.importorskip("xgboost", reason="xgboost not installed")

from chrono_ltv.models.cox_ph import CoxPHModel  # noqa: E402
from chrono_ltv.models.xgb_survival import XGBSurvivalModel, _to_xgb_labels  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_survival_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """50-sample synthetic dataset for fast model tests."""
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame(
        {
            "feature_1": rng.standard_normal(n),
            "feature_2": rng.standard_normal(n),
            "feature_3": rng.uniform(0, 1, n),
        }
    )
    durations = rng.exponential(scale=200, size=n).astype(np.float64)
    events = rng.random(n) > 0.3
    y = np.empty(n, dtype=SURVIVAL_DTYPE)
    y["event"] = events
    y["duration"] = durations
    return X, y


@pytest.fixture(scope="module")
def fitted_cox(
    small_survival_dataset: tuple[pd.DataFrame, np.ndarray],
) -> CoxPHModel:
    X, y = small_survival_dataset
    return CoxPHModel(alpha=1.0).fit(X, y)


@pytest.fixture(scope="module")
def fitted_xgb(
    small_survival_dataset: tuple[pd.DataFrame, np.ndarray],
) -> XGBSurvivalModel:
    X, y = small_survival_dataset
    return XGBSurvivalModel(n_estimators=20, early_stopping_rounds=None).fit(X, y)


# ---------------------------------------------------------------------------
# Abstract interface contract
# ---------------------------------------------------------------------------


class TestSurvivalModelInterface:
    def test_cox_is_survival_model(self) -> None:
        assert isinstance(CoxPHModel(), SurvivalModel)

    def test_xgb_is_survival_model(self) -> None:
        assert isinstance(XGBSurvivalModel(), SurvivalModel)

    def test_cox_repr_contains_name(self) -> None:
        m = CoxPHModel(alpha=0.5)
        assert "CoxPHModel" in repr(m)
        assert "0.5" in repr(m)

    def test_cox_get_params_keys(self) -> None:
        params = CoxPHModel().get_params()
        assert {"alpha", "ties", "n_iter", "tol"} == set(params.keys())

    def test_xgb_get_params_keys(self) -> None:
        params = XGBSurvivalModel().get_params()
        assert "n_estimators" in params
        assert "learning_rate" in params


# ---------------------------------------------------------------------------
# CoxPHModel
# ---------------------------------------------------------------------------


class TestCoxPHModel:
    def test_fit_returns_self(
        self, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, y = small_survival_dataset
        model = CoxPHModel()
        result = model.fit(X, y)
        assert result is model

    def test_predict_risk_score_shape(
        self, fitted_cox: CoxPHModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        scores = fitted_cox.predict_risk_score(X)
        assert scores.shape == (len(X),)

    def test_predict_survival_function_shape(
        self, fitted_cox: CoxPHModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        surv = fitted_cox.predict_survival_function(X)
        assert surv.shape[0] == len(X)
        assert surv.shape[1] > 0

    def test_survival_values_in_unit_interval(
        self, fitted_cox: CoxPHModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        surv = fitted_cox.predict_survival_function(X)
        assert (surv >= 0).all()
        assert (surv <= 1).all()

    def test_predict_median_survival_shape(
        self, fitted_cox: CoxPHModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        medians = fitted_cox.predict_median_survival_time(X)
        assert medians.shape == (len(X),)

    def test_predict_before_fit_raises(
        self, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        with pytest.raises(RuntimeError, match="fit"):
            CoxPHModel().predict_risk_score(X)

    def test_coef_shape(
        self, fitted_cox: CoxPHModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        assert fitted_cox.coef_.shape == (X.shape[1],)

    def test_predict_survival_with_custom_times(
        self, fitted_cox: CoxPHModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        times = np.array([30.0, 90.0, 180.0, 365.0])
        surv = fitted_cox.predict_survival_function(X, times=times)
        assert surv.shape == (len(X), 4)

    def test_survival_non_increasing_per_sample(
        self, fitted_cox: CoxPHModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        surv = fitted_cox.predict_survival_function(X)
        # Each row should be non-increasing over time
        diffs = np.diff(surv, axis=1)
        assert (diffs <= 1e-9).all(), "Survival function is not non-increasing"


# ---------------------------------------------------------------------------
# XGBSurvivalModel
# ---------------------------------------------------------------------------


class TestXGBSurvivalModel:
    def test_fit_returns_self(
        self, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, y = small_survival_dataset
        model = XGBSurvivalModel(n_estimators=10, early_stopping_rounds=None)
        result = model.fit(X, y)
        assert result is model

    def test_predict_risk_score_shape(
        self, fitted_xgb: XGBSurvivalModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        scores = fitted_xgb.predict_risk_score(X)
        assert scores.shape == (len(X),)

    def test_predict_survival_function_shape(
        self, fitted_xgb: XGBSurvivalModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        surv = fitted_xgb.predict_survival_function(X)
        assert surv.shape[0] == len(X)
        assert surv.shape[1] > 0

    def test_survival_values_in_unit_interval(
        self, fitted_xgb: XGBSurvivalModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        surv = fitted_xgb.predict_survival_function(X)
        assert (surv >= 0).all()
        assert (surv <= 1).all()

    def test_predict_median_survival_shape(
        self, fitted_xgb: XGBSurvivalModel, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        medians = fitted_xgb.predict_median_survival_time(X)
        assert medians.shape == (len(X),)

    def test_predict_before_fit_raises(
        self, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        with pytest.raises(RuntimeError, match="fit"):
            XGBSurvivalModel().predict_risk_score(X)

    def test_to_xgb_labels_event_positive(self) -> None:
        y = np.empty(3, dtype=SURVIVAL_DTYPE)
        y["event"] = [True, False, True]
        y["duration"] = [100.0, 200.0, 50.0]
        labels = _to_xgb_labels(y)
        assert labels[0] == pytest.approx(100.0)
        assert labels[1] == pytest.approx(-200.0)
        assert labels[2] == pytest.approx(50.0)
