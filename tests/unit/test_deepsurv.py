"""Unit tests for DeepSurvModel — skipped unless torch is installed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")

from chrono_ltv.features.pipeline import SURVIVAL_DTYPE  # noqa: E402
from chrono_ltv.models.base import SurvivalModel  # noqa: E402
from chrono_ltv.models.deepsurv import DeepSurvModel, _build_network  # noqa: E402


@pytest.fixture(scope="module")
def small_survival_dataset() -> tuple[pd.DataFrame, np.ndarray]:
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


class TestDeepSurvModel:
    def test_build_network_output_shape(self) -> None:
        import torch as t

        net = _build_network(
            n_features=10,
            hidden_dims=[32, 16],
            dropout=0.0,
            batch_norm=False,
            activation="relu",
        )
        x = t.randn(5, 10)
        out = net(x)
        assert out.shape == (5, 1)

    def test_fit_returns_self(
        self, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, y = small_survival_dataset
        model = DeepSurvModel(
            hidden_dims=[16, 8], max_epochs=3, batch_size=16, dropout=0.0, batch_norm=False
        )
        result = model.fit(X, y)
        assert result is model

    def test_predict_risk_score_shape(
        self, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, y = small_survival_dataset
        model = DeepSurvModel(
            hidden_dims=[16, 8], max_epochs=3, batch_size=16, dropout=0.0, batch_norm=False
        )
        model.fit(X, y)
        scores = model.predict_risk_score(X)
        assert scores.shape == (len(X),)

    def test_predict_survival_function_shape(
        self, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, y = small_survival_dataset
        model = DeepSurvModel(
            hidden_dims=[16, 8], max_epochs=3, batch_size=16, dropout=0.0, batch_norm=False
        )
        model.fit(X, y)
        surv = model.predict_survival_function(X)
        assert surv.shape[0] == len(X)

    def test_predict_before_fit_raises(
        self, small_survival_dataset: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        X, _ = small_survival_dataset
        with pytest.raises(RuntimeError, match="fit"):
            DeepSurvModel().predict_risk_score(X)

    def test_get_params_complete(self) -> None:
        params = DeepSurvModel().get_params()
        assert "hidden_dims" in params
        assert "max_epochs" in params
        assert "learning_rate" in params

    def test_deepsurv_is_survival_model(self) -> None:
        assert isinstance(DeepSurvModel(), SurvivalModel)
