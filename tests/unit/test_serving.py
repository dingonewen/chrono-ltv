"""Unit tests for the FastAPI serving layer and ModelPredictor."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
httpx = pytest.importorskip("httpx", reason="httpx not installed")
sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")

from fastapi.testclient import TestClient  # noqa: E402

from chrono_ltv.features.pipeline import SURVIVAL_DTYPE  # noqa: E402
from chrono_ltv.models.cox_ph import CoxPHModel  # noqa: E402
from chrono_ltv.serving.api import create_app  # noqa: E402
from chrono_ltv.serving.predictor import ModelPredictor, PredictionOutput  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TIME_POINTS = [30, 90, 180]
_FEATURES = {f"f{i}": float(i) for i in range(5)}


@pytest.fixture(scope="module")
def fitted_cox() -> CoxPHModel:
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(5)})
    y = np.empty(n, dtype=SURVIVAL_DTYPE)
    y["event"] = rng.random(n) > 0.3
    y["duration"] = rng.exponential(scale=200, size=n)
    return CoxPHModel(alpha=1.0).fit(X, y)


@pytest.fixture(scope="module")
def loaded_predictor(fitted_cox: CoxPHModel) -> ModelPredictor:
    pred = ModelPredictor(model_uri="models:/mock/1", time_points=_TIME_POINTS)
    pred._model = fitted_cox  # bypass MLflow load
    return pred


class _UnloadedPredictor:
    """Stub predictor whose is_loaded is False."""

    model_uri = "models:/mock/1"
    model_version = "models:/mock/1"
    is_loaded = False


class _ErrorPredictor:
    """Stub predictor that raises on predict()."""

    model_uri = "models:/mock/1"
    model_version = "models:/mock/1"
    is_loaded = True

    def predict(self, features: dict[str, float]) -> PredictionOutput:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# ModelPredictor — unit tests (no MLflow needed)
# ---------------------------------------------------------------------------


class TestModelPredictor:
    def test_not_loaded_initially(self) -> None:
        pred = ModelPredictor(model_uri="models:/mock/1")
        assert pred.is_loaded is False

    def test_predict_raises_when_not_loaded(self) -> None:
        pred = ModelPredictor(model_uri="models:/mock/1")
        with pytest.raises(RuntimeError, match="not loaded"):
            pred.predict(_FEATURES)

    def test_predict_returns_prediction_output(self, loaded_predictor: ModelPredictor) -> None:
        result = loaded_predictor.predict(_FEATURES)
        assert isinstance(result, PredictionOutput)

    def test_predict_risk_score_is_finite(self, loaded_predictor: ModelPredictor) -> None:
        result = loaded_predictor.predict(_FEATURES)
        assert np.isfinite(result.risk_score)

    def test_predict_median_is_positive(self, loaded_predictor: ModelPredictor) -> None:
        result = loaded_predictor.predict(_FEATURES)
        assert result.median_survival_days > 0.0

    def test_predict_survival_times_match_time_points(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        result = loaded_predictor.predict(_FEATURES)
        assert result.survival_times == _TIME_POINTS

    def test_predict_probs_length_matches_time_points(
        self, loaded_predictor: ModelPredictor
    ) -> None:
        result = loaded_predictor.predict(_FEATURES)
        assert len(result.survival_probs) == len(_TIME_POINTS)

    def test_predict_probs_in_unit_interval(self, loaded_predictor: ModelPredictor) -> None:
        result = loaded_predictor.predict(_FEATURES)
        for p in result.survival_probs:
            assert 0.0 <= p <= 1.0

    def test_load_calls_mlflow(self, fitted_cox: CoxPHModel) -> None:
        mock_mlflow_sklearn = MagicMock()
        mock_mlflow_sklearn.load_model.return_value = fitted_cox

        pred = ModelPredictor(model_uri="models:/mock/1")
        # Inject mock into sys.modules so the local import inside load() resolves
        import sys

        fake_mlflow = MagicMock()
        fake_mlflow.sklearn = mock_mlflow_sklearn
        original = sys.modules.get("mlflow")
        original_sklearn = sys.modules.get("mlflow.sklearn")
        try:
            sys.modules["mlflow"] = fake_mlflow
            sys.modules["mlflow.sklearn"] = mock_mlflow_sklearn
            pred.load()
        finally:
            if original is None:
                sys.modules.pop("mlflow", None)
            else:
                sys.modules["mlflow"] = original
            if original_sklearn is None:
                sys.modules.pop("mlflow.sklearn", None)
            else:
                sys.modules["mlflow.sklearn"] = original_sklearn

        mock_mlflow_sklearn.load_model.assert_called_once_with("models:/mock/1")
        assert pred.is_loaded


# ---------------------------------------------------------------------------
# FastAPI app — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_ok(loaded_predictor: ModelPredictor) -> Any:
    test_app = create_app(model_uri="models:/mock/1", time_points=_TIME_POINTS)
    test_app.state.predictor = loaded_predictor
    with TestClient(test_app) as c:
        yield c


@pytest.fixture()
def client_degraded() -> Any:
    test_app = create_app(model_uri="models:/mock/1", time_points=_TIME_POINTS)
    test_app.state.predictor = _UnloadedPredictor()
    with TestClient(test_app) as c:
        yield c


@pytest.fixture()
def client_no_predictor() -> Any:
    test_app = create_app(model_uri="models:/mock/1", time_points=_TIME_POINTS)
    # Predictor deliberately not set — lifespan will try to load and fail
    # gracefully, leaving an unloaded predictor.
    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, client_ok: Any) -> None:
        assert client_ok.get("/health").status_code == 200

    def test_status_ok_when_loaded(self, client_ok: Any) -> None:
        assert client_ok.get("/health").json()["status"] == "ok"

    def test_model_loaded_true(self, client_ok: Any) -> None:
        assert client_ok.get("/health").json()["model_loaded"] is True

    def test_status_degraded_when_not_loaded(self, client_degraded: Any) -> None:
        resp = client_degraded.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"

    def test_model_loaded_false_when_degraded(self, client_degraded: Any) -> None:
        assert client_degraded.get("/health").json()["model_loaded"] is False

    def test_model_uri_present(self, client_ok: Any) -> None:
        assert "model_uri" in client_ok.get("/health").json()


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_returns_200(self, client_ok: Any) -> None:
        assert client_ok.get("/metrics").status_code == 200

    def test_counts_are_non_negative(self, client_ok: Any) -> None:
        data = client_ok.get("/metrics").json()
        assert data["requests_total"] >= 0
        assert data["requests_failed"] >= 0

    def test_avg_latency_non_negative(self, client_ok: Any) -> None:
        assert client_ok.get("/metrics").json()["avg_latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    _payload = {"customer_id": "cust_001", "features": _FEATURES}

    def test_returns_200(self, client_ok: Any) -> None:
        assert client_ok.post("/predict", json=self._payload).status_code == 200

    def test_response_has_customer_id(self, client_ok: Any) -> None:
        data = client_ok.post("/predict", json=self._payload).json()
        assert data["customer_id"] == "cust_001"

    def test_risk_score_is_float(self, client_ok: Any) -> None:
        data = client_ok.post("/predict", json=self._payload).json()
        assert isinstance(data["risk_score"], float)

    def test_median_survival_is_positive(self, client_ok: Any) -> None:
        data = client_ok.post("/predict", json=self._payload).json()
        assert data["median_survival_days"] > 0.0

    def test_survival_curve_times_match_time_points(self, client_ok: Any) -> None:
        data = client_ok.post("/predict", json=self._payload).json()
        assert data["survival_curve"]["time_days"] == _TIME_POINTS

    def test_survival_probs_length(self, client_ok: Any) -> None:
        data = client_ok.post("/predict", json=self._payload).json()
        assert len(data["survival_curve"]["probabilities"]) == len(_TIME_POINTS)

    def test_survival_probs_in_unit_interval(self, client_ok: Any) -> None:
        data = client_ok.post("/predict", json=self._payload).json()
        for p in data["survival_curve"]["probabilities"]:
            assert 0.0 <= p <= 1.0

    def test_model_version_present(self, client_ok: Any) -> None:
        data = client_ok.post("/predict", json=self._payload).json()
        assert "model_version" in data

    def test_503_when_model_not_loaded(self, client_degraded: Any) -> None:
        resp = client_degraded.post("/predict", json=self._payload)
        assert resp.status_code == 503

    def test_500_on_model_error(self, loaded_predictor: ModelPredictor) -> None:
        test_app = create_app(model_uri="models:/mock/1", time_points=_TIME_POINTS)
        test_app.state.predictor = _ErrorPredictor()
        with TestClient(test_app, raise_server_exceptions=False) as c:
            resp = c.post("/predict", json=self._payload)
        assert resp.status_code == 500

    def test_failed_request_increments_failed_count(self, loaded_predictor: ModelPredictor) -> None:
        test_app = create_app(model_uri="models:/mock/1", time_points=_TIME_POINTS)
        test_app.state.predictor = _ErrorPredictor()
        with TestClient(test_app, raise_server_exceptions=False) as c:
            c.post("/predict", json=self._payload)
            data = c.get("/metrics").json()
        assert data["requests_failed"] >= 1

    def test_successful_request_increments_total_count(self, client_ok: Any) -> None:
        before = client_ok.get("/metrics").json()["requests_total"]
        client_ok.post("/predict", json=self._payload)
        after = client_ok.get("/metrics").json()["requests_total"]
        assert after == before + 1
