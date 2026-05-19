"""Invariance tests: predictions must not change under semantically-neutral transformations."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")

from chrono_ltv.models.cox_ph import CoxPHModel  # noqa: E402

_FeatMatrix = tuple[pd.DataFrame, npt.NDArray[Any]]
_TIME_POINTS = np.array([90.0, 180.0, 365.0])


class TestRowOrderInvariance:
    def test_shuffled_rows_same_risk_scores(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))

        scores_orig = fitted_cox.predict_risk_score(X)
        scores_shuffled = fitted_cox.predict_risk_score(X.iloc[idx])

        np.testing.assert_allclose(scores_orig[idx], scores_shuffled, rtol=1e-6)

    def test_shuffled_rows_same_survival_function(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        rng = np.random.default_rng(1)
        idx = rng.permutation(len(X))

        S_orig = fitted_cox.predict_survival_function(X, times=_TIME_POINTS)
        S_shuffled = fitted_cox.predict_survival_function(X.iloc[idx], times=_TIME_POINTS)

        np.testing.assert_allclose(S_orig[idx], S_shuffled, rtol=1e-6)


class TestIdenticalRowInvariance:
    def test_identical_rows_same_risk_score(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        row = X.iloc[[0]]
        duped = pd.concat([row, row, row], ignore_index=True)
        scores = fitted_cox.predict_risk_score(duped)
        assert np.allclose(scores[0], scores[1]) and np.allclose(scores[0], scores[2])

    def test_identical_rows_same_survival_curve(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        row = X.iloc[[0]]
        duped = pd.concat([row, row], ignore_index=True)
        S = fitted_cox.predict_survival_function(duped, times=_TIME_POINTS)
        np.testing.assert_allclose(S[0], S[1], rtol=1e-8)

    def test_identical_rows_same_median(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        row = X.iloc[[0]]
        duped = pd.concat([row, row], ignore_index=True)
        medians = fitted_cox.predict_median_survival_time(duped)
        assert medians[0] == medians[1]


class TestDeterminism:
    def test_predictions_deterministic_on_rerun(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        scores_1 = fitted_cox.predict_risk_score(X)
        scores_2 = fitted_cox.predict_risk_score(X)
        np.testing.assert_array_equal(scores_1, scores_2)

    def test_single_row_consistent_with_batch(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        """Score for one row in isolation must equal its score in a batch."""
        X, _ = behavioral_feature_matrix
        batch_scores = fitted_cox.predict_risk_score(X)
        for i in [0, 1, 5]:
            single_score = fitted_cox.predict_risk_score(X.iloc[[i]])[0]
            assert np.isclose(single_score, batch_scores[i], rtol=1e-6)
