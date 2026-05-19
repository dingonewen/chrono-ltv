"""Minimum-functionality (sanity) tests for SurvivalModel predictions.

These tests assert basic mathematical contracts that every survival model
must satisfy regardless of the data it was trained on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")

from sksurv.metrics import concordance_index_censored  # noqa: E402

from chrono_ltv.models.cox_ph import CoxPHModel  # noqa: E402

_FeatMatrix = tuple[pd.DataFrame, npt.NDArray[Any]]
_TIME_POINTS = np.array([30.0, 90.0, 180.0, 365.0])


class TestRiskScores:
    def test_shape_matches_input(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, y = behavioral_feature_matrix
        scores = fitted_cox.predict_risk_score(X)
        assert scores.shape == (len(X),)

    def test_all_finite(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        scores = fitted_cox.predict_risk_score(X)
        assert np.all(np.isfinite(scores))

    def test_not_all_identical(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        scores = fitted_cox.predict_risk_score(X)
        assert np.std(scores) > 0.0, "Risk scores must vary across customers"

    def test_concordance_above_chance(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, y = behavioral_feature_matrix
        scores = fitted_cox.predict_risk_score(X)
        c_idx, *_ = concordance_index_censored(
            y["event"].astype(bool), y["duration"].astype(float), scores
        )
        assert c_idx > 0.55, f"C-index {c_idx:.3f} is not better than chance"


class TestSurvivalFunction:
    def test_shape(self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix) -> None:
        X, _ = behavioral_feature_matrix
        S = fitted_cox.predict_survival_function(X, times=_TIME_POINTS)
        assert S.shape == (len(X), len(_TIME_POINTS))

    def test_values_in_unit_interval(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        S = fitted_cox.predict_survival_function(X, times=_TIME_POINTS)
        assert np.all(S >= 0.0) and np.all(S <= 1.0)

    def test_monotone_nonincreasing(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        S = fitted_cox.predict_survival_function(X, times=_TIME_POINTS)
        diffs = np.diff(S, axis=1)
        # Allow tiny floating-point violations
        assert np.all(diffs <= 1e-9), "Survival function must be non-increasing over time"

    def test_at_time_zero_near_one(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        very_early = np.array([1.0])
        S = fitted_cox.predict_survival_function(X, times=very_early)
        assert float(np.mean(S[:, 0])) > 0.5


class TestMedianSurvival:
    def test_shape_matches_input(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        medians = fitted_cox.predict_median_survival_time(X)
        assert medians.shape == (len(X),)

    def test_finite_values_all_positive(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        X, _ = behavioral_feature_matrix
        medians = fitted_cox.predict_median_survival_time(X)
        finite = medians[np.isfinite(medians)]
        assert np.all(finite > 0.0)

    def test_risk_ranking_consistent_with_median(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        """Higher risk score must correspond to shorter median survival."""
        from scipy.stats import spearmanr

        X, _ = behavioral_feature_matrix
        scores = fitted_cox.predict_risk_score(X)
        medians = fitted_cox.predict_median_survival_time(X)
        finite = np.isfinite(medians)
        if finite.sum() < 20:
            pytest.skip("Too few finite median survival times")

        corr, _ = spearmanr(scores[finite], medians[finite])
        assert corr < 0, "Higher risk score must rank with shorter median survival"
