"""Directional tests: model predictions must move in the expected direction.

These tests encode domain knowledge about the ChronoLTV problem:
- Customers who churned (event=True) are higher-risk than those still active.
- Risk scores should rank customers by their actual survival time.
- Features with known churn-driving semantics should move risk in the right direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.stats import spearmanr

sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")

from chrono_ltv.models.cox_ph import CoxPHModel  # noqa: E402

_FeatMatrix = tuple[pd.DataFrame, npt.NDArray[Any]]


class TestSurvivalOutcomeDirection:
    def test_churned_have_higher_mean_risk_than_censored(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        """Customers who actually churned must rank higher-risk on average."""
        X, y = behavioral_feature_matrix
        scores = fitted_cox.predict_risk_score(X)
        churned = y["event"].astype(bool)

        if churned.sum() < 5 or (~churned).sum() < 5:
            pytest.skip("Too few events or censored customers")

        mean_churned = float(np.mean(scores[churned]))
        mean_censored = float(np.mean(scores[~churned]))
        assert mean_churned > mean_censored, (
            f"Churned mean risk {mean_churned:.3f} must exceed "
            f"censored mean risk {mean_censored:.3f}"
        )

    def test_risk_scores_rank_survival_duration(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        """Among churned customers, higher risk must correlate with shorter survival."""
        X, y = behavioral_feature_matrix
        scores = fitted_cox.predict_risk_score(X)
        churned = y["event"].astype(bool)

        if churned.sum() < 20:
            pytest.skip("Too few events for rank correlation test")

        corr, _ = spearmanr(scores[churned], y["duration"][churned].astype(float))
        assert corr < 0, (
            f"Expected negative Spearman correlation (higher risk → shorter duration), "
            f"got {corr:.3f}"
        )

    def test_high_risk_customers_shorter_predicted_median(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        """Top-quartile risk customers must have shorter median survival than bottom quartile."""
        X, _ = behavioral_feature_matrix
        scores = fitted_cox.predict_risk_score(X)
        medians = fitted_cox.predict_median_survival_time(X)

        finite = np.isfinite(medians)
        if finite.sum() < 20:
            pytest.skip("Too few finite median survival times")

        q25, q75 = np.percentile(scores[finite], [25, 75])
        low_risk_median = float(np.mean(medians[finite & (scores <= q25)]))
        high_risk_median = float(np.mean(medians[finite & (scores >= q75)]))

        assert high_risk_median < low_risk_median, (
            f"High-risk median {high_risk_median:.1f}d must be shorter than "
            f"low-risk median {low_risk_median:.1f}d"
        )


class TestFeatureDirectionality:
    def test_higher_recency_days_quartile_has_higher_risk(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        """More days since last order (higher recency_days) → higher churn risk."""
        X, _ = behavioral_feature_matrix
        col = "recency_days"
        if col not in X.columns:
            pytest.skip(f"{col!r} not in feature matrix")

        scores = fitted_cox.predict_risk_score(X)
        recency = X[col].values

        q25, q75 = np.percentile(recency, [25, 75])
        recent_risk = float(np.mean(scores[recency <= q25]))
        old_risk = float(np.mean(scores[recency >= q75]))

        assert old_risk > recent_risk, (
            f"Old-buyer risk {old_risk:.3f} must exceed recent-buyer risk {recent_risk:.3f}"
        )

    def test_higher_frequency_quartile_has_lower_risk(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        """More orders (higher frequency) → lower churn risk."""
        X, _ = behavioral_feature_matrix
        col = "frequency"
        if col not in X.columns:
            pytest.skip(f"{col!r} not in feature matrix")

        scores = fitted_cox.predict_risk_score(X)
        freq = X[col].values

        q25, q75 = np.percentile(freq, [25, 75])
        low_freq_risk = float(np.mean(scores[freq <= q25]))
        high_freq_risk = float(np.mean(scores[freq >= q75]))

        assert high_freq_risk < low_freq_risk, (
            f"High-frequency risk {high_freq_risk:.3f} must be lower than "
            f"low-frequency risk {low_freq_risk:.3f}"
        )

    def test_higher_monetary_quartile_has_lower_risk(
        self, fitted_cox: CoxPHModel, behavioral_feature_matrix: _FeatMatrix
    ) -> None:
        """Higher-spending customers must have lower churn risk."""
        X, _ = behavioral_feature_matrix
        col = "monetary_total"
        if col not in X.columns:
            pytest.skip(f"{col!r} not in feature matrix")

        scores = fitted_cox.predict_risk_score(X)
        monetary = X[col].values

        q25, q75 = np.percentile(monetary, [25, 75])
        low_spend_risk = float(np.mean(scores[monetary <= q25]))
        high_spend_risk = float(np.mean(scores[monetary >= q75]))

        assert high_spend_risk < low_spend_risk, (
            f"High-spend risk {high_spend_risk:.3f} must be lower than "
            f"low-spend risk {low_spend_risk:.3f}"
        )
