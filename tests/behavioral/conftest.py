"""Session-scoped fixtures for the behavioral test suite.

Uses a larger, dedicated dataset (500 customers, 2 years) so the fitted
model has enough signal for directional and invariance assertions.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")
sksurv = pytest.importorskip("sksurv", reason="scikit-survival not installed")

from chrono_ltv.data.simulator import EcommerceSimulator, NoiseConfig, SimulatorConfig  # noqa: E402
from chrono_ltv.features.pipeline import FeaturePipeline  # noqa: E402
from chrono_ltv.models.cox_ph import CoxPHModel  # noqa: E402

_FeatMatrix = tuple[pd.DataFrame, npt.NDArray[Any]]


@pytest.fixture(scope="session")
def behavioral_datasets() -> dict[str, Any]:
    """500 clean customers over two years — enough signal for Cox to learn."""
    cfg = SimulatorConfig(
        n_customers=500,
        start_date="2022-01-01",
        end_date="2023-12-31",
        random_seed=7,
        noise=NoiseConfig(
            missing_rate=0.0,
            duplicate_rate=0.0,
            outlier_rate=0.0,
            future_date_rate=0.0,
            negative_amount_rate=0.0,
        ),
    )
    return EcommerceSimulator(cfg).run(persist=False)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def behavioral_feature_matrix(behavioral_datasets: dict[str, Any]) -> _FeatMatrix:
    """Full feature matrix (X: DataFrame, y: structured array) from the pipeline."""
    d = behavioral_datasets
    pipe = FeaturePipeline(scale=True)
    return pipe.fit_transform(  # type: ignore[no-any-return]
        customers=d["customers"],
        transactions=d["transactions"],
        clickstream=d["clickstream"],
        tickets=d["support_tickets"],
        labels=d["survival_labels"],
    )


@pytest.fixture(scope="session")
def fitted_cox(behavioral_feature_matrix: _FeatMatrix) -> CoxPHModel:
    """CoxPH model fitted on the full behavioral feature matrix."""
    X, y = behavioral_feature_matrix
    return CoxPHModel(alpha=0.5).fit(X, y)
