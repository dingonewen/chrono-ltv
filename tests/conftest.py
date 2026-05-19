"""Shared Pytest fixtures for all test suites."""

from __future__ import annotations

import pytest

from chrono_ltv.data.simulator import EcommerceSimulator, NoiseConfig, SimulatorConfig


@pytest.fixture(scope="session")
def clean_sim_cfg() -> SimulatorConfig:
    """A small, noise-free simulation config for fast unit tests."""
    return SimulatorConfig(
        n_customers=200,
        start_date="2023-01-01",
        end_date="2023-12-31",
        random_seed=0,
        noise=NoiseConfig(
            missing_rate=0.0,
            duplicate_rate=0.0,
            outlier_rate=0.0,
            future_date_rate=0.0,
            negative_amount_rate=0.0,
        ),
    )


@pytest.fixture(scope="session")
def noisy_sim_cfg() -> SimulatorConfig:
    """A small simulation config with full noise injection."""
    return SimulatorConfig(
        n_customers=200,
        start_date="2023-01-01",
        end_date="2023-12-31",
        random_seed=1,
        noise=NoiseConfig(
            missing_rate=0.10,
            duplicate_rate=0.05,
            outlier_rate=0.05,
            future_date_rate=0.02,
            negative_amount_rate=0.02,
        ),
    )


@pytest.fixture(scope="session")
def clean_datasets(clean_sim_cfg: SimulatorConfig) -> dict:
    """Run the full simulator once (no I/O) and cache for the session."""
    sim = EcommerceSimulator(clean_sim_cfg)
    return sim.run(persist=False)


@pytest.fixture(scope="session")
def noisy_datasets(noisy_sim_cfg: SimulatorConfig) -> dict:
    sim = EcommerceSimulator(noisy_sim_cfg)
    return sim.run(persist=False)
