"""Unit tests for the Data Stream Simulator.

Test categories
---------------
1. Schema / structural  — correct columns, dtypes, non-null primary keys.
2. Statistical          — output distributions stay within plausible bounds.
3. Survival labels      — ground-truth logic is internally consistent.
4. Noise injection      — faults are actually present after injection.
5. Reproducibility      — same seed → identical output.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chrono_ltv.data.simulator import (
    EcommerceSimulator,
    NoiseConfig,
    SimulatorConfig,
)

# ===========================================================================
# 1. Schema / Structural Tests
# ===========================================================================


class TestCustomerSchema:
    def test_required_columns(self, clean_datasets: dict) -> None:
        expected = {
            "customer_id",
            "first_name",
            "last_name",
            "email",
            "country",
            "acquisition_channel",
            "registration_date",
            "age",
            "gender",
            "loyalty_tier",
        }
        assert expected.issubset(clean_datasets["customers"].columns)

    def test_row_count(self, clean_datasets: dict, clean_sim_cfg: SimulatorConfig) -> None:
        assert len(clean_datasets["customers"]) == clean_sim_cfg.n_customers

    def test_customer_ids_are_unique(self, clean_datasets: dict) -> None:
        cids = clean_datasets["customers"]["customer_id"]
        assert cids.nunique() == len(cids), "Duplicate customer IDs detected"

    def test_customer_ids_are_valid_uuids(self, clean_datasets: dict) -> None:
        for cid in clean_datasets["customers"]["customer_id"].head(20):
            uuid.UUID(str(cid))  # raises ValueError if invalid

    def test_no_null_primary_keys(self, clean_datasets: dict) -> None:
        assert clean_datasets["customers"]["customer_id"].notna().all()


class TestTransactionSchema:
    def test_required_columns(self, clean_datasets: dict) -> None:
        expected = {
            "transaction_id",
            "customer_id",
            "event_timestamp",
            "order_value",
            "num_items",
            "product_category",
            "payment_method",
            "is_returned",
            "discount_applied",
        }
        assert expected.issubset(clean_datasets["transactions"].columns)

    def test_every_transaction_has_known_customer(self, clean_datasets: dict) -> None:
        known_ids = set(clean_datasets["customers"]["customer_id"])
        txn_ids = set(clean_datasets["transactions"]["customer_id"])
        assert txn_ids.issubset(known_ids)

    def test_at_least_one_transaction_per_customer(self, clean_datasets: dict) -> None:
        customers_with_txns = clean_datasets["transactions"]["customer_id"].nunique()
        n_customers = len(clean_datasets["customers"])
        # At minimum 95 % of customers should have at least one purchase
        assert customers_with_txns / n_customers >= 0.95


class TestClickstreamSchema:
    def test_required_columns(self, clean_datasets: dict) -> None:
        expected = {
            "session_id",
            "customer_id",
            "event_timestamp",
            "page_type",
            "time_on_page_seconds",
            "device_type",
            "added_to_cart",
        }
        assert expected.issubset(clean_datasets["clickstream"].columns)


class TestSupportTicketSchema:
    def test_required_columns(self, clean_datasets: dict) -> None:
        expected = {
            "ticket_id",
            "customer_id",
            "created_at",
            "topic",
            "raw_text",
            "sentiment_score",
            "resolved",
        }
        assert expected.issubset(clean_datasets["support_tickets"].columns)


class TestSurvivalLabelSchema:
    def test_required_columns(self, clean_datasets: dict) -> None:
        expected = {
            "customer_id",
            "first_purchase_date",
            "last_purchase_date",
            "duration_days",
            "event_observed",
            "total_orders",
            "total_revenue",
        }
        assert expected.issubset(clean_datasets["survival_labels"].columns)


# ===========================================================================
# 2. Statistical Distribution Tests
# ===========================================================================


class TestStatisticalProperties:
    def test_order_value_is_positive(self, clean_datasets: dict) -> None:
        assert (clean_datasets["transactions"]["order_value"] > 0).all()

    def test_order_value_median_in_range(self, clean_datasets: dict) -> None:
        # Log-normal(4.2, 0.8) → median ≈ exp(4.2) ≈ $66.7
        median = clean_datasets["transactions"]["order_value"].median()
        assert 20 < median < 200, f"Order value median out of expected range: {median:.2f}"

    def test_discount_between_zero_and_one(self, clean_datasets: dict) -> None:
        disc = clean_datasets["transactions"]["discount_applied"]
        assert (disc >= 0.0).all() and (disc <= 1.0).all()

    def test_age_range(self, clean_datasets: dict) -> None:
        ages = clean_datasets["customers"]["age"].dropna()
        assert (ages >= 18).all() and (ages <= 100).all()

    def test_sentiment_score_range(self, clean_datasets: dict) -> None:
        scores = clean_datasets["support_tickets"]["sentiment_score"]
        assert (scores >= -1.0).all() and (scores <= 1.0).all()

    def test_acquisition_channel_distribution(self, clean_datasets: dict) -> None:
        channels = clean_datasets["customers"]["acquisition_channel"].value_counts(normalize=True)
        # "organic_search" has weight 0.30 — expect roughly 20–40 %
        assert 0.15 < channels.get("organic_search", 0) < 0.50

    def test_churn_rate_is_plausible(self, clean_datasets: dict) -> None:
        churn_rate = clean_datasets["survival_labels"]["event_observed"].mean()
        # Sanity check only: churn logic is wired up (not 0 %) and not degenerate (not 100 %)
        assert 0.05 < churn_rate < 1.0, f"Implausible churn rate: {churn_rate:.1%}"

    def test_page_type_values_are_valid(self, clean_datasets: dict) -> None:
        valid = {
            "home",
            "category",
            "product",
            "cart",
            "checkout",
            "confirmation",
            "search",
            "account",
        }
        unique_pages = set(clean_datasets["clickstream"]["page_type"].unique())
        assert unique_pages.issubset(valid)


# ===========================================================================
# 3. Survival Label Logic Tests
# ===========================================================================


class TestSurvivalLabels:
    def test_duration_is_non_negative(self, clean_datasets: dict) -> None:
        assert (clean_datasets["survival_labels"]["duration_days"] >= 0).all()

    def test_first_before_last_purchase(self, clean_datasets: dict) -> None:
        labels = clean_datasets["survival_labels"]
        assert (
            pd.to_datetime(labels["first_purchase_date"])
            <= pd.to_datetime(labels["last_purchase_date"])
        ).all()

    def test_total_orders_positive(self, clean_datasets: dict) -> None:
        assert (clean_datasets["survival_labels"]["total_orders"] >= 1).all()

    def test_total_revenue_positive(self, clean_datasets: dict) -> None:
        assert (clean_datasets["survival_labels"]["total_revenue"] > 0).all()

    def test_labels_cover_all_customers(self, clean_datasets: dict) -> None:
        customers_with_txns = clean_datasets["transactions"]["customer_id"].nunique()
        n_labels = len(clean_datasets["survival_labels"])
        assert n_labels == customers_with_txns

    def test_churned_customers_have_correct_duration(self) -> None:
        """Churned duration must equal (last_purchase − first_purchase) + threshold."""
        cfg = SimulatorConfig(
            n_customers=50,
            start_date="2023-01-01",
            end_date="2023-12-31",
            churn_threshold_days=90,
            random_seed=99,
            noise=NoiseConfig(missing_rate=0, duplicate_rate=0),
        )
        sim = EcommerceSimulator(cfg)
        datasets = sim.run(persist=False)
        labels = datasets["survival_labels"]
        churned = labels[labels["event_observed"]].copy()

        if churned.empty:
            pytest.skip("No churned customers in this seed — increase n_customers.")

        first = pd.to_datetime(churned["first_purchase_date"])
        last = pd.to_datetime(churned["last_purchase_date"])
        expected_duration = (last - first).dt.days + cfg.churn_threshold_days
        actual_duration = churned["duration_days"].values

        np.testing.assert_array_almost_equal(actual_duration, expected_duration.values, decimal=0)


# ===========================================================================
# 4. Noise Injection Tests
# ===========================================================================


class TestNoiseInjection:
    def test_missing_values_exist_in_transactions(self, noisy_datasets: dict) -> None:
        assert noisy_datasets["transactions"].isna().any().any()

    def test_duplicate_rows_exist(
        self, noisy_datasets: dict, noisy_sim_cfg: SimulatorConfig
    ) -> None:
        txns = noisy_datasets["transactions"]
        # With 5 % duplicate rate on IDs (plus added rows), total should exceed n_unique
        assert len(txns) > txns["transaction_id"].nunique()

    def test_negative_amounts_injected(self, noisy_datasets: dict) -> None:
        amounts = noisy_datasets["transactions"]["order_value"].dropna()
        assert (amounts < 0).any(), "No negative amounts injected"

    def test_fault_report_keys(self, noisy_sim_cfg: SimulatorConfig) -> None:
        sim = EcommerceSimulator(noisy_sim_cfg)
        sim.run(persist=False)
        report = sim.fault_report()
        assert "transactions" in report
        assert "missing" in report["transactions"]

    def test_no_noise_leaves_clean_data(self, clean_datasets: dict) -> None:
        assert not clean_datasets["transactions"].isna().any().any()
        txns = clean_datasets["transactions"]
        assert len(txns) == txns["transaction_id"].nunique()


# ===========================================================================
# 5. Reproducibility Tests
# ===========================================================================


class TestReproducibility:
    def _run_sim(self, seed: int) -> dict:
        cfg = SimulatorConfig(
            n_customers=100,
            start_date="2023-01-01",
            end_date="2023-12-31",
            random_seed=seed,
            noise=NoiseConfig(missing_rate=0, duplicate_rate=0),
        )
        return EcommerceSimulator(cfg).run(persist=False)

    def test_same_seed_produces_identical_customers(self) -> None:
        d1 = self._run_sim(42)
        d2 = self._run_sim(42)
        pd.testing.assert_frame_equal(d1["customers"], d2["customers"])

    def test_same_seed_produces_identical_transactions(self) -> None:
        d1 = self._run_sim(42)
        d2 = self._run_sim(42)
        pd.testing.assert_frame_equal(d1["transactions"], d2["transactions"])

    def test_different_seeds_produce_different_data(self) -> None:
        d1 = self._run_sim(10)
        d2 = self._run_sim(99)
        # Customer emails should differ
        assert not d1["customers"]["email"].equals(d2["customers"]["email"])


# ===========================================================================
# 6. Persistence Tests
# ===========================================================================


class TestPersistence:
    def test_persist_writes_all_parquet_files(self, tmp_path: Path) -> None:
        cfg = SimulatorConfig(
            n_customers=30,
            start_date="2023-01-01",
            end_date="2023-06-30",
            random_seed=7,
            noise=NoiseConfig(missing_rate=0, duplicate_rate=0),
            output_dir=tmp_path,
        )
        EcommerceSimulator(cfg).run(persist=True)

        expected_files = {
            "customers.parquet",
            "transactions.parquet",
            "clickstream.parquet",
            "support_tickets.parquet",
            "survival_labels.parquet",
        }
        written = {p.name for p in tmp_path.iterdir()}
        assert expected_files == written

    def test_persisted_customers_roundtrip(self, tmp_path: Path) -> None:
        cfg = SimulatorConfig(
            n_customers=30,
            start_date="2023-01-01",
            end_date="2023-06-30",
            random_seed=7,
            noise=NoiseConfig(missing_rate=0, duplicate_rate=0),
            output_dir=tmp_path,
        )
        datasets = EcommerceSimulator(cfg).run(persist=True)
        loaded = pd.read_parquet(tmp_path / "customers.parquet")
        assert len(loaded) == len(datasets["customers"])
