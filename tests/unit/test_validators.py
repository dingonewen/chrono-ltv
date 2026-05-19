"""Unit tests for DataValidator and ValidationSummary."""

from __future__ import annotations

import pytest

pytest.importorskip("great_expectations", reason="great-expectations not installed")

import pandas as pd  # noqa: E402

from chrono_ltv.data.validators import (  # noqa: E402
    DataValidator,
    ValidationSummary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _customers_df(**overrides: object) -> pd.DataFrame:
    base: dict[str, list[object]] = {
        "customer_id": ["00000000-0000-4000-8000-000000000001"],
        "first_name": ["Alice"],
        "last_name": ["Smith"],
        "email": ["alice@example.com"],
        "country": ["US"],
        "acquisition_channel": ["organic_search"],
        "registration_date": ["2023-01-01"],
        "age": [30],
        "gender": ["F"],
        "loyalty_tier": ["Gold"],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def _transactions_df(**overrides: object) -> pd.DataFrame:
    base: dict[str, list[object]] = {
        "transaction_id": ["00000000-0000-4000-8000-000000000002"],
        "customer_id": ["00000000-0000-4000-8000-000000000001"],
        "event_timestamp": ["2023-03-15"],
        "order_value": [49.99],
        "num_items": [2],
        "product_category": ["Electronics"],
        "payment_method": ["credit_card"],
        "is_returned": [False],
        "discount_applied": [0.1],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def _survival_df(**overrides: object) -> pd.DataFrame:
    base: dict[str, list[object]] = {
        "customer_id": ["00000000-0000-4000-8000-000000000001"],
        "first_purchase_date": ["2023-01-01"],
        "last_purchase_date": ["2023-04-01"],
        "duration_days": [180.0],
        "event_observed": [True],
        "total_orders": [5],
        "total_revenue": [249.95],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# 1. ValidationSummary unit tests
# ---------------------------------------------------------------------------


class TestValidationSummary:
    def test_pass_rate_all_pass(self) -> None:
        s = ValidationSummary("x", True, 5, 5, 0)
        assert s.pass_rate == 1.0

    def test_pass_rate_partial(self) -> None:
        s = ValidationSummary("x", False, 4, 3, 1)
        assert s.pass_rate == pytest.approx(0.75)

    def test_pass_rate_zero_expectations(self) -> None:
        s = ValidationSummary("x", True, 0, 0, 0)
        assert s.pass_rate == 1.0


# ---------------------------------------------------------------------------
# 2. validate_all
# ---------------------------------------------------------------------------


class TestValidateAll:
    def test_returns_all_five_keys(self, clean_datasets: dict) -> None:
        summaries = DataValidator().validate_all(clean_datasets)
        assert set(summaries.keys()) == {
            "customers",
            "transactions",
            "clickstream",
            "support_tickets",
            "survival_labels",
        }

    def test_all_clean_data_passes(self, clean_datasets: dict) -> None:
        summaries = DataValidator().validate_all(clean_datasets)
        failures = {k: v.failures for k, v in summaries.items() if not v.success}
        assert not failures, f"Unexpected failures: {failures}"

    def test_missing_dataset_is_skipped(self, clean_datasets: dict) -> None:
        partial = {k: v for k, v in clean_datasets.items() if k != "clickstream"}
        summaries = DataValidator().validate_all(partial)
        assert "clickstream" not in summaries
        assert len(summaries) == 4


# ---------------------------------------------------------------------------
# 3. Per-dataset: clean data passes
# ---------------------------------------------------------------------------


class TestValidateClean:
    def test_customers(self, clean_datasets: dict) -> None:
        s = DataValidator().validate_customers(clean_datasets["customers"])
        assert s.success, s.failures

    def test_transactions(self, clean_datasets: dict) -> None:
        s = DataValidator().validate_transactions(clean_datasets["transactions"])
        assert s.success, s.failures

    def test_clickstream(self, clean_datasets: dict) -> None:
        s = DataValidator().validate_clickstream(clean_datasets["clickstream"])
        assert s.success, s.failures

    def test_support_tickets(self, clean_datasets: dict) -> None:
        s = DataValidator().validate_support_tickets(clean_datasets["support_tickets"])
        assert s.success, s.failures

    def test_survival_labels(self, clean_datasets: dict) -> None:
        s = DataValidator().validate_survival_labels(clean_datasets["survival_labels"])
        assert s.success, s.failures


# ---------------------------------------------------------------------------
# 4. Violation detection: deliberately bad DataFrames
# ---------------------------------------------------------------------------


class TestValidateViolations:
    def test_customers_catches_invalid_channel(self) -> None:
        df = _customers_df(acquisition_channel="billboard")
        s = DataValidator().validate_customers(df)
        assert not s.success
        assert s.n_failed >= 1

    def test_customers_catches_invalid_gender(self) -> None:
        df = _customers_df(gender="X")
        s = DataValidator().validate_customers(df)
        assert not s.success

    def test_transactions_catches_all_negative_order_values(self) -> None:
        # 100 % negative values — well below the 95 % mostly threshold
        df = pd.DataFrame(
            {
                "transaction_id": [f"id-{i}" for i in range(20)],
                "customer_id": ["c1"] * 20,
                "event_timestamp": ["2023-01-01"] * 20,
                "order_value": [-50.0] * 20,
                "num_items": [1] * 20,
                "product_category": ["Books"] * 20,
                "payment_method": ["credit_card"] * 20,
                "is_returned": [False] * 20,
                "discount_applied": [0.0] * 20,
            }
        )
        s = DataValidator().validate_transactions(df)
        assert not s.success

    def test_survival_labels_catches_negative_duration(self) -> None:
        df = _survival_df(duration_days=-10.0)
        s = DataValidator().validate_survival_labels(df)
        assert not s.success

    def test_survival_labels_catches_duplicate_customer_id(self) -> None:
        df = pd.concat([_survival_df(), _survival_df()], ignore_index=True)
        s = DataValidator().validate_survival_labels(df)
        assert not s.success

    def test_summary_structure_is_consistent(self) -> None:
        df = _customers_df()
        s = DataValidator().validate_customers(df)
        assert s.n_passed + s.n_failed == s.n_expectations
        assert isinstance(s.failures, list)
