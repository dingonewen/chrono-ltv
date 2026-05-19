"""Unit tests for feature extractors and FeaturePipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from chrono_ltv.features.encoders import (
    BehavioralFeatureExtractor,
    CustomerFeatureExtractor,
    RFMFeatureExtractor,
    TicketFeatureExtractor,
)

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

from chrono_ltv.features.pipeline import SURVIVAL_DTYPE, FeaturePipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Shared minimal fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_customer_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "customer_id": ["c1", "c1", "c2", "c2"],
            "event_timestamp": pd.to_datetime(
                ["2023-01-10", "2023-03-01", "2023-02-15", "2023-04-20"]
            ),
            "order_value": [50.0, 80.0, 30.0, 120.0],
            "num_items": [2, 3, 1, 4],
            "product_category": ["Books", "Electronics", "Books", "Books"],
            "payment_method": ["credit_card", "paypal", "credit_card", "credit_card"],
            "is_returned": [False, False, True, False],
            "discount_applied": [0.0, 0.1, 0.0, 0.2],
        }
    )


@pytest.fixture()
def two_customer_clickstream() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s3", "s3"],
            "customer_id": ["c1", "c1", "c1", "c2", "c2"],
            "event_timestamp": pd.to_datetime(
                ["2023-01-10"] * 2 + ["2023-02-01"] + ["2023-02-15"] * 2
            ),
            "page_type": ["home", "product", "cart", "home", "checkout"],
            "time_on_page_seconds": [30.0, 90.0, 45.0, 20.0, 60.0],
            "device_type": ["desktop", "desktop", "mobile", "mobile", "mobile"],
            "added_to_cart": [False, True, True, False, False],
            "search_query": [None, None, None, None, None],
        }
    )


@pytest.fixture()
def two_customer_tickets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticket_id": ["tk1", "tk2"],
            "customer_id": ["c1", "c1"],
            "created_at": pd.to_datetime(["2023-02-01", "2023-03-15"]),
            "topic": ["billing", "shipping"],
            "raw_text": ["I was charged twice", "My order is late"],
            "sentiment_score": [-0.5, -0.2],
            "resolved": [True, False],
            "resolution_days": [2.0, None],
        }
    )


@pytest.fixture()
def two_customer_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "age": [28.0, 45.0],
            "gender": ["F", "M"],
            "loyalty_tier": ["Gold", "Bronze"],
            "acquisition_channel": ["organic_search", "paid_search"],
        }
    )


@pytest.fixture()
def two_customer_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "first_purchase_date": pd.to_datetime(["2023-01-10", "2023-02-15"]),
            "last_purchase_date": pd.to_datetime(["2023-04-01", "2023-04-20"]),
            "duration_days": [80.0, 64.0],
            "event_observed": [True, False],
            "total_orders": [2, 2],
            "total_revenue": [130.0, 150.0],
        }
    )


# ---------------------------------------------------------------------------
# RFMFeatureExtractor
# ---------------------------------------------------------------------------


class TestRFMFeatureExtractor:
    def test_output_shape(self, two_customer_transactions: pd.DataFrame) -> None:
        rfm = RFMFeatureExtractor()
        result = rfm.fit_transform(two_customer_transactions)
        assert result.shape[0] == 2
        assert "recency_days" in result.columns
        assert "frequency" in result.columns
        assert "monetary_total" in result.columns

    def test_frequency_counts_transactions(self, two_customer_transactions: pd.DataFrame) -> None:
        rfm = RFMFeatureExtractor()
        result = rfm.fit_transform(two_customer_transactions)
        freq = result.set_index("customer_id")["frequency"]
        assert freq["c1"] == 2
        assert freq["c2"] == 2

    def test_monetary_total_correct(self, two_customer_transactions: pd.DataFrame) -> None:
        rfm = RFMFeatureExtractor()
        result = rfm.fit_transform(two_customer_transactions)
        totals = result.set_index("customer_id")["monetary_total"]
        assert totals["c1"] == pytest.approx(130.0)
        assert totals["c2"] == pytest.approx(150.0)

    def test_explicit_reference_date(self, two_customer_transactions: pd.DataFrame) -> None:
        ref = pd.Timestamp("2023-05-01")
        rfm = RFMFeatureExtractor(reference_date=ref)
        result = rfm.fit_transform(two_customer_transactions)
        rec = result.set_index("customer_id")["recency_days"]
        # c1's last tx: 2023-03-01 → 61 days before 2023-05-01
        assert rec["c1"] == 61

    def test_transform_without_fit_raises(self, two_customer_transactions: pd.DataFrame) -> None:
        rfm = RFMFeatureExtractor()
        with pytest.raises(RuntimeError, match="fit"):
            rfm.transform(two_customer_transactions)

    def test_monetary_std_zero_for_single_transaction(self) -> None:
        df = pd.DataFrame(
            {
                "transaction_id": ["t1"],
                "customer_id": ["c1"],
                "event_timestamp": pd.to_datetime(["2023-01-01"]),
                "order_value": [100.0],
                "num_items": [1],
                "product_category": ["Books"],
                "payment_method": ["credit_card"],
                "is_returned": [False],
                "discount_applied": [0.0],
            }
        )
        rfm = RFMFeatureExtractor()
        result = rfm.fit_transform(df)
        assert result["monetary_std"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BehavioralFeatureExtractor
# ---------------------------------------------------------------------------


class TestBehavioralFeatureExtractor:
    def test_output_has_expected_columns(self, two_customer_clickstream: pd.DataFrame) -> None:
        result = BehavioralFeatureExtractor().transform(two_customer_clickstream)
        for col in ["customer_id", "n_sessions", "avg_session_duration_s", "cart_session_rate"]:
            assert col in result.columns

    def test_n_sessions_correct(self, two_customer_clickstream: pd.DataFrame) -> None:
        result = BehavioralFeatureExtractor().transform(two_customer_clickstream)
        ns = result.set_index("customer_id")["n_sessions"]
        assert ns["c1"] == 2  # s1, s2
        assert ns["c2"] == 1  # s3

    def test_device_pct_columns_present(self, two_customer_clickstream: pd.DataFrame) -> None:
        result = BehavioralFeatureExtractor().transform(two_customer_clickstream)
        device_cols = [c for c in result.columns if c.startswith("device_")]
        assert len(device_cols) >= 1

    def test_page_pct_columns_present(self, two_customer_clickstream: pd.DataFrame) -> None:
        result = BehavioralFeatureExtractor().transform(two_customer_clickstream)
        page_cols = [c for c in result.columns if c.startswith("page_")]
        assert len(page_cols) >= 1


# ---------------------------------------------------------------------------
# TicketFeatureExtractor
# ---------------------------------------------------------------------------


class TestTicketFeatureExtractor:
    def test_only_ticketed_customer_has_rows(self, two_customer_tickets: pd.DataFrame) -> None:
        result = TicketFeatureExtractor().transform(two_customer_tickets)
        # only c1 has tickets in fixture
        assert set(result["customer_id"]) == {"c1"}

    def test_n_tickets_correct(self, two_customer_tickets: pd.DataFrame) -> None:
        result = TicketFeatureExtractor().transform(two_customer_tickets)
        assert result.set_index("customer_id").loc["c1", "n_tickets"] == 2

    def test_avg_resolution_days_fills_nan(self, two_customer_tickets: pd.DataFrame) -> None:
        result = TicketFeatureExtractor().transform(two_customer_tickets)
        # pandas mean() skips NaN: mean([2.0, NaN]) == 2.0
        val = result.set_index("customer_id").loc["c1", "avg_resolution_days"]
        assert val == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# CustomerFeatureExtractor
# ---------------------------------------------------------------------------


class TestCustomerFeatureExtractor:
    def test_loyalty_tier_ordinal_ordering(self, two_customer_customers: pd.DataFrame) -> None:
        result = CustomerFeatureExtractor().transform(two_customer_customers)
        vals = result.set_index("customer_id")["loyalty_tier_ordinal"]
        # Gold (index 2) > Bronze (index 0)
        assert vals["c1"] > vals["c2"]

    def test_gender_dummies_present(self, two_customer_customers: pd.DataFrame) -> None:
        result = CustomerFeatureExtractor().transform(two_customer_customers)
        gender_cols = [c for c in result.columns if c.startswith("gender_")]
        assert len(gender_cols) >= 2

    def test_channel_dummies_present(self, two_customer_customers: pd.DataFrame) -> None:
        result = CustomerFeatureExtractor().transform(two_customer_customers)
        channel_cols = [c for c in result.columns if c.startswith("channel_")]
        assert len(channel_cols) >= 2

    def test_no_loyalty_tier_column_in_output(self, two_customer_customers: pd.DataFrame) -> None:
        result = CustomerFeatureExtractor().transform(two_customer_customers)
        assert "loyalty_tier" not in result.columns


# ---------------------------------------------------------------------------
# FeaturePipeline (requires scikit-learn)
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    def test_fit_transform_returns_dataframe_and_structured_array(
        self,
        two_customer_customers: pd.DataFrame,
        two_customer_transactions: pd.DataFrame,
        two_customer_clickstream: pd.DataFrame,
        two_customer_tickets: pd.DataFrame,
        two_customer_labels: pd.DataFrame,
    ) -> None:
        pipe = FeaturePipeline(scale=False)
        X, y = pipe.fit_transform(
            two_customer_customers,
            two_customer_transactions,
            two_customer_clickstream,
            two_customer_tickets,
            two_customer_labels,
        )
        assert isinstance(X, pd.DataFrame)
        assert y.dtype == SURVIVAL_DTYPE

    def test_X_has_no_nans_after_imputation(
        self,
        two_customer_customers: pd.DataFrame,
        two_customer_transactions: pd.DataFrame,
        two_customer_clickstream: pd.DataFrame,
        two_customer_tickets: pd.DataFrame,
        two_customer_labels: pd.DataFrame,
    ) -> None:
        pipe = FeaturePipeline(scale=False)
        X, _ = pipe.fit_transform(
            two_customer_customers,
            two_customer_transactions,
            two_customer_clickstream,
            two_customer_tickets,
            two_customer_labels,
        )
        assert not X.isnull().any().any(), "NaNs remain after imputation"

    def test_y_aligned_to_X_index(
        self,
        two_customer_customers: pd.DataFrame,
        two_customer_transactions: pd.DataFrame,
        two_customer_clickstream: pd.DataFrame,
        two_customer_tickets: pd.DataFrame,
        two_customer_labels: pd.DataFrame,
    ) -> None:
        pipe = FeaturePipeline(scale=False)
        X, y = pipe.fit_transform(
            two_customer_customers,
            two_customer_transactions,
            two_customer_clickstream,
            two_customer_tickets,
            two_customer_labels,
        )
        assert len(X) == len(y)

    def test_y_events_match_labels(
        self,
        two_customer_customers: pd.DataFrame,
        two_customer_transactions: pd.DataFrame,
        two_customer_clickstream: pd.DataFrame,
        two_customer_tickets: pd.DataFrame,
        two_customer_labels: pd.DataFrame,
    ) -> None:
        pipe = FeaturePipeline(scale=False)
        X, y = pipe.fit_transform(
            two_customer_customers,
            two_customer_transactions,
            two_customer_clickstream,
            two_customer_tickets,
            two_customer_labels,
        )
        # c1 is churned (True), c2 is censored (False) in the fixture
        events = dict(zip(X.index, y["event"], strict=True))
        assert bool(events["c1"]) is True
        assert bool(events["c2"]) is False

    def test_feature_names_populated_after_fit(
        self,
        two_customer_customers: pd.DataFrame,
        two_customer_transactions: pd.DataFrame,
        two_customer_clickstream: pd.DataFrame,
        two_customer_tickets: pd.DataFrame,
        two_customer_labels: pd.DataFrame,
    ) -> None:
        pipe = FeaturePipeline()
        pipe.fit(
            two_customer_customers,
            two_customer_transactions,
            two_customer_clickstream,
            two_customer_tickets,
            two_customer_labels,
        )
        assert len(pipe.feature_names) > 0

    def test_transform_without_fit_raises(
        self,
        two_customer_customers: pd.DataFrame,
        two_customer_transactions: pd.DataFrame,
        two_customer_clickstream: pd.DataFrame,
        two_customer_tickets: pd.DataFrame,
        two_customer_labels: pd.DataFrame,
    ) -> None:
        pipe = FeaturePipeline()
        with pytest.raises(RuntimeError, match="fit"):
            pipe.transform(
                two_customer_customers,
                two_customer_transactions,
                two_customer_clickstream,
                two_customer_tickets,
                two_customer_labels,
            )

    def test_scale_true_produces_standardized_columns(
        self,
        two_customer_customers: pd.DataFrame,
        two_customer_transactions: pd.DataFrame,
        two_customer_clickstream: pd.DataFrame,
        two_customer_tickets: pd.DataFrame,
        two_customer_labels: pd.DataFrame,
    ) -> None:
        pipe = FeaturePipeline(scale=True)
        X, _ = pipe.fit_transform(
            two_customer_customers,
            two_customer_transactions,
            two_customer_clickstream,
            two_customer_tickets,
            two_customer_labels,
        )
        # With only 2 rows, StandardScaler produces values symmetrically around 0
        assert X.shape[0] == 2

    def test_pipeline_works_on_full_simulated_data(self, clean_datasets: dict) -> None:
        pipe = FeaturePipeline(scale=True)
        X, y = pipe.fit_transform(
            clean_datasets["customers"],
            clean_datasets["transactions"],
            clean_datasets["clickstream"],
            clean_datasets["support_tickets"],
            clean_datasets["survival_labels"],
        )
        assert X.shape[0] > 0
        assert X.shape[1] > 10
        assert y.dtype == SURVIVAL_DTYPE
        assert not X.isnull().any().any(), "NaNs in full-data transform"
        assert (y["duration"] >= 0).all()
