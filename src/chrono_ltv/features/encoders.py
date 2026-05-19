"""Per-table feature extractors for the ChronoLTV feature engineering step.

Each extractor accepts one or more DataFrames, applies domain-specific
aggregations, and returns a customer-indexed DataFrame ready for merging.

All extractors are pure-pandas — no sklearn dependency — so they can be
tested without the `features` extras group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# RFM features (from transactions)
# ---------------------------------------------------------------------------


class RFMFeatureExtractor:
    """Recency, Frequency, Monetary features derived from transaction history.

    Parameters
    ----------
    reference_date : pd.Timestamp | None
        Date from which recency is measured.  Defaults to the max
        ``event_timestamp`` in the training set so the value is
        data-driven rather than wall-clock dependent.
    """

    def __init__(self, reference_date: pd.Timestamp | None = None) -> None:
        self.reference_date = reference_date
        self._fitted_reference_date: pd.Timestamp | None = None

    def fit(self, transactions: pd.DataFrame) -> RFMFeatureExtractor:
        if self.reference_date is not None:
            self._fitted_reference_date = self.reference_date
        else:
            self._fitted_reference_date = pd.to_datetime(transactions["event_timestamp"]).max()
        return self

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if self._fitted_reference_date is None:
            raise RuntimeError("Call fit() before transform().")
        ts = pd.to_datetime(transactions["event_timestamp"])
        df = transactions.assign(event_timestamp=ts)

        agg = (
            df.groupby("customer_id")
            .agg(
                recency_days=(
                    "event_timestamp",
                    lambda s: (self._fitted_reference_date - s.max()).days,
                ),
                frequency=("transaction_id", "count"),
                monetary_total=("order_value", "sum"),
                monetary_mean=("order_value", "mean"),
                monetary_std=("order_value", "std"),
                avg_items_per_order=("num_items", "mean"),
                return_rate=("is_returned", "mean"),
                avg_discount=("discount_applied", "mean"),
                n_categories=("product_category", "nunique"),
            )
            .reset_index()
        )
        agg["monetary_std"] = agg["monetary_std"].fillna(0.0)
        return agg

    def fit_transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        return self.fit(transactions).transform(transactions)


# ---------------------------------------------------------------------------
# Behavioral features (from clickstream)
# ---------------------------------------------------------------------------


class BehavioralFeatureExtractor:
    """Session and engagement features from clickstream events."""

    def transform(self, clickstream: pd.DataFrame) -> pd.DataFrame:
        df = clickstream.copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

        # Session-level aggregation
        sessions = (
            df.groupby(["customer_id", "session_id"])
            .agg(
                session_duration_s=("time_on_page_seconds", "sum"),
                pages_per_session=("session_id", "count"),
                added_to_cart_in_session=("added_to_cart", "max"),
            )
            .reset_index()
        )

        # Customer-level aggregation
        agg = (
            sessions.groupby("customer_id")
            .agg(
                n_sessions=("session_id", "count"),
                avg_session_duration_s=("session_duration_s", "mean"),
                avg_pages_per_session=("pages_per_session", "mean"),
                cart_session_rate=("added_to_cart_in_session", "mean"),
            )
            .reset_index()
        )

        # Device-type dummies as proportions
        device_counts = (
            df.groupby(["customer_id", "device_type"])
            .size()
            .unstack(fill_value=0)
            .rename(columns=lambda c: f"device_{c}_pct")
        )
        device_pct = device_counts.div(device_counts.sum(axis=1), axis=0).reset_index()

        # Page-type proportions
        page_counts = (
            df.groupby(["customer_id", "page_type"])
            .size()
            .unstack(fill_value=0)
            .rename(columns=lambda c: f"page_{c}_pct")
        )
        page_pct = page_counts.div(page_counts.sum(axis=1), axis=0).reset_index()

        result = agg.merge(device_pct, on="customer_id", how="left")
        result = result.merge(page_pct, on="customer_id", how="left")
        return result


# ---------------------------------------------------------------------------
# Support-ticket features
# ---------------------------------------------------------------------------


class TicketFeatureExtractor:
    """Aggregate support-ticket features per customer.

    Text embedding is intentionally deferred to the pipeline layer so this
    extractor stays pure-pandas and fast to test.
    """

    def transform(self, tickets: pd.DataFrame) -> pd.DataFrame:
        df = tickets.copy()
        df["created_at"] = pd.to_datetime(df["created_at"])

        agg = (
            df.groupby("customer_id")
            .agg(
                n_tickets=("ticket_id", "count"),
                avg_sentiment=("sentiment_score", "mean"),
                min_sentiment=("sentiment_score", "min"),
                pct_resolved=("resolved", "mean"),
                avg_resolution_days=("resolution_days", "mean"),
                n_topics=("topic", "nunique"),
            )
            .reset_index()
        )
        agg["avg_resolution_days"] = agg["avg_resolution_days"].fillna(0.0)
        return agg


# ---------------------------------------------------------------------------
# Customer demographic features
# ---------------------------------------------------------------------------


class CustomerFeatureExtractor:
    """Numeric/categorical encoding of the customer master table.

    Ordinal encoding of ``loyalty_tier`` is fitted on the training set to
    preserve the ordering: Bronze < Silver < Gold < Platinum.
    """

    _TIER_ORDER: list[str] = ["Bronze", "Silver", "Gold", "Platinum"]
    _CHANNEL_ORDER: list[str] = [
        "organic_search",
        "paid_search",
        "social_media",
        "email",
        "referral",
        "direct",
    ]

    def transform(self, customers: pd.DataFrame) -> pd.DataFrame:
        df = customers[
            ["customer_id", "age", "gender", "loyalty_tier", "acquisition_channel"]
        ].copy()

        df["loyalty_tier_ordinal"] = pd.Categorical(
            df["loyalty_tier"], categories=self._TIER_ORDER, ordered=True
        ).codes.astype(float)
        # Replace -1 (unknown) with NaN so imputation can handle it
        df["loyalty_tier_ordinal"] = df["loyalty_tier_ordinal"].replace(-1, np.nan)

        # One-hot: gender and acquisition_channel
        gender_dummies = pd.get_dummies(df["gender"], prefix="gender")
        channel_dummies = pd.get_dummies(df["acquisition_channel"], prefix="channel")

        result = pd.concat(
            [
                df[["customer_id", "age", "loyalty_tier_ordinal"]],
                gender_dummies,
                channel_dummies,
            ],
            axis=1,
        )
        return result
