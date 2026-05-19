"""Unit tests for Pydantic v2 schemas in chrono_ltv.data.schemas."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from chrono_ltv.data.schemas import (
    ClickstreamRecord,
    CustomerRecord,
    SupportTicketRecord,
    SurvivalLabel,
    TransactionRecord,
)

_NOW = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
_CID = uuid.uuid4()


class TestCustomerRecord:
    def _valid(self) -> dict:
        return {
            "customer_id": uuid.uuid4(),
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "alice@example.com",
            "country": "US",
            "acquisition_channel": "organic_search",
            "registration_date": _NOW,
            "age": 30,
            "gender": "F",
            "loyalty_tier": "Gold",
        }

    def test_valid_instance(self) -> None:
        rec = CustomerRecord(**self._valid())
        assert rec.first_name == "Alice"

    def test_age_none_allowed(self) -> None:
        data = self._valid()
        data["age"] = None
        rec = CustomerRecord(**data)
        assert rec.age is None

    def test_age_below_min_raises(self) -> None:
        data = self._valid()
        data["age"] = 17
        with pytest.raises(ValidationError):
            CustomerRecord(**data)

    def test_invalid_channel_raises(self) -> None:
        data = self._valid()
        data["acquisition_channel"] = "billboard"
        with pytest.raises(ValidationError):
            CustomerRecord(**data)

    def test_invalid_gender_raises(self) -> None:
        data = self._valid()
        data["gender"] = "X"
        with pytest.raises(ValidationError):
            CustomerRecord(**data)


class TestTransactionRecord:
    def _valid(self) -> dict:
        return {
            "transaction_id": uuid.uuid4(),
            "customer_id": _CID,
            "event_timestamp": _NOW,
            "order_value": 49.99,
            "num_items": 2,
            "product_category": "Electronics",
            "payment_method": "credit_card",
            "is_returned": False,
            "discount_applied": 0.1,
        }

    def test_valid_instance(self) -> None:
        rec = TransactionRecord(**self._valid())
        assert rec.order_value == 49.99

    def test_negative_order_value_raises(self) -> None:
        data = self._valid()
        data["order_value"] = -1.0
        with pytest.raises(ValidationError):
            TransactionRecord(**data)

    def test_infinite_order_value_raises(self) -> None:
        data = self._valid()
        data["order_value"] = math.inf
        with pytest.raises(ValidationError):
            TransactionRecord(**data)

    def test_discount_above_one_raises(self) -> None:
        data = self._valid()
        data["discount_applied"] = 1.5
        with pytest.raises(ValidationError):
            TransactionRecord(**data)

    def test_zero_items_raises(self) -> None:
        data = self._valid()
        data["num_items"] = 0
        with pytest.raises(ValidationError):
            TransactionRecord(**data)


class TestClickstreamRecord:
    def _valid(self) -> dict:
        return {
            "session_id": uuid.uuid4(),
            "customer_id": _CID,
            "event_timestamp": _NOW,
            "page_type": "product",
            "time_on_page_seconds": 45.0,
            "device_type": "desktop",
            "added_to_cart": True,
            "search_query": None,
        }

    def test_valid_instance(self) -> None:
        rec = ClickstreamRecord(**self._valid())
        assert rec.page_type == "product"

    def test_invalid_page_type_raises(self) -> None:
        data = self._valid()
        data["page_type"] = "landing"
        with pytest.raises(ValidationError):
            ClickstreamRecord(**data)

    def test_search_query_optional(self) -> None:
        data = self._valid()
        data["search_query"] = "running shoes"
        rec = ClickstreamRecord(**data)
        assert rec.search_query == "running shoes"


class TestSupportTicketRecord:
    def _valid(self) -> dict:
        return {
            "ticket_id": uuid.uuid4(),
            "customer_id": _CID,
            "created_at": _NOW,
            "topic": "Billing",
            "raw_text": "My order was charged twice.",
            "sentiment_score": -0.5,
            "resolved": True,
            "resolution_days": 2.0,
        }

    def test_valid_instance(self) -> None:
        rec = SupportTicketRecord(**self._valid())
        assert rec.resolved is True

    def test_sentiment_out_of_range_raises(self) -> None:
        data = self._valid()
        data["sentiment_score"] = 1.5
        with pytest.raises(ValidationError):
            SupportTicketRecord(**data)

    def test_resolution_days_none_allowed(self) -> None:
        data = self._valid()
        data["resolution_days"] = None
        rec = SupportTicketRecord(**data)
        assert rec.resolution_days is None


class TestSurvivalLabel:
    def _valid(self) -> dict:
        return {
            "customer_id": _CID,
            "first_purchase_date": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "last_purchase_date": datetime(2023, 5, 1, tzinfo=timezone.utc),
            "duration_days": 120.0,
            "event_observed": True,
            "total_orders": 5,
            "total_revenue": 249.95,
        }

    def test_valid_instance(self) -> None:
        rec = SurvivalLabel(**self._valid())
        assert rec.total_orders == 5

    def test_negative_duration_raises(self) -> None:
        data = self._valid()
        data["duration_days"] = -1.0
        with pytest.raises(ValidationError):
            SurvivalLabel(**data)

    def test_negative_revenue_raises(self) -> None:
        data = self._valid()
        data["total_revenue"] = -0.01
        with pytest.raises(ValidationError):
            SurvivalLabel(**data)
