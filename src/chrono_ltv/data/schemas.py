"""Pydantic v2 schemas for every simulated entity.

These schemas serve a dual purpose:
  1. Runtime validation of rows flowing through the pipeline.
  2. Auto-generated OpenAPI documentation for the serving layer.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: UUID
    first_name: str
    last_name: str
    email: str
    country: str
    acquisition_channel: Literal[
        "organic_search",
        "paid_search",
        "social_media",
        "email",
        "referral",
        "direct",
    ]
    registration_date: datetime
    age: int | None = Field(default=None, ge=18, le=100)
    gender: Literal["M", "F", "Non-binary", "Unknown"]
    loyalty_tier: Literal["Bronze", "Silver", "Gold", "Platinum"]


class TransactionRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: UUID
    customer_id: UUID
    event_timestamp: datetime
    order_value: float = Field(ge=0.0)
    num_items: int = Field(ge=1)
    product_category: str
    payment_method: Literal["credit_card", "debit_card", "paypal", "crypto", "buy_now_pay_later"]
    is_returned: bool
    discount_applied: float = Field(ge=0.0, le=1.0)

    @field_validator("order_value")
    @classmethod
    def order_value_must_be_finite(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError("order_value must be finite")
        return v


class ClickstreamRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: UUID
    customer_id: UUID
    event_timestamp: datetime
    page_type: Literal[
        "home",
        "category",
        "product",
        "cart",
        "checkout",
        "confirmation",
        "search",
        "account",
    ]
    time_on_page_seconds: float = Field(ge=0.0)
    device_type: Literal["desktop", "mobile", "tablet"]
    added_to_cart: bool
    search_query: str | None = None


class SupportTicketRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ticket_id: UUID
    customer_id: UUID
    created_at: datetime
    topic: str
    raw_text: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    resolved: bool
    resolution_days: float | None = Field(default=None, ge=0.0)


class SurvivalLabel(BaseModel):
    """Ground-truth survival analysis targets.

    duration  — days from first purchase to churn event or censoring date.
    event     — True if churn was observed; False if the customer was
                 censored (still active at observation window end).
    """

    customer_id: UUID
    first_purchase_date: datetime
    last_purchase_date: datetime
    duration_days: float = Field(ge=0.0)
    event_observed: bool
    total_orders: int = Field(ge=0)
    total_revenue: float = Field(ge=0.0)
