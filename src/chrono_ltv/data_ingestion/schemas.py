"""Pydantic v2 schemas for the multi-source data ingestion layer.

Three heterogeneous e-commerce data streams are modelled here:
- :class:`ShopifyWebhookPayload` — DTC storefront order events
- :class:`AmazonOrderReport`    — Seller Central MWS/SP-API order rows
- :class:`Logistics3PLInvoice`  — last-mile shipping invoice lines

Note: do NOT add ``from __future__ import annotations`` to this file.
Pydantic v2 evaluates field annotations at runtime; deferred evaluation
breaks ``ConfigDict`` and ``field_validator`` introspection.
"""

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Shopify ───────────────────────────────────────────────────────────────────


class ShopifyWebhookPayload(BaseModel):
    """A single order event delivered by the Shopify Webhooks API.

    Fields mirror the ``orders/create`` and ``orders/updated`` webhook topics.
    Marketing tags (UTM params, campaign slugs) are surfaced in
    ``marketing_tags`` for attribution downstream.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    checkout_id: str
    order_id: str
    customer_id: str
    email: str
    total_price: float = Field(ge=0.0)
    subtotal_price: float = Field(ge=0.0)
    total_discounts: float = Field(ge=0.0, default=0.0)
    financial_status: Literal[
        "paid", "pending", "refunded", "partially_refunded", "voided"
    ]
    fulfillment_status: Literal[
        "fulfilled", "partial", "unfulfilled", "restocked"
    ] | None = None
    marketing_tags: list[str] = Field(default_factory=list)
    line_item_count: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    @field_validator("total_price", "subtotal_price")
    @classmethod
    def _must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("price must be a finite number")
        return v

    @field_validator("updated_at")
    @classmethod
    def _updated_not_before_created(cls, v: datetime, info: Any) -> datetime:
        created = info.data.get("created_at")
        if created is not None and v < created:
            raise ValueError("updated_at cannot precede created_at")
        return v


# ── Amazon ────────────────────────────────────────────────────────────────────


class AmazonOrderReport(BaseModel):
    """A single order row from an Amazon Seller Central flat-file report.

    ``fulfillment_channel``
        ``"AFN"`` = Amazon-fulfilled (FBA); ``"MFN"`` = merchant-fulfilled (FBM).

    ``fba_fee`` and ``referral_fee``
        Populated from the *Fee Preview* or *Settlement* report columns.
        Both are zero for MFN orders (seller handles fulfilment).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    amazon_order_id: str
    merchant_order_id: str | None = None
    customer_id: str | None = None  # populated when seller controls identity
    purchase_date: datetime
    last_updated_date: datetime
    order_status: Literal["Pending", "Unshipped", "Shipped", "Delivered", "Cancelled"]
    fulfillment_channel: Literal["AFN", "MFN"]
    sales_channel: str
    asin: str
    sku: str | None = None
    quantity: int = Field(ge=1)
    item_price: float = Field(ge=0.0)
    item_tax: float = Field(ge=0.0, default=0.0)
    shipping_price: float = Field(ge=0.0, default=0.0)
    fba_fee: float = Field(ge=0.0, default=0.0)
    referral_fee: float = Field(ge=0.0, default=0.0)
    is_business_order: bool = False

    @field_validator("item_price", "fba_fee", "referral_fee")
    @classmethod
    def _must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("monetary value must be a finite number")
        return v

    @property
    def total_amazon_fees(self) -> float:
        return self.fba_fee + self.referral_fee


# ── 3PL / Logistics ───────────────────────────────────────────────────────────


class Logistics3PLInvoice(BaseModel):
    """A single shipment invoice line from a third-party logistics provider.

    ``order_reference``
        The originating order identifier.  For Shopify orders this matches
        ``ShopifyWebhookPayload.checkout_id``; for Amazon orders it matches
        ``AmazonOrderReport.amazon_order_id``.

    ``volumetric_weight_lbs``
        Dimensional weight using the standard DIM divisor of 139
        (L × W × H in cubic inches / 139).  The ``billable_weight_lbs``
        property returns whichever is higher — actual vs. dimensional.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    invoice_id: str
    tracking_number: str
    order_reference: str
    carrier: str
    service_level: str
    ship_date: datetime
    delivery_date: datetime | None = None
    weight_lbs: float = Field(gt=0.0)
    length_in: float = Field(gt=0.0)
    width_in: float = Field(gt=0.0)
    height_in: float = Field(gt=0.0)
    zone: int = Field(ge=1, le=8)
    base_rate: float = Field(ge=0.0)
    fuel_surcharge: float = Field(ge=0.0, default=0.0)
    residential_surcharge: float = Field(ge=0.0, default=0.0)
    total_charge: float = Field(ge=0.0)
    carrier_delay_status: bool = False
    delivery_status: Literal["in_transit", "delivered", "exception", "returned"]

    @field_validator("total_charge")
    @classmethod
    def _total_covers_base(cls, v: float, info: Any) -> float:
        base = info.data.get("base_rate", 0.0)
        if v < base:
            raise ValueError(
                f"total_charge ({v}) cannot be less than base_rate ({base})"
            )
        return v

    @property
    def volumetric_weight_lbs(self) -> float:
        """Dimensional weight: L × W × H (cubic in) / 139."""
        return (self.length_in * self.width_in * self.height_in) / 139.0

    @property
    def billable_weight_lbs(self) -> float:
        """Max of actual weight and dimensional weight."""
        return max(self.weight_lbs, self.volumetric_weight_lbs)
