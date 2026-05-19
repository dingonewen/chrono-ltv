"""Unit tests for the multi-source data ingestion layer.

Coverage:
- Schema validation (Pydantic models): valid payloads, field constraints,
  custom validators, and computed properties.
- MultiSourceAggregator: normalisation, 3PL join, net margin, empty inputs,
  and to_transactions_df() column layout.
- IngestionValidator (GE-backed): per-stream suites and cross-stream join
  integrity (both matched and unmatched cases).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from chrono_ltv.data_ingestion.schemas import (
    AmazonOrderReport,
    Logistics3PLInvoice,
    ShopifyWebhookPayload,
)
from chrono_ltv.data_ingestion.aggregator import MultiSourceAggregator

# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

_TS_EARLY = datetime(2024, 1, 10, 8, 0, tzinfo=timezone.utc)
_TS_LATE = datetime(2024, 1, 10, 9, 0, tzinfo=timezone.utc)


def _shopify(**overrides: object) -> ShopifyWebhookPayload:
    defaults: dict[str, object] = {
        "checkout_id": "chk_001",
        "order_id": "ord_001",
        "customer_id": "cust_001",
        "email": "alice@example.com",
        "total_price": 120.00,
        "subtotal_price": 110.00,
        "total_discounts": 10.00,
        "financial_status": "paid",
        "fulfillment_status": "fulfilled",
        "marketing_tags": ["utm_source=google"],
        "line_item_count": 2,
        "created_at": _TS_EARLY,
        "updated_at": _TS_LATE,
    }
    defaults.update(overrides)
    return ShopifyWebhookPayload(**defaults)  # type: ignore[arg-type]


def _amazon(**overrides: object) -> AmazonOrderReport:
    defaults: dict[str, object] = {
        "amazon_order_id": "111-2222222-3333333",
        "purchase_date": _TS_EARLY,
        "last_updated_date": _TS_LATE,
        "order_status": "Delivered",
        "fulfillment_channel": "AFN",
        "sales_channel": "Amazon.com",
        "asin": "B01EXAMPLE",
        "quantity": 1,
        "item_price": 89.99,
        "fba_fee": 7.50,
        "referral_fee": 13.50,
    }
    defaults.update(overrides)
    return AmazonOrderReport(**defaults)  # type: ignore[arg-type]


def _tpl(**overrides: object) -> Logistics3PLInvoice:
    defaults: dict[str, object] = {
        "invoice_id": "inv_001",
        "tracking_number": "1Z999AA10123456784",
        "order_reference": "chk_001",
        "carrier": "UPS",
        "service_level": "Ground",
        "ship_date": _TS_EARLY,
        "delivery_date": _TS_LATE,
        "weight_lbs": 2.5,
        "length_in": 10.0,
        "width_in": 8.0,
        "height_in": 6.0,
        "zone": 4,
        "base_rate": 8.50,
        "fuel_surcharge": 1.20,
        "residential_surcharge": 3.65,
        "total_charge": 13.35,
        "carrier_delay_status": False,
        "delivery_status": "delivered",
    }
    defaults.update(overrides)
    return Logistics3PLInvoice(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# Schema tests
# ===========================================================================


class TestShopifyWebhookPayload:
    def test_valid_payload(self) -> None:
        p = _shopify()
        assert p.checkout_id == "chk_001"
        assert p.total_price == 120.00
        assert p.marketing_tags == ["utm_source=google"]

    def test_missing_required_field_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ShopifyWebhookPayload(  # type: ignore[call-arg]
                order_id="x",
                customer_id="y",
                email="z@z.com",
                total_price=10.0,
                subtotal_price=10.0,
                financial_status="paid",
                line_item_count=1,
                created_at=_TS_EARLY,
                updated_at=_TS_LATE,
                # checkout_id intentionally omitted
            )

    def test_negative_total_price_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _shopify(total_price=-1.0)

    def test_updated_before_created_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _shopify(created_at=_TS_LATE, updated_at=_TS_EARLY)

    def test_invalid_financial_status_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _shopify(financial_status="settled")

    def test_whitespace_stripped(self) -> None:
        p = _shopify(checkout_id="  chk_001  ")
        assert p.checkout_id == "chk_001"

    def test_default_marketing_tags_empty_list(self) -> None:
        p = _shopify(marketing_tags=[])
        assert p.marketing_tags == []

    def test_fulfillment_status_nullable(self) -> None:
        p = _shopify(fulfillment_status=None)
        assert p.fulfillment_status is None


class TestAmazonOrderReport:
    def test_valid_report(self) -> None:
        r = _amazon()
        assert r.amazon_order_id == "111-2222222-3333333"
        assert r.fulfillment_channel == "AFN"

    def test_total_amazon_fees_property(self) -> None:
        r = _amazon(fba_fee=7.50, referral_fee=13.50)
        assert r.total_amazon_fees == pytest.approx(21.00)

    def test_fba_fee_negative_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _amazon(fba_fee=-1.0)

    def test_invalid_fulfillment_channel_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _amazon(fulfillment_channel="XYZ")

    def test_invalid_order_status_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _amazon(order_status="Returned")

    def test_merchant_order_id_optional(self) -> None:
        r = _amazon(merchant_order_id=None)
        assert r.merchant_order_id is None

    def test_customer_id_optional(self) -> None:
        r = _amazon()
        assert r.customer_id is None

    def test_quantity_must_be_at_least_one(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _amazon(quantity=0)


class TestLogistics3PLInvoice:
    def test_valid_invoice(self) -> None:
        inv = _tpl()
        assert inv.tracking_number == "1Z999AA10123456784"
        assert inv.zone == 4

    def test_volumetric_weight_property(self) -> None:
        inv = _tpl(length_in=10.0, width_in=8.0, height_in=6.0)
        expected = (10.0 * 8.0 * 6.0) / 139.0
        assert inv.volumetric_weight_lbs == pytest.approx(expected)

    def test_billable_weight_is_max(self) -> None:
        inv = _tpl(weight_lbs=0.1, length_in=20.0, width_in=20.0, height_in=20.0)
        assert inv.billable_weight_lbs == pytest.approx(inv.volumetric_weight_lbs)

        heavy = _tpl(weight_lbs=100.0, length_in=5.0, width_in=5.0, height_in=5.0)
        assert heavy.billable_weight_lbs == pytest.approx(100.0)

    def test_total_charge_less_than_base_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _tpl(base_rate=10.0, total_charge=5.0)

    def test_zone_out_of_range_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _tpl(zone=9)

    def test_invalid_delivery_status_raises(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _tpl(delivery_status="lost")

    def test_delivery_date_optional(self) -> None:
        inv = _tpl(delivery_date=None)
        assert inv.delivery_date is None


# ===========================================================================
# Aggregator tests
# ===========================================================================


class TestMultiSourceAggregatorEmpty:
    def test_all_empty_returns_empty_dataframe(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate([], [], [])
        assert result.empty

    def test_to_transactions_df_on_empty_returns_correct_columns(self) -> None:
        agg = MultiSourceAggregator()
        tx = agg.to_transactions_df(pd.DataFrame())
        assert set(tx.columns) == {
            "customer_id",
            "transaction_id",
            "event_timestamp",
            "order_value",
            "num_items",
            "product_category",
            "is_returned",
            "discount_applied",
        }
        assert len(tx) == 0


class TestMultiSourceAggregatorShopifyOnly:
    def test_single_shopify_row_no_tpl(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate([_shopify()], [], [])
        assert len(result) == 1
        assert result["channel"].iloc[0] == "shopify"
        assert result["universal_order_id"].iloc[0] == "chk_001"

    def test_net_margin_no_fees_no_tpl(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate([_shopify(total_price=100.0)], [], [])
        # gross=100, platform_fees=0, shipping_cost=0 → margin=100
        assert result["net_contribution_margin"].iloc[0] == pytest.approx(100.0)

    def test_tpl_not_matched_sets_tpl_matched_false(self) -> None:
        agg = MultiSourceAggregator()
        tpl = _tpl(order_reference="chk_OTHER")  # does not match chk_001
        result = agg.aggregate([_shopify()], [], [tpl])
        assert not result["tpl_matched"].iloc[0]

    def test_tpl_matched_sets_shipping_cost(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate([_shopify()], [], [_tpl(total_charge=13.35)])
        assert result["shipping_cost"].iloc[0] == pytest.approx(13.35)
        assert result["tpl_matched"].iloc[0]

    def test_net_margin_deducts_shipping(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate(
            [_shopify(total_price=100.0)], [], [_tpl(total_charge=10.0)]
        )
        assert result["net_contribution_margin"].iloc[0] == pytest.approx(90.0)


class TestMultiSourceAggregatorAmazonOnly:
    def test_single_amazon_fba_row(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate([], [_amazon()], [])
        assert len(result) == 1
        assert result["channel"].iloc[0] == "amazon"
        assert result["fulfillment_type"].iloc[0] == "fba"

    def test_amazon_fbm_fulfillment_type(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate([], [_amazon(fulfillment_channel="MFN")], [])
        assert result["fulfillment_type"].iloc[0] == "fbm"

    def test_platform_fees_deducted_in_margin(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate(
            [], [_amazon(item_price=100.0, fba_fee=5.0, referral_fee=15.0)], []
        )
        # gross=100, platform_fees=20, shipping=0 → margin=80
        assert result["net_contribution_margin"].iloc[0] == pytest.approx(80.0)

    def test_customer_id_fallback_to_amazon_order_id(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate(
            [], [_amazon(customer_id=None, merchant_order_id=None)], []
        )
        assert result["customer_id"].iloc[0] == "111-2222222-3333333"


class TestMultiSourceAggregatorMixed:
    def test_shopify_and_amazon_concatenated(self) -> None:
        agg = MultiSourceAggregator()
        result = agg.aggregate([_shopify()], [_amazon()], [])
        assert len(result) == 2
        assert set(result["channel"]) == {"shopify", "amazon"}

    def test_tpl_joins_only_matching_rows(self) -> None:
        agg = MultiSourceAggregator()
        # Two Shopify orders; only one has a 3PL record
        s1 = _shopify(checkout_id="chk_A")
        s2 = _shopify(checkout_id="chk_B")
        tpl = _tpl(order_reference="chk_A", base_rate=4.0, total_charge=5.0)
        result = agg.aggregate([s1, s2], [], [tpl])

        matched = result[result["universal_order_id"] == "chk_A"]
        unmatched = result[result["universal_order_id"] == "chk_B"]
        assert matched["tpl_matched"].iloc[0]
        assert not unmatched["tpl_matched"].iloc[0]
        assert unmatched["shipping_cost"].iloc[0] == pytest.approx(0.0)


class TestToTransactionsDf:
    def test_column_layout_matches_feature_pipeline(self) -> None:
        agg = MultiSourceAggregator()
        orders = agg.aggregate([_shopify()], [_amazon()], [_tpl()])
        tx = agg.to_transactions_df(orders)
        required = {
            "customer_id",
            "transaction_id",
            "event_timestamp",
            "order_value",
            "num_items",
            "product_category",
            "is_returned",
            "discount_applied",
        }
        assert required.issubset(set(tx.columns))

    def test_order_value_clipped_to_zero(self) -> None:
        """Net margin can go negative if fees exceed revenue; clip to 0."""
        agg = MultiSourceAggregator()
        # item_price=1, fba=5, referral=5 → margin = -9
        orders = agg.aggregate(
            [], [_amazon(item_price=1.0, fba_fee=5.0, referral_fee=5.0)], []
        )
        tx = agg.to_transactions_df(orders)
        assert tx["order_value"].iloc[0] == pytest.approx(0.0)

    def test_returned_orders_flagged(self) -> None:
        agg = MultiSourceAggregator()
        orders = agg.aggregate(
            [_shopify()], [], [_tpl(delivery_status="returned")]
        )
        tx = agg.to_transactions_df(orders)
        assert bool(tx["is_returned"].iloc[0]) is True


# ===========================================================================
# Validator tests (require great-expectations)
# ===========================================================================

gx = pytest.importorskip("great_expectations", reason="great-expectations not installed")

from chrono_ltv.data_ingestion.validators import IngestionValidator  # noqa: E402


def _shopify_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [_shopify(checkout_id=f"chk_{i}", customer_id=f"cust_{i}").model_dump() for i in range(n)]
    )


def _amazon_df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _amazon(amazon_order_id=f"111-{i:07d}-{i:07d}").model_dump()
            for i in range(n)
        ]
    )


def _tpl_df(order_refs: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _tpl(
                invoice_id=f"inv_{i}",
                tracking_number=f"TRACK{i:010d}",
                order_reference=ref,
            ).model_dump()
            for i, ref in enumerate(order_refs)
        ]
    )


class TestIngestionValidatorShopify:
    def test_valid_df_passes(self) -> None:
        v = IngestionValidator()
        summary = v.validate_shopify(_shopify_df())
        assert summary.success
        assert summary.n_failed == 0

    def test_missing_column_fails(self) -> None:
        v = IngestionValidator()
        df = _shopify_df().drop(columns=["checkout_id"])
        summary = v.validate_shopify(df)
        assert not summary.success
        assert summary.n_failed > 0

    def test_invalid_financial_status_fails(self) -> None:
        v = IngestionValidator()
        df = _shopify_df()
        df.loc[0, "financial_status"] = "invalid_status"
        summary = v.validate_shopify(df)
        assert not summary.success


class TestIngestionValidatorAmazon:
    def test_valid_df_passes(self) -> None:
        v = IngestionValidator()
        summary = v.validate_amazon(_amazon_df())
        assert summary.success

    def test_invalid_fulfillment_channel_fails(self) -> None:
        v = IngestionValidator()
        df = _amazon_df()
        df.loc[0, "fulfillment_channel"] = "INVALID"
        summary = v.validate_amazon(df)
        assert not summary.success


class TestIngestionValidatorTPL:
    def test_valid_df_passes(self) -> None:
        v = IngestionValidator()
        df = _tpl_df(["chk_0", "chk_1"])
        summary = v.validate_tpl(df)
        assert summary.success

    def test_missing_tracking_number_fails(self) -> None:
        v = IngestionValidator()
        df = _tpl_df(["chk_0"])
        df.loc[0, "tracking_number"] = None
        summary = v.validate_tpl(df)
        assert not summary.success


class TestJoinIntegrity:
    def test_all_fulfilled_orders_matched(self) -> None:
        v = IngestionValidator()
        shopify_df = _shopify_df(3)  # checkout_ids: chk_0, chk_1, chk_2
        tpl_df = _tpl_df(["chk_0", "chk_1", "chk_2"])
        summary = v.validate_join_integrity(shopify_df, tpl_df)
        assert summary.success
        assert summary.n_failed == 0

    def test_unmatched_fulfilled_orders_captured(self) -> None:
        v = IngestionValidator()
        shopify_df = _shopify_df(3)  # chk_0, chk_1, chk_2
        tpl_df = _tpl_df(["chk_0"])  # only chk_0 matched
        summary = v.validate_join_integrity(shopify_df, tpl_df)
        assert not summary.success
        assert summary.n_failed == 2
        assert any("chk_1" in f or "chk_2" in f for f in summary.failures)

    def test_no_fulfilled_orders_is_success(self) -> None:
        v = IngestionValidator()
        shopify_df = _shopify_df(2)
        shopify_df["fulfillment_status"] = "unfulfilled"
        tpl_df = pd.DataFrame(columns=["order_reference"])
        summary = v.validate_join_integrity(shopify_df, tpl_df)
        assert summary.success
        assert summary.n_failed == 0

    def test_missing_column_does_not_raise(self) -> None:
        v = IngestionValidator()
        shopify_df = pd.DataFrame({"order_id": ["x"]})  # missing required columns
        tpl_df = _tpl_df(["chk_0"])
        summary = v.validate_join_integrity(shopify_df, tpl_df)
        assert not summary.success
        assert len(summary.failures) > 0
