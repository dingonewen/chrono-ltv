"""Multi-source order aggregation engine for ChronoLTV.

:class:`MultiSourceAggregator` ingests three heterogeneous data streams,
aligns them into a unified order DataFrame, deducts last-mile and platform
fees to produce a Net Contribution Margin, and exposes a convenience method
that maps the result to the exact column layout expected by
:class:`~chrono_ltv.features.pipeline.FeaturePipeline`.

Usage
-----
>>> from chrono_ltv.data_ingestion.aggregator import MultiSourceAggregator
>>> agg = MultiSourceAggregator()
>>> orders = agg.aggregate(shopify_payloads, amazon_reports, tpl_invoices)
>>> transactions = agg.to_transactions_df(orders)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from chrono_ltv.utils.logging import get_logger

if TYPE_CHECKING:
    from chrono_ltv.data_ingestion.schemas import (
        AmazonOrderReport,
        Logistics3PLInvoice,
        ShopifyWebhookPayload,
    )

logger = get_logger(__name__)

# Columns present in every unified order row
_ORDER_COLS = [
    "universal_order_id",
    "source_order_id",
    "customer_id",
    "channel",
    "fulfillment_type",
    "order_date",
    "gross_revenue",
    "platform_fees",
    "financial_status",
    "fulfillment_status",
]

# Columns appended by the 3PL join
_TPL_COLS = [
    "tracking_number",
    "carrier",
    "shipping_cost",
    "carrier_delay_status",
    "delivery_status",
    "volumetric_weight_lbs",
    "zone",
]


class MultiSourceAggregator:
    """Ingest, align, and enrich orders from Shopify, Amazon, and a 3PL provider.

    The aggregation pipeline is:
    1. Normalise each source into a common schema (``_normalize_*``).
    2. Concatenate Shopify and Amazon rows.
    3. Left-join with 3PL invoices on ``universal_order_id``.
    4. Compute ``net_contribution_margin``.

    ``universal_order_id`` mapping
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Shopify → ``ShopifyWebhookPayload.checkout_id``
    - Amazon  → ``AmazonOrderReport.amazon_order_id``
    - 3PL     → ``Logistics3PLInvoice.order_reference`` (matches either of the above)
    """

    # ── public ────────────────────────────────────────────────────────────────

    def aggregate(
        self,
        shopify: list[ShopifyWebhookPayload],
        amazon: list[AmazonOrderReport],
        invoices: list[Logistics3PLInvoice],
    ) -> pd.DataFrame:
        """Return a unified, enriched order DataFrame.

        Parameters
        ----------
        shopify:
            Validated :class:`~chrono_ltv.data_ingestion.schemas.ShopifyWebhookPayload`
            objects.
        amazon:
            Validated :class:`~chrono_ltv.data_ingestion.schemas.AmazonOrderReport`
            objects.
        invoices:
            Validated :class:`~chrono_ltv.data_ingestion.schemas.Logistics3PLInvoice`
            objects.

        Returns
        -------
        pd.DataFrame
            One row per order with columns from :data:`_ORDER_COLS`,
            :data:`_TPL_COLS`, ``tpl_matched``, and
            ``net_contribution_margin``.
        """
        shopify_df = self._normalize_shopify(shopify)
        amazon_df = self._normalize_amazon(amazon)

        if shopify_df.empty and amazon_df.empty:
            logger.warning("No orders to aggregate — all source lists are empty.")
            return pd.DataFrame()

        orders_df = pd.concat(
            [df for df in (shopify_df, amazon_df) if not df.empty],
            ignore_index=True,
        )

        tpl_df = self._normalize_tpl(invoices)
        merged = self._merge_with_tpl(orders_df, tpl_df)
        result = self._compute_net_margin(merged)

        logger.info(
            f"Aggregated {len(result)} orders "
            f"({shopify_df.shape[0]} Shopify, {amazon_df.shape[0]} Amazon); "
            f"3PL matched: {result['tpl_matched'].sum()}/{len(result)}"
        )
        return result

    def to_transactions_df(self, aggregated: pd.DataFrame) -> pd.DataFrame:
        """Map the aggregated output to the column layout expected by FeaturePipeline.

        The :class:`~chrono_ltv.features.encoders.RFMFeatureExtractor` requires
        the following columns: ``customer_id``, ``transaction_id``,
        ``event_timestamp``, ``order_value``, ``num_items``, ``product_category``,
        ``is_returned``, ``discount_applied``.

        ``order_value`` is the clipped ``net_contribution_margin`` (floors at 0
        to avoid negative values confusing the monetary aggregation).
        """
        if aggregated.empty:
            return pd.DataFrame(
                columns=[
                    "customer_id",
                    "transaction_id",
                    "event_timestamp",
                    "order_value",
                    "num_items",
                    "product_category",
                    "is_returned",
                    "discount_applied",
                ]
            )

        is_returned = (
            aggregated["delivery_status"].eq("returned")
            if "delivery_status" in aggregated.columns
            else pd.Series(False, index=aggregated.index)
        )

        return pd.DataFrame(
            {
                "customer_id": aggregated["customer_id"],
                "transaction_id": aggregated["universal_order_id"],
                "event_timestamp": aggregated["order_date"],
                "order_value": aggregated["net_contribution_margin"].clip(lower=0.0),
                "num_items": 1,
                "product_category": aggregated["channel"],
                "is_returned": is_returned.fillna(False).astype(bool),
                "discount_applied": 0.0,
            }
        ).reset_index(drop=True)

    # ── private ───────────────────────────────────────────────────────────────

    def _normalize_shopify(
        self, records: list[ShopifyWebhookPayload]
    ) -> pd.DataFrame:
        """Convert Shopify payloads to the unified order schema."""
        if not records:
            return pd.DataFrame()

        rows = [
            {
                "universal_order_id": r.checkout_id,
                "source_order_id": r.order_id,
                "customer_id": r.customer_id,
                "channel": "shopify",
                "fulfillment_type": "shopify_own",
                "order_date": r.created_at,
                "gross_revenue": r.total_price,
                "platform_fees": 0.0,
                "financial_status": r.financial_status,
                "fulfillment_status": r.fulfillment_status or "unfulfilled",
            }
            for r in records
        ]
        return pd.DataFrame(rows)

    def _normalize_amazon(
        self, records: list[AmazonOrderReport]
    ) -> pd.DataFrame:
        """Convert Amazon order rows to the unified order schema."""
        if not records:
            return pd.DataFrame()

        rows = [
            {
                "universal_order_id": r.amazon_order_id,
                "source_order_id": r.amazon_order_id,
                # Use explicit customer_id if available, else fall back to
                # merchant_order_id, then use the Amazon order ID as a proxy.
                "customer_id": r.customer_id
                or r.merchant_order_id
                or r.amazon_order_id,
                "channel": "amazon",
                "fulfillment_type": "fba" if r.fulfillment_channel == "AFN" else "fbm",
                "order_date": r.purchase_date,
                "gross_revenue": r.item_price,
                "platform_fees": r.total_amazon_fees,
                "financial_status": (
                    "paid"
                    if r.order_status in ("Shipped", "Delivered")
                    else r.order_status.lower()
                ),
                "fulfillment_status": (
                    "fulfilled"
                    if r.order_status in ("Shipped", "Delivered")
                    else "unfulfilled"
                ),
            }
            for r in records
        ]
        return pd.DataFrame(rows)

    def _normalize_tpl(
        self, records: list[Logistics3PLInvoice]
    ) -> pd.DataFrame:
        """Convert 3PL invoices to the join-ready shipping cost table."""
        if not records:
            return pd.DataFrame()

        rows = [
            {
                "order_reference": r.order_reference,
                "tracking_number": r.tracking_number,
                "carrier": r.carrier,
                "shipping_cost": r.total_charge,
                "carrier_delay_status": r.carrier_delay_status,
                "delivery_status": r.delivery_status,
                "volumetric_weight_lbs": r.volumetric_weight_lbs,
                "zone": r.zone,
            }
            for r in records
        ]
        return pd.DataFrame(rows)

    def _merge_with_tpl(
        self, orders_df: pd.DataFrame, tpl_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Left-join orders with 3PL invoices; fill unmatched rows with safe defaults."""
        if tpl_df.empty:
            for col in _TPL_COLS:
                orders_df[col] = None
            orders_df["shipping_cost"] = 0.0
            orders_df["carrier_delay_status"] = False
            orders_df["tpl_matched"] = False
            return orders_df

        merged = orders_df.merge(
            tpl_df.rename(columns={"order_reference": "universal_order_id"}),
            on="universal_order_id",
            how="left",
        )
        merged["tpl_matched"] = merged["tracking_number"].notna()
        merged["shipping_cost"] = merged["shipping_cost"].fillna(0.0)
        merged["carrier_delay_status"] = merged["carrier_delay_status"].fillna(False)

        n_unmatched = int((~merged["tpl_matched"]).sum())
        if n_unmatched:
            logger.warning(
                f"{n_unmatched} order(s) have no matching 3PL invoice "
                f"— shipping_cost defaulted to 0"
            )
        return merged

    @staticmethod
    def _compute_net_margin(df: pd.DataFrame) -> pd.DataFrame:
        """Add ``net_contribution_margin = gross_revenue - platform_fees - shipping_cost``."""
        df = df.copy()
        df["net_contribution_margin"] = (
            df["gross_revenue"] - df["platform_fees"] - df["shipping_cost"]
        )
        return df
