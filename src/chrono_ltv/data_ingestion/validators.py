"""Great Expectations validation suites for the multi-source ingestion layer.

Three per-stream suites validate schema correctness and value ranges.
A fourth cross-stream check verifies that every fulfilled Shopify order
can be JOIN-matched with a 3PL tracking record — mismatches are reported
in the returned :class:`ValidationSummary` but never raise exceptions.

Requires: pip install -e ".[validation]"   (great-expectations >= 1.0)

Usage
-----
>>> from chrono_ltv.data_ingestion.validators import IngestionValidator
>>> validator = IngestionValidator()
>>> summary = validator.validate_shopify(shopify_df)
>>> join_summary = validator.validate_join_integrity(shopify_df, tpl_df)
>>> assert join_summary.success or not join_summary.failures  # silent mismatch
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chrono_ltv.data.validators import ValidationSummary
from chrono_ltv.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


class IngestionValidator:
    """GE-backed validation for Shopify, Amazon, and 3PL ingestion DataFrames.

    All ``validate_*`` methods accept a DataFrame containing flattened fields
    (i.e., the Pydantic models already serialised via ``model_dump()`` or
    equivalent).  They return a :class:`~chrono_ltv.data.validators.ValidationSummary`
    and never raise — failures are recorded in the summary object.
    """

    # ── public ────────────────────────────────────────────────────────────────

    def validate_shopify(self, df: pd.DataFrame) -> ValidationSummary:
        """Validate a DataFrame of flattened :class:`ShopifyWebhookPayload` records."""
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToNotBeNull,
        )

        expectations = [
            ExpectColumnToExist(column="checkout_id"),
            ExpectColumnToExist(column="order_id"),
            ExpectColumnToExist(column="customer_id"),
            ExpectColumnValuesToNotBeNull(column="checkout_id"),
            ExpectColumnValuesToNotBeNull(column="customer_id"),
            ExpectColumnValuesToNotBeNull(column="order_id"),
            ExpectColumnValuesToBeBetween(column="total_price", min_value=0.0, mostly=0.99),
            ExpectColumnValuesToBeBetween(
                column="subtotal_price", min_value=0.0, mostly=0.99
            ),
            ExpectColumnValuesToBeBetween(
                column="total_discounts", min_value=0.0, mostly=0.99
            ),
            ExpectColumnValuesToBeBetween(column="line_item_count", min_value=1, mostly=0.99),
            ExpectColumnValuesToBeInSet(
                column="financial_status",
                value_set=["paid", "pending", "refunded", "partially_refunded", "voided"],
            ),
        ]
        return self._run(df, expectations, "shopify")

    def validate_amazon(self, df: pd.DataFrame) -> ValidationSummary:
        """Validate a DataFrame of flattened :class:`AmazonOrderReport` records."""
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToNotBeNull,
        )

        expectations = [
            ExpectColumnToExist(column="amazon_order_id"),
            ExpectColumnValuesToNotBeNull(column="amazon_order_id"),
            ExpectColumnValuesToBeInSet(
                column="order_status",
                value_set=["Pending", "Unshipped", "Shipped", "Delivered", "Cancelled"],
            ),
            ExpectColumnValuesToBeInSet(
                column="fulfillment_channel",
                value_set=["AFN", "MFN"],
            ),
            ExpectColumnValuesToBeBetween(column="item_price", min_value=0.0, mostly=0.99),
            ExpectColumnValuesToBeBetween(column="fba_fee", min_value=0.0, mostly=0.99),
            ExpectColumnValuesToBeBetween(column="referral_fee", min_value=0.0, mostly=0.99),
            ExpectColumnValuesToBeBetween(column="quantity", min_value=1, mostly=0.99),
        ]
        return self._run(df, expectations, "amazon")

    def validate_tpl(self, df: pd.DataFrame) -> ValidationSummary:
        """Validate a DataFrame of flattened :class:`Logistics3PLInvoice` records."""
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToNotBeNull,
        )

        expectations = [
            ExpectColumnToExist(column="invoice_id"),
            ExpectColumnToExist(column="tracking_number"),
            ExpectColumnToExist(column="order_reference"),
            ExpectColumnValuesToNotBeNull(column="invoice_id"),
            ExpectColumnValuesToNotBeNull(column="tracking_number"),
            ExpectColumnValuesToNotBeNull(column="order_reference"),
            ExpectColumnValuesToBeBetween(column="weight_lbs", min_value=0.0, mostly=0.99),
            ExpectColumnValuesToBeBetween(column="base_rate", min_value=0.0, mostly=0.99),
            ExpectColumnValuesToBeBetween(column="total_charge", min_value=0.0, mostly=0.99),
            ExpectColumnValuesToBeBetween(column="zone", min_value=1, max_value=8, mostly=0.99),
            ExpectColumnValuesToBeInSet(
                column="delivery_status",
                value_set=["in_transit", "delivered", "exception", "returned"],
            ),
        ]
        return self._run(df, expectations, "tpl")

    def validate_join_integrity(
        self,
        shopify_df: pd.DataFrame,
        tpl_df: pd.DataFrame,
        fulfilled_statuses: list[str] | None = None,
    ) -> ValidationSummary:
        """Check that every fulfilled Shopify order has a matching 3PL record.

        This is a pure-pandas cross-stream check (GE cannot join two DataFrames).
        Mismatches are logged and captured in the returned summary — no exception
        is ever raised, letting the pipeline continue with partial data.

        Parameters
        ----------
        shopify_df:
            Flattened Shopify records; must contain ``checkout_id`` and
            ``fulfillment_status`` columns.
        tpl_df:
            Flattened 3PL invoice records; must contain ``order_reference``.
        fulfilled_statuses:
            Which ``fulfillment_status`` values count as fulfilled.
            Defaults to ``["fulfilled"]``.
        """
        if fulfilled_statuses is None:
            fulfilled_statuses = ["fulfilled"]

        dataset_name = "shopify_tpl_join_integrity"

        try:
            fulfilled_mask = shopify_df["fulfillment_status"].isin(fulfilled_statuses)
            fulfilled_ids: set[str] = set(
                shopify_df.loc[fulfilled_mask, "checkout_id"].astype(str)
            )
        except KeyError as exc:
            return ValidationSummary(
                dataset=dataset_name,
                success=False,
                n_expectations=0,
                n_passed=0,
                n_failed=1,
                failures=[f"Missing required column in shopify_df: {exc}"],
            )

        try:
            tpl_refs: set[str] = set(tpl_df["order_reference"].astype(str))
        except KeyError as exc:
            return ValidationSummary(
                dataset=dataset_name,
                success=False,
                n_expectations=len(fulfilled_ids),
                n_passed=0,
                n_failed=len(fulfilled_ids),
                failures=[f"Missing required column in tpl_df: {exc}"],
            )

        unmatched = sorted(fulfilled_ids - tpl_refs)
        n_total = len(fulfilled_ids)
        n_failed = len(unmatched)
        n_passed = n_total - n_failed

        failures = [
            f"fulfilled order has no 3PL invoice: checkout_id={oid}"
            for oid in unmatched[:50]  # cap to avoid bloated summary objects
        ]

        if unmatched:
            logger.warning(
                f"JOIN integrity: {n_failed}/{n_total} fulfilled Shopify orders "
                f"have no matching 3PL record — pipeline continues with partial data"
            )
        else:
            logger.info(
                f"JOIN integrity: all {n_total} fulfilled Shopify orders matched "
                f"to a 3PL invoice"
            )

        return ValidationSummary(
            dataset=dataset_name,
            success=n_failed == 0,
            n_expectations=n_total,
            n_passed=n_passed,
            n_failed=n_failed,
            failures=failures,
        )

    def validate_all(
        self,
        shopify_df: pd.DataFrame,
        amazon_df: pd.DataFrame,
        tpl_df: pd.DataFrame,
    ) -> dict[str, ValidationSummary]:
        """Run all three stream validators and the cross-stream join check."""
        results: dict[str, ValidationSummary] = {}
        for name, runner, df in (
            ("shopify", self.validate_shopify, shopify_df),
            ("amazon", self.validate_amazon, amazon_df),
            ("tpl", self.validate_tpl, tpl_df),
        ):
            summary: ValidationSummary = runner(df)
            results[name] = summary
            icon = "✓" if summary.success else "✗"
            logger.info(
                f"[{icon}] {name}: {summary.n_passed}/{summary.n_expectations} "
                f"expectations passed  (pass_rate={summary.pass_rate:.1%})"
            )

        results["join_integrity"] = self.validate_join_integrity(shopify_df, tpl_df)
        return results

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _run(df: pd.DataFrame, expectations: list[Any], name: str) -> ValidationSummary:
        """Execute *expectations* against *df* in an ephemeral GX context."""
        import great_expectations as gx

        context = gx.get_context(mode="ephemeral")
        ds = context.data_sources.add_pandas("ingestion")
        asset = ds.add_dataframe_asset(name)
        batch_def = asset.add_batch_definition_whole_dataframe(f"{name}_batch")

        suite = context.suites.add(gx.ExpectationSuite(name=f"{name}_suite"))
        for exp in expectations:
            suite.add_expectation(exp)

        validation_def = context.validation_definitions.add(
            gx.ValidationDefinition(
                name=f"{name}_validation",
                data=batch_def,
                suite=suite,
            )
        )
        result = validation_def.run(batch_parameters={"dataframe": df})
        return IngestionValidator._to_summary(name, result)

    @staticmethod
    def _to_summary(name: str, result: Any) -> ValidationSummary:
        """Convert a GX ValidationResult into a ValidationSummary."""
        n_total = len(result.results)
        n_passed = sum(1 for r in result.results if r.success)
        n_failed = n_total - n_passed
        failures: list[str] = []
        for r in result.results:
            if not r.success:
                cfg = r.expectation_config
                exp_type: str = str(
                    getattr(cfg, "type", None) or getattr(cfg, "expectation_type", "unknown")
                )
                kwargs: dict[str, Any] = getattr(cfg, "kwargs", {}) or {}
                col = kwargs.get("column", "?")
                failures.append(f"{exp_type}(column={col})")
        return ValidationSummary(
            dataset=name,
            success=bool(result.success),
            n_expectations=n_total,
            n_passed=n_passed,
            n_failed=n_failed,
            failures=failures,
        )
