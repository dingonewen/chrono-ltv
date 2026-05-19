"""Great Expectations validation suites for all five ChronoLTV datasets.

Requires: pip install -e ".[validation]"   (great-expectations >= 1.0)

Usage
-----
>>> from chrono_ltv.data.validators import DataValidator
>>> validator = DataValidator()
>>> summaries = validator.validate_all(datasets)
>>> assert all(s.success for s in summaries.values())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ValidationSummary:
    """Structured result for a single dataset's validation run."""

    dataset: str
    success: bool
    n_expectations: int
    n_passed: int
    n_failed: int
    failures: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.n_expectations == 0:
            return 1.0
        return self.n_passed / self.n_expectations


class DataValidator:
    """Runs GE expectation suites against all ChronoLTV datasets.

    Each public ``validate_*`` method accepts a DataFrame and returns a
    ``ValidationSummary``.  ``validate_all`` is the pipeline entry-point.
    """

    # ── public ───────────────────────────────────────────────────────────

    def validate_all(self, datasets: dict[str, pd.DataFrame]) -> dict[str, ValidationSummary]:
        """Validate every dataset; return per-name summaries."""
        runners: dict[str, Any] = {
            "customers": self.validate_customers,
            "transactions": self.validate_transactions,
            "clickstream": self.validate_clickstream,
            "support_tickets": self.validate_support_tickets,
            "survival_labels": self.validate_survival_labels,
        }
        results: dict[str, ValidationSummary] = {}
        for name, runner in runners.items():
            df = datasets.get(name)
            if df is None:
                logger.warning(f"Dataset '{name}' not found — skipping validation.")
                continue
            summary: ValidationSummary = runner(df)
            results[name] = summary
            icon = "✓" if summary.success else "✗"
            logger.info(
                f"[{icon}] {name}: {summary.n_passed}/{summary.n_expectations} "
                f"expectations passed  (pass_rate={summary.pass_rate:.1%})"
            )
        return results

    def validate_customers(self, df: pd.DataFrame) -> ValidationSummary:
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToBeUnique,
            ExpectColumnValuesToNotBeNull,
        )

        expectations = [
            ExpectColumnToExist(column="customer_id"),
            ExpectColumnValuesToNotBeNull(column="customer_id"),
            ExpectColumnValuesToBeUnique(column="customer_id"),
            ExpectColumnValuesToBeInSet(
                column="acquisition_channel",
                value_set=[
                    "organic_search",
                    "paid_search",
                    "social_media",
                    "email",
                    "referral",
                    "direct",
                ],
            ),
            ExpectColumnValuesToBeInSet(
                column="gender",
                value_set=["M", "F", "Non-binary", "Unknown"],
            ),
            ExpectColumnValuesToBeInSet(
                column="loyalty_tier",
                value_set=["Bronze", "Silver", "Gold", "Platinum"],
            ),
            # mostly=0.99 — age can be NULL for a small fraction of records
            ExpectColumnValuesToBeBetween(column="age", min_value=18, max_value=100, mostly=0.99),
        ]
        return self._run(df, expectations, "customers")

    def validate_transactions(self, df: pd.DataFrame) -> ValidationSummary:
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToNotBeNull,
        )

        expectations = [
            ExpectColumnToExist(column="transaction_id"),
            ExpectColumnValuesToNotBeNull(column="transaction_id"),
            ExpectColumnValuesToBeInSet(
                column="payment_method",
                value_set=[
                    "credit_card",
                    "debit_card",
                    "paypal",
                    "crypto",
                    "buy_now_pay_later",
                ],
            ),
            # mostly < 1.0 — noise injection adds negatives and outliers
            ExpectColumnValuesToBeBetween(column="order_value", min_value=0.0, mostly=0.95),
            ExpectColumnValuesToBeBetween(
                column="discount_applied", min_value=0.0, max_value=1.0, mostly=0.99
            ),
            ExpectColumnValuesToBeBetween(column="num_items", min_value=1, mostly=0.98),
        ]
        return self._run(df, expectations, "transactions")

    def validate_clickstream(self, df: pd.DataFrame) -> ValidationSummary:
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToNotBeNull,
        )

        expectations = [
            ExpectColumnToExist(column="session_id"),
            ExpectColumnValuesToNotBeNull(column="session_id"),
            ExpectColumnValuesToBeInSet(
                column="page_type",
                value_set=[
                    "home",
                    "category",
                    "product",
                    "cart",
                    "checkout",
                    "confirmation",
                    "search",
                    "account",
                ],
            ),
            ExpectColumnValuesToBeInSet(
                column="device_type",
                value_set=["desktop", "mobile", "tablet"],
            ),
            ExpectColumnValuesToBeBetween(
                column="time_on_page_seconds", min_value=0.0, mostly=0.99
            ),
        ]
        return self._run(df, expectations, "clickstream")

    def validate_support_tickets(self, df: pd.DataFrame) -> ValidationSummary:
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToNotBeNull,
        )

        expectations = [
            ExpectColumnToExist(column="ticket_id"),
            ExpectColumnValuesToNotBeNull(column="ticket_id"),
            ExpectColumnValuesToNotBeNull(column="raw_text"),
            ExpectColumnValuesToBeBetween(
                column="sentiment_score",
                min_value=-1.0,
                max_value=1.0,
                mostly=0.99,
            ),
        ]
        return self._run(df, expectations, "support_tickets")

    def validate_survival_labels(self, df: pd.DataFrame) -> ValidationSummary:
        from great_expectations.expectations import (
            ExpectColumnToExist,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeUnique,
        )

        expectations = [
            ExpectColumnToExist(column="customer_id"),
            ExpectColumnValuesToBeUnique(column="customer_id"),
            ExpectColumnValuesToBeBetween(column="duration_days", min_value=0.0),
            ExpectColumnValuesToBeBetween(column="total_orders", min_value=1),
            ExpectColumnValuesToBeBetween(column="total_revenue", min_value=0.0),
        ]
        return self._run(df, expectations, "survival_labels")

    # ── private ──────────────────────────────────────────────────────────

    @staticmethod
    def _run(df: pd.DataFrame, expectations: list[Any], name: str) -> ValidationSummary:
        """Execute *expectations* against *df* in an ephemeral GX context."""
        import great_expectations as gx

        context = gx.get_context(mode="ephemeral")
        ds = context.data_sources.add_pandas("chrono_ltv")
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
        return DataValidator._to_summary(name, result)

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
                # .type in GX 1.x; .expectation_type in 0.18.x
                exp_type: str = getattr(cfg, "type", None) or getattr(
                    cfg, "expectation_type", "unknown"
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
