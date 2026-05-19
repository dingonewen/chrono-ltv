"""Data Stream Simulator for the ChronoLTV pipeline.

Generates statistically realistic, survival-analysis-ready customer datasets
that mimic a DTC e-commerce platform at Shopify transaction scale (~10k
customers, ~150k transactions).  Every dataset contains intentional noise
(missing values, duplicates, outliers, future-dated records) so that
downstream Great Expectations validators and drift monitors have real work
to do.

Architecture
------------
EcommerceSimulator            — orchestrates the full simulation
  ├── _CustomerFactory        — synthesises customer master records
  ├── _TransactionFactory     — synthesises purchase events per customer
  ├── _ClickstreamFactory     — synthesises navigation / session events
  ├── _SupportTicketFactory   — synthesises CRM support ticket text
  ├── _SurvivalLabelBuilder   — derives ground-truth survival targets
  └── _NoiseInjector          — injects configurable data-quality faults
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

from chrono_ltv.utils.io import save_parquet

# ---------------------------------------------------------------------------
# Configuration dataclass (mirrors conf/data/simulator.yaml)
# ---------------------------------------------------------------------------


@dataclass
class NoiseConfig:
    missing_rate: float = 0.04
    duplicate_rate: float = 0.01
    outlier_rate: float = 0.02
    future_date_rate: float = 0.005
    negative_amount_rate: float = 0.005


@dataclass
class SimulatorConfig:
    n_customers: int = 10_000
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    random_seed: int = 42
    churn_threshold_days: int = 90
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    output_dir: Path = Path("data/raw")

    # Catalogue
    product_categories: list[str] = field(default_factory=lambda: [
        "Electronics", "Clothing", "Home & Garden", "Beauty",
        "Sports", "Books", "Food & Beverage", "Toys",
    ])
    acquisition_channels: dict[str, float] = field(default_factory=lambda: {
        "organic_search": 0.30, "paid_search": 0.20, "social_media": 0.18,
        "email": 0.12, "referral": 0.10, "direct": 0.10,
    })
    support_topics: list[str] = field(default_factory=lambda: [
        "delayed shipping", "wrong item delivered", "refund request",
        "product defect", "account access issue", "billing discrepancy",
        "product inquiry", "general feedback",
    ])

    def __post_init__(self) -> None:
        self.start_dt = datetime.fromisoformat(self.start_date)
        self.end_dt = datetime.fromisoformat(self.end_date)
        self.observation_days = (self.end_dt - self.start_dt).days


# ---------------------------------------------------------------------------
# Internal factory helpers
# ---------------------------------------------------------------------------


class _CustomerFactory:
    """Generates the customer master table."""

    _LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
    _TIER_WEIGHTS = [0.50, 0.28, 0.15, 0.07]
    _GENDERS = ["M", "F", "Non-binary", "Unknown"]
    _GENDER_WEIGHTS = [0.46, 0.46, 0.05, 0.03]

    def __init__(self, cfg: SimulatorConfig, rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._rng = rng
        self._fake = Faker()
        Faker.seed(cfg.random_seed)

    def build(self) -> pd.DataFrame:
        logger.info(f"Generating {self._cfg.n_customers:,} customer records …")
        channels = list(self._cfg.acquisition_channels.keys())
        weights = list(self._cfg.acquisition_channels.values())

        records = []
        for _ in range(self._cfg.n_customers):
            reg_offset = self._rng.integers(0, self._cfg.observation_days - 1)
            reg_date = self._cfg.start_dt + timedelta(days=int(reg_offset))
            records.append({
                "customer_id": str(uuid.uuid4()),
                "first_name": self._fake.first_name(),
                "last_name": self._fake.last_name(),
                "email": self._fake.email(),
                "country": self._fake.country_code(representation="alpha-2"),
                "acquisition_channel": self._rng.choice(channels, p=weights),
                "registration_date": reg_date,
                "age": int(self._rng.integers(18, 80)),
                "gender": self._rng.choice(self._GENDERS, p=self._GENDER_WEIGHTS),
                "loyalty_tier": self._rng.choice(
                    self._LOYALTY_TIERS, p=self._TIER_WEIGHTS
                ),
            })
        return pd.DataFrame(records)


class _TransactionFactory:
    """Generates purchase events, modelling per-customer purchase intensity
    via a Negative-Binomial process (realistic heavy-tail order distribution).
    """

    _PAYMENT_METHODS = [
        "credit_card", "debit_card", "paypal", "crypto", "buy_now_pay_later",
    ]
    _PAYMENT_WEIGHTS = [0.40, 0.25, 0.20, 0.05, 0.10]

    def __init__(self, cfg: SimulatorConfig, rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._rng = rng

    def build(self, customers: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating transactions …")
        rows: list[dict[str, Any]] = []

        for _, cust in customers.iterrows():
            reg_date: datetime = pd.Timestamp(cust["registration_date"]).to_pydatetime()
            days_active = max(1, (self._cfg.end_dt - reg_date).days)

            # Negative-Binomial: mean=12 orders/year, overdispersion=0.4
            n_orders = int(
                self._rng.negative_binomial(n=5, p=0.30)
                * (days_active / 365)
            )
            n_orders = max(1, n_orders)

            # Spread orders randomly over the active window
            offsets = sorted(self._rng.integers(0, days_active, size=n_orders).tolist())
            for offset in offsets:
                event_dt = reg_date + timedelta(days=int(offset))
                # Order value ~ log-normal (μ=4.2, σ=0.8) → median ≈ $67
                order_value = float(self._rng.lognormal(mean=4.2, sigma=0.8))
                rows.append({
                    "transaction_id": str(uuid.uuid4()),
                    "customer_id": cust["customer_id"],
                    "event_timestamp": event_dt,
                    "order_value": round(order_value, 2),
                    "num_items": int(self._rng.integers(1, 8)),
                    "product_category": self._rng.choice(self._cfg.product_categories),
                    "payment_method": self._rng.choice(
                        self._PAYMENT_METHODS, p=self._PAYMENT_WEIGHTS
                    ),
                    "is_returned": bool(self._rng.binomial(1, 0.08)),
                    "discount_applied": round(
                        float(self._rng.beta(1.5, 8.0)), 3
                    ),  # right-skewed towards 0
                })

        logger.info(f"Generated {len(rows):,} transaction records.")
        return pd.DataFrame(rows)


class _ClickstreamFactory:
    """Generates clickstream / session events correlated with transactions."""

    _PAGE_TYPES = [
        "home", "category", "product", "cart",
        "checkout", "confirmation", "search", "account",
    ]
    _PAGE_WEIGHTS = [0.15, 0.20, 0.30, 0.12, 0.08, 0.03, 0.10, 0.02]
    _DEVICES = ["desktop", "mobile", "tablet"]
    _DEVICE_WEIGHTS = [0.45, 0.45, 0.10]

    def __init__(self, cfg: SimulatorConfig, rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._rng = rng

    def build(self, transactions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating clickstream events …")
        rows: list[dict[str, Any]] = []

        for _, txn in transactions.iterrows():
            # Each transaction generates 3-15 page views in the session
            n_clicks = int(self._rng.integers(3, 16))
            session_id = str(uuid.uuid4())
            txn_dt: datetime = pd.Timestamp(txn["event_timestamp"]).to_pydatetime()
            device = self._rng.choice(self._DEVICES, p=self._DEVICE_WEIGHTS)

            for click_idx in range(n_clicks):
                page = self._rng.choice(self._PAGE_TYPES, p=self._PAGE_WEIGHTS)
                time_offset = timedelta(seconds=int(self._rng.integers(0, 3600)))
                rows.append({
                    "session_id": session_id,
                    "customer_id": txn["customer_id"],
                    "event_timestamp": txn_dt - timedelta(hours=1) + time_offset,
                    "page_type": page,
                    "time_on_page_seconds": round(
                        float(self._rng.exponential(scale=90.0)), 1
                    ),
                    "device_type": device,
                    "added_to_cart": bool(
                        page == "product" and self._rng.binomial(1, 0.25)
                    ),
                    "search_query": (
                        self._rng.choice(self._cfg.product_categories)
                        if page == "search"
                        else None
                    ),
                })

        logger.info(f"Generated {len(rows):,} clickstream records.")
        return pd.DataFrame(rows)


class _SupportTicketFactory:
    """Synthesises CRM support ticket text using templates.

    In production, embeddings are computed by a sentence-transformer.
    Here we generate the raw text; the embedding step is a separate
    feature-engineering stage so it can be swapped or cached cheaply.
    """

    # Topic → sentiment bias (negative = dissatisfied customer)
    _TOPIC_SENTIMENT: dict[str, float] = {
        "delayed shipping":       -0.6,
        "wrong item delivered":   -0.7,
        "refund request":         -0.5,
        "product defect":         -0.8,
        "account access issue":   -0.3,
        "billing discrepancy":    -0.5,
        "product inquiry":        +0.1,
        "general feedback":       +0.2,
    }

    # Template sentences per topic
    _TEMPLATES: dict[str, list[str]] = {
        "delayed shipping": [
            "My order #{order_id} placed {days} days ago still hasn't arrived.",
            "Where is my package? It's been {days} days since I ordered.",
            "Tracking shows no updates for {days} days on order #{order_id}.",
        ],
        "wrong item delivered": [
            "I received the wrong product in my order #{order_id}.",
            "Order #{order_id}: I ordered {category} but got something else.",
            "Please help—wrong item was sent. Order #{order_id}.",
        ],
        "refund request": [
            "I would like to return my order #{order_id} and get a full refund.",
            "Requesting a refund for #{order_id}. The item did not meet expectations.",
            "How do I process a return for order #{order_id}?",
        ],
        "product defect": [
            "The {category} I received in order #{order_id} is defective.",
            "My order #{order_id} arrived damaged. The {category} is broken.",
            "There's a quality issue with the {category} from order #{order_id}.",
        ],
        "account access issue": [
            "I can't log into my account. Please reset my password.",
            "My account seems locked. I've been unable to access it for {days} days.",
            "Two-factor authentication is not working for my account.",
        ],
        "billing discrepancy": [
            "I was charged twice for order #{order_id}.",
            "My invoice for #{order_id} shows an incorrect amount.",
            "There's an unauthorised charge of ${amount} on my account.",
        ],
        "product inquiry": [
            "Does the {category} come in different sizes or colours?",
            "What is the warranty on the {category} product?",
            "Is the {category} compatible with model XYZ?",
        ],
        "general feedback": [
            "I love your {category} products. Very satisfied with my purchase.",
            "Your website is easy to navigate and checkout was smooth.",
            "Delivery was faster than expected. Great experience overall.",
        ],
    }

    def __init__(self, cfg: SimulatorConfig, rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._rng = rng

    def build(self, customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating support tickets …")

        # Only ~15 % of customers ever open a ticket
        n_tickets_per_customer = self._rng.negative_binomial(n=1, p=0.85, size=len(customers))
        rows: list[dict[str, Any]] = []

        cust_to_txns = transactions.groupby("customer_id")

        for (_, cust), n_tickets in zip(customers.iterrows(), n_tickets_per_customer):
            if n_tickets == 0:
                continue
            cust_txns = cust_to_txns.get_group(cust["customer_id"]) if cust["customer_id"] in cust_to_txns.groups else None
            for _ in range(n_tickets):
                topic = self._rng.choice(self._cfg.support_topics)
                sentiment_base = self._TOPIC_SENTIMENT.get(topic, 0.0)
                sentiment = float(
                    np.clip(self._rng.normal(sentiment_base, 0.15), -1.0, 1.0)
                )
                # Pick a random transaction to reference (if one exists)
                ref_order = (
                    cust_txns.sample(1, random_state=int(self._rng.integers(0, 2**31))).iloc[0]
                    if cust_txns is not None
                    else None
                )
                raw_text = self._render_template(topic, ref_order, self._cfg.product_categories)

                # Ticket created 1-14 days after the referenced transaction
                base_dt = (
                    pd.Timestamp(ref_order["event_timestamp"]).to_pydatetime()
                    if ref_order is not None
                    else self._cfg.start_dt
                )
                created_at = base_dt + timedelta(days=int(self._rng.integers(1, 15)))

                resolved = bool(self._rng.binomial(1, 0.82))
                rows.append({
                    "ticket_id": str(uuid.uuid4()),
                    "customer_id": cust["customer_id"],
                    "created_at": created_at,
                    "topic": topic,
                    "raw_text": raw_text,
                    "sentiment_score": round(sentiment, 4),
                    "resolved": resolved,
                    "resolution_days": (
                        round(float(self._rng.exponential(scale=3.0)), 1)
                        if resolved
                        else None
                    ),
                })

        logger.info(f"Generated {len(rows):,} support ticket records.")
        return pd.DataFrame(rows)

    def _render_template(
        self,
        topic: str,
        txn: pd.Series | None,
        categories: list[str],
    ) -> str:
        templates = self._TEMPLATES.get(topic, ["I need help with my order."])
        template = random.choice(templates)  # noqa: S311  (non-cryptographic use)
        return template.format(
            order_id=str(uuid.uuid4())[:8].upper(),
            days=random.randint(1, 14),  # noqa: S311
            category=random.choice(categories),  # noqa: S311
            amount=round(random.uniform(10, 500), 2),  # noqa: S311
        )


class _SurvivalLabelBuilder:
    """Derives ground-truth survival analysis labels from the transaction log.

    duration_days  : days from first purchase to either the churn event
                     (last purchase + churn_threshold) or the observation
                     end date (whichever comes first).
    event_observed : True if the customer churned within the window.
    """

    def __init__(self, cfg: SimulatorConfig) -> None:
        self._cfg = cfg

    def build(self, transactions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Deriving survival labels …")

        agg = (
            transactions.groupby("customer_id")
            .agg(
                first_purchase_date=("event_timestamp", "min"),
                last_purchase_date=("event_timestamp", "max"),
                total_orders=("transaction_id", "count"),
                total_revenue=("order_value", "sum"),
            )
            .reset_index()
        )

        records = []
        for _, row in agg.iterrows():
            last_purchase = pd.Timestamp(row["last_purchase_date"]).to_pydatetime()
            churn_date = last_purchase + timedelta(days=self._cfg.churn_threshold_days)
            churned = churn_date <= self._cfg.end_dt

            first_purchase = pd.Timestamp(row["first_purchase_date"]).to_pydatetime()
            end_point = churn_date if churned else self._cfg.end_dt
            duration = max(0.0, (end_point - first_purchase).days)

            records.append({
                "customer_id": row["customer_id"],
                "first_purchase_date": first_purchase,
                "last_purchase_date": last_purchase,
                "duration_days": duration,
                "event_observed": churned,
                "total_orders": int(row["total_orders"]),
                "total_revenue": round(float(row["total_revenue"]), 2),
            })

        df = pd.DataFrame(records)
        churn_rate = df["event_observed"].mean()
        logger.info(
            f"Survival labels: {len(df):,} customers, "
            f"churn rate={churn_rate:.1%}, "
            f"median duration={df['duration_days'].median():.0f} days"
        )
        return df


class _NoiseInjector:
    """Injects configurable data-quality faults into DataFrames.

    Each fault type can be individually enabled/disabled via ``NoiseConfig``.
    Faults are tracked so tests can assert on injection counts.
    """

    def __init__(self, cfg: NoiseConfig, rng: np.random.Generator) -> None:
        self._cfg = cfg
        self._rng = rng
        self.fault_report: dict[str, int] = {}

    def inject(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Apply all configured noise types and return the mutated DataFrame."""
        df = df.copy()
        n = len(df)

        df, n_missing = self._inject_missing(df, n)
        df, n_dupes = self._inject_duplicates(df, n)
        df, n_outliers = self._inject_outliers(df, n)
        df, n_future = self._inject_future_dates(df, n)
        df, n_negative = self._inject_negative_amounts(df, n)

        self.fault_report[dataset_name] = {
            "missing": n_missing,
            "duplicates": n_dupes,
            "outliers": n_outliers,
            "future_dates": n_future,
            "negative_amounts": n_negative,
        }
        logger.info(
            f"[{dataset_name}] Noise injected — "
            f"missing={n_missing}, dupes={n_dupes}, outliers={n_outliers}, "
            f"future_dates={n_future}, neg_amounts={n_negative}"
        )
        return df

    # ── private helpers ───────────────────────────────────────────────────

    def _inject_missing(self, df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, int]:
        if self._cfg.missing_rate <= 0:
            return df, 0
        # Only target nullable columns (skip IDs and dates)
        nullable_cols = [
            c for c in df.columns
            if c not in {"customer_id", "transaction_id", "session_id",
                         "ticket_id", "event_timestamp", "registration_date"}
            and df[c].dtype == object or df[c].dtype in [np.float64, np.int64]
        ]
        if not nullable_cols:
            return df, 0
        n_cells = int(n * len(nullable_cols) * self._cfg.missing_rate)
        rows = self._rng.integers(0, n, size=n_cells)
        cols = self._rng.choice(nullable_cols, size=n_cells)
        for r, c in zip(rows, cols):
            df.at[df.index[r], c] = np.nan
        return df, n_cells

    def _inject_duplicates(self, df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, int]:
        if self._cfg.duplicate_rate <= 0:
            return df, 0
        n_dupes = max(1, int(n * self._cfg.duplicate_rate))
        dupe_idx = self._rng.choice(df.index, size=n_dupes, replace=False)
        dupes = df.loc[dupe_idx].copy()
        df = pd.concat([df, dupes], ignore_index=True)
        return df, n_dupes

    def _inject_outliers(self, df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, int]:
        if self._cfg.outlier_rate <= 0:
            return df, 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df, 0
        n_outliers = max(1, int(n * self._cfg.outlier_rate))
        rows = self._rng.integers(0, n, size=n_outliers)
        cols = self._rng.choice(numeric_cols, size=n_outliers)
        for r, c in zip(rows, cols):
            col_max = df[c].max()
            df.at[df.index[r], c] = col_max * self._rng.uniform(10, 100)
        return df, n_outliers

    def _inject_future_dates(self, df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, int]:
        if self._cfg.future_date_rate <= 0:
            return df, 0
        date_cols = [c for c in df.columns if "timestamp" in c or "date" in c]
        if not date_cols:
            return df, 0
        n_future = max(1, int(n * self._cfg.future_date_rate))
        rows = self._rng.integers(0, n, size=n_future)
        col = date_cols[0]
        future_offset = timedelta(days=365 * 5)
        for r in rows:
            current = df.at[df.index[r], col]
            if isinstance(current, datetime):
                df.at[df.index[r], col] = current + future_offset
        return df, n_future

    def _inject_negative_amounts(self, df: pd.DataFrame, n: int) -> tuple[pd.DataFrame, int]:
        if self._cfg.negative_amount_rate <= 0:
            return df, 0
        amount_cols = [c for c in df.columns if "value" in c or "revenue" in c or "amount" in c]
        if not amount_cols:
            return df, 0
        n_neg = max(1, int(n * self._cfg.negative_amount_rate))
        rows = self._rng.integers(0, n, size=n_neg)
        col = amount_cols[0]
        for r in rows:
            df.at[df.index[r], col] = -abs(df.at[df.index[r], col])
        return df, n_neg


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class EcommerceSimulator:
    """Orchestrates the full simulation and persists results as Parquet.

    Usage
    -----
    >>> cfg = SimulatorConfig(n_customers=1_000)
    >>> sim = EcommerceSimulator(cfg)
    >>> datasets = sim.run()
    """

    def __init__(self, cfg: SimulatorConfig | None = None) -> None:
        self._cfg = cfg or SimulatorConfig()
        np.random.seed(self._cfg.random_seed)
        random.seed(self._cfg.random_seed)
        self._rng = np.random.default_rng(self._cfg.random_seed)

        self._customer_factory = _CustomerFactory(self._cfg, self._rng)
        self._transaction_factory = _TransactionFactory(self._cfg, self._rng)
        self._clickstream_factory = _ClickstreamFactory(self._cfg, self._rng)
        self._ticket_factory = _SupportTicketFactory(self._cfg, self._rng)
        self._label_builder = _SurvivalLabelBuilder(self._cfg)
        self._noise_injector = _NoiseInjector(self._cfg.noise, self._rng)

    # ── public API ────────────────────────────────────────────────────────

    def run(self, *, persist: bool = True) -> dict[str, pd.DataFrame]:
        """Run the full simulation.

        Parameters
        ----------
        persist:
            If True, each dataset is saved to ``cfg.output_dir`` as Parquet.

        Returns
        -------
        dict mapping dataset name → DataFrame (post-noise-injection).
        """
        logger.info(
            f"Starting simulation: n_customers={self._cfg.n_customers:,}, "
            f"window={self._cfg.start_date}→{self._cfg.end_date}"
        )

        # 1. Generate clean data
        customers = self._customer_factory.build()
        transactions = self._transaction_factory.build(customers)
        clickstream = self._clickstream_factory.build(transactions)
        tickets = self._ticket_factory.build(customers, transactions)
        labels = self._label_builder.build(transactions)

        # 2. Inject noise (customers table is kept clean — it's the master key)
        noisy_transactions = self._noise_injector.inject(transactions, "transactions")
        noisy_clickstream = self._noise_injector.inject(clickstream, "clickstream")
        noisy_tickets = self._noise_injector.inject(tickets, "support_tickets")

        datasets: dict[str, pd.DataFrame] = {
            "customers": customers,
            "transactions": noisy_transactions,
            "clickstream": noisy_clickstream,
            "support_tickets": noisy_tickets,
            "survival_labels": labels,
        }

        # 3. Persist
        if persist:
            self._persist(datasets)

        logger.info("Simulation complete.")
        self._log_summary(datasets)
        return datasets

    def fault_report(self) -> dict[str, Any]:
        """Return the noise-injection fault report from the last ``run()``."""
        return self._noise_injector.fault_report

    # ── private helpers ───────────────────────────────────────────────────

    def _persist(self, datasets: dict[str, pd.DataFrame]) -> None:
        output_dir = Path(self._cfg.output_dir)
        file_map = {
            "customers": "customers.parquet",
            "transactions": "transactions.parquet",
            "clickstream": "clickstream.parquet",
            "support_tickets": "support_tickets.parquet",
            "survival_labels": "survival_labels.parquet",
        }
        for name, df in datasets.items():
            save_parquet(df, output_dir / file_map[name])

    @staticmethod
    def _log_summary(datasets: dict[str, pd.DataFrame]) -> None:
        logger.info("── Dataset Summary ─────────────────────────────────")
        for name, df in datasets.items():
            logger.info(f"  {name:<20} {len(df):>10,} rows × {len(df.columns):>3} cols")
        labels = datasets["survival_labels"]
        churn_rate = labels["event_observed"].mean()
        logger.info(f"  Observed churn rate : {churn_rate:.1%}")
        logger.info(f"  Median LTV (revenue): ${labels['total_revenue'].median():,.2f}")
