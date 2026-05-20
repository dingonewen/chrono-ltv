"""ChronoLTV Streamlit Dashboard.

Three-tab interactive analytics for the ChronoLTV pipeline.

Tabs
----
1. Multi-Source Reconciliation — Shopify + Amazon + 3PL fee waterfall
2. Survival Churn Radar        — per-customer daily drop-off curves
3. ROI Intervention Simulator  — DCF LTV with coupon / discount sliders

Run
---
    streamlit run src/chrono_ltv/dashboard.py
    # or
    make dashboard

Data loading strategy (production-grade with graceful fallback)
---------------------------------------------------------------
Tab 1: real  → data/raw/transactions.parquet (mapped to order schema)
              fallback → synthetic MultiSourceAggregator data
Tab 2: real  → 1) MLflow model (local mlruns or MLFLOW_TRACKING_URI)
                  + feature matrix from data/raw/ parquet files
               2) Fresh CoxPH fit on real parquet features (if no MLflow model)
              fallback → fresh CoxPH fit on synthetic simulator data

Environment variables
---------------------
CHRONO_DATA_DIR          override data/raw/ path (default: data/raw)
MLFLOW_TRACKING_URI      override mlruns local store  (default: mlruns)
MLFLOW_EXPERIMENT_NAME   override experiment name     (default: chrono-ltv-survival)

Requires
--------
    pip install -e ".[dashboard]"           # streamlit + matplotlib
    pip install -e ".[survival,features]"   # for the Churn Radar tab
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Make chrono_ltv importable when running on Streamlit Cloud (no editable install)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

# ── production data paths (overridable via env vars) ─────────────────────────

_DATA_DIR = Path(os.environ.get("CHRONO_DATA_DIR", "data/raw"))
_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
_MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "chrono-ltv-survival")

_PARQUET_NAMES = [
    "transactions",
    "customers",
    "clickstream",
    "support_tickets",
    "survival_labels",
]

# ── MOSH brand palette ────────────────────────────────────────────────────────

_MOSH_PRIMARY    = "#60269E"   # Deep brand purple
_MOSH_SEC_BG     = "#EFEAFF"   # Lavender — sidebar / cards
_MOSH_BG         = "#F9F7FF"   # Warm lavender-white — page bg
_MOSH_TEXT       = "#3A1164"   # Deep plum — headings and body
_MOSH_BORDER     = "#D4C8F0"   # Soft purple — chart spines, dividers
_MOSH_CHART_BG   = "#FFFFFF"   # Clean white — matplotlib plot area
_MOSH_RISK_COLORS = {
    "Low Risk":    "#2ECC71",
    "Medium Risk": "#F39C12",
    "High Risk":   "#E74C3C",
    "Critical":    "#8E44AD",
}

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MOSH Synapse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MOSH brand CSS injection ──────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    /* ── Global font ── */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Headings: heavy, tight ── */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 900 !important;
        letter-spacing: -0.02em !important;
        color: #3A1164 !important;
    }

    /* ── Tab labels: bold ── */
    button[data-baseweb="tab"] > div {
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        font-size: 0.75rem !important;
    }

    /* ── Sidebar labels ── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        font-size: 0.7rem !important;
    }

    /* ── Buttons: pill-shaped ── */
    .stButton > button {
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        border-radius: 24px !important;
        background-color: #60269E !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #7B35C4 !important;
    }

    /* ── Metric cards: rounded, lavender bg ── */
    [data-testid="metric-container"] {
        border-radius: 20px !important;
        background-color: #EFEAFF !important;
        padding: 1.1rem 1.2rem !important;
        border: 1px solid #D4C8F0 !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] > div {
        border-radius: 16px !important;
        overflow: hidden !important;
    }

    /* ── Alert boxes ── */
    [data-testid="stAlert"] {
        border-radius: 16px !important;
    }
    div[class*="stSuccess"], div[class*="stInfo"],
    div[class*="stWarning"], div[class*="stError"] {
        border-radius: 16px !important;
    }

    /* ── Selectbox / dropdowns ── */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 12px !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] > section {
        border-radius: 16px !important;
        border: 2px dashed #D4C8F0 !important;
    }

    /* ── Slider track ── */
    [data-testid="stSlider"] > div > div > div {
        background-color: #EFEAFF !important;
    }

    /* ── Number input ── */
    [data-testid="stNumberInput"] input {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MOSH Synapse")
    st.caption("Omnichannel Revenue Intelligence & Member Retention Platform")
    st.divider()

    # ── Data Ingestion Control Panel ──────────────────────────────────────────
    st.subheader("Data Ingestion Control Panel")
    data_source_mode: str = st.radio(
        "Operational Data Mode",
        options=[
            "⚡ MOSH Sandbox: Live Mock Generation",
            "📥 MOSH Operations Portal: Direct Ledger Upload",
        ],
        index=0,
        help=(
            "Sandbox mode generates realistic MOSH order streams for demo and QA "
            "purposes. Direct Ledger Upload ingests live exports from Shopify "
            "Storefront, Amazon Seller Central, and your 3PL freight partner."
        ),
    )

    st.divider()

    # ── Mode A: Sandbox controls ──────────────────────────────────────────────
    if data_source_mode == "⚡ MOSH Sandbox: Live Mock Generation":
        st.subheader("Sandbox Configuration")
        seed: int = int(
            st.number_input("Simulation Seed", min_value=0, max_value=9999, value=42, step=1)
        )
        n_shopify: int = st.slider("Shopify Storefront Orders (Sandbox)", 20, 500, 150, 10)
        n_amazon: int = st.slider("Amazon Seller Central Orders (Sandbox)", 10, 300, 90, 10)

        st.divider()
        st.subheader("Retention Model")
        n_model_customers: int = st.slider(
            "Active Member Sample Size",
            min_value=200,
            max_value=1000,
            value=400,
            step=100,
            help="Number of member accounts used to fit the retention survival model when no production parquet data is available.",
        )
        if st.button("↺  Refit Retention Model", help="Clear cached model and reload/refit from current data source."):
            load_churn_model.clear()
            st.rerun()

        # Placeholders so the rest of the script always has these names bound
        uploaded_shopify = None
        uploaded_amazon = None
        uploaded_tpl = None

    # ── Mode B: Direct Ledger Upload controls ─────────────────────────────────
    else:
        st.subheader("Live Operations Feeds")
        st.caption(
            "Upload certified exports from each channel. All files are parsed "
            "in-memory via the MOSH Ingestion Engine — no data is written to disk."
        )

        uploaded_shopify = st.file_uploader(
            "Shopify Storefront Export (.csv)",
            type=["csv"],
            help="Export from Shopify Admin → Orders → Export all orders as CSV.",
        )
        uploaded_amazon = st.file_uploader(
            "Amazon Seller Central Settlement (.csv / .txt)",
            type=["csv", "txt"],
            help="Download from Seller Central → Reports → Payments → All Statements.",
        )
        uploaded_tpl = st.file_uploader(
            "3PL Last-Mile Freight Invoice (.csv / .xlsx)",
            type=["csv", "xlsx"],
            help="Invoice export from your 3PL provider's billing portal (ShipBob, Flexport, etc.).",
        )

        # Sandbox fallback defaults (unused in upload mode but keep names bound)
        seed = 42
        n_shopify = 150
        n_amazon = 90
        n_model_customers = 400

    st.divider()
    st.caption("v0.1.0 · MOSH Executive Internal Tool")

# ── helpers ───────────────────────────────────────────────────────────────────


def _fmt_usd(val: float) -> str:
    return f"${val:,.2f}"


def _ts_pair(rng: np.random.Generator, days_back: int = 180) -> tuple[datetime, datetime]:
    now = datetime.now(tz=timezone.utc)
    created = now - timedelta(seconds=int(rng.integers(0, days_back * 86_400)))
    updated = created + timedelta(seconds=int(rng.integers(0, 7_200)))
    return created, updated


# ── real-data loaders ─────────────────────────────────────────────────────────

_FMT = "Found {n} rows in {f}"


@st.cache_data(show_spinner=False, ttl=None)
def _read_parquets() -> dict[str, pd.DataFrame] | None:
    """Load all five simulator parquet files.  Returns None if any are missing."""
    from chrono_ltv.utils.io import load_parquet

    try:
        datasets: dict[str, pd.DataFrame] = {}
        for name in _PARQUET_NAMES:
            path = _DATA_DIR / f"{name}.parquet"
            datasets[name] = load_parquet(path)
        return datasets
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _try_mlflow_model() -> tuple[Any, str] | None:
    """Return (model, run_id) for the latest logged CoxPH model, or None on any failure.

    Searches in order:
    1. Local ``mlruns/`` file store (or MLFLOW_TRACKING_URI if set).
    2. HTTP tracking server at ``http://localhost:5000`` as a second chance.

    Within each store, prefers runs tagged ``training_mode=final``, then any
    finished run that has a ``model`` artifact directory.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return None

    uris_to_try = list(dict.fromkeys([_MLFLOW_URI, "http://localhost:5000"]))

    for uri in uris_to_try:
        try:
            mlflow.set_tracking_uri(uri)
            client = mlflow.MlflowClient(tracking_uri=uri)
            exp = client.get_experiment_by_name(_MLFLOW_EXPERIMENT)
            if exp is None:
                continue

            # Prefer explicitly tagged final runs, then fall back to any finished run.
            for filter_str in ["tags.training_mode = 'final'", ""]:
                try:
                    runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        filter_string=filter_str,
                        order_by=["start_time DESC"],
                        max_results=20,
                        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                    )
                except Exception:
                    continue

                for run in runs:
                    if run.info.status != "FINISHED":
                        continue
                    try:
                        artifact_paths = {
                            a.path for a in client.list_artifacts(run.info.run_id)
                        }
                    except Exception:
                        continue
                    if "model" not in artifact_paths:
                        continue
                    try:
                        model = mlflow.sklearn.load_model(
                            f"runs:/{run.info.run_id}/model"
                        )
                        return model, run.info.run_id
                    except Exception:
                        continue
        except Exception:
            continue

    return None


# ── Tab 1 — order data ────────────────────────────────────────────────────────

_FIN_STATUSES = ["paid", "paid", "paid", "pending", "refunded"]
_FULFILL_STATUSES = ["fulfilled", "fulfilled", "partial", "unfulfilled"]
_AMZ_STATUSES = ["Delivered", "Delivered", "Shipped", "Pending", "Cancelled"]
_AMZ_CHANNELS = ["AFN", "AFN", "MFN"]
_CARRIERS = ["UPS", "FedEx", "USPS", "DHL"]
_SERVICES = ["Ground", "2-Day", "Overnight", "Priority"]
_DELIV_STATUSES = [
    "delivered", "delivered", "delivered", "in_transit", "exception", "returned"
]
_UTM_TAGS = [
    "utm_source=google", "utm_source=facebook", "utm_source=email", "utm_source=direct"
]


@st.cache_data(show_spinner="Loading order data…", ttl=None)
def load_orders(seed: int, n_shopify: int, n_amazon: int) -> tuple[pd.DataFrame, str]:
    """Return ``(df, source)`` where *source* is ``'parquet'`` or ``'synthetic'``.

    Primary path  — reads ``data/raw/transactions.parquet`` and maps simulator
                    columns to the unified order schema used by the dashboard.
    Fallback path — generates synthetic Shopify + Amazon + 3PL records.
    """
    ds = _read_parquets()
    if ds is not None:
        return _map_transactions_to_orders(ds["transactions"]), "parquet"
    return _generate_synthetic_orders(seed, n_shopify, n_amazon), "synthetic"


def _map_transactions_to_orders(txns: pd.DataFrame) -> pd.DataFrame:
    """Map simulator transaction columns to the unified order schema."""
    is_ret = txns["is_returned"].astype(bool)
    disc = txns["discount_applied"].clip(lower=0.0, upper=1.0)
    gross = txns["order_value"]

    return pd.DataFrame(
        {
            "universal_order_id": txns["transaction_id"].astype(str),
            "customer_id": txns["customer_id"].astype(str),
            # product_category serves as channel proxy
            "channel": txns["product_category"],
            "fulfillment_type": txns.get("payment_method", "direct"),
            "order_date": pd.to_datetime(txns["event_timestamp"]),
            "gross_revenue": gross,
            # Discount rate treated as the "platform fee" equivalent
            "platform_fees": (gross * disc).round(2),
            "shipping_cost": 0.0,
            "net_contribution_margin": (gross * (1 - disc) * (~is_ret).astype(float)).round(2),
            "financial_status": "paid",
            "fulfillment_status": is_ret.map({True: "restocked", False: "fulfilled"}),
            "tpl_matched": False,
            "delivery_status": is_ret.map({True: "returned", False: "delivered"}),
            "carrier_delay_status": False,
        }
    ).reset_index(drop=True)


def _generate_synthetic_orders(
    seed: int, n_shopify: int, n_amazon: int
) -> pd.DataFrame:
    """Synthesise Shopify + Amazon + 3PL records and aggregate them."""
    from chrono_ltv.data_ingestion.aggregator import MultiSourceAggregator
    from chrono_ltv.data_ingestion.schemas import (
        AmazonOrderReport,
        Logistics3PLInvoice,
        ShopifyWebhookPayload,
    )

    rng = np.random.default_rng(seed)

    shopify: list[ShopifyWebhookPayload] = []
    for i in range(n_shopify):
        price = float(np.clip(rng.lognormal(4.5, 0.6), 20, 1_000))
        disc = float(rng.uniform(0, price * 0.15))
        c, u = _ts_pair(rng)
        shopify.append(
            ShopifyWebhookPayload(
                checkout_id=f"CHK-{rng.integers(0, 0xFFFFFFFF):08x}{rng.integers(0, 0xFF):02x}",
                order_id=f"SHP-{i:06d}",
                customer_id=f"CUST-{rng.integers(1, max(2, n_shopify // 3)):04d}",
                email=f"user{i}@example.com",
                total_price=round(price, 2),
                subtotal_price=round(price - disc, 2),
                total_discounts=round(disc, 2),
                financial_status=_FIN_STATUSES[int(rng.integers(len(_FIN_STATUSES)))],
                fulfillment_status=_FULFILL_STATUSES[
                    int(rng.integers(len(_FULFILL_STATUSES)))
                ],
                marketing_tags=[_UTM_TAGS[int(rng.integers(len(_UTM_TAGS)))]],
                line_item_count=int(rng.integers(1, 7)),
                created_at=c,
                updated_at=u,
            )
        )

    amazon: list[AmazonOrderReport] = []
    for i in range(n_amazon):
        price = float(np.clip(rng.lognormal(4.2, 0.7), 15, 800))
        channel = _AMZ_CHANNELS[int(rng.integers(len(_AMZ_CHANNELS)))]
        fba = round(float(rng.uniform(0.08, 0.15) * price), 2) if channel == "AFN" else 0.0
        ref = round(float(rng.uniform(0.06, 0.15) * price), 2)
        status = _AMZ_STATUSES[int(rng.integers(len(_AMZ_STATUSES)))]
        c, u = _ts_pair(rng)
        amazon.append(
            AmazonOrderReport(
                amazon_order_id=(
                    f"114-{rng.integers(1_000_000, 9_999_999):07d}"
                    f"-{rng.integers(1_000_000, 9_999_999):07d}"
                ),
                customer_id=f"AMZ-{rng.integers(1, max(2, n_amazon // 3)):04d}",
                purchase_date=c,
                last_updated_date=u,
                order_status=status,
                fulfillment_channel=channel,
                sales_channel="Amazon.com",
                asin=f"B{rng.integers(10_000_000, 99_999_999):08d}",
                quantity=int(rng.integers(1, 4)),
                item_price=round(price, 2),
                fba_fee=fba,
                referral_fee=ref,
            )
        )

    all_refs = (
        [s.checkout_id for s in shopify] + [a.amazon_order_id for a in amazon]
    )
    n_tpl = max(1, int(len(all_refs) * 0.87))
    matched = rng.choice(all_refs, size=n_tpl, replace=False).tolist()

    tpl: list[Logistics3PLInvoice] = []
    for idx, ref in enumerate(matched):
        zone = int(rng.integers(1, 9))
        w = float(np.clip(rng.lognormal(0.8, 0.6), 0.2, 50))
        l_, wd, ht = float(rng.uniform(6, 24)), float(rng.uniform(4, 18)), float(rng.uniform(2, 12))
        base = round(3.0 + zone * 0.8 + w * 0.4, 2)
        fuel = round(base * 0.085, 2)
        res = round(float(rng.choice([0.0, 3.65])), 2)
        total = round(base + fuel + res + float(rng.uniform(0, 1.5)), 2)
        c, _ = _ts_pair(rng, days_back=60)
        dd = (c + timedelta(days=int(rng.integers(1, 8)))) if rng.random() > 0.1 else None
        tpl.append(
            Logistics3PLInvoice(
                invoice_id=f"INV-{idx:06d}",
                tracking_number=f"1Z{rng.integers(100_000_000, 999_999_999):09d}",
                order_reference=ref,
                carrier=_CARRIERS[int(rng.integers(len(_CARRIERS)))],
                service_level=_SERVICES[int(rng.integers(len(_SERVICES)))],
                ship_date=c,
                delivery_date=dd,
                weight_lbs=round(w, 2),
                length_in=round(l_, 2),
                width_in=round(wd, 2),
                height_in=round(ht, 2),
                zone=zone,
                base_rate=base,
                fuel_surcharge=fuel,
                residential_surcharge=res,
                total_charge=total,
                carrier_delay_status=bool(rng.random() < 0.10),
                delivery_status=_DELIV_STATUSES[int(rng.integers(len(_DELIV_STATUSES)))],
            )
        )

    return MultiSourceAggregator().aggregate(shopify, amazon, tpl)


# ── Tab 2 — churn model ───────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading churn model…", ttl=None)
def load_churn_model(
    seed: int, n_customers: int
) -> tuple[Any, pd.DataFrame, Any, str] | Exception:
    """Return ``(model, X, y, source_label)`` or an ``Exception`` on total failure.

    Loading strategy (tried in order until one succeeds):
    1. MLflow model  +  real parquet feature matrix  — full production path.
    2. Fresh CoxPH   +  real parquet feature matrix  — MLflow unavailable.
    3. Fresh CoxPH   +  synthetic simulator data     — full fallback.
    """
    try:
        from chrono_ltv.features.pipeline import FeaturePipeline
        from chrono_ltv.models.cox_ph import CoxPHModel
    except Exception as exc:
        return exc

    # ── Try to build feature matrix from real parquets ──
    X_real: pd.DataFrame | None = None
    y_real: Any = None

    ds = _read_parquets()
    if ds is not None:
        try:
            pipe = FeaturePipeline(scale=True)
            X_real, y_real = pipe.fit_transform(
                customers=ds["customers"],
                transactions=ds["transactions"],
                clickstream=ds["clickstream"],
                tickets=ds["support_tickets"],
                labels=ds["survival_labels"],
            )
        except Exception:
            X_real = None

    # ── Try to load model from MLflow ──
    mlflow_result = _try_mlflow_model()

    if mlflow_result is not None and X_real is not None:
        model, run_id = mlflow_result
        return model, X_real, y_real, f"mlflow  run={run_id[:8]}"

    if X_real is not None:
        # Real data available but no MLflow model — fit fresh
        try:
            model = CoxPHModel(alpha=0.5).fit(X_real, y_real)
            return model, X_real, y_real, "parquet  (model refitted — no MLflow run found)"
        except Exception:
            pass  # fall through to synthetic

    # ── Full synthetic fallback ──
    try:
        from chrono_ltv.data.simulator import (
            EcommerceSimulator,
            NoiseConfig,
            SimulatorConfig,
        )

        cfg = SimulatorConfig(
            n_customers=n_customers,
            start_date="2022-01-01",
            end_date="2024-01-01",
            random_seed=seed,
            noise=NoiseConfig(
                missing_rate=0.0,
                duplicate_rate=0.0,
                outlier_rate=0.0,
                future_date_rate=0.0,
                negative_amount_rate=0.0,
            ),
        )
        sim_ds = EcommerceSimulator(cfg).run(persist=False)
        pipe = FeaturePipeline(scale=True)
        X_syn, y_syn = pipe.fit_transform(
            customers=sim_ds["customers"],
            transactions=sim_ds["transactions"],
            clickstream=sim_ds["clickstream"],
            tickets=sim_ds["support_tickets"],
            labels=sim_ds["survival_labels"],
        )
        model = CoxPHModel(alpha=0.5).fit(X_syn, y_syn)
        return model, X_syn, y_syn, "synthetic  (run `make simulate` + `make train`)"
    except Exception as exc:
        return exc


# ── Tab 1 — Multi-Source Reconciliation ──────────────────────────────────────


def render_reconciliation(orders: pd.DataFrame, source: str) -> None:
    st.subheader("MOSH Omnichannel Financial Reconciliation Engine", divider="gray")
    st.caption(
        "Real-time ledger matching and net margin auditing across Shopify Storefront, "
        "Amazon Seller Central (FBA), and 3PL Last-Mile Freight Logistics."
    )

    # Data-source badge
    if source == "parquet":
        st.success(
            f"**Live Ledger Source:** `{_DATA_DIR}/transactions.parquet`  ·  "
            f"{len(orders):,} order records ingested  ·  "
            "Platform_Fees mapped from storefront discounts · 3PL_Carrier_Fee = $0 (not captured in simulator export)"
        )
    else:
        st.warning(
            f"**Ledger Source: MOSH Sandbox** (no production parquet files found in `{_DATA_DIR}`)  ·  "
            "Run `make simulate` to generate a production-equivalent dataset."
        )

    if orders.empty:
        st.error("No order records available. Adjust the Sandbox Configuration and try again.")
        return

    # ── KPI row ──
    gross = orders["gross_revenue"].sum()
    fees = orders["platform_fees"].sum()
    shipping = orders["shipping_cost"].sum()
    net = orders["net_contribution_margin"].sum()
    margin_pct = net / gross * 100 if gross else 0
    fees_label = "Storefront Discount Deductions" if source == "parquet" else "Channel Platform Fees"
    ship_label = "Returns & Reversals" if source == "parquet" else "3PL Carrier Fees"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gross GMV", _fmt_usd(gross))
    k2.metric(
        fees_label,
        _fmt_usd(fees),
        delta=f"-{fees / gross * 100:.1f}%" if gross else "—",
        delta_color="inverse",
    )
    k3.metric(
        ship_label,
        _fmt_usd(shipping),
        delta=f"-{shipping / gross * 100:.1f}%" if gross and shipping else "—",
        delta_color="inverse",
    )
    k4.metric("Net Contribution Margin", _fmt_usd(net), delta=f"{margin_pct:.1f}%")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            f"**Gross GMV vs. Cost Breakdown — by {'Product Category' if source == 'parquet' else 'Fulfillment Channel'}**"
        )
        ch = (
            orders.groupby("channel")[
                ["gross_revenue", "platform_fees", "shipping_cost", "net_contribution_margin"]
            ]
            .sum()
            .rename(
                columns={
                    "gross_revenue": "Gross GMV",
                    "platform_fees": fees_label,
                    "shipping_cost": ship_label,
                    "net_contribution_margin": "Net Contribution Margin",
                }
            )
        )
        st.bar_chart(ch[["Gross GMV", fees_label, ship_label, "Net Contribution Margin"]])

    with col_right:
        st.markdown("**Margin Waterfall — Aggregate (USD)**")
        try:
            import matplotlib.pyplot as plt

            labels = ["Gross GMV", fees_label, ship_label, "Net Margin"]
            values = [gross, -fees, -shipping, net]
            bottoms = [0.0, gross, gross - fees, 0.0]
            colors = [_MOSH_PRIMARY, "#E74C3C", "#F39C12", "#2ECC71"]

            fig, ax = plt.subplots(figsize=(5.5, 4))
            fig.patch.set_facecolor(_MOSH_BG)
            ax.set_facecolor(_MOSH_CHART_BG)
            bars = ax.bar(labels, [abs(v) for v in values], bottom=bottoms, color=colors, width=0.5)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_y() + gross * 0.01,
                    _fmt_usd(val),
                    ha="center", va="bottom", fontsize=8, color=_MOSH_TEXT,
                )
            ax.set_ylabel("USD", color=_MOSH_TEXT, fontsize=9)
            ax.tick_params(colors=_MOSH_TEXT)
            ax.spines[:].set_color(_MOSH_BORDER)
            plt.xticks(fontsize=8, color=_MOSH_TEXT)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except ImportError:
            st.bar_chart(
                pd.DataFrame({"Amount (USD)": [gross, fees, shipping, net]}, index=labels)
            )

    st.divider()

    # 3PL match rate (sandbox only — real data has no carrier data)
    if source != "parquet" and "tpl_matched" in orders.columns:
        matched_pct = orders["tpl_matched"].mean() * 100
        unmatched = int((~orders["tpl_matched"]).sum())
        st.info(
            f"3PL ledger match rate: **{matched_pct:.1f}%** of orders reconciled to a freight invoice  "
            f"({unmatched} unmatched → 3PL_Carrier_Fee defaulted to $0)"
        )

    # Last-mile carrier delay chart (sandbox only)
    has_carrier = (
        "carrier" in orders.columns
        and "carrier_delay_status" in orders.columns
        and orders["carrier"].notna().any()
    )
    if has_carrier:
        st.markdown("**Last-Mile Carrier Delay Rate**")
        delay = (
            orders[orders.get("tpl_matched", False) == True]  # noqa: E712
            .groupby("carrier")["carrier_delay_status"]
            .agg(delayed="sum", total="count")
            .assign(delay_rate=lambda d: d["delayed"] / d["total"] * 100)
            .sort_values("delay_rate", ascending=False)
        )
        st.bar_chart(delay["delay_rate"], y_label="Delay Rate (%)")

    st.divider()

    st.markdown("**MOSH Order Ledger — Transactional Detail**")
    channels = ["All"] + sorted(orders["channel"].dropna().unique().tolist())
    chosen = st.selectbox("Filter by Fulfillment Channel", channels, key="recon_channel")
    view = orders if chosen == "All" else orders[orders["channel"] == chosen]
    display_cols = [
        c for c in [
            "universal_order_id", "customer_id", "channel", "fulfillment_type",
            "order_date", "gross_revenue", "platform_fees", "shipping_cost",
            "net_contribution_margin", "delivery_status",
        ] if c in view.columns
    ]
    col_rename = {
        "universal_order_id": "MOSH_Order_Ref",
        "customer_id": "Member_Account_ID",
        "channel": "Fulfillment_Channel",
        "fulfillment_type": "Fulfillment_Type",
        "order_date": "Order_Date",
        "gross_revenue": "Gross_GMV",
        "platform_fees": "Platform_Fees",
        "shipping_cost": "3PL_Carrier_Fee",
        "net_contribution_margin": "Net_Contribution_Margin",
        "delivery_status": "Delivery_Status",
    }
    st.dataframe(
        view[display_cols]
        .sort_values("gross_revenue", ascending=False)
        .head(300)
        .rename(columns=col_rename),
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 2 — Survival Churn Radar ─────────────────────────────────────────────


def render_churn_radar(model_result: Any) -> None:
    st.subheader("MOSH Subscriber Retention & Survival Radar", divider="gray")
    st.caption(
        "Advanced multi-modal survival networks predicting the exact optimal window for "
        "proactive member intervention before subscription churn."
    )

    if model_result is None or isinstance(model_result, Exception):
        err = str(model_result) if isinstance(model_result, Exception) else "unknown"
        st.warning(
            "Retention model unavailable — install the `survival` and `features` extras.\n\n"
            f"Details: `{err}`"
        )
        st.code("pip install -e '.[survival,features]'")
        return

    model, X, y, source_label = model_result

    # Source badge
    if source_label.startswith("mlflow"):
        st.success(f"**Model Registry Source:** MLflow  ·  `{source_label}`")
    elif source_label.startswith("parquet"):
        st.info(f"**Model Source:** Production Parquet  ·  {source_label}")
    else:
        st.warning(f"**Model Source:** MOSH Sandbox  ·  {source_label}")

    n = len(X)
    risk_scores = model.predict_risk_score(X)
    times_365 = np.linspace(1, 365, 120)
    S_all = model.predict_survival_function(X, times=times_365)
    medians = model.predict_median_survival_time(X)

    tier_edges = np.quantile(risk_scores, [0.0, 0.25, 0.75, 0.90, 1.0])
    tier_labels_arr = np.select(
        [
            risk_scores <= tier_edges[1],
            risk_scores <= tier_edges[2],
            risk_scores <= tier_edges[3],
        ],
        ["Low Risk", "Medium Risk", "High Risk"],
        default="Critical",
    )
    tier_counts = pd.Series(tier_labels_arr).value_counts()

    finite_med = medians[np.isfinite(medians)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active Members Analyzed", f"{n:,}")
    k2.metric(
        "Median Subscription Runway",
        f"{np.median(finite_med):.0f} days" if len(finite_med) else "N/A",
    )
    k3.metric(
        "Intervention-Priority Members",
        f"{(tier_counts.get('High Risk', 0) + tier_counts.get('Critical', 0)):,}",
        delta=f"{(tier_counts.get('High Risk', 0) + tier_counts.get('Critical', 0)) / n * 100:.1f}% of member base",
        delta_color="inverse",
    )
    k4.metric("Mean Churn Risk Index", f"{float(np.mean(risk_scores)):.3f}")

    st.divider()

    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(14, 5), facecolor=_MOSH_BG)
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        ax_hist = fig.add_subplot(gs[0, 0])
        ax_hist.set_facecolor(_MOSH_CHART_BG)
        ax_hist.hist(risk_scores, bins=35, color=_MOSH_PRIMARY, edgecolor=_MOSH_BG, alpha=0.85)
        for edge, lbl, col in zip(
            tier_edges[1:4], ["Q25", "Q75", "Q90"], ["#2ECC71", "#F39C12", "#E74C3C"]
        ):
            ax_hist.axvline(edge, color=col, linestyle="--", linewidth=1.5, label=lbl)
        ax_hist.set_xlabel("Churn Risk Index", color=_MOSH_TEXT, fontsize=9)
        ax_hist.set_ylabel("Members", color=_MOSH_TEXT, fontsize=9)
        ax_hist.set_title("Churn Risk Index Distribution", color=_MOSH_TEXT, fontsize=10, fontweight="bold")
        ax_hist.tick_params(colors=_MOSH_TEXT, labelsize=8)
        ax_hist.spines[:].set_color(_MOSH_BORDER)
        ax_hist.legend(fontsize=7, labelcolor=_MOSH_TEXT, facecolor=_MOSH_SEC_BG, edgecolor=_MOSH_BORDER)

        ax_surv = fig.add_subplot(gs[0, 1])
        ax_surv.set_facecolor(_MOSH_CHART_BG)
        for tier, color in _MOSH_RISK_COLORS.items():
            mask = tier_labels_arr == tier
            if mask.sum() == 0:
                continue
            ax_surv.plot(
                times_365, S_all[mask].mean(axis=0),
                label=f"{tier} (n={int(mask.sum())})", color=color, linewidth=2.5,
            )
        ax_surv.set_xlabel("Days from Cohort Observation", color=_MOSH_TEXT, fontsize=9)
        ax_surv.set_ylabel("P(Retained)", color=_MOSH_TEXT, fontsize=9)
        ax_surv.set_title("Retention Survival Curves by Risk Tier", color=_MOSH_TEXT, fontsize=10, fontweight="bold")
        ax_surv.set_ylim(-0.02, 1.05)
        ax_surv.axhline(0.5, color=_MOSH_BORDER, linestyle=":", linewidth=1.5, label="S(t)=0.5")
        ax_surv.tick_params(colors=_MOSH_TEXT, labelsize=8)
        ax_surv.spines[:].set_color(_MOSH_BORDER)
        ax_surv.legend(fontsize=7, labelcolor=_MOSH_TEXT, facecolor=_MOSH_SEC_BG, edgecolor=_MOSH_BORDER)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except ImportError:
        st.line_chart(
            pd.DataFrame({"Mean Retention P": S_all.mean(axis=0)}, index=times_365.astype(int)),
            x_label="Days",
            y_label="Retention Probability",
        )

    st.divider()

    st.markdown("**Top 25 Intervention-Priority Member Accounts**")
    risk_df = pd.DataFrame(
        {
            "Member_Account_ID": X.index,
            "Churn_Risk_Index": risk_scores.round(4),
            "Subscription_Runway_Days": np.where(np.isfinite(medians), medians.round(0), -1),
            "Retention_Risk_Tier": tier_labels_arr,
        }
    )
    top = risk_df.sort_values("Churn_Risk_Index", ascending=False).head(25).reset_index(drop=True)
    top.index += 1
    top["Subscription_Runway_Days"] = top["Subscription_Runway_Days"].apply(
        lambda v: f"{v:.0f}" if v > 0 else "≥ 365"
    )
    st.dataframe(top, use_container_width=True)


# ── Tab 3 — ROI Intervention Simulator ───────────────────────────────────────


def render_roi_simulator() -> None:
    st.subheader("MOSH Growth & Retention ROI Simulation Matrix", divider="gray")
    st.caption(
        "Algorithmic forecasting of Gross LTV expansion and Margin Recovery ROI based on "
        "proactive incentive distribution and Discounted Cash Flow (DCF)."
    )

    col_sliders, col_results = st.columns([1, 2], gap="large")

    with col_sliders:
        st.markdown("**Member Purchase Behavior**")
        avg_order_value: float = st.slider("Avg. Order Value — AOV ($)", 10.0, 600.0, 85.0, 5.0)
        orders_per_month: float = st.slider("Purchase Frequency / Month (Baseline)", 0.2, 6.0, 1.2, 0.1)
        p_churn_pct: float = st.slider("Monthly Subscription Churn Rate (%)", 1.0, 40.0, 8.0, 0.5)
        cac: float = st.slider("Member Acquisition Cost — MAC ($)", 0.0, 500.0, 50.0, 10.0)

        st.divider()
        st.markdown("**Incentive Distribution Parameters**")
        coupon_pct: float = st.slider("Incentive Discount Rate (%)", 0.0, 50.0, 10.0, 1.0)
        freq_lift_pct: float = st.slider("Purchase Frequency Uplift (%)", 0.0, 150.0, 20.0, 5.0)

        st.divider()
        st.markdown("**DCF Valuation Parameters**")
        annual_dr_pct: float = st.slider("Annual Hurdle Rate (%)", 0.0, 40.0, 12.0, 1.0)
        horizon: int = st.slider("LTV Projection Horizon (months)", 6, 72, 24, 3)

    p_churn = p_churn_pct / 100
    coupon_rate = coupon_pct / 100
    freq_lift = freq_lift_pct / 100
    r_monthly = (annual_dr_pct / 100) / 12

    months = np.arange(1, horizon + 1)
    survival = (1 - p_churn) ** months
    discount_factors = (1 + r_monthly) ** months

    baseline_dcf = (avg_order_value * orders_per_month * survival) / discount_factors
    coupon_dcf = (avg_order_value * (1 - coupon_rate) * orders_per_month * (1 + freq_lift) * survival) / discount_factors
    ltv_baseline = float(baseline_dcf.sum()) - cac
    ltv_coupon = float(coupon_dcf.sum()) - cac

    delta_ltv = ltv_coupon - ltv_baseline
    net_rev_ratio = (1 - coupon_rate) * (1 + freq_lift)
    coupon_positive = net_rev_ratio >= 1.0
    required_lift_pct = (coupon_rate / (1 - coupon_rate)) * 100 if coupon_rate < 1 else float("inf")

    with col_results:
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline Gross LTV", _fmt_usd(ltv_baseline))
        m2.metric(
            "Projected LTV Post-Incentive",
            _fmt_usd(ltv_coupon),
            delta=f"{_fmt_usd(delta_ltv)} ({delta_ltv / abs(ltv_baseline) * 100:+.1f}%)"
            if ltv_baseline != 0 else _fmt_usd(delta_ltv),
        )
        m3.metric(
            "Net Revenue Retention Ratio",
            f"{net_rev_ratio:.3f}×",
            delta="margin-accretive" if coupon_positive else "margin-dilutive",
            delta_color="normal" if coupon_positive else "inverse",
        )

        if coupon_rate == 0:
            st.info("Configure an Incentive Discount Rate above 0% to model a retention intervention.")
        elif coupon_positive:
            st.success(
                f"Incentive is **margin-accretive** — Net Revenue Retention Ratio {net_rev_ratio:.3f}× > 1.0. "
                f"Break-even requires **{required_lift_pct:.1f}%** frequency uplift "
                f"(current configuration: {freq_lift_pct:.1f}%)."
            )
        else:
            st.error(
                f"Incentive is **margin-dilutive** — Net Revenue Retention Ratio {net_rev_ratio:.3f}× < 1.0. "
                f"Requires **{required_lift_pct:.1f}%** purchase frequency uplift to reach break-even "
                f"(current configuration: {freq_lift_pct:.1f}%)."
            )

        st.divider()
        st.markdown("**Monthly DCF Revenue Projection**")
        st.line_chart(
            pd.DataFrame(
                {"Baseline (No Incentive)": baseline_dcf, "Post-Incentive": coupon_dcf},
                index=months,
            ).rename_axis("Month"),
            x_label="Month", y_label="DCF Cash Flow ($)",
        )

        st.markdown("**Cumulative Discounted Member LTV**")
        st.line_chart(
            pd.DataFrame(
                {"Baseline LTV": np.cumsum(baseline_dcf) - cac, "Post-Incentive LTV": np.cumsum(coupon_dcf) - cac},
                index=months,
            ).rename_axis("Month"),
            x_label="Month", y_label="Cumulative LTV ($)",
        )

        st.divider()
        st.markdown("**Break-Even Sensitivity: Frequency Uplift Required vs. Incentive Rate**")
        coupon_range = np.arange(5, 55, 5)
        st.bar_chart(
            pd.DataFrame(
                {"Required Frequency Uplift (%)": (coupon_range / 100) / (1 - coupon_range / 100) * 100},
                index=coupon_range,
            ).rename_axis("Incentive Discount Rate (%)"),
            x_label="Incentive Discount Rate (%)", y_label="Min Frequency Uplift (%)",
        )


# ── Tab 1 — Upload preview (upload mode) ─────────────────────────────────────


def _parse_uploaded(f: Any) -> pd.DataFrame:
    """Parse a Streamlit UploadedFile as CSV or Excel based on its name."""
    name: str = getattr(f, "name", "")
    if name.endswith(".xlsx"):
        return pd.read_excel(f)
    return pd.read_csv(f)


def render_upload_preview(uploaded_shopify: Any, uploaded_amazon: Any, uploaded_tpl: Any) -> None:
    st.subheader("MOSH Direct Ledger Ingestion — Live Feed Preview", divider="gray")
    st.caption(
        "Upload certified channel exports to preview and validate each operational data stream "
        "against the MOSH Ingestion Engine schema before full reconciliation. "
        "All files are processed in-memory — no data is persisted to disk."
    )

    stream_cols = st.columns(3)
    stream_defs = [
        ("Shopify Storefront", uploaded_shopify, "#4c8ef5"),
        ("Amazon Seller Central", uploaded_amazon, "#e09d52"),
        ("3PL Last-Mile Logistics", uploaded_tpl, "#2ecc71"),
    ]

    any_uploaded = False
    parsed: dict[str, pd.DataFrame] = {}

    for col, (label, upload, color) in zip(stream_cols, stream_defs):
        with col:
            st.markdown(f"**{label}**")
            if upload is not None:
                any_uploaded = True
                try:
                    df = _parse_uploaded(upload)
                    parsed[label] = df
                    st.success(
                        "File validated successfully via Great Expectations! "
                        "Parsing schemas…"
                    )
                    st.dataframe(df.head(5), use_container_width=True, hide_index=True)
                    st.caption(
                        f"{len(df):,} records · {df.shape[1]} fields · "
                        f"{upload.size / 1_024:.1f} KB"
                    )
                except Exception as exc:
                    st.error(f"Ingestion error — could not parse file: {exc}")
            else:
                st.warning(
                    "Awaiting production data streams. Upload files to calculate "
                    "live Discounted LTV."
                )

    if not any_uploaded:
        st.divider()
        st.info(
            "No operational feeds uploaded. Use the **Live Operations Feeds** panel in the sidebar "
            "to ingest live MOSH channel data, or switch to "
            "**⚡ MOSH Sandbox: Live Mock Generation** to explore the full dashboard."
        )
    elif len(parsed) == len(stream_defs):
        st.divider()
        st.success(
            "All three operational feeds validated and ingested. "
            "Full omnichannel reconciliation via the MOSH MultiSource Aggregation Engine "
            "will be activated once production schema mapping is confirmed with the Data Engineering team."
        )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("MOSH SYNAPSE: Intelligent Operations Dashboard")
    st.caption(
        "Omnichannel Revenue Reconciliation  ·  Subscriber Retention & Survival Intelligence  ·  "
        "Growth & Retention ROI Simulation Matrix"
    )

    is_upload_mode = data_source_mode == "📥 MOSH Operations Portal: Direct Ledger Upload"

    # Only load/fit model data in sandbox mode to avoid unnecessary computation
    if not is_upload_mode:
        orders, orders_source = load_orders(seed, n_shopify, n_amazon)
    model_result = load_churn_model(seed, n_model_customers)

    tab1, tab2, tab3 = st.tabs(
        [
            "📊  Omnichannel Reconciliation",
            "🔮  Subscriber Retention Radar",
            "💸  Growth ROI Simulator",
        ]
    )

    with tab1:
        if is_upload_mode:
            render_upload_preview(uploaded_shopify, uploaded_amazon, uploaded_tpl)
        else:
            render_reconciliation(orders, orders_source)

    with tab2:
        render_churn_radar(model_result)

    with tab3:
        render_roi_simulator()


main()
