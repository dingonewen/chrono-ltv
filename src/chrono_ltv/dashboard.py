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

Requires
--------
    pip install -e ".[dashboard]"           # streamlit + matplotlib
    pip install -e ".[survival,features]"   # for the Churn Radar tab
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChronoLTV",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("ChronoLTV")
    st.caption("Multi-source CLV & Churn Intelligence")
    st.divider()

    st.subheader("Data Controls")
    seed: int = int(
        st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    )
    n_shopify: int = st.slider("Shopify orders", 20, 500, 150, 10)
    n_amazon: int = st.slider("Amazon orders", 10, 300, 90, 10)

    st.divider()
    st.subheader("Churn Model")
    n_model_customers: int = st.slider(
        "Training customers",
        min_value=200,
        max_value=1000,
        value=400,
        step=100,
        help="More customers ⟹ better accuracy. First run takes ~30 s.",
    )
    if "refit_counter" not in st.session_state:
        st.session_state["refit_counter"] = 0
    if st.button("↺  Refit Churn Model", help="Clear cached model and refit with current params."):
        st.session_state["refit_counter"] += 1

    st.divider()
    st.caption("v0.1.0 · Internal Demo")

# ── helpers ───────────────────────────────────────────────────────────────────


def _ts_pair(rng: np.random.Generator, days_back: int = 180) -> tuple[datetime, datetime]:
    """Return a (created_at, updated_at) pair within the past *days_back* days."""
    now = datetime.now(tz=timezone.utc)
    created = now - timedelta(seconds=int(rng.integers(0, days_back * 86_400)))
    updated = created + timedelta(seconds=int(rng.integers(0, 7_200)))
    return created, updated


def _fmt_usd(val: float) -> str:
    return f"${val:,.2f}"


# ── Tab 1 data ────────────────────────────────────────────────────────────────

_FIN_STATUSES = ["paid", "paid", "paid", "pending", "refunded"]
_FULFILL_STATUSES = ["fulfilled", "fulfilled", "partial", "unfulfilled"]
_AMZ_STATUSES = ["Delivered", "Delivered", "Shipped", "Pending", "Cancelled"]
_AMZ_CHANNELS = ["AFN", "AFN", "MFN"]
_CARRIERS = ["UPS", "FedEx", "USPS", "DHL"]
_SERVICES = ["Ground", "2-Day", "Overnight", "Priority"]
_DELIV_STATUSES = ["delivered", "delivered", "delivered", "in_transit", "exception", "returned"]
_UTM_TAGS = ["utm_source=google", "utm_source=facebook", "utm_source=email", "utm_source=direct"]


@st.cache_data(show_spinner="Generating order data…")
def load_orders(seed: int, n_shopify: int, n_amazon: int) -> pd.DataFrame:
    """Synthesise Shopify + Amazon + 3PL records and aggregate them."""
    from chrono_ltv.data_ingestion.aggregator import MultiSourceAggregator
    from chrono_ltv.data_ingestion.schemas import (
        AmazonOrderReport,
        Logistics3PLInvoice,
        ShopifyWebhookPayload,
    )

    rng = np.random.default_rng(seed)

    # ── Shopify ──
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

    # ── Amazon ──
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

    # ── 3PL (≈87 % coverage) ──
    all_refs = [s.checkout_id for s in shopify] + [a.amazon_order_id for a in amazon]
    n_tpl = max(1, int(len(all_refs) * 0.87))
    matched = rng.choice(all_refs, size=n_tpl, replace=False).tolist()

    tpl: list[Logistics3PLInvoice] = []
    for idx, ref in enumerate(matched):
        zone = int(rng.integers(1, 9))
        w = float(np.clip(rng.lognormal(0.8, 0.6), 0.2, 50))
        l_, wd, ht = (
            float(rng.uniform(6, 24)),
            float(rng.uniform(4, 18)),
            float(rng.uniform(2, 12)),
        )
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


# ── Tab 2 data ────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Fitting CoxPH churn model… (~30 s first run)")
def load_churn_model(
    seed: int, n_customers: int, _counter: int = 0
) -> tuple[Any, pd.DataFrame, Any] | None:
    """Simulate *n_customers*, run feature pipeline, fit CoxPH. Returns (model, X, y)."""
    try:
        from chrono_ltv.data.simulator import (
            EcommerceSimulator,
            NoiseConfig,
            SimulatorConfig,
        )
        from chrono_ltv.features.pipeline import FeaturePipeline
        from chrono_ltv.models.cox_ph import CoxPHModel

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
        ds = EcommerceSimulator(cfg).run(persist=False)
        pipe = FeaturePipeline(scale=True)
        X, y = pipe.fit_transform(
            customers=ds["customers"],
            transactions=ds["transactions"],
            clickstream=ds["clickstream"],
            tickets=ds["support_tickets"],
            labels=ds["survival_labels"],
        )
        model = CoxPHModel(alpha=0.5).fit(X, y)
        return model, X, y
    except Exception as exc:
        return exc  # surface the error in the tab


# ── Tab 1 — Multi-Source Reconciliation ──────────────────────────────────────


def render_reconciliation(orders: pd.DataFrame) -> None:
    st.subheader("Unified Order Ledger", divider="gray")

    if orders.empty:
        st.error("No orders generated. Adjust sidebar controls and try again.")
        return

    # ── KPI row ──
    gross = orders["gross_revenue"].sum()
    fees = orders["platform_fees"].sum()
    shipping = orders["shipping_cost"].sum()
    net = orders["net_contribution_margin"].sum()
    margin_pct = net / gross * 100 if gross else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gross Revenue", _fmt_usd(gross))
    k2.metric("Platform Fees", _fmt_usd(fees), delta=f"-{fees / gross * 100:.1f}%", delta_color="inverse")
    k3.metric("3PL Shipping", _fmt_usd(shipping), delta=f"-{shipping / gross * 100:.1f}%", delta_color="inverse")
    k4.metric("Net Contribution Margin", _fmt_usd(net), delta=f"{margin_pct:.1f}%")

    st.divider()

    # ── Revenue by channel ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Revenue vs. Cost by Channel**")
        ch = (
            orders.groupby("channel")[
                ["gross_revenue", "platform_fees", "shipping_cost", "net_contribution_margin"]
            ]
            .sum()
            .rename(
                columns={
                    "gross_revenue": "Gross Revenue",
                    "platform_fees": "Platform Fees",
                    "shipping_cost": "3PL Shipping",
                    "net_contribution_margin": "Net Margin",
                }
            )
        )
        st.bar_chart(ch[["Gross Revenue", "Platform Fees", "3PL Shipping", "Net Margin"]])

    with col_right:
        st.markdown("**Fee Waterfall — Aggregate (USD)**")
        try:
            import matplotlib.pyplot as plt

            labels = ["Gross Revenue", "Platform Fees", "3PL Shipping", "Net Margin"]
            values = [gross, -fees, -shipping, net]
            bottoms = [0.0, gross, gross - fees, 0.0]
            colors = ["#4c8ef5", "#e05252", "#e09d52", "#2ecc71"]

            fig, ax = plt.subplots(figsize=(5.5, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            bars = ax.bar(labels, [abs(v) for v in values], bottom=bottoms, color=colors, width=0.5)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_y() + gross * 0.01,
                    _fmt_usd(val),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="white",
                )
            ax.set_ylabel("USD", color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333")
            plt.xticks(fontsize=8, color="white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except ImportError:
            wf = pd.DataFrame(
                {"Component": labels, "Amount (USD)": [gross, fees, shipping, net]}
            ).set_index("Component")
            st.bar_chart(wf)

    st.divider()

    # ── 3PL match rate ──
    if "tpl_matched" in orders.columns:
        matched_pct = orders["tpl_matched"].mean() * 100
        unmatched = int((~orders["tpl_matched"]).sum())
        st.info(
            f"3PL match rate: **{matched_pct:.1f}%** of orders "
            f"({unmatched} unmatched → shipping cost defaulted to $0)"
        )

    # ── Carrier delay breakdown ──
    if "carrier_delay_status" in orders.columns and "carrier" in orders.columns:
        st.markdown("**Carrier Delay Rate**")
        delay = (
            orders[orders["tpl_matched"] == True]  # noqa: E712
            .groupby("carrier")["carrier_delay_status"]
            .agg(delayed="sum", total="count")
            .assign(delay_rate=lambda d: d["delayed"] / d["total"] * 100)
            .sort_values("delay_rate", ascending=False)
        )
        st.bar_chart(delay["delay_rate"], y_label="Delay Rate (%)")

    st.divider()

    # ── Detailed table ──
    st.markdown("**Order Detail (filterable)**")
    channels = ["All"] + sorted(orders["channel"].unique().tolist())
    chosen_channel = st.selectbox("Filter by channel", channels, key="recon_channel")
    view = orders if chosen_channel == "All" else orders[orders["channel"] == chosen_channel]

    display_cols = [
        c
        for c in [
            "universal_order_id",
            "channel",
            "fulfillment_type",
            "order_date",
            "gross_revenue",
            "platform_fees",
            "shipping_cost",
            "net_contribution_margin",
            "tpl_matched",
            "carrier",
            "delivery_status",
        ]
        if c in view.columns
    ]
    st.dataframe(
        view[display_cols].sort_values("gross_revenue", ascending=False).head(200),
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 2 — Survival Churn Radar ─────────────────────────────────────────────


def render_churn_radar(model_result: Any) -> None:
    st.subheader("Customer Churn Survival Radar", divider="gray")

    if model_result is None or isinstance(model_result, Exception):
        err = str(model_result) if isinstance(model_result, Exception) else "unknown"
        st.warning(
            "Churn model unavailable — install `survival` and `features` extras.\n\n"
            f"Details: `{err}`"
        )
        st.code("pip install -e '.[survival,features]'")
        return

    model, X, y = model_result
    n = len(X)

    # ── Compute risk scores and median survival ──
    risk_scores = model.predict_risk_score(X)
    times_365 = np.linspace(1, 365, 120)
    S_all = model.predict_survival_function(X, times=times_365)  # (n, 120)
    medians = model.predict_median_survival_time(X)

    # ── Risk tier segmentation ──
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

    # ── KPI row ──
    finite_med = medians[np.isfinite(medians)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers Modelled", f"{n:,}")
    k2.metric("Median Survival", f"{np.median(finite_med):.0f} days" if len(finite_med) else "N/A")
    k3.metric(
        "High + Critical Risk",
        f"{(tier_counts.get('High Risk', 0) + tier_counts.get('Critical', 0)):,}",
        delta=f"{(tier_counts.get('High Risk', 0) + tier_counts.get('Critical', 0)) / n * 100:.1f}% of base",
        delta_color="inverse",
    )
    k4.metric("Avg Risk Score", f"{float(np.mean(risk_scores)):.3f}")

    st.divider()

    # ── Two-panel matplotlib figure ──
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        _TIER_COLORS = {
            "Low Risk": "#2ecc71",
            "Medium Risk": "#f39c12",
            "High Risk": "#e74c3c",
            "Critical": "#8e44ad",
        }

        fig = plt.figure(figsize=(14, 5), facecolor="#0e1117")
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        # Left — risk score histogram
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_hist.set_facecolor("#0e1117")
        ax_hist.hist(risk_scores, bins=35, color="#4c8ef5", edgecolor="#0e1117", alpha=0.85)
        for edge, lbl, col in zip(
            tier_edges[1:4],
            ["Q25", "Q75", "Q90"],
            ["#2ecc71", "#f39c12", "#e74c3c"],
        ):
            ax_hist.axvline(edge, color=col, linestyle="--", linewidth=1, label=lbl)
        ax_hist.set_xlabel("Risk Score", color="white", fontsize=9)
        ax_hist.set_ylabel("Customers", color="white", fontsize=9)
        ax_hist.set_title("Risk Score Distribution", color="white", fontsize=10)
        ax_hist.tick_params(colors="white", labelsize=8)
        ax_hist.spines[:].set_color("#333")
        ax_hist.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e", edgecolor="#333")

        # Right — survival curves per tier
        ax_surv = fig.add_subplot(gs[0, 1])
        ax_surv.set_facecolor("#0e1117")

        for tier, color in _TIER_COLORS.items():
            mask = tier_labels_arr == tier
            if mask.sum() == 0:
                continue
            mean_curve = S_all[mask].mean(axis=0)
            count = int(mask.sum())
            ax_surv.plot(
                times_365, mean_curve, label=f"{tier} (n={count})", color=color, linewidth=2
            )

        ax_surv.set_xlabel("Days from Observation", color="white", fontsize=9)
        ax_surv.set_ylabel("P(Survived)", color="white", fontsize=9)
        ax_surv.set_title("Survival Curves by Risk Tier", color="white", fontsize=10)
        ax_surv.set_ylim(-0.02, 1.05)
        ax_surv.axhline(0.5, color="#aaa", linestyle=":", linewidth=1, label="S(t)=0.5")
        ax_surv.tick_params(colors="white", labelsize=8)
        ax_surv.spines[:].set_color("#333")
        ax_surv.legend(fontsize=7, labelcolor="white", facecolor="#1a1a2e", edgecolor="#333")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    except ImportError:
        # Fallback: Streamlit native line chart for average survival curve
        mean_S = S_all.mean(axis=0)
        st.line_chart(
            pd.DataFrame({"Mean Survival P": mean_S}, index=times_365.astype(int)),
            x_label="Days",
            y_label="Survival Probability",
        )

    st.divider()

    # ── At-Risk Customers Table ──
    st.markdown("**Top 25 At-Risk Customers**")
    risk_df = pd.DataFrame(
        {
            "customer_id": X.index,
            "risk_score": risk_scores,
            "median_survival_days": np.where(np.isfinite(medians), medians, -1),
            "risk_tier": tier_labels_arr,
        }
    )
    top_at_risk = (
        risk_df.sort_values("risk_score", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )
    top_at_risk.index += 1
    top_at_risk["median_survival_days"] = top_at_risk["median_survival_days"].apply(
        lambda v: f"{v:.0f}" if v > 0 else "≥ 365"
    )
    top_at_risk["risk_score"] = top_at_risk["risk_score"].round(4)
    st.dataframe(top_at_risk, use_container_width=True)


# ── Tab 3 — ROI Intervention Simulator ───────────────────────────────────────


def render_roi_simulator() -> None:
    st.subheader("Discounted Cash-Flow LTV Simulator", divider="gray")
    st.caption(
        "Model the financial impact of coupon interventions on customer lifetime value. "
        "All values are forward-looking projections."
    )

    # ── Two-column layout ──
    col_sliders, col_results = st.columns([1, 2], gap="large")

    with col_sliders:
        st.markdown("**Customer & Order Parameters**")
        avg_order_value: float = st.slider(
            "Avg. Order Value ($)", 10.0, 600.0, 85.0, 5.0,
            help="Mean net revenue per transaction."
        )
        orders_per_month: float = st.slider(
            "Orders / Month (baseline)", 0.2, 6.0, 1.2, 0.1
        )
        p_churn_pct: float = st.slider(
            "Monthly Churn Probability (%)", 1.0, 40.0, 8.0, 0.5
        )
        cac: float = st.slider(
            "Customer Acquisition Cost ($)", 0.0, 500.0, 50.0, 10.0
        )

        st.divider()
        st.markdown("**Coupon Intervention**")
        coupon_pct: float = st.slider(
            "Coupon Discount (%)", 0.0, 50.0, 10.0, 1.0,
            help="Revenue reduction per order (e.g. 10 % off)."
        )
        freq_lift_pct: float = st.slider(
            "Frequency Lift (%)", 0.0, 150.0, 20.0, 5.0,
            help="Expected % increase in orders/month due to the coupon."
        )

        st.divider()
        st.markdown("**Financial Parameters**")
        annual_dr_pct: float = st.slider(
            "Annual Discount Rate (%)", 0.0, 40.0, 12.0, 1.0,
            help="Weighted average cost of capital (WACC)."
        )
        horizon: int = st.slider(
            "Projection Horizon (months)", 6, 72, 24, 3
        )

    # ── Calculations ──
    p_churn = p_churn_pct / 100
    coupon_rate = coupon_pct / 100
    freq_lift = freq_lift_pct / 100
    r_monthly = (annual_dr_pct / 100) / 12

    months = np.arange(1, horizon + 1)
    survival = (1 - p_churn) ** months
    discount_factors = (1 + r_monthly) ** months

    # Baseline (no coupon)
    baseline_cf = avg_order_value * orders_per_month * survival
    baseline_dcf = baseline_cf / discount_factors
    ltv_baseline = float(baseline_dcf.sum()) - cac

    # With coupon
    adj_val = avg_order_value * (1 - coupon_rate)
    adj_freq = orders_per_month * (1 + freq_lift)
    coupon_cf = adj_val * adj_freq * survival
    coupon_dcf = coupon_cf / discount_factors
    ltv_coupon = float(coupon_dcf.sum()) - cac

    delta_ltv = ltv_coupon - ltv_baseline
    net_rev_ratio = (1 - coupon_rate) * (1 + freq_lift)
    coupon_positive = net_rev_ratio >= 1.0
    required_lift_pct = (coupon_rate / (1 - coupon_rate)) * 100 if coupon_rate < 1 else float("inf")

    with col_results:
        # ── KPI metrics ──
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline LTV", _fmt_usd(ltv_baseline))
        m2.metric(
            "LTV with Coupon",
            _fmt_usd(ltv_coupon),
            delta=f"{_fmt_usd(delta_ltv)} ({delta_ltv / abs(ltv_baseline) * 100:+.1f}%)"
            if ltv_baseline != 0
            else _fmt_usd(delta_ltv),
            delta_color="normal",
        )
        m3.metric(
            "Net Revenue Ratio",
            f"{net_rev_ratio:.3f}×",
            delta="profitable" if coupon_positive else "unprofitable",
            delta_color="normal" if coupon_positive else "inverse",
        )

        # ── Break-even banner ──
        if coupon_rate == 0:
            st.info("Set a coupon discount above 0 % to model an intervention.")
        elif coupon_positive:
            st.success(
                f"Coupon is **net-positive** — revenue ratio {net_rev_ratio:.3f}× > 1.0. "
                f"Required frequency lift to break even: **{required_lift_pct:.1f} %** "
                f"(current: {freq_lift_pct:.1f} %)."
            )
        else:
            st.error(
                f"Coupon is **net-negative** — revenue ratio {net_rev_ratio:.3f}× < 1.0. "
                f"Need **{required_lift_pct:.1f} %** frequency lift to break even "
                f"(current: {freq_lift_pct:.1f} %)."
            )

        st.divider()

        # ── DCF cash-flow chart ──
        st.markdown("**Monthly DCF Cash-Flow Projection**")
        cf_df = pd.DataFrame(
            {
                "Baseline (No Coupon)": baseline_dcf,
                "With Coupon": coupon_dcf,
            },
            index=months,
        )
        cf_df.index.name = "Month"
        st.line_chart(cf_df, x_label="Month", y_label="DCF Cash Flow ($)")

        # ── Cumulative LTV chart ──
        st.markdown("**Cumulative Discounted LTV**")
        cum_df = pd.DataFrame(
            {
                "Baseline": np.cumsum(baseline_dcf) - cac,
                "With Coupon": np.cumsum(coupon_dcf) - cac,
            },
            index=months,
        )
        cum_df.index.name = "Month"
        st.line_chart(cum_df, x_label="Month", y_label="Cumulative LTV ($)")

        # ── Sensitivity: net_rev_ratio at various coupon/lift combos ──
        st.divider()
        st.markdown("**Break-Even Sensitivity: Freq. Lift Needed to Break Even**")
        coupon_range = np.arange(5, 55, 5)
        be_lifts = (coupon_range / 100) / (1 - coupon_range / 100) * 100
        be_df = pd.DataFrame(
            {"Required Freq. Lift (%)": be_lifts},
            index=coupon_range,
        )
        be_df.index.name = "Coupon Discount (%)"
        st.bar_chart(be_df, x_label="Coupon Discount (%)", y_label="Min Frequency Lift (%)")


# ── main layout ───────────────────────────────────────────────────────────────


def main() -> None:
    st.title("ChronoLTV — Analytics Dashboard")
    st.caption(
        "Real-time multi-source revenue reconciliation · Churn survival radar · "
        "Coupon ROI simulator"
    )

    # Load data (cached)
    orders = load_orders(seed, n_shopify, n_amazon)
    model_result = load_churn_model(
        seed, n_model_customers, st.session_state["refit_counter"]
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊  Multi-Source Reconciliation", "🎯  Survival Churn Radar", "💰  ROI Intervention Simulator"]
    )

    with tab1:
        render_reconciliation(orders)

    with tab2:
        render_churn_radar(model_result)

    with tab3:
        render_roi_simulator()


main()
