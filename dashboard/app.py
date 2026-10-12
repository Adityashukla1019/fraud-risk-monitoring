"""Streamlit risk dashboard.  Run:  streamlit run dashboard/app.py"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.run_sql_analysis import parse_queries  # noqa: E402

st.set_page_config(page_title="Fraud Risk Monitor", layout="wide", page_icon="🛡️")


@st.cache_data
def sql_results():
    con = sqlite3.connect(ROOT / "data" / "fraud.db")
    qs = parse_queries((ROOT / "sql" / "analysis.sql").read_text())
    out = {k: pd.read_sql_query(v, con) for k, v in qs.items()}
    con.close()
    return out


@st.cache_data
def scored():
    return pd.read_csv(ROOT / "reports" / "scored_test_set.csv")


try:
    res, df = sql_results(), scored()
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
except FileNotFoundError:
    st.error("Run the pipeline first:  `make all`  (see README).")
    st.stop()

st.title("🛡️ Transaction Fraud Risk Monitor")
st.caption("Synthetic PaySim-style data · XGBoost risk scoring · time-based hold-out")

# ---- Filters -------------------------------------------------------------
st.sidebar.header("Filters (scored test period)")
types = st.sidebar.multiselect("Transaction type", sorted(df["type"].unique()), default=sorted(df["type"].unique()))
tiers = st.sidebar.multiselect("Risk tier", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"])
min_amt = st.sidebar.number_input("Minimum amount", 0, int(df.amount.max()), 0, step=10_000)
f = df[df["type"].isin(types) & df["risk_tier"].isin(tiers) & (df["amount"] >= min_amt)]

# ---- KPIs --------------------------------------------------------------------
all_tx = res["fraud_by_type"]
xgb = next(m for m in metrics["models"] if m["model"] == "XGBoost")
c = st.columns(5)
c[0].metric("Transactions (all)", f"{all_tx.n_tx.sum():,}")
c[1].metric("Fraud rate", f"{all_tx.n_fraud.sum() / all_tx.n_tx.sum():.2%}")
c[2].metric("Fraud value", f"{all_tx.fraud_value.sum() / 1e6:,.0f}M")
c[3].metric("Fraud caught in top 1%", f"{xgb['fraud_captured_in_top_1pct']:.0%}",
            delta=f"vs {metrics['legacy_rule']['recall']:.0%} legacy rule")
c[4].metric("Model PR-AUC", f"{xgb['pr_auc']:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends & Drivers", "🎯 Risk Scorecard", "🚩 Flagged Transactions", "🧪 Model Performance"])

with tab1:
    a, b = st.columns(2)
    d = res["daily_fraud_trend_with_7d_avg"]
    a.plotly_chart(px.line(d, x="day", y=["n_fraud", "fraud_7d_avg"], title="Daily fraud count & 7-day average"), width="stretch")
    h = res["fraud_by_hour"]
    b.plotly_chart(px.bar(h, x="hour_of_day", y="fraud_rate_pct", title="Fraud rate by hour of day (%)"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(res["fraud_by_type"], x="type", y="fraud_value", title="Fraud value by transaction type"), width="stretch")
    b.plotly_chart(px.bar(res["fraud_by_amount_band"], x="amount_band", y="fraud_rate_pct", title="Fraud rate by amount band (TRANSFER/CASH_OUT, %)"), width="stretch")
    st.subheader("Root-cause: drained origin accounts")
    st.dataframe(res["drained_account_signal"], hide_index=True)
    st.subheader("Mule-account candidates (top destinations by fraud value)")
    st.dataframe(res["top_risky_destinations"], hide_index=True)

with tab2:
    sc = (f.groupby("risk_tier", observed=True)
            .agg(transactions=("isFraud", "size"), confirmed_fraud=("isFraud", "sum"), value=("amount", "sum"))
            .reindex(["Critical", "High", "Medium", "Low"]).dropna().reset_index())
    sc["fraud_rate_%"] = (100 * sc.confirmed_fraud / sc.transactions).round(2)
    st.dataframe(sc, hide_index=True, width="stretch")
    st.plotly_chart(px.bar(sc, x="risk_tier", y="fraud_rate_%", color="risk_tier", title="Observed fraud rate by risk tier",
                           color_discrete_map={"Critical": "#b91c1c", "High": "#ea580c", "Medium": "#ca8a04", "Low": "#16a34a"}),
                    width="stretch")
    st.caption("Tiers use an alert budget: Critical = top 0.5% of scores, High = next 0.5%, Medium = next 4%, Low = rest.")

with tab3:
    st.write(f"{len(f):,} transactions match the filters. Showing the 500 highest-risk.")
    cols = ["step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig", "risk_score", "risk_tier", "isFraud"]
    top = f.head(500)[cols]
    st.dataframe(top, hide_index=True, width="stretch")
    st.download_button("⬇️ Download flagged list (CSV)", top.to_csv(index=False), "flagged_transactions.csv")

with tab4:
    st.dataframe(pd.DataFrame(metrics["models"]).drop(columns=["threshold_at_90_precision"]), hide_index=True)
    st.json(metrics["legacy_rule"])
    st.image(str(ROOT / "reports/figures/pr_curve.png"))
    st.image(str(ROOT / "reports/figures/feature_importance.png"))
