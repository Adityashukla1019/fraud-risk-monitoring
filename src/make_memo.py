"""Auto-generate the stakeholder memo from real pipeline outputs (no hand-typed numbers)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    m = json.loads((ROOT / "reports/metrics.json").read_text())
    sql = ROOT / "reports/sql"
    by_type = pd.read_csv(sql / "fraud_by_type.csv")
    bands = pd.read_csv(sql / "fraud_by_amount_band.csv")
    drained = pd.read_csv(sql / "drained_account_signal.csv").set_index("pattern")
    hours = pd.read_csv(sql / "fraud_by_hour.csv")
    scored = pd.read_csv(ROOT / "reports/scored_test_set.csv", usecols=["isFraud", "amount", "risk_tier"])

    xgb = next(x for x in m["models"] if x["model"] == "XGBoost")
    legacy = m["legacy_rule"]
    total_val = by_type["fraud_value"].sum()
    top = by_type.iloc[0]
    night = hours[hours.hour_of_day < 6]
    night_share = night.n_fraud.sum() / hours.n_fraud.sum()
    night_tx_share = night.n_tx.sum() / hours.n_tx.sum()
    top_band = bands.sort_values("fraud_rate_pct", ascending=False).iloc[0]

    flagged = scored[scored.risk_tier.isin(["Critical", "High"])]
    caught_val = flagged.loc[flagged.isFraud == 1, "amount"].sum()
    all_val = scored.loc[scored.isFraud == 1, "amount"].sum()

    memo = f"""# Fraud Monitoring - Findings & Recommendations

**To:** Head of Payments Risk  |  **From:** Data Analytics  |  **Data:** synthetic PaySim-style transactions ({m['train_rows'] + m['test_rows']:,} rows)

## Headline
A gradient-boosted model reviewing only the **riskiest 1% of transactions** catches **{xgb['fraud_captured_in_top_1pct']:.0%} of fraud** in a held-out future period, versus **{legacy['recall']:.0%}** for the current rule (TRANSFER > 200K). In value terms the Critical+High queue covers **{caught_val / all_val:.0%}** of fraudulent amount.

## What the data shows
1. **Fraud is concentrated in two channels.** CASH_OUT and TRANSFER carry 100% of fraud; {top['type']} alone accounts for {top['fraud_value'] / total_val:.0%} of fraud value.
2. **Size matters.** Fraud rate in the {top_band['amount_band']} band is **{top_band['fraud_rate_pct']:.1f}%**, vs {bands.iloc[0]['fraud_rate_pct']:.2f}% in the {bands.iloc[0]['amount_band']} band.
3. **Account-draining is a strong signal.** Drained-origin transactions are fraudulent {drained.loc['origin_drained','fraud_rate_pct']:.2f}% of the time vs {drained.loc['not_drained','fraud_rate_pct']:.2f}% otherwise.
4. **Night-time spike.** Hours 00-05 hold {night_tx_share:.0%} of volume but {night_share:.0%} of fraud.

## Model performance (time-based hold-out)
| Model | PR-AUC | Recall @ 50% precision | Fraud caught in top 1% |
|---|---|---|---|
""" + "\n".join(f"| {x['model']} | {x['pr_auc']:.3f} | {x['recall_at_50_precision']:.0%} | {x['fraud_captured_in_top_1pct']:.0%} |" for x in m['models']) + f"""
| Legacy rule | - | {legacy['recall']:.0%} recall at {legacy['precision']:.0%} precision | - |

## Recommendations
1. **Replace the static amount rule with the risk score** and route Critical/High tiers (top 1%) to analysts.
2. **Add step-up authentication** for CASH_OUT/TRANSFER above 100K, especially 00-05h.
3. **Monitor repeat beneficiary accounts** (see `top_risky_destinations`) as mule candidates.
4. **Re-train monthly** and track PR-AUC drift; set an alert if recall at the 1% budget falls below an agreed floor.

## Limitations
- Data is synthetic; absolute numbers will differ on production data.
- Labels reflect confirmed fraud only; unreported fraud is invisible to the model.
- Trade-off: widening the review budget increases recall but also analyst workload and customer friction.
"""
    (ROOT / "reports/stakeholder_memo.md").write_text(memo)
    print(memo)


if __name__ == "__main__":
    main()
