# Fraud Monitoring - Findings & Recommendations

**To:** Head of Payments Risk  |  **From:** Data Analytics  |  **Data:** synthetic PaySim-style transactions (600,000 rows)

## Headline
A gradient-boosted model reviewing only the **riskiest 1% of transactions** catches **69% of fraud** in a held-out future period, versus **10%** for the current rule (TRANSFER > 200K). In value terms the Critical+High queue covers **88%** of fraudulent amount.

## What the data shows
1. **Fraud is concentrated in two channels.** CASH_OUT and TRANSFER carry 100% of fraud; CASH_OUT alone accounts for 81% of fraud value.
2. **Size matters.** Fraud rate in the 5) >1M band is **35.2%**, vs 0.01% in the 1) <10K band.
3. **Account-draining is a strong signal.** Drained-origin transactions are fraudulent 2.47% of the time vs 0.27% otherwise.
4. **Night-time spike.** Hours 00-05 hold 25% of volume but 59% of fraud.

## Model performance (time-based hold-out)
| Model | PR-AUC | Recall @ 50% precision | Fraud caught in top 1% |
|---|---|---|---|
| Logistic Regression | 0.435 | 42% | 68% |
| XGBoost | 0.478 | 47% | 69% |
| Legacy rule | - | 10% recall at 17% precision | - |

## Recommendations
1. **Replace the static amount rule with the risk score** and route Critical/High tiers (top 1%) to analysts.
2. **Add step-up authentication** for CASH_OUT/TRANSFER above 100K, especially 00-05h.
3. **Monitor repeat beneficiary accounts** (see `top_risky_destinations`) as mule candidates.
4. **Re-train monthly** and track PR-AUC drift; set an alert if recall at the 1% budget falls below an agreed floor.

## Limitations
- Data is synthetic; absolute numbers will differ on production data.
- Labels reflect confirmed fraud only; unreported fraud is invisible to the model.
- Trade-off: widening the review budget increases recall but also analyst workload and customer friction.
