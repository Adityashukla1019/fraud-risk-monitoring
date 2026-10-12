-- Each query is tagged with "-- name:" so src/run_sql_analysis.py can run them individually.

-- name: fraud_by_type
SELECT type,
       COUNT(*)                                   AS n_tx,
       SUM(isFraud)                               AS n_fraud,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 3)  AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN isFraud=1 THEN amount ELSE 0 END), 0) AS fraud_value
FROM transactions
GROUP BY type
ORDER BY fraud_value DESC;

-- name: fraud_by_hour
SELECT (step - 1) % 24 AS hour_of_day,
       COUNT(*)        AS n_tx,
       SUM(isFraud)    AS n_fraud,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- name: fraud_by_amount_band
WITH banded AS (
    SELECT *,
        CASE WHEN amount <   10000 THEN '1) <10K'
             WHEN amount <  100000 THEN '2) 10K-100K'
             WHEN amount <  500000 THEN '3) 100K-500K'
             WHEN amount < 1000000 THEN '4) 500K-1M'
             ELSE                       '5) >1M' END AS amount_band
    FROM transactions
    WHERE type IN ('TRANSFER','CASH_OUT')
)
SELECT amount_band,
       COUNT(*) AS n_tx,
       SUM(isFraud) AS n_fraud,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 3) AS fraud_rate_pct
FROM banded
GROUP BY amount_band
ORDER BY amount_band;

-- name: drained_account_signal
-- Root-cause style query: does emptying the origin account predict fraud?
SELECT CASE WHEN newbalanceOrig = 0 AND oldbalanceOrg > 0 THEN 'origin_drained' ELSE 'not_drained' END AS pattern,
       COUNT(*) AS n_tx,
       SUM(isFraud) AS n_fraud,
       ROUND(100.0 * SUM(isFraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
WHERE type IN ('TRANSFER','CASH_OUT')
GROUP BY pattern;

-- name: daily_fraud_trend_with_7d_avg
-- Window function: rolling 7-day average of daily fraud count
WITH daily AS (
    SELECT (step - 1) / 24 + 1 AS day,
           COUNT(*) AS n_tx,
           SUM(isFraud) AS n_fraud
    FROM transactions
    GROUP BY day
)
SELECT day, n_tx, n_fraud,
       ROUND(AVG(n_fraud) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS fraud_7d_avg
FROM daily
ORDER BY day;

-- name: top_risky_destinations
-- Destination accounts receiving the most fraudulent value (mule candidates)
SELECT nameDest,
       COUNT(*)  AS n_fraud_tx,
       ROUND(SUM(amount), 0) AS fraud_value,
       RANK() OVER (ORDER BY SUM(amount) DESC) AS value_rank
FROM transactions
WHERE isFraud = 1
GROUP BY nameDest
ORDER BY value_rank
LIMIT 10;

-- name: legacy_rule_performance
-- How good is the existing naive rule (isFlaggedFraud) vs reality?
SELECT SUM(CASE WHEN isFlaggedFraud=1 AND isFraud=1 THEN 1 ELSE 0 END) AS true_positives,
       SUM(CASE WHEN isFlaggedFraud=1 AND isFraud=0 THEN 1 ELSE 0 END) AS false_positives,
       SUM(CASE WHEN isFlaggedFraud=0 AND isFraud=1 THEN 1 ELSE 0 END) AS missed_fraud,
       ROUND(100.0 * SUM(CASE WHEN isFlaggedFraud=1 AND isFraud=1 THEN 1 ELSE 0 END)
             / NULLIF(SUM(isFraud),0), 2) AS legacy_recall_pct
FROM transactions;
