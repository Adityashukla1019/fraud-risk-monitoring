"""Generate a synthetic, PaySim-style mobile-money transaction dataset.

Why synthetic? It lets the project run end-to-end with zero downloads.
Using the real data instead? Download PaySim from Kaggle
(ealaxi/paysim1) and save it as data/transactions.csv - the rest of the
pipeline uses the exact same column names.

Fraud patterns baked in (mirroring what is publicly known about PaySim):
  * fraud occurs only in TRANSFER and CASH_OUT
  * fraudsters tend to drain the origin account
  * fraud amounts skew high; destination accounts are often empty "mule" accounts
  * a share of legitimate traffic looks suspicious too (noise), so the
    problem is not trivially separable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"]
TYPE_P = [0.34, 0.22, 0.35, 0.08, 0.01]


def generate(n_rows: int = 600_000, fraud_rate: float = 0.004, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows

    step = np.sort(rng.integers(1, 745, n))  # 31 days of hourly steps
    tx_type = rng.choice(TYPES, n, p=TYPE_P)
    amount = np.round(rng.lognormal(mean=9.5, sigma=1.4, size=n), 2)

    old_org = np.round(rng.lognormal(mean=10.0, sigma=2.0, size=n), 2)
    old_org[rng.random(n) < 0.30] = 0.0  # many empty accounts
    old_dest = np.round(rng.lognormal(mean=11.0, sigma=2.0, size=n), 2)
    old_dest[rng.random(n) < 0.35] = 0.0

    # --- assign fraud (only TRANSFER / CASH_OUT) -------------------------
    eligible = np.isin(tx_type, ["TRANSFER", "CASH_OUT"])
    n_fraud = int(n * fraud_rate)
    fraud_idx = rng.choice(np.where(eligible)[0], n_fraud, replace=False)
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[fraud_idx] = 1

    # fraud amounts skew high; origin accounts get drained (with noise)
    amount[fraud_idx] = np.round(rng.lognormal(mean=12.2, sigma=1.1, size=n_fraud), 2)
    drain = rng.random(n_fraud) < 0.80
    old_org[fraud_idx[drain]] = amount[fraud_idx[drain]]
    old_org[fraud_idx[~drain]] = amount[fraud_idx[~drain]] * rng.uniform(1.0, 3.0, (~drain).sum())
    mule = rng.random(n_fraud) < 0.65
    old_dest[fraud_idx[mule]] = 0.0

    # night-time bias for fraud (hours 0-5)
    night = rng.random(n_fraud) < 0.45
    fi = fraud_idx[night]
    step[fi] = ((step[fi] - 1) // 24) * 24 + rng.integers(0, 6, night.sum()) + 1

    # legitimate "look-alikes": big transfers / drained accounts that are fine
    legit_elig = np.where(eligible & (is_fraud == 0))[0]
    lookalike = rng.choice(legit_elig, int(len(legit_elig) * 0.004), replace=False)
    amount[lookalike] = np.round(rng.lognormal(mean=11.8, sigma=1.0, size=len(lookalike)), 2)
    old_org[lookalike] = amount[lookalike]
    is_look = np.zeros(n, dtype=bool)
    is_look[lookalike] = True

    # --- balances ---------------------------------------------------------
    debit_types = np.isin(tx_type, ["PAYMENT", "CASH_OUT", "TRANSFER", "DEBIT"])
    # normal customers must have enough funds; fraud & look-alikes are set above
    needs_funds = debit_types & (old_org < amount) & (is_fraud == 0) & ~is_look
    old_org = np.where(needs_funds, amount + old_org, old_org)
    new_org = np.where(debit_types, np.maximum(old_org - amount, 0), old_org)
    new_org = np.where(tx_type == "CASH_IN", old_org + amount, new_org)

    merchant = tx_type == "PAYMENT"  # PaySim hides merchant balances
    new_dest = np.where(np.isin(tx_type, ["TRANSFER", "CASH_OUT"]), old_dest + amount, old_dest)
    old_dest = np.where(merchant, 0.0, old_dest)
    new_dest = np.where(merchant, 0.0, new_dest)

    # a few percent of balance-update glitches in legit data
    glitch = rng.random(n) < 0.02
    new_org = np.where(glitch & (is_fraud == 0), old_org, new_org)

    name_orig = np.char.add("C", rng.integers(10_000_000, 99_999_999, n).astype(str))
    dest_prefix = np.where(merchant, "M", "C")
    name_dest = np.char.add(dest_prefix, rng.integers(10_000_000, 99_999_999, n).astype(str))
    # a small pool of repeat "mule" destinations for fraud, to make the SQL ranking interesting
    mule_pool = np.char.add("C", rng.integers(90_000_000, 90_000_400, n_fraud).astype(str))
    name_dest[fraud_idx] = mule_pool

    flagged = ((tx_type == "TRANSFER") & (amount > 200_000)).astype(int)  # naive legacy rule

    return pd.DataFrame({
        "step": step, "type": tx_type, "amount": amount,
        "nameOrig": name_orig, "oldbalanceOrg": np.round(old_org, 2), "newbalanceOrig": np.round(new_org, 2),
        "nameDest": name_dest, "oldbalanceDest": np.round(old_dest, 2), "newbalanceDest": np.round(new_dest, 2),
        "isFraud": is_fraud, "isFlaggedFraud": flagged,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600_000)
    ap.add_argument("--fraud-rate", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/transactions.csv")
    a = ap.parse_args()
    df = generate(a.rows, a.fraud_rate, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} rows -> {a.out}  (fraud rate {df.isFraud.mean():.3%})")
