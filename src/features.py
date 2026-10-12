"""Feature engineering shared by training and scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

FEATURES = [
    "amount", "log_amount", "hour", "is_night",
    "err_orig", "err_dest", "orig_drained", "dest_empty_before",
    "amount_to_balance", "orig_balance_zero",
] + [f"type_{t}" for t in TYPES]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["amount"] = df["amount"]
    X["log_amount"] = np.log1p(df["amount"])
    X["hour"] = (df["step"] - 1) % 24
    X["is_night"] = (X["hour"] < 6).astype(int)
    # accounting-identity errors: legit transactions usually balance; fraud often does not
    X["err_orig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    X["err_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    X["orig_drained"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
    X["dest_empty_before"] = ((df["oldbalanceDest"] == 0) & (df["amount"] > 0)).astype(int)
    X["amount_to_balance"] = (df["amount"] / (df["oldbalanceOrg"] + 1)).clip(upper=10)
    X["orig_balance_zero"] = (df["oldbalanceOrg"] == 0).astype(int)
    for t in TYPES:
        X[f"type_{t}"] = (df["type"] == t).astype(int)
    return X[FEATURES]
