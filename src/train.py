"""Train + evaluate fraud models with a time-based split.

Why a time split? In production we score *future* transactions; a random split
would leak future patterns into training and overstate performance.

Metrics reported: PR-AUC (primary), ROC-AUC, recall @ >=90% precision,
and the share of fraud caught in the top 1% highest-risk transactions.
Accuracy is deliberately NOT used - with 0.4% fraud, "predict never fraud"
is 99.6% accurate and useless.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

try:  # allow both `python -m src.train` and `python src/train.py`
    from src.features import FEATURES, build_features
except ImportError:  # pragma: no cover
    from features import FEATURES, build_features

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def recall_at_precision(y, p, min_precision=0.90):
    prec, rec, thr = precision_recall_curve(y, p)
    ok = prec[:-1] >= min_precision
    if not ok.any():
        return 0.0, None
    i = np.where(ok)[0][np.argmax(rec[:-1][ok])]
    return float(rec[i]), float(thr[i])


def top_k_capture(y, p, frac=0.01):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    return float(np.asarray(y)[idx].sum() / np.asarray(y).sum())


def evaluate(name, y, p):
    rec, thr = recall_at_precision(y, p, 0.90)
    rec50, thr50 = recall_at_precision(y, p, 0.50)
    return {
        "model": name,
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "recall_at_90_precision": round(rec, 4),
        "threshold_at_90_precision": thr,
        "recall_at_50_precision": round(rec50, 4),
        "fraud_captured_in_top_1pct": round(top_k_capture(y, p, 0.01), 4),
    }


def main(db: Path | None = None):
    db = db or ROOT / "data" / "fraud.db"
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    con = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM transactions", con)
    con.close()

    cutoff = df["step"].quantile(0.70)
    train, test = df[df["step"] <= cutoff], df[df["step"] > cutoff]
    Xtr, ytr = build_features(train), train["isFraud"]
    Xte, yte = build_features(test), test["isFraud"]
    print(f"Train {len(train):,} ({ytr.mean():.3%} fraud) | Test {len(test):,} ({yte.mean():.3%} fraud)")

    # --- baseline: logistic regression with class weights ------------------
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    # --- XGBoost with scale_pos_weight for class imbalance -----------------
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=spw ** 0.5, eval_metric="aucpr",
                        n_jobs=-1, random_state=SEED, tree_method="hist")
    xgb.fit(Xtr, ytr)
    p_xgb = xgb.predict_proba(Xte)[:, 1]

    # --- legacy rule baseline ---------------------------------------------
    legacy = test["isFlaggedFraud"].values
    tp = int(((legacy == 1) & (yte == 1)).sum())
    legacy_metrics = {"model": "Legacy rule (TRANSFER > 200K)",
                      "recall": round(tp / int(yte.sum()), 4),
                      "precision": round(tp / max(1, int(legacy.sum())), 4)}

    results = [evaluate("Logistic Regression", yte, p_lr), evaluate("XGBoost", yte, p_xgb)]
    metrics = {"train_rows": len(train), "test_rows": len(test),
               "test_fraud_rate": round(float(yte.mean()), 5),
               "models": results, "legacy_rule": legacy_metrics}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # --- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, p in [("Logistic Regression", p_lr), ("XGBoost", p_xgb)]:
        pr, rc, _ = precision_recall_curve(yte, p)
        ax.plot(rc, pr, label=f"{name} (PR-AUC {average_precision_score(yte, p):.3f})")
    ax.axhline(yte.mean(), ls="--", c="grey", label="No-skill baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall curve (time-based test set)")
    ax.legend(); fig.tight_layout(); fig.savefig(ROOT / "reports/figures/pr_curve.png", dpi=140); plt.close(fig)

    imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot.barh(ax=ax); ax.set_title("XGBoost feature importance")
    fig.tight_layout(); fig.savefig(ROOT / "reports/figures/feature_importance.png", dpi=140); plt.close(fig)

    # --- scored test set for the dashboard -------------------------------------
    # Risk tiers follow an "alert budget": analysts can only review a limited share of traffic.
    scored = test.copy()
    scored["risk_score"] = p_xgb
    pct = scored["risk_score"].rank(pct=True, ascending=False)  # 0 = riskiest
    scored["risk_tier"] = pd.cut(pct, bins=[0, 0.005, 0.01, 0.05, 1.0],
                                 labels=["Critical", "High", "Medium", "Low"], include_lowest=True)
    scored = scored.sort_values("risk_score", ascending=False)
    scored.to_csv(ROOT / "reports/scored_test_set.csv", index=False)
    scored[scored["risk_tier"].isin(["Critical", "High"])].to_csv(ROOT / "reports/flagged_transactions.csv", index=False)

    joblib.dump(xgb, ROOT / "models" / "xgb_fraud.joblib")
    return metrics


if __name__ == "__main__":
    main()
