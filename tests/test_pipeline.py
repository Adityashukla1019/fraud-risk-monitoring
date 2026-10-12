import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate_data import generate
from src.features import FEATURES, build_features
from src.run_sql_analysis import parse_queries
from src.train import recall_at_precision, top_k_capture


@pytest.fixture(scope="module")
def df():
    return generate(n_rows=20_000, fraud_rate=0.01, seed=1)


def test_fraud_only_in_transfer_and_cash_out(df):
    assert set(df.loc[df.isFraud == 1, "type"]) <= {"TRANSFER", "CASH_OUT"}


def test_fraud_rate_close_to_target(df):
    assert 0.008 <= df.isFraud.mean() <= 0.012


def test_no_nulls_or_negative_amounts(df):
    assert df.isna().sum().sum() == 0
    assert (df.amount > 0).all()


def test_feature_columns_and_no_leakage(df):
    X = build_features(df)
    assert list(X.columns) == FEATURES
    assert "isFraud" not in X.columns and "isFlaggedFraud" not in X.columns
    assert len(X) == len(df)


def test_sql_file_parses_all_queries():
    qs = parse_queries((Path(__file__).resolve().parents[1] / "sql" / "analysis.sql").read_text())
    assert {"fraud_by_type", "fraud_by_hour", "legacy_rule_performance"} <= set(qs)
    def body(q):  # drop leading SQL comment lines
        return "\n".join(l for l in q.splitlines() if not l.strip().startswith("--")).strip().upper()
    assert all(body(q).startswith(("SELECT", "WITH")) for q in qs.values())


def test_recall_at_precision_perfect_model():
    y = np.array([0] * 90 + [1] * 10)
    p = np.array([0.1] * 90 + [0.9] * 10)
    rec, _ = recall_at_precision(y, p, 0.9)
    assert rec == 1.0


def test_top_k_capture():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert top_k_capture(y, p, 0.1) == 0.5
