"""Load the CSV into a SQLite database using sql/schema.sql."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(csv: Path, db: Path) -> int:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript((ROOT / "sql" / "schema.sql").read_text())
    total = 0
    for chunk in pd.read_csv(csv, chunksize=200_000):
        chunk.to_sql("transactions", con, if_exists="append", index=False)
        total += len(chunk)
    con.commit()
    con.close()
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "transactions.csv"))
    ap.add_argument("--db", default=str(ROOT / "data" / "fraud.db"))
    a = ap.parse_args()
    print(f"Loaded {load(Path(a.csv), Path(a.db)):,} rows -> {a.db}")
