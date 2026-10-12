"""Run every named query in sql/analysis.sql and save results as CSVs in reports/sql/."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_queries(text: str) -> dict[str, str]:
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run(db: Path | None = None, verbose: bool = True) -> dict[str, pd.DataFrame]:
    db = db or ROOT / "data" / "fraud.db"
    queries = parse_queries((ROOT / "sql" / "analysis.sql").read_text())
    out_dir = ROOT / "reports" / "sql"
    out_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    results = {}
    for name, q in queries.items():
        df = pd.read_sql_query(q, con)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        results[name] = df
        if verbose:
            print(f"\n=== {name} ===\n{df.head(10).to_string(index=False)}")
    con.close()
    return results


if __name__ == "__main__":
    run()
