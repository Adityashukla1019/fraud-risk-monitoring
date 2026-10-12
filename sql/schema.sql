-- Schema (SQLite syntax; works in PostgreSQL with minor type tweaks: REAL -> NUMERIC, INTEGER -> BIGINT)
DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    tx_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    step            INTEGER NOT NULL,          -- hour of simulation (1..744)
    type            TEXT    NOT NULL,
    amount          REAL    NOT NULL,
    nameOrig        TEXT    NOT NULL,
    oldbalanceOrg   REAL,
    newbalanceOrig  REAL,
    nameDest        TEXT    NOT NULL,
    oldbalanceDest  REAL,
    newbalanceDest  REAL,
    isFraud         INTEGER NOT NULL CHECK (isFraud IN (0,1)),
    isFlaggedFraud  INTEGER NOT NULL CHECK (isFlaggedFraud IN (0,1))
);
CREATE INDEX idx_tx_type  ON transactions(type);
CREATE INDEX idx_tx_step  ON transactions(step);
CREATE INDEX idx_tx_fraud ON transactions(isFraud);
