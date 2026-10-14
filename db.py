import sqlite3
from contextlib import closing

import pandas as pd

from config import DB_PATH, PROCESSED_PATH

SEGMENTS = {"category", "channel", "country", "risk_tier"}


def build_database():
    df = pd.read_csv(PROCESSED_PATH)
    with closing(sqlite3.connect(DB_PATH)) as con:
        df.to_sql("transactions", con, if_exists="replace", index=False)
        for col in ("date", "merchant_id", "user_id", "category"):
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON transactions({col})")
        con.commit()


def run(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as con:
        return pd.read_sql(sql, con, params=params)


def where(start, end, categories=(), channels=(), tiers=()):
    clauses, params = ["date BETWEEN ? AND ?"], [start, end]
    for col, values in (("category", categories), ("channel", channels), ("risk_tier", tiers)):
        if values:
            clauses.append(f"{col} IN ({','.join('?' * len(values))})")
            params += list(values)
    return "WHERE " + " AND ".join(clauses), params


def bounds():
    return run("SELECT MIN(date) AS lo, MAX(date) AS hi FROM transactions").iloc[0]


def daily(w, p):
    return run(f"""
        SELECT date, COUNT(*) AS txns, SUM(is_fraud) AS fraud, SUM(amount) AS volume,
               SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_amount,
               SUM(risk_score) AS risk_sum,
               SUM(CASE WHEN risk_tier IN ('High', 'Critical') THEN 1 ELSE 0 END) AS high_cnt
        FROM transactions {w} GROUP BY date ORDER BY date""", p)


def segment(col, w, p):
    assert col in SEGMENTS
    return run(f"""
        SELECT {col} AS segment, COUNT(*) AS txns, SUM(is_fraud) AS fraud,
               100.0 * AVG(is_fraud) AS fraud_rate
        FROM transactions {w} GROUP BY {col} ORDER BY fraud_rate DESC""", p)


def heat(w, p):
    return run(f"""
        SELECT weekday, hour, COUNT(*) AS txns, SUM(is_fraud) AS fraud
        FROM transactions {w} GROUP BY weekday, hour""", p)


def merchants(w, p):
    return run(f"""
        SELECT merchant_id, category, COUNT(*) AS txns, SUM(is_fraud) AS fraud_cases,
               100.0 * AVG(is_fraud) AS fraud_rate, AVG(risk_score) AS avg_risk, SUM(amount) AS volume
        FROM transactions {w} GROUP BY merchant_id, category HAVING COUNT(*) >= 20
        ORDER BY fraud_rate DESC, fraud_cases DESC LIMIT 15""", p)


def users(w, p):
    return run(f"""
        SELECT user_id, home_country, COUNT(*) AS txns, SUM(is_fraud) AS fraud_cases,
               100.0 * AVG(is_fraud) AS fraud_rate, AVG(risk_score) AS avg_risk,
               SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_amount
        FROM transactions {w} GROUP BY user_id, home_country HAVING COUNT(*) >= 8
        ORDER BY fraud_rate DESC, fraud_cases DESC LIMIT 15""", p)
