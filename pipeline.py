import pandas as pd

from config import (CATEGORY_POINTS, CHANNELS, DB_PATH, PROCESSED_PATH, RAW_PATH,
                    TIER_BINS, TIER_ORDER)
from db import build_database
from generate_data import write_raw


def clean(df):
    df = df.drop_duplicates("txn_id").copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "amount"])
    df["amount"] = df["amount"].abs()
    df["channel"] = df["channel"].str.strip().str.upper().map(CHANNELS)
    df = df.dropna(subset=["channel"])
    df["country"] = df["country"].fillna("UNK")
    return df.reset_index(drop=True)


def engineer(df):
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.dayofweek
    df["is_night"] = (df["hour"] <= 4).astype(int)
    df["geo_mismatch"] = (df["country"] != df["home_country"]).astype(int)
    df["new_account"] = (df["account_age_days"] < 30).astype(int)
    df["high_amount"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
    df["daily_user_txns"] = df.groupby(["user_id", "date"])["txn_id"].transform("count")
    df["velocity_flag"] = (df["daily_user_txns"] >= 4).astype(int)
    score = (20 * df["is_night"] + 25 * df["high_amount"] + 20 * df["new_account"]
             + 15 * df["geo_mismatch"] + 15 * df["velocity_flag"] + df["category"].map(CATEGORY_POINTS))
    df["risk_score"] = score.clip(0, 100)
    df["risk_tier"] = pd.cut(df["risk_score"], TIER_BINS, labels=TIER_ORDER).astype(str)
    return df


def run_all():
    if not RAW_PATH.exists():
        write_raw()
    df = engineer(clean(pd.read_csv(RAW_PATH)))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    build_database()
    return df


def ensure_ready():
    if not DB_PATH.exists():
        run_all()


if __name__ == "__main__":
    out = run_all()
    print(f"{len(out):,} transactions | fraud rate {out['is_fraud'].mean():.2%}")
