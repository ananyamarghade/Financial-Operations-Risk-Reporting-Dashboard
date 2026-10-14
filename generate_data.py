import numpy as np
import pandas as pd

from config import CATEGORY_BASE_RISK, COUNTRIES, DAYS, RAW_PATH, START


def generate(n=60000, seed=7):
    rng = np.random.default_rng(seed)
    n_users, n_merchants = 2500, 300

    users = pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(n_users)],
        "home_country": rng.choice(COUNTRIES, n_users, p=[0.5, 0.15, 0.1, 0.1, 0.08, 0.07]),
        "account_age_days": rng.integers(3, 2200, n_users),
    })
    merchants = pd.DataFrame({
        "merchant_id": [f"M{i:04d}" for i in range(n_merchants)],
        "category": rng.choice(list(CATEGORY_BASE_RISK), n_merchants),
        "risk_mult": rng.uniform(0.7, 1.6, n_merchants),
    })

    weights = rng.gamma(1.2, size=n_users)
    weights /= weights.sum()
    uid = rng.choice(n_users, n, p=weights)
    mid = rng.integers(0, n_merchants, n)
    day = rng.integers(0, DAYS, n)
    hour_p = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 8, 6, 4, 2], float)
    hour = rng.choice(24, n, p=hour_p / hour_p.sum())

    ring_users = rng.choice(n_users, 25, replace=False)
    ring_merchants = rng.choice(n_merchants, 6, replace=False)
    ring = (day >= 40) & (day <= 46) & (rng.random(n) < 0.14)
    uid[ring] = rng.choice(ring_users, ring.sum())
    mid[ring] = rng.choice(ring_merchants, ring.sum())
    hour[ring] = rng.choice([0, 1, 2, 3, 23], ring.sum())

    u = users.iloc[uid].reset_index(drop=True)
    m = merchants.iloc[mid].reset_index(drop=True)
    home = u["home_country"].to_numpy()
    country = np.where(rng.random(n) < 0.93, home, rng.choice(COUNTRIES, n))
    channel = rng.choice(["Online", "POS", "ATM", "Mobile"], n, p=[0.4, 0.3, 0.1, 0.2])
    scale = m["category"].map({"Electronics": 3, "Travel": 4, "Crypto": 3, "Gaming": 1.5}).fillna(1).to_numpy()
    amount = np.round(rng.lognormal(3.8, 1.0, n) * scale, 2)

    p = m["category"].map(CATEGORY_BASE_RISK).to_numpy() * m["risk_mult"].to_numpy()
    p = p * np.where(hour <= 4, 2.8, 1) * np.where(amount > np.quantile(amount, 0.95), 2.5, 1)
    p = p * np.where(channel == "Online", 1.6, 1) * np.where(u["account_age_days"].to_numpy() < 30, 3, 1)
    p = p * np.where(country != home, 2.2, 1)
    p = np.where(ring, 0.65, np.clip(p, 0, 0.9))

    ts = (pd.Timestamp(START) + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h")
          + pd.to_timedelta(rng.integers(0, 3600, n), unit="s"))

    df = pd.DataFrame({
        "txn_id": [f"T{i:07d}" for i in range(n)],
        "ts": ts.astype(str),
        "user_id": u["user_id"],
        "merchant_id": m["merchant_id"],
        "category": m["category"],
        "channel": channel,
        "country": country,
        "home_country": home,
        "amount": amount,
        "account_age_days": u["account_age_days"],
        "is_fraud": (rng.random(n) < p).astype(int),
    })

    df = pd.concat([df, df.sample(frac=0.012, random_state=seed)], ignore_index=True)
    k = len(df)
    df.loc[rng.random(k) < 0.008, "amount"] = np.nan
    negative = rng.random(k) < 0.002
    df.loc[negative, "amount"] = -df.loc[negative, "amount"]
    df.loc[rng.random(k) < 0.005, "country"] = None
    noisy = rng.random(k) < 0.05
    df.loc[noisy, "channel"] = df.loc[noisy, "channel"].str.lower() + " "
    return df


def write_raw():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(RAW_PATH, index=False)
