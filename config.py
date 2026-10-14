from pathlib import Path

ROOT = Path(__file__).parent
RAW_PATH = ROOT / "data" / "raw" / "transactions_raw.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "transactions.csv"
DB_PATH = ROOT / "data" / "risk.db"

START = "2026-03-01"
DAYS = 61
COUNTRIES = ["IN", "US", "GB", "AE", "SG", "DE"]
CHANNELS = {"ONLINE": "Online", "POS": "POS", "ATM": "ATM", "MOBILE": "Mobile"}
CATEGORY_BASE_RISK = {
    "Grocery": 0.004,
    "Electronics": 0.014,
    "Travel": 0.012,
    "Fashion": 0.008,
    "Gaming": 0.018,
    "Crypto": 0.030,
    "Utilities": 0.003,
    "Dining": 0.005,
}
CATEGORY_POINTS = {k: int(round(v * 500)) for k, v in CATEGORY_BASE_RISK.items()}
TIER_ORDER = ["Low", "Medium", "High", "Critical"]
TIER_BINS = [-1, 24, 49, 74, 100]
