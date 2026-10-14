# Financial Operations Risk Reporting Dashboard

Analytics system for financial transaction data covering operational reporting and risk monitoring (Mar 2026 to Apr 2026).

## Stack
Python, Pandas, NumPy, SQL (SQLite), Matplotlib, Streamlit

## Structure
- `generate_data.py` builds a raw transaction file with realistic defects and embedded fraud patterns
- `pipeline.py` cleans the data, engineers KPIs and a rule-based risk score, and loads SQLite
- `db.py` holds the SQL aggregation layer used by the dashboard
- `app.py` is the Streamlit dashboard

## Run
```
pip install -r requirements.txt
python pipeline.py
streamlit run app.py
```
The first dashboard launch builds the data automatically if `python pipeline.py` was skipped.

## Dashboard
- KPI cards with week-over-week change: volume, fraud rate, fraud loss, high-risk share
- Trends: daily volume, 7-day fraud rate, anomaly day detection, hour and weekday patterns
- Risk segments: tier distribution, category, channel and country breakdowns, weekday by hour heatmap
- Watchlist: merchants and users ranked by fraud rate, with CSV export
- Key findings: auto-generated observations that update with the filters

## Risk score
Points per transaction: night activity 20, high amount 25, new account 20, country mismatch 15,
repeat activity in one day 15, plus a merchant category weight. Tiers are Low, Medium, High and Critical.
