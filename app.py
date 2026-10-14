import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import db
from config import TIER_ORDER
from pipeline import ensure_ready

INK, RED, TEAL, SLATE, GRID = "#14213d", "#d1495b", "#2a9d8f", "#9aa5b8", "#e3e7ee"
TIER_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#f4a261", "Critical": "#d1495b"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="Risk Operations Dashboard", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1280px;}
    .banner {background: #14213d; color: #f6f7f9; padding: 1.4rem 1.8rem; border-radius: 6px;
             border-left: 6px solid #d1495b; margin-bottom: 1.4rem;}
    .banner h1 {margin: 0; font-size: 1.7rem; font-weight: 650; letter-spacing: -0.01em;}
    .banner p {margin: .35rem 0 0; color: #b8c2d6; font-size: .95rem;}
    div[data-testid="stMetric"] {background: #fff; border: 1px solid #e3e7ee; border-top: 3px solid #14213d;
             padding: .9rem 1rem; border-radius: 4px;}
    div[data-testid="stMetricValue"] {font-size: 1.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing data")
def prepare():
    ensure_ready()
    return db.bounds()


@st.cache_data(show_spinner=False)
def load(start, end, cats, chs, tiers):
    w, p = db.where(start, end, cats, chs, tiers)
    return {
        "daily": db.daily(w, p),
        "category": db.segment("category", w, p),
        "channel": db.segment("channel", w, p),
        "country": db.segment("country", w, p),
        "tier": db.segment("risk_tier", w, p),
        "heat": db.heat(w, p),
        "merchants": db.merchants(w, p),
        "users": db.users(w, p),
    }


def show(fig, target=st):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=170, bbox_inches="tight")
    plt.close(fig)
    target.image(buf, width="stretch")


def money(v):
    for unit, div in (("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"₹{v / div:,.2f}{unit}"
    return f"₹{v:,.0f}"


def axes(w=9, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd3e0")
    ax.tick_params(colors="#4a5568", labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def score_days(d):
    d = d.copy()
    d["rate"] = d["fraud"] / d["txns"] * 100
    d["roll"] = d["rate"].rolling(7, min_periods=1).mean()
    sd = d["rate"].std(ddof=0) or 1
    d["z"] = (d["rate"] - d["rate"].mean()) / sd
    d["day"] = pd.to_datetime(d["date"])
    return d


def daily_chart(d):
    d = score_days(d)
    fig, ax = axes(10, 3.8)
    ax.bar(d["day"], d["txns"], color="#d5dbe6", width=0.8)
    ax.set_ylabel("Transactions", color="#4a5568")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2 = ax.twinx()
    ax2.plot(d["day"], d["rate"], color="#e8a0aa", linewidth=1)
    ax2.plot(d["day"], d["roll"], color=RED, linewidth=2.4, label="Fraud rate, 7-day average")
    spikes = d[d["z"] > 2]
    ax2.scatter(spikes["day"], spikes["rate"], color=INK, s=38, zorder=5, label="Anomalous day")
    ax2.set_ylabel("Fraud rate (%)", color=RED)
    ax2.tick_params(colors=RED, labelsize=9)
    for side in ("top", "left"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color("#cbd3e0")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def bar_chart(df, title, w=5, h=3.4, color_by_rate=True, overall=None):
    df = df.sort_values("fraud_rate")
    fig, ax = axes(w, h)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    colors = [RED if (overall is not None and r > overall) else SLATE for r in df["fraud_rate"]]
    ax.barh(df["segment"].astype(str), df["fraud_rate"], color=colors, height=0.62)
    for y, v in enumerate(df["fraud_rate"]):
        ax.text(v, y, f"  {v:.2f}%", va="center", fontsize=9, color=INK)
    if overall is not None:
        ax.axvline(overall, color=INK, linestyle="--", linewidth=1)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlim(0, df["fraud_rate"].max() * 1.22 or 1)
    fig.tight_layout()
    return fig


def hour_chart(h):
    g = h.groupby("hour")[["txns", "fraud"]].sum().reindex(range(24), fill_value=0)
    rate = (g["fraud"] / g["txns"].replace(0, np.nan) * 100).fillna(0)
    fig, ax = axes(5, 3.4)
    ax.bar(rate.index, rate.values, color=[RED if i <= 4 else SLATE for i in rate.index], width=0.75)
    ax.set_xticks(range(0, 24, 3))
    ax.set_title("Fraud rate by hour of day (%)", loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    return fig


def weekday_chart(h):
    g = h.groupby("weekday")[["txns", "fraud"]].sum().reindex(range(7), fill_value=0)
    rate = (g["fraud"] / g["txns"].replace(0, np.nan) * 100).fillna(0)
    fig, ax = axes(5, 3.4)
    ax.bar(DAYS, rate.values, color=TEAL, width=0.65)
    ax.set_title("Fraud rate by weekday (%)", loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    return fig


def donut(tier):
    t = tier.set_index("segment").reindex(TIER_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fig.patch.set_alpha(0)
    ax.pie(t["txns"], colors=[TIER_COLORS[i] for i in t.index], startangle=90, counterclock=False,
           wedgeprops={"width": 0.36, "edgecolor": "white"})
    share = t["txns"] / t["txns"].sum() * 100
    ax.legend([f"{i}  {share[i]:.1f}%" for i in t.index], loc="center left",
              bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10)
    hi = share[["High", "Critical"]].sum()
    ax.text(0, 0, f"{hi:.1f}%\nhigh-risk", ha="center", va="center", fontsize=12, color=INK, fontweight="bold")
    ax.set_title("Risk tier distribution", loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    return fig


def tier_yield(tier):
    t = tier.set_index("segment").reindex(TIER_ORDER).fillna(0).reset_index()
    fig, ax = axes(5, 3.4)
    ax.bar(t["segment"], t["fraud_rate"], color=[TIER_COLORS[i] for i in t["segment"]], width=0.6)
    for x, v in enumerate(t["fraud_rate"]):
        ax.text(x, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title("Confirmed fraud rate by risk tier", loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    return fig


def heatmap(h):
    g = h.pivot_table(index="weekday", columns="hour", values=["txns", "fraud"], aggfunc="sum", fill_value=0)
    txns = g["txns"].reindex(index=range(7), columns=range(24), fill_value=0)
    fraud = g["fraud"].reindex(index=range(7), columns=range(24), fill_value=0)
    rate = (fraud / txns.replace(0, np.nan) * 100).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_alpha(0)
    im = ax.imshow(rate.values, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 2))
    ax.tick_params(labelsize=9, colors="#4a5568", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Fraud rate (%)", color="#4a5568")
    ax.set_title("When fraud happens: weekday by hour", loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    return fig


def pct_shift(last, prev):
    return None if not prev else (last - prev) / prev * 100


bounds = prepare()
lo, hi = pd.to_datetime(bounds["lo"]).date(), pd.to_datetime(bounds["hi"]).date()

with st.sidebar:
    st.header("Filters")
    picked = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi)
    cats = st.multiselect("Merchant category", ["Crypto", "Gaming", "Electronics", "Travel", "Fashion", "Dining", "Grocery", "Utilities"])
    chs = st.multiselect("Channel", ["Online", "Mobile", "POS", "ATM"])
    tiers = st.multiselect("Risk tier", TIER_ORDER)
    st.caption("Leave a filter empty to include everything.")

if not isinstance(picked, tuple) or len(picked) != 2:
    st.info("Pick both a start and an end date to continue.")
    st.stop()

start, end = (x.strftime("%Y-%m-%d") for x in picked)
data = load(start, end, tuple(cats), tuple(chs), tuple(tiers))
d = data["daily"]

st.markdown(
    f"""<div class="banner"><h1>Financial Operations Risk Dashboard</h1>
    <p>Transaction monitoring from {start} to {end}. Fraud rate, volume and risk exposure across users and merchants.</p></div>""",
    unsafe_allow_html=True,
)

if d.empty:
    st.warning("No transactions match these filters. Widen the date range or clear a filter.")
    st.stop()

txns, fraud = int(d["txns"].sum()), int(d["fraud"].sum())
rate = fraud / txns * 100
loss = d["fraud_amount"].sum()
high_share = d["high_cnt"].sum() / txns * 100

deltas = {"txns": None, "rate": None, "loss": None}
if len(d) >= 14:
    last, prev = d.tail(7), d.iloc[-14:-7]
    t = pct_shift(last["txns"].sum(), prev["txns"].sum())
    l = pct_shift(last["fraud_amount"].sum(), prev["fraud_amount"].sum())
    r = last["fraud"].sum() / last["txns"].sum() * 100 - prev["fraud"].sum() / prev["txns"].sum() * 100
    deltas = {
        "txns": None if t is None else f"{t:+.1f}% vs prior week",
        "rate": f"{r:+.2f} pp vs prior week",
        "loss": None if l is None else f"{l:+.1f}% vs prior week",
    }

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Transactions", f"{txns:,}", deltas["txns"])
c2.metric("Volume", money(d["volume"].sum()))
c3.metric("Fraud rate", f"{rate:.2f}%", deltas["rate"], delta_color="inverse")
c4.metric("Fraud loss", money(loss), deltas["loss"], delta_color="inverse")
c5.metric("High-risk share", f"{high_share:.1f}%")

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Risk segments", "Watchlist", "Key findings"])

with tab1:
    show(daily_chart(d), st)
    a, b = st.columns(2)
    show(hour_chart(data["heat"]), a)
    show(weekday_chart(data["heat"]), b)

with tab2:
    a, b = st.columns(2)
    show(donut(data["tier"]), a)
    show(tier_yield(data["tier"]), b)
    a, b = st.columns(2)
    show(bar_chart(data["category"], "Fraud rate by merchant category", overall=rate), a)
    show(bar_chart(data["channel"], "Fraud rate by channel", overall=rate), b)
    show(heatmap(data["heat"]), st)
    show(bar_chart(data["country"], "Fraud rate by country", w=10, h=3, overall=rate), st)

with tab3:
    cfg = {
        "fraud_rate": st.column_config.ProgressColumn("Fraud rate %", format="%.1f", min_value=0, max_value=100),
        "avg_risk": st.column_config.ProgressColumn("Avg risk score", format="%.0f", min_value=0, max_value=100),
        "volume": st.column_config.NumberColumn("Volume", format="₹%.0f"),
        "fraud_amount": st.column_config.NumberColumn("Fraud amount", format="₹%.0f"),
        "txns": "Transactions",
        "fraud_cases": "Fraud cases",
    }
    st.subheader("Merchants to review")
    st.dataframe(data["merchants"], hide_index=True, width="stretch", column_config=cfg)
    st.subheader("Users to review")
    st.dataframe(data["users"], hide_index=True, width="stretch", column_config=cfg)
    st.download_button("Download merchant watchlist", data["merchants"].to_csv(index=False),
                       "merchant_watchlist.csv", "text/csv")
    st.download_button("Download user watchlist", data["users"].to_csv(index=False),
                       "user_watchlist.csv", "text/csv")

with tab4:
    sd = score_days(d)
    spikes = sd[sd["z"] > 2]
    cat = data["category"].iloc[0]
    hh = data["heat"].groupby("hour")[["txns", "fraud"]].sum()
    hh["rate"] = hh["fraud"] / hh["txns"] * 100
    pool = hh[hh["txns"] >= 50]
    pool = pool if len(pool) else hh
    night, day_ = hh[hh.index <= 4], hh[hh.index > 4]
    top_m = data["merchants"].head(10)
    notes = [
        f"**{cat['segment']}** is the riskiest category at {cat['fraud_rate']:.2f}% fraud, "
        f"{cat['fraud_rate'] / rate:.1f}x the overall rate of {rate:.2f}%.",
        f"Fraud peaks at **{int(pool['rate'].idxmax()):02d}:00** with a {pool['rate'].max():.2f}% rate.",
    ]
    if night["txns"].sum() and day_["txns"].sum():
        nr = night["fraud"].sum() / night["txns"].sum() * 100
        dr = day_["fraud"].sum() / day_["txns"].sum() * 100
        notes.append(f"Transactions between 00:00 and 04:59 are fraudulent {nr / dr:.1f}x as often as daytime ones.")
    if len(spikes):
        worst = spikes.sort_values("rate").iloc[-1]
        notes.append(f"**{len(spikes)} day(s)** ran well above normal fraud levels. The worst was "
                     f"**{worst['date']}** at {worst['rate']:.2f}%, which points to a coordinated burst.")
    if fraud and len(top_m):
        notes.append(f"The 10 highest-rate merchants account for **{top_m['fraud_cases'].sum() / fraud:.0%}** of all fraud cases.")
    tier = data["tier"].set_index("segment")
    if {"Low", "Critical"} <= set(tier.index) and tier.loc["Low", "fraud_rate"] > 0:
        notes.append(f"Critical-tier transactions are confirmed as fraud {tier.loc['Critical', 'fraud_rate'] / tier.loc['Low', 'fraud_rate']:.0f}x "
                     "as often as low-tier ones, so the risk score is separating good from bad traffic.")
    for n in notes:
        st.markdown(f"- {n}")
