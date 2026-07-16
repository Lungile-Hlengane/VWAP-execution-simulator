"""
app.py
-------
Streamlit VWAP execution simulator for BTC/USDT.

User enters an order size (USD) and a target date. The app simulates how a
volume-shape-driven VWAP execution algo would have filled that order against
that day's real tick data, using a GA to pick the best lookback window over
prior days' hourly volume shapes.

Speed trick: prior days' hourly volume shapes never change once the day is
over, so instead of re-loading and re-parsing full tick data for every prior
day on every request, we read a small cached shape file from S3
(parquets/shape_cache/hourly_shapes.json) and only fall back to a live
per-day computation for any day that hasn't been backfilled into that cache
yet. Only the TARGET date gets a full tick-data load, since that's the only
day the fill simulation actually needs tick-level detail for.
"""

import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import s3fs
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_BUCKET = os.environ.get("S3_BUCKET", "madot-algo-data")
AGGTRADES_PREFIX = "parquets/aggTrades"
SHAPE_CACHE_KEY = "parquets/shape_cache/hourly_shapes.json"

AGGTRADES_S3_PATH = f"s3://{S3_BUCKET}/{AGGTRADES_PREFIX}"
SHAPE_CACHE_S3_PATH = f"s3://{S3_BUCKET}/{SHAPE_CACHE_KEY}"

HISTORY_START = "2026-04-01"  # earliest day of tick data we have
START_TIME_STR = "00:00:00"
END_TIME_STR = "23:59:59"

SLICE_MINUTES = 5
N_TERMINALS = int(
    ((pd.Timestamp("2000-01-01 23:59:59") - pd.Timestamp("2000-01-01 00:00:00")).seconds + 1)
    / 60
    / SLICE_MINUTES
)

IMPACT_COEFFICIENT = 10.0
IMPACT_EXPONENT = 0.5
MIN_LOOKBACK_DAYS = 3
MAX_LOOKBACK_DAYS = 60

AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")
os.environ.setdefault("AWS_DEFAULT_REGION", AWS_DEFAULT_REGION)


@st.cache_resource
def get_fs():
    # AWS credentials come from Streamlit Cloud secrets (see deployment notes),
    # which get exported as environment variables, so s3fs picks them up
    # automatically - no keys handled in this file.
    return s3fs.S3FileSystem()


# ---------------------------------------------------------------------------
# Tick data + shape computation
# ---------------------------------------------------------------------------
def load_day_trades(date_str, base_path=AGGTRADES_S3_PATH):
    """Loads ONLY 1 day of BTCUSDT aggTrades tick data from S3.
    This should only ever be called for the TARGET date - every prior day's
    shape comes from the cache instead (see build_shape_history)."""
    fs = get_fs()
    pattern = f"{base_path}/*{date_str}.parquet"
    files = sorted(fs.glob(pattern))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_parquet(
            f"s3://{f}", columns=["price", "quantity", "transact_time", "is_buyer_maker"]
        )
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="unsigned")
        df["transact_time"] = pd.to_datetime(df["transact_time"], unit="ms")
        dfs.append(df)

    df_day = pd.concat(dfs, ignore_index=True)
    df_day = df_day.drop_duplicates(subset=["transact_time", "price", "quantity"])
    return df_day


def get_hourly_volume_shape(df_day, start_time_str, end_time_str):
    """Returns 24 hourly % shares of volume."""
    day_date = df_day["transact_time"].dt.date.iloc[0]
    window_start = pd.Timestamp(f"{day_date} {start_time_str}")
    window_end = pd.Timestamp(f"{day_date} {end_time_str}")
    window = df_day[
        (df_day["transact_time"] >= window_start) & (df_day["transact_time"] <= window_end)
    ]

    hourly_edges = pd.date_range(window_start, window_end, periods=25)
    total_qty = window["quantity"].sum()
    if total_qty == 0:
        return np.full(24, np.nan)

    shape = []
    for i in range(24):
        slice_qty = window[
            (window["transact_time"] >= hourly_edges[i])
            & (window["transact_time"] < hourly_edges[i + 1])
        ]["quantity"].sum()
        shape.append(slice_qty / total_qty)
    return np.array(shape)


def compute_shape_for_day(date_str):
    """Slow path: loads one day's tick data and computes its shape."""
    df_day = load_day_trades(date_str)
    if df_day.empty:
        return None
    shape = get_hourly_volume_shape(df_day, START_TIME_STR, END_TIME_STR)
    del df_day
    if np.isnan(shape).all():
        return None
    return shape


# ---------------------------------------------------------------------------
# Shape cache (the fast path)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_shape_cache():
    """Reads the small pre-built shape cache from S3. Cached for an hour so
    it's only actually fetched from S3 once per hour, not once per request."""
    fs = get_fs()
    key = SHAPE_CACHE_S3_PATH.replace("s3://", "")
    if not fs.exists(key):
        return {}
    with fs.open(key, "r") as f:
        return json.load(f)


def save_shape_cache(cache_dict):
    fs = get_fs()
    key = SHAPE_CACHE_S3_PATH.replace("s3://", "")
    with fs.open(key, "w") as f:
        json.dump(cache_dict, f)


def build_shape_history(target_date_str, history_start=HISTORY_START):
    """Builds the list of prior-day shapes needed for the GA, reading from the
    cache wherever possible and only computing (+ caching) live for any day
    that hasn't been backfilled yet."""
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    cache = load_shape_cache()
    newly_computed = {}

    shape_history = []
    d = datetime.strptime(history_start, "%Y-%m-%d")
    while d < target_date:
        date_str = d.strftime("%Y-%m-%d")
        if date_str in cache:
            shape = np.array(cache[date_str])
        else:
            shape = compute_shape_for_day(date_str)
            if shape is None:
                d += timedelta(days=1)
                continue
            newly_computed[date_str] = shape.tolist()
        shape_history.append(shape)
        d += timedelta(days=1)

    if newly_computed:
        cache.update(newly_computed)
        save_shape_cache(cache)
        load_shape_cache.clear()  # invalidate the cached read so it picks up the update next time

    return shape_history, len(newly_computed)


# ---------------------------------------------------------------------------
# GA lookback search + prediction
# ---------------------------------------------------------------------------
def weighted_mape(actual, predicted):
    mask = actual != 0
    errors = np.abs((actual[mask] - predicted[mask]) / actual[mask])
    weights = actual[mask] / actual[mask].sum()
    return np.sum(errors * weights)


def ga_optimize_period(shape_history, min_period, max_period, pop_size=10, generations=15, mutation_rate=0.2):
    n_days = len(shape_history)
    max_period = min(max_period, n_days - 1)
    if max_period < min_period:
        return None, None

    def fitness(period):
        errors = []
        for i in range(period, n_days):
            past = shape_history[i - period:i]
            predicted = np.mean(past, axis=0)
            actual = shape_history[i]
            errors.append(weighted_mape(actual, predicted))
        return np.mean(errors) if errors else np.inf

    population = [random.randint(min_period, max_period) for _ in range(pop_size)]
    best_period, best_score = None, np.inf
    for _ in range(generations):
        scored = sorted([(p, fitness(p)) for p in population], key=lambda x: x[1])
        if scored[0][1] < best_score:
            best_period, best_score = scored[0]
        survivors = [p for p, _ in scored[: pop_size // 2]]
        new_population = survivors.copy()
        while len(new_population) < pop_size:
            parent = random.choice(survivors)
            child = parent
            if random.random() < mutation_rate:
                child = max(min_period, min(max_period, parent + random.choice([-1, 1])))
            new_population.append(child)
        population = new_population

    return best_period, best_score


def predict_terminal_targets(shape_history, min_period, max_period, n_terminals, **ga_kwargs):
    best_period, best_score = ga_optimize_period(shape_history, min_period, max_period, **ga_kwargs)
    if best_period is None:
        return None, None, None

    past = shape_history[-best_period:]
    predicted_hourly = np.mean(past, axis=0)
    bins_per_hour = n_terminals // 24
    predicted_fine = np.repeat(predicted_hourly / bins_per_hour, bins_per_hour)
    return predicted_fine, best_period, best_score


# ---------------------------------------------------------------------------
# Strategy: pick top-volume slices, simulate fills with market impact
# ---------------------------------------------------------------------------
def select_top_volume_windows(predicted_shape, capture_pct=0.60):
    ranked_idx = np.argsort(predicted_shape)[::-1]
    cumulative = 0.0
    selected = []
    for idx in ranked_idx:
        selected.append(idx)
        cumulative += predicted_shape[idx]
        if cumulative >= capture_pct:
            break
    selected_sorted = sorted(selected)
    selected_weights = predicted_shape[selected_sorted]
    selected_weights = selected_weights / selected_weights.sum()
    return selected_sorted, selected_weights


def run_vwap_strategy(df_day, shape_history, total_usd, start_time_str, end_time_str,
                       n_terminals, impact_coef, impact_exp, min_period, max_period,
                       capture_pct=0.60):
    day_date = df_day["transact_time"].dt.date.iloc[0]
    window_start = pd.Timestamp(f"{day_date} {start_time_str}")
    window_end = pd.Timestamp(f"{day_date} {end_time_str}")
    window = df_day[
        (df_day["transact_time"] >= window_start) & (df_day["transact_time"] < window_end)
    ].copy()

    if window.empty:
        return pd.DataFrame(), None, None

    full_vwap = (window["price"] * window["quantity"]).sum() / window["quantity"].sum()

    predicted_shape, best_period, ga_mape = predict_terminal_targets(
        shape_history, min_period, max_period, n_terminals
    )
    if predicted_shape is None:
        predicted_shape = np.full(n_terminals, 1 / n_terminals)
        best_period, ga_mape = None, None

    selected_terminals, selected_weights = select_top_volume_windows(predicted_shape, capture_pct)
    target_usd_per_selected = total_usd * selected_weights
    edges = pd.date_range(window_start, window_end, periods=n_terminals + 1)

    rows = []
    for pos, term_idx in enumerate(selected_terminals):
        t_start, t_end = edges[term_idx], edges[term_idx + 1]
        slice_df = window[(window["transact_time"] >= t_start) & (window["transact_time"] < t_end)]
        market_qty = slice_df["quantity"].sum()

        if slice_df.empty or market_qty == 0:
            rows.append({
                "date": day_date, "terminal": term_idx + 1, "position_num": pos + 1,
                "ga_period_used": best_period, "ga_mape": ga_mape,
                "terminal_vwap": np.nan, "market_qty": 0, "your_qty": np.nan,
                "target_usd": np.nan, "participation_pct": np.nan, "impact_bps": np.nan,
                "fill_price": np.nan, "slippage_vs_vwap_bps": np.nan, "cost_usd": np.nan,
            })
            continue

        terminal_vwap = (slice_df["price"] * slice_df["quantity"]).sum() / market_qty
        your_usd = target_usd_per_selected[pos]
        your_qty = your_usd / terminal_vwap
        participation_pct = your_qty / market_qty

        impact_bps = impact_coef * (participation_pct ** impact_exp)
        fill_price = terminal_vwap * (1 + impact_bps / 10000)

        slippage_vs_vwap_bps = ((fill_price - full_vwap) / full_vwap) * 10000
        cost_usd = (slippage_vs_vwap_bps / 10000) * your_usd

        rows.append({
            "date": day_date, "terminal": term_idx + 1, "position_num": pos + 1,
            "ga_period_used": best_period, "ga_mape": ga_mape,
            "terminal_vwap": terminal_vwap, "market_qty": market_qty, "your_qty": your_qty,
            "target_usd": your_usd, "participation_pct": participation_pct * 100,
            "impact_bps": impact_bps, "fill_price": fill_price,
            "slippage_vs_vwap_bps": slippage_vs_vwap_bps, "cost_usd": cost_usd,
        })

    return pd.DataFrame(rows), best_period, ga_mape


def rate_impact_bps(impact_bps):
    if impact_bps < 2:
        return "Excellent"
    elif impact_bps < 5:
        return "Good"
    elif impact_bps < 10:
        return "Acceptable"
    elif impact_bps < 20:
        return "Poor"
    else:
        return "Bad"


def compute_cost_metrics(results_df):
    if results_df.empty or "fill_price" not in results_df.columns:
        return None
    df = results_df.dropna(subset=["fill_price"]).copy()
    if df.empty:
        return None

    arrival_price = df.iloc[0]["terminal_vwap"]
    total_qty = df["your_qty"].sum()
    our_vwap = (df["fill_price"] * df["your_qty"]).sum() / total_qty
    total_usd = df["target_usd"].sum()

    df["impact_cost_usd"] = df["your_qty"] * df["terminal_vwap"] * (df["impact_bps"] / 10000.0)
    total_impact = df["impact_cost_usd"].sum()
    impact_bps_total = (total_impact / total_usd) * 10000

    timing_cost = total_qty * (our_vwap - arrival_price) - total_impact
    timing_bps_total = (timing_cost / total_usd) * 10000
    total_slippage = total_qty * (our_vwap - arrival_price)
    total_bps = (total_slippage / total_usd) * 10000

    return {
        "arrival_price": arrival_price,
        "our_vwap": our_vwap,
        "total_usd": total_usd,
        "impact_bps": impact_bps_total,
        "impact_usd": total_impact,
        "timing_bps": timing_bps_total,
        "timing_usd": timing_cost,
        "total_bps": total_bps,
        "total_slippage_usd": total_slippage,
        "positions_taken": len(df),
        "avg_participation_pct": df["participation_pct"].mean(),
        "max_participation_pct": df["participation_pct"].max(),
        "rating": rate_impact_bps(impact_bps_total),
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="BTC/USDT VWAP Execution Simulator", layout="wide")
st.title("BTC/USDT VWAP Execution Simulator")
st.caption(
    "Simulates how a volume-shape-driven VWAP algo would have filled your order "
    "against real BTC/USDT tick data on the date you choose."
)

col1, col2, col3 = st.columns(3)
with col1:
    total_order_usd = st.number_input(
        "Order size (USD)", min_value=1000.0, value=1_000_000.0, step=10_000.0, format="%.0f"
    )
with col2:
    target_date = st.date_input(
        "Execution date",
        value=datetime.strptime("2026-06-15", "%Y-%m-%d"),
        min_value=datetime.strptime(HISTORY_START, "%Y-%m-%d") + timedelta(days=MIN_LOOKBACK_DAYS),
    )
with col3:
    capture_pct = st.slider("Volume capture target (%)", 30, 90, 60) / 100.0

run = st.button("Simulate execution", type="primary")

if run:
    target_date_str = target_date.strftime("%Y-%m-%d")

    with st.spinner("Loading prior-day volume shapes (from cache where possible)..."):
        shape_history, n_live_computed = build_shape_history(target_date_str)

    if n_live_computed:
        st.info(
            f"{n_live_computed} day(s) weren't in the shape cache yet, so they were "
            f"computed live and added to the cache for next time."
        )

    if len(shape_history) < MIN_LOOKBACK_DAYS:
        st.warning(
            f"Not enough history yet ({len(shape_history)} days) - need at least "
            f"{MIN_LOOKBACK_DAYS} prior days of data before {target_date_str}."
        )
        st.stop()

    with st.spinner(f"Loading tick data for {target_date_str} and simulating fills..."):
        df_target = load_day_trades(target_date_str)

    if df_target.empty:
        st.error(f"No tick data found for {target_date_str}. Check the date and try again.")
        st.stop()

    results, best_period, ga_mape = run_vwap_strategy(
        df_target, shape_history, total_order_usd, START_TIME_STR, END_TIME_STR,
        N_TERMINALS, IMPACT_COEFFICIENT, IMPACT_EXPONENT,
        MIN_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS, capture_pct=capture_pct,
    )

    metrics = compute_cost_metrics(results)
    if metrics is None:
        st.warning("No fills were generated for this day - try a different date.")
        st.stop()

    st.subheader(f"Results — {target_date_str}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Order size", f"${metrics['total_usd']:,.0f}")
    m2.metric("Slices traded", f"{metrics['positions_taken']} / {N_TERMINALS}")
    m3.metric("Arrival price", f"${metrics['arrival_price']:,.2f}")
    m4.metric("Your VWAP", f"${metrics['our_vwap']:,.2f}")
    m5.metric("Impact cost", f"{metrics['impact_bps']:.2f} bps", metrics["rating"])

    st.markdown("#### Cost breakdown (implementation shortfall vs. arrival price)")
    c1, c2, c3 = st.columns(3)
    c1.metric("1. Market impact (your footprint)", f"${metrics['impact_usd']:,.2f}", f"{metrics['impact_bps']:.2f} bps")
    c2.metric("2. Timing / drift (market luck)", f"${metrics['timing_usd']:,.2f}", f"{metrics['timing_bps']:.2f} bps")
    c3.metric("3. Total vs. arrival", f"${metrics['total_slippage_usd']:,.2f}", f"{metrics['total_bps']:.2f} bps")

    if best_period is not None:
        st.caption(f"GA-selected lookback window: {best_period} days | GA fit error (weighted MAPE): {ga_mape:.4f}")

    st.markdown("#### Price vs. fills")
    plot_df = results.dropna(subset=["fill_price"]).sort_values("terminal")
    chart_df = plot_df.set_index("terminal")[["terminal_vwap", "fill_price"]]
    chart_df.columns = ["Market VWAP (slice)", "Your fill price"]
    st.line_chart(chart_df)

    st.markdown("#### Participation rate by slice")
    st.bar_chart(plot_df.set_index("terminal")[["participation_pct"]])

    with st.expander("Full fill-by-fill detail"):
        st.dataframe(results, use_container_width=True)
