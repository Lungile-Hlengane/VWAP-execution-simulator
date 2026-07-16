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

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import s3fs
import streamlit as st
from matplotlib.patches import Patch

# --- Shared visual style: dark HUD / cyberpunk theme ----------------------
plt.rcParams.update({
    "figure.facecolor": "#04060A",
    "axes.facecolor": "#04060A",
    "axes.edgecolor": "#1B2430",
    "axes.labelcolor": "#B9C4D4",
    "axes.titleweight": "bold",
    "axes.titlesize": 11.5,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.color": "#101823",
    "grid.linewidth": 0.7,
    "grid.linestyle": (0, (1, 3)),
    "font.family": "monospace",
    "font.size": 10,
    "text.color": "#C7D0DA",
    "xtick.color": "#7C8898",
    "ytick.color": "#7C8898",
    "legend.frameon": True,
    "legend.facecolor": "#080C12",
    "legend.edgecolor": "#1F2C3A",
    "legend.framealpha": 0.9,
    "legend.fontsize": 8.5,
    "savefig.facecolor": "#04060A",
})

PALETTE = {
    "market": "#00D9FF",       # neon cyan - market VWAP line
    "ours": "#FFB627",         # amber - our VWAP / allocation
    "fill": "#FF3E7D",         # neon pink - our fills
    "arrival": "#39FF88",      # neon green - arrival price reference
    "high_part": "#FF3E7D",    # elevated participation
    "low_part": "#00D9FF",     # normal participation
    "grid_bg": "#0A0F16",
    "session_shade": "#0E2530",
    "text_muted": "#7C8898",
}


def _clean_axis(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#1B2430")
    ax.tick_params(length=3, colors="#7C8898")


def _glow_line(ax, x, y, color, lw=1.6, layers=4, zorder=3, **kwargs):
    """Fake a neon glow by stacking translucent, progressively thicker lines
    under a crisp core line."""
    for i in range(layers, 0, -1):
        ax.plot(x, y, color=color, linewidth=lw + i * 2.2, alpha=0.05,
                 solid_capstyle="round", zorder=zorder - 1, solid_joinstyle="round")
    return ax.plot(x, y, color=color, linewidth=lw, zorder=zorder, **kwargs)[0]


def _glow_scatter(ax, x, y, color, s=28, layers=3, zorder=6, **kwargs):
    for i in range(layers, 0, -1):
        ax.scatter(x, y, color=color, s=s + i * 55, alpha=0.05, zorder=zorder - 1, linewidths=0)
    return ax.scatter(x, y, color=color, s=s, zorder=zorder, **kwargs)


def _glow_hline(ax, y, color, lw=1.3, layers=3, zorder=3, **kwargs):
    for i in range(layers, 0, -1):
        ax.axhline(y, color=color, linewidth=lw + i * 2.5, alpha=0.05, zorder=zorder - 1)
    return ax.axhline(y, color=color, linewidth=lw, zorder=zorder, **kwargs)


def _hud_frame(fig):
    """Thin neon border + corner brackets around the whole figure, for that
    heads-up-display look."""
    frame_ax = fig.add_axes([0, 0, 1, 1])
    frame_ax.axis("off")
    frame_ax.set_xlim(0, 1)
    frame_ax.set_ylim(0, 1)

    edge = "#1C3B45"
    frame_ax.add_patch(plt.Rectangle((0.006, 0.006), 0.988, 0.988, fill=False,
                                      edgecolor=edge, linewidth=1.1, alpha=0.9, zorder=10))

    bracket = 0.018
    bracket_color = PALETTE["market"]
    corners = [(0.006, 0.006, 1, 1), (0.994, 0.006, -1, 1),
               (0.006, 0.994, 1, -1), (0.994, 0.994, -1, -1)]
    for cx, cy, dx, dy in corners:
        frame_ax.plot([cx, cx + dx * bracket], [cy, cy], color=bracket_color,
                       linewidth=2.0, alpha=0.85, zorder=11, solid_capstyle="round")
        frame_ax.plot([cx, cx], [cy, cy + dy * bracket], color=bracket_color,
                       linewidth=2.0, alpha=0.85, zorder=11, solid_capstyle="round")
    return frame_ax


def _shade_sessions(ax, segments, color=PALETTE["session_shade"], alpha=0.35):
    for start, end in segments:
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=alpha, zorder=0, linewidth=0)


def plot_execution_summary(results_df, n_terminals):
    """3-panel HUD-styled execution figure, returned as a matplotlib Figure
    so Streamlit can render it with st.pyplot."""
    df = results_df.dropna(subset=["fill_price"]).copy().sort_values("terminal").reset_index(drop=True)
    if df.empty:
        return None

    df["market_notional_usd"] = df["market_qty"] * df["terminal_vwap"]

    # Insert NaN break-points across gaps in the selected terminals so the
    # price line doesn't join across skipped slices.
    gap_threshold = 1
    plot_rows = []
    for i, row in df.iterrows():
        if i > 0 and (row["terminal"] - df.loc[i - 1, "terminal"]) > gap_threshold:
            blank = row.copy()
            blank["terminal_vwap"] = np.nan
            blank["fill_price"] = np.nan
            blank["terminal"] = df.loc[i - 1, "terminal"] + 0.5
            plot_rows.append(blank)
        plot_rows.append(row)
    df_plot = pd.DataFrame(plot_rows).reset_index(drop=True)

    # Contiguous traded segments, used to shade "active" trading windows.
    segments = []
    seg_start = df["terminal"].iloc[0]
    prev = seg_start
    for t in df["terminal"].iloc[1:]:
        if t - prev > gap_threshold:
            segments.append((seg_start, prev))
            seg_start = t
        prev = t
    segments.append((seg_start, prev))

    arrival_price = df.iloc[0]["terminal_vwap"]
    our_vwap = (df["fill_price"] * df["your_qty"]).sum() / df["your_qty"].sum()
    total_impact_usd = (df["your_qty"] * df["terminal_vwap"] * df["impact_bps"] / 10000).sum()
    total_usd = df["target_usd"].sum()
    impact_bps_total = (total_impact_usd / total_usd) * 10000
    rating = rate_impact_bps(impact_bps_total)
    rating_color = {
        "Excellent": "#39FF88", "Good": "#9BE564", "Acceptable": "#FFD54F",
        "Poor": "#FF9F45", "Bad": "#FF3E7D",
    }.get(rating, "#7C8898")

    fig, axes = plt.subplots(3, 1, figsize=(15, 11.6), sharex=True,
                              gridspec_kw={"height_ratios": [2.1, 1.4, 1]})
    fig.patch.set_facecolor("#04060A")

    _hud_frame(fig)

    fig.suptitle(f"► EXECUTION SUMMARY — {df['date'].iloc[0]}", fontsize=14.5, fontweight="bold",
                 x=0.025, ha="left", y=0.99, color="#E8F6FF", family="monospace")

    # --- KPI cards ---
    kpi_vals = [
        ("ORDER", f"${total_usd:,.0f}", PALETTE["text_muted"]),
        ("SLICES TRADED", f"{len(df)} / {n_terminals}", PALETTE["text_muted"]),
        ("ARRIVAL PRICE", f"${arrival_price:,.2f}", PALETTE["arrival"]),
        ("OUR VWAP", f"${our_vwap:,.2f}", PALETTE["ours"]),
        ("IMPACT COST", f"{impact_bps_total:.2f} bps — {rating.upper()}", rating_color),
    ]
    kpi_ax = fig.add_axes([0.02, 0.90, 0.965, 0.055])
    kpi_ax.axis("off")
    kpi_ax.set_xlim(0, 1)
    kpi_ax.set_ylim(0, 1)
    n = len(kpi_vals)
    pad = 0.006
    card_w = (1 - pad * (n + 1)) / n
    for i, (label, val, color) in enumerate(kpi_vals):
        x0 = pad + i * (card_w + pad)
        kpi_ax.add_patch(plt.Rectangle((x0, 0.05), card_w, 0.9, fill=True,
                                        facecolor="#080C12", edgecolor="#1F2C3A",
                                        linewidth=1.0, zorder=1))
        kpi_ax.add_patch(plt.Rectangle((x0, 0.05), card_w, 0.06, fill=True,
                                        facecolor=color, edgecolor="none", alpha=0.9, zorder=2))
        kpi_ax.text(x0 + 0.02, 0.72, label, fontsize=8, color="#6E7C8C",
                    ha="left", va="center", family="monospace", zorder=3)
        kpi_ax.text(x0 + 0.02, 0.32, val, fontsize=13, fontweight="bold", color=color,
                    ha="left", va="center", family="monospace", zorder=3)

    # --- Panel 1: price ---
    ax1 = axes[0]
    _shade_sessions(ax1, segments)
    _glow_line(ax1, df_plot["terminal"], df_plot["terminal_vwap"], PALETTE["market"],
               lw=1.6, label="Market VWAP (selected terminals)", zorder=3)
    _glow_scatter(ax1, df["terminal"], df["fill_price"], PALETTE["fill"], s=26,
                  edgecolor="#04060A", linewidth=0.6, label="Your fill price (incl. impact)", zorder=6)
    _glow_hline(ax1, arrival_price, PALETTE["arrival"], lw=1.3, linestyle=":",
                label=f"Arrival price (${arrival_price:,.2f})")
    _glow_hline(ax1, our_vwap, PALETTE["ours"], lw=1.3, linestyle="--",
                label=f"Your VWAP (${our_vwap:,.2f})")
    ax1.set_ylabel("BTC PRICE (USD)")
    ax1.set_title("▍ PRICE VS. FILLS", loc="left", pad=8, color="#E8F6FF")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax1.legend(loc="upper left", ncol=2)
    _clean_axis(ax1)

    # --- Panel 2: notional volume, log scale ---
    ax2 = axes[1]
    _shade_sessions(ax2, segments)
    ax2.bar(df["terminal"], df["market_notional_usd"], color=PALETTE["market"], alpha=0.30,
            width=1.0, label="Market volume (USD)", zorder=2)
    ax2.bar(df["terminal"], df["target_usd"], color=PALETTE["ours"], width=0.6,
            label="Your allocation (USD)", zorder=3, edgecolor=PALETTE["ours"], linewidth=0.3)
    ax2.set_ylabel("USD NOTIONAL (LOG)")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.set_title("▍ ORDER SIZE VS. MARKET DEPTH", loc="left", pad=8, color="#E8F6FF")
    ax2.legend(loc="upper left")
    _clean_axis(ax2)

    # --- Panel 3: participation ---
    ax3 = axes[2]
    _shade_sessions(ax3, segments)
    median_part = df["participation_pct"].median()
    colors = [PALETTE["high_part"] if p > median_part * 2 else PALETTE["low_part"]
              for p in df["participation_pct"]]
    ax3.bar(df["terminal"], df["participation_pct"], color=colors, width=1.0, alpha=0.9, zorder=2)
    ax3.axhline(df["participation_pct"].mean(), color="#9AA6B4", linestyle="--", linewidth=1, zorder=3)
    ax3.set_ylabel("PARTICIPATION %")
    ax3.set_title("▍ PARTICIPATION RATE  (pink = >2x median — impact-heavy slices)",
                  loc="left", pad=8, color="#E8F6FF")
    ax3.legend(loc="upper right", handles=[
        plt.Line2D([0], [0], color="#9AA6B4", linestyle="--", linewidth=1,
                   label=f"Avg: {df['participation_pct'].mean():.2f}%"),
        Patch(facecolor=PALETTE["low_part"], label="Normal"),
        Patch(facecolor=PALETTE["high_part"], label="Elevated (>2x median)"),
    ])
    _clean_axis(ax3)

    # --- Shared x-axis: show clock time instead of raw terminal index ---
    slice_minutes = (24 * 60) / n_terminals

    def _terminal_to_clock(t, _pos):
        total_min = (t - 1) * slice_minutes
        hh = int(total_min // 60) % 24
        mm = int(total_min % 60)
        return f"{hh:02d}:{mm:02d}"

    ax3.set_xlabel("TIME OF DAY (UTC)")
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(_terminal_to_clock))
    tick_step = max(1, round(n_terminals / 12))
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    plt.setp(ax3.get_xticklabels(), rotation=0, ha="center")

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.885])
    return fig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Streamlit secrets (Settings -> Secrets) show up in st.secrets, NOT in
# os.environ automatically. We pull them from st.secrets here and export them
# as env vars ourselves, since boto3/s3fs both read credentials from the
# environment. Falls back to actual environment variables (e.g. for local
# dev via `export AWS_ACCESS_KEY_ID=...`) if st.secrets isn't configured.
def _get_secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


_aws_access_key = _get_secret("AWS_ACCESS_KEY_ID")
_aws_secret_key = _get_secret("AWS_SECRET_ACCESS_KEY")
_aws_region = _get_secret("AWS_DEFAULT_REGION", "eu-north-1")

if _aws_access_key:
    os.environ["AWS_ACCESS_KEY_ID"] = _aws_access_key
if _aws_secret_key:
    os.environ["AWS_SECRET_ACCESS_KEY"] = _aws_secret_key
os.environ["AWS_DEFAULT_REGION"] = _aws_region

S3_BUCKET = _get_secret("S3_BUCKET", "madot-algo-data")
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

if not _aws_access_key or not _aws_secret_key:
    st.error(
        "AWS credentials not found. Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
        "in this app's Settings -> Secrets, then reload."
    )
    st.stop()


@st.cache_resource
def get_fs():
    # Pass credentials explicitly rather than relying on s3fs/aiobotocore to
    # pick them up from the environment - more reliable across platforms.
    return s3fs.S3FileSystem(
        key=_aws_access_key,
        secret=_aws_secret_key,
        client_kwargs={"region_name": _aws_region},
    )


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

    fig = plot_execution_summary(results, N_TERMINALS)
    if fig is None:
        st.warning("Nothing to plot - every selected terminal came back empty for this day.")
    else:
        st.pyplot(fig)

    with st.expander("Full fill-by-fill detail"):
        st.dataframe(results, use_container_width=True)
