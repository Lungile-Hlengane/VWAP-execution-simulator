# BTC/USDT VWAP Execution Simulator

A Streamlit app that simulates how a **volume-shape-driven VWAP execution algorithm**
would have filled a large BTC/USDT order against real historical tick data — including
a genetic-algorithm-tuned lookback window, a market-impact model, and a full
implementation-shortfall cost breakdown.

You enter an order size and a date; the app tells you how the algo would have sliced
that order across the day, what price it would have filled at, and how much of the
cost was your own footprint versus the market simply moving.

---

## Table of contents

- [What it does](#what-it-does)
- [How the strategy works](#how-the-strategy-works)
- [Live output](#live-output)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Data requirements](#data-requirements)
- [Deploying on Streamlit Cloud](#deploying-on-streamlit-cloud)
- [Key formulas](#key-formulas)
- [Known limitations](#known-limitations)

---

## What it does

1. You pick an **order size (USD)**, a **target execution date**, and a **volume
   capture target** (what % of the day's predicted volume the algo should try to
   trade during).
2. The app builds a history of prior days' hourly volume "shapes" (what % of a
   day's volume typically trades in each hour), reading from a pre-built cache
   where possible and falling back to a live per-day computation for any day not
   yet cached.
3. A small **genetic algorithm** searches over lookback windows (e.g. "average the
   last 12 days") to find the one that best predicts the target day's volume shape,
   scored by weighted MAPE against known history.
4. Using the predicted shape, the algo greedily selects the smallest set of 5-minute
   "terminals" (time slices) that together cover your chosen capture %, and
   allocates your order across them proportional to predicted volume.
5. For each selected terminal, it computes the real market VWAP from tick data,
   simulates your fill price with a **market impact penalty** based on your
   participation rate, and rolls everything up into cost metrics.
6. Results are shown as KPIs, a 3-panel execution chart, a cost breakdown
   (impact vs. timing/drift), and the full fill-by-fill table.

## How the strategy works

```
Prior days' tick data ──► hourly volume shape ──► shape cache (S3, JSON)
                                                         │
                                                         ▼
                                        GA searches lookback windows (3-60 days)
                                        to find the one that best predicts today
                                                         │
                                                         ▼
                                    Predicted 24-hour shape, split into N terminals
                                                         │
                                                         ▼
                        Greedily select terminals until capture_pct of volume covered
                                                         │
                                                         ▼
              Allocate order $ across selected terminals, proportional to predicted volume
                                                         │
                                                         ▼
        For each terminal: real tick data → terminal VWAP → your fill price (VWAP + impact)
                                                         │
                                                         ▼
                    Cost breakdown: market impact (your footprint) vs. timing (market drift)
```

Only the **target date** requires a full tick-data load — every prior day's
contribution to the shape history is a single cached 24-number array, which is
what keeps this fast even over months of history (see the caching notes in
`app.py`'s module docstring).

## Live output

The results view includes:

- **KPI strip** — order size, slices traded, arrival price, your VWAP, impact cost (bps + rating)
- **Cost breakdown** — market impact vs. timing/drift vs. total vs. arrival price, plus a separate slippage-vs-day-VWAP metric, in both USD and bps
- **Price vs. Fills** chart — market VWAP per terminal vs. your simulated fill price, with arrival price and your VWAP as reference lines
- **Order Size vs. Market Depth** chart — your allocation vs. total market volume per terminal (log scale)
- **Participation Rate** chart — your participation % per terminal, flagging terminals above 2x the median (higher impact risk)
- **Full fill-by-fill table** — every terminal's raw numbers, expandable at the bottom

The chart panel uses a dark, high-contrast "HUD" visual style (neon line/marker
glow, monospace labels, bordered KPI cards, shaded trading-session bands) defined
at the top of `app.py` in the `PALETTE` dict and the `_glow_*` / `_hud_frame` /
`_shade_sessions` helper functions — tweak those to restyle the chart.

## Project structure

```
.
├── app.py               # Streamlit app: UI, strategy, cost model, chart
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

Everything lives in a single `app.py`, organized top to bottom as:

| Section | What's in it |
|---|---|
| Visual style & chart helpers | `PALETTE`, glow/frame helpers, `plot_execution_summary()` |
| Config | secrets/env loading, S3 paths, strategy constants |
| Tick data + shape computation | `load_day_trades()`, `get_hourly_volume_shape()`, `compute_shape_for_day()` |
| Shape cache | `load_shape_cache()`, `save_shape_cache()`, `build_shape_history()` |
| GA lookback search | `weighted_mape()`, `ga_optimize_period()`, `predict_terminal_targets()` |
| Strategy + fills | `select_top_volume_windows()`, `run_vwap_strategy()` |
| Cost metrics | `rate_impact_bps()`, `compute_cost_metrics()` |
| Streamlit UI | inputs, run button, results rendering |

## Setup

**Requirements:** Python 3.9+

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
```

Set your AWS credentials as environment variables for local development:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=eu-north-1   # or your bucket's region
```

Run it:

```bash
streamlit run app.py
```

## Configuration

These constants live near the top of the "Config" section in `app.py` and control
strategy behavior:

| Constant | Default | Meaning |
|---|---|---|
| `S3_BUCKET` | `madot-algo-data` | Bucket holding tick data + shape cache (overridable via `S3_BUCKET` secret) |
| `HISTORY_START` | `2026-04-01` | Earliest date of tick data available |
| `SLICE_MINUTES` | `5` | Length of each execution "terminal" |
| `N_TERMINALS` | `288` | Derived: number of 5-minute slices in a day |
| `IMPACT_COEFFICIENT` | `100.0` | Scales the market impact cost model |
| `IMPACT_EXPONENT` | `0.5` | Controls how impact scales with participation rate (square-root impact model) |
| `MIN_LOOKBACK_DAYS` | `3` | Minimum GA lookback window |
| `MAX_LOOKBACK_DAYS` | `60` | Maximum GA lookback window |

> **Calibration note:** `IMPACT_COEFFICIENT` was originally `10.0`, which made every
> simulated order rate as "Excellent" (<2 bps) regardless of order size. That
> happened because the strategy intentionally trades in the highest-volume
> terminals (see `select_top_volume_windows()`), so participation rates on
> BTC/USDT typically land around 0.01-0.3% even for multi-million-dollar orders -
> and at that participation range, a coefficient of 10 produces well under 1 bp of
> impact no matter what. `100.0` keeps the same square-root shape but spreads
> results across the actual rating bands (roughly 1-3 bps at typical participation,
> up to 10-30+ bps if you push order size or capture % high enough to force
> trading into thinner slices). Neither value is empirically fit to real fills -
> if you have historical execution data, refitting this constant against it would
> be the correct next step rather than picking a rounder number.

The **volume capture target** (30–90%) and **order size** are exposed as sliders/
inputs in the UI itself rather than hardcoded.

## Data requirements

The app expects two things in your S3 bucket:

1. **Tick data** — one or more Parquet files per day at
   `parquets/aggTrades/*{YYYY-MM-DD}.parquet`, with at least these columns:
   - `price` (float)
   - `quantity` (float)
   - `transact_time` (int64, epoch milliseconds)
   - `is_buyer_maker` (bool)

2. **Shape cache** (optional but recommended) — a JSON file at
   `parquets/shape_cache/hourly_shapes.json` mapping `"YYYY-MM-DD"` to a 24-element
   list of hourly volume shares. Any date missing from this cache is computed live
   from tick data on first request and written back to the cache automatically.

## Deploying on Streamlit Cloud

1. Push `app.py` and `requirements.txt` to the **root** of your GitHub repo (a
   `requirements.txt` anywhere else won't be picked up).
2. In the app's **Settings → Secrets**, add:
   ```toml
   AWS_ACCESS_KEY_ID = "..."
   AWS_SECRET_ACCESS_KEY = "..."
   AWS_DEFAULT_REGION = "eu-north-1"
   S3_BUCKET = "madot-algo-data"
   ```
3. Deploy / reboot the app. If it can't find your credentials, it will show an
   error and stop rather than fail silently.

## Key formulas

**Participation rate** (per terminal):
```
participation_pct = your_qty / market_qty
```

**Market impact** (bps), a square-root impact model:
```
impact_bps = IMPACT_COEFFICIENT * participation_pct ** IMPACT_EXPONENT
fill_price = terminal_vwap * (1 + impact_bps / 10000)
```

**Slippage** — the app shows two different slippage numbers, benchmarked against
two different reference prices (both appear in the "Cost breakdown" row):

1. **Slippage vs. arrival price** — metrics 1-3 in the cost breakdown ("Market
   impact", "Timing / drift", "Total vs. arrival"):
   ```
   total_slippage_usd = total_qty * (your_vwap - arrival_price)
   total_bps          = (total_slippage_usd / total_usd) * 10000
   ```
   This is further split into the part you caused (impact) and the part the
   market caused on its own (timing/drift) — see the decomposition below.

2. **Slippage vs. day VWAP** — metric 4 ("Slippage vs. day VWAP"), aggregated
   from the per-terminal `slippage_vs_vwap_bps` / `cost_usd` columns computed in
   `run_vwap_strategy()`:
   ```
   slippage_vs_vwap_bps = ((fill_price - full_vwap) / full_vwap) * 10000
   cost_usd             = (slippage_vs_vwap_bps / 10000) * your_usd
   vwap_slippage_usd    = Σ cost_usd
   vwap_slippage_bps    = (vwap_slippage_usd / total_usd) * 10000
   ```
   `full_vwap` here is the VWAP over the *entire* execution window, not just one
   terminal — so this answers "how did I do against the market's overall average
   price today", a different (and commonly used) benchmark from arrival price.

**Cost decomposition vs. arrival price** — the arrival-price slippage above is
further split into the part you caused (impact) and the part the market caused
on its own (timing/drift):
```
total_impact_usd = Σ (your_qty * terminal_vwap * impact_bps / 10000)
timing_cost      = total_slippage_usd - total_impact_usd
```

Impact is rated `Excellent` (<2 bps), `Good` (<5 bps), `Acceptable` (<10 bps),
`Poor` (<20 bps), or `Bad` (≥20 bps) — see `rate_impact_bps()`.

## Known limitations

- The market impact model is a simplified square-root heuristic, not calibrated
  against real fill data — treat impact/rating numbers as directionally useful,
  not as a precise cost forecast.
- The GA searches lookback *period* only; it doesn't optimize other hyperparameters
  (impact coefficient/exponent, capture threshold) — those are fixed constants.
- Shape history and volume prediction use hourly buckets even though execution
  happens in 5-minute terminals, so within-hour volume is assumed uniform.
- Requires network access to S3 on every run for the target date's tick data;
  there's no local/offline data path built in.
