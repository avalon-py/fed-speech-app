"""
scripts/update_data.py

Runs daily via GitHub Actions. Refreshes dataset/macro_indicators.csv and
dataset/price_action.csv so the deployed Streamlit app always has fresh
trailing-window data to build features from, without ever calling FRED or
Yahoo Finance itself.

Design:
- Genuinely incremental: fetch from the last date already recorded in each
  CSV (not a fixed lookback window), plus a small buffer so a skipped run
  (holiday, Action failure, etc.) gets automatically made up the next day.
- Apply the SAME transforms used in TOOLS_get_macros.ipynb / TOOLS_get_prices.ipynb
  and respected by FE_main.ipynb: reindex onto a full daily calendar and
  ffill (this is what makes weekends/holidays NOT show up as NaN — it's the
  same mechanism your training data went through, not something bolted on).
  unemployment/growth_rate are left UNshifted here, matching the raw CSVs —
  FE_main.ipynb applies its own 30-day shift after loading, at training/
  inference time, not before saving.
- Validate the result (columns, no unexpected NaNs in the recent tail, date
  recency, sane value ranges) BEFORE touching the real CSVs.
- Write to temp files and atomically replace the real files only if BOTH
  datasets pass validation. If anything fails, exit non-zero, leave the
  last-known-good CSVs untouched, and the workflow's commit step is skipped.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from fredapi import Fred

DATASET_DIR = "dataset"
MACRO_PATH = os.path.join(DATASET_DIR, "macro_indicators.csv")
PRICE_PATH = os.path.join(DATASET_DIR, "price_action.csv")

BOOTSTRAP_START = "1996-01-01"  # only used the very first time either CSV
                                 # doesn't exist yet — every run after that
                                 # is incremental from the last recorded date

# Self-heal buffer: how far before the last recorded date we re-fetch, so a
# missed run (weekend outage, one failed Action, whatever) gets backfilled
# by the next successful run instead of leaving a permanent gap. A week is
# plenty — if the job is down longer than that, something needs a human.
SELF_HEAL_DAYS = 7

# FRED-only: observation_start limits what comes back, but doesn't change
# when a series was last *released*. If we ask FRED for "just the last 7
# days," a monthly series with no release in that window comes back empty,
# and ffill has nothing to anchor to. So the FRED fetch itself reaches back
# further than SELF_HEAL_DAYS purely to find a real prior value — but only
# the last SELF_HEAL_DAYS of the *result* actually gets merged into the CSV.
MACRO_FFILL_ANCHOR_DAYS = 120

TICKERS = {
    "SPX": "^GSPC",
    "TNX": "^TNX",
    "GOLD": "GC=F",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
}


def log(msg: str) -> None:
    print(f"[update_data] {msg}", flush=True)


def fail(msg: str) -> None:
    log(f"VALIDATION FAILED: {msg}")
    sys.exit(1)


def last_recorded_date(path: str) -> pd.Timestamp | None:
    if not os.path.exists(path):
        return None
    existing = pd.read_csv(path, usecols=["date"])
    if existing.empty:
        return None
    return pd.to_datetime(existing["date"]).max()


# ---------------------------------------------------------------------------
# Macro indicators (FRED)
# ---------------------------------------------------------------------------

def fetch_macro(today: pd.Timestamp, last_date: pd.Timestamp | None) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        fail("FRED_API_KEY env var is not set")

    fred = Fred(api_key=api_key)

    if last_date is None:
        # First run ever for this repo — bootstrap full history once, same
        # as TOOLS_get_macros.ipynb did originally.
        obs_start = BOOTSTRAP_START
        merge_floor = None
        log("no existing macro CSV — bootstrapping full history (one-time only)")
    else:
        # Fetch further back than we'll actually merge, purely so ffill has
        # a real value to anchor to at the start of the merge window.
        obs_start = (last_date - timedelta(days=MACRO_FFILL_ANCHOR_DAYS)).strftime("%Y-%m-%d")
        merge_floor = last_date - timedelta(days=SELF_HEAL_DAYS)

    unemp = fred.get_series("UNRATE", observation_start=obs_start).to_frame("unemployment")
    ffr = fred.get_series("FEDFUNDS", observation_start=obs_start).to_frame("interest_rate")
    gdp = fred.get_series("A191RL1Q225SBEA", observation_start=obs_start).to_frame("growth_rate")

    macro = pd.concat([unemp, ffr, gdp], axis=1).sort_index()
    if macro.empty:
        fail("FRED returned no rows at all — check FRED_API_KEY / series IDs")

    # Same conversion as the training notebook: full daily calendar, ffill.
    # This is what makes the format match — monthly/quarterly FRED releases
    # become a value for every single day, identical to what training saw.
    daily_range = pd.date_range(start=macro.index.min(), end=today, freq="D")
    macro_daily = macro.reindex(daily_range).ffill()
    macro_daily = macro_daily.reset_index().rename(columns={"index": "date"})

    if merge_floor is not None:
        # Trim off the ffill-anchor padding — only the genuinely new window
        # (plus the small self-heal buffer) actually gets merged.
        macro_daily = macro_daily[macro_daily["date"] >= merge_floor].reset_index(drop=True)

    return macro_daily


def validate_macro(df: pd.DataFrame, today: pd.Timestamp) -> None:
    expected_cols = {"date", "unemployment", "interest_rate", "growth_rate"}
    if set(df.columns) != expected_cols:
        fail(f"macro columns mismatch: got {set(df.columns)}, expected {expected_cols}")

    if df.empty:
        fail("macro dataframe is empty")

    last_date = pd.to_datetime(df["date"]).max()
    if (today - last_date) > timedelta(days=5):
        fail(f"macro data is stale: last date {last_date.date()}, today {today.date()}")

    tail = df.tail(30)
    if tail[["unemployment", "interest_rate", "growth_rate"]].isna().any().any():
        fail("NaNs found in the last 30 rows of macro data — every calendar day, "
             "including weekends, should have a forward-filled value")

    if not pd.to_datetime(df["date"]).is_monotonic_increasing:
        fail("macro dates are not sorted ascending")

    if pd.to_datetime(df["date"]).duplicated().any():
        fail("duplicate dates found in macro data")

    log(f"macro OK — {len(df)} rows, last date {last_date.date()}")


# ---------------------------------------------------------------------------
# Price action (yfinance)
# ---------------------------------------------------------------------------

def fetch_prices(today: pd.Timestamp, last_date: pd.Timestamp | None) -> pd.DataFrame:
    if last_date is None:
        start = BOOTSTRAP_START
        log("no existing price CSV — bootstrapping full history (one-time only). "
            "NOTE: this will NOT replicate the pre-2000 XAUUSD gold patch from "
            "TOOLS_get_prices.ipynb — that's only relevant for historical backfill, "
            "not going forward, but flagging it in case this branch ever actually runs.")
    else:
        start = (last_date - timedelta(days=SELF_HEAL_DAYS)).strftime("%Y-%m-%d")

    end = (today + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    data = {}
    for name, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            fail(f"yfinance returned no data for {name} ({ticker})")
        data[name] = df["Close"]

    prices = pd.concat(data, axis=1, sort=True)  # explicit, silences Pandas4Warning
    prices.columns = TICKERS.keys()
    prices["TNX"] = prices["TNX"] / 10
    prices = prices.sort_index()

    # Same conversion as the training notebook: reindex onto every calendar
    # day (trading days only come back from yfinance) and ffill. This is
    # what makes weekends/holidays carry forward Friday's close instead of
    # showing up as NaN.
    full_index = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq="D")
    prices = prices.reindex(full_index).ffill()
    prices = prices.reset_index().rename(columns={"index": "date"})

    return prices


def validate_prices(df: pd.DataFrame, today: pd.Timestamp) -> None:
    expected_cols = {"date", "SPX", "TNX", "GOLD", "VIX", "DXY"}
    if set(df.columns) != expected_cols:
        fail(f"price columns mismatch: got {set(df.columns)}, expected {expected_cols}")

    if df.empty:
        fail("price dataframe is empty")

    last_date = pd.to_datetime(df["date"]).max()
    if (today - last_date) > timedelta(days=5):
        fail(f"price data is stale: last date {last_date.date()}, today {today.date()}")

    tail = df.tail(35)  # >30 so the 30-day rolling window has no gaps at inference
    numeric_cols = ["SPX", "TNX", "GOLD", "VIX", "DXY"]
    if tail[numeric_cols].isna().any().any():
        fail("NaNs found in the last 35 rows of price data — every calendar day, "
             "including weekends/holidays, should have a forward-filled value")

    # Sanity bounds — catches unit errors (e.g. forgetting the TNX /10) or a
    # bad ticker swap, not just missing data.
    bounds = {
        "SPX": (500, 20000),
        "TNX": (0, 20),
        "GOLD": (100, 20000),
        "VIX": (5, 150),
        "DXY": (50, 200),
    }
    for col, (lo, hi) in bounds.items():
        recent = df[col].tail(10)
        if not recent.between(lo, hi).all():
            fail(f"{col} values out of sane range in last 10 rows: {recent.tolist()}")

    if not pd.to_datetime(df["date"]).is_monotonic_increasing:
        fail("price dates are not sorted ascending")

    if pd.to_datetime(df["date"]).duplicated().any():
        fail("duplicate dates found in price data")

    log(f"price OK — {len(df)} rows, last date {last_date.date()}")


# ---------------------------------------------------------------------------
# Merge-and-write helpers
# ---------------------------------------------------------------------------

def merge_into_existing(existing_path: str, fresh_df: pd.DataFrame) -> pd.DataFrame:
    """Combine freshly fetched rows with whatever's already on disk, keeping
    the fresh values wherever dates overlap (fresh data may include late
    revisions), and preserving all older history the fresh pull didn't touch.
    """
    fresh_df = fresh_df.copy()
    fresh_df["date"] = pd.to_datetime(fresh_df["date"])

    if not os.path.exists(existing_path):
        log(f"no existing file at {existing_path}, writing fresh data only")
        return fresh_df.sort_values("date").reset_index(drop=True)

    existing_df = pd.read_csv(existing_path)
    existing_df["date"] = pd.to_datetime(existing_df["date"])

    combined = pd.concat([existing_df, fresh_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last")  # fresh wins on overlap
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def atomic_write(df: pd.DataFrame, real_path: str) -> None:
    tmp_path = real_path + ".tmp"
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(tmp_path, index=False)
    os.replace(tmp_path, real_path)  # atomic on POSIX
    log(f"wrote {real_path} ({len(out)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    log(f"run date (UTC): {today.date()}")

    os.makedirs(DATASET_DIR, exist_ok=True)

    macro_last = last_recorded_date(MACRO_PATH)
    price_last = last_recorded_date(PRICE_PATH)
    log(f"last recorded macro date: {macro_last.date() if macro_last is not None else 'none (bootstrap)'}")
    log(f"last recorded price date: {price_last.date() if price_last is not None else 'none (bootstrap)'}")

    # --- Macro ---
    macro_fresh = fetch_macro(today, macro_last)
    macro_combined = merge_into_existing(MACRO_PATH, macro_fresh)
    validate_macro(macro_combined, today)

    # --- Prices ---
    price_fresh = fetch_prices(today, price_last)
    price_combined = merge_into_existing(PRICE_PATH, price_fresh)
    validate_prices(price_combined, today)

    # Only write once BOTH datasets are validated — never leave the repo in
    # a half-updated state where macro moved forward but price didn't.
    atomic_write(macro_combined, MACRO_PATH)
    atomic_write(price_combined, PRICE_PATH)

    log("done.")


if __name__ == "__main__":
    main()
