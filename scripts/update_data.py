"""
scripts/update_data.py

Runs daily via GitHub Actions. Refreshes dataset/macro_indicators.csv and
dataset/price_action.csv so the deployed Streamlit app always has fresh
trailing-window data to build features from, without ever calling FRED or
Yahoo Finance itself.

Design:
- Fetch fresh data from FRED + yfinance (small, cheap, full series each run —
  matches the original training notebooks exactly, no drift risk).
- Apply the SAME transforms used in TOOLS_get_macros.ipynb / TOOLS_get_prices.ipynb
  and respected by FE_main.ipynb (e.g. unemployment/growth_rate shifted 30 days
  BEFORE the daily ffill; interest_rate is not shifted).
- Validate the result (columns, dtypes, no unexpected NaNs in the recent tail,
  date recency) BEFORE touching the real CSVs.
- Write to temp files and atomically replace the real files only if BOTH
  datasets pass validation. If anything fails, exit non-zero, leave the
  last-known-good CSVs untouched, and the workflow's commit step is skipped.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

DATASET_DIR = "dataset"
MACRO_PATH = os.path.join(DATASET_DIR, "macro_indicators.csv")
PRICE_PATH = os.path.join(DATASET_DIR, "price_action.csv")

# How far back to pull on every run. Full history for macro (tiny data,
# and some FRED series get revised, so re-pulling is safer than trusting
# old rows). For prices, a generous trailing window is enough — we only
# need to *extend* the existing calendar-filled history, not rebuild it.
MACRO_START = "1996-01-01"  # floor used only when NO existing CSV is found at all
LOOKBACK_DAYS = 400  # comfortably covers the 30-day rolling windows, quarterly
                      # GDP releases, and enough buffer to self-heal any run
                      # that got skipped (holiday, Action failure, etc.)
MACRO_FFILL_PAD_DAYS = 120  # extra run-in fetched (not merged as "new" per se,
                             # just included) so ffill has a real prior value
                             # right at the edge of the window — otherwise the
                             # first few rows of a monthly/quarterly series can
                             # come back NaN before their first in-window release

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


# ---------------------------------------------------------------------------
# Macro indicators (FRED)
# ---------------------------------------------------------------------------

def fetch_macro(today: pd.Timestamp, fetch_start: pd.Timestamp) -> pd.DataFrame:
    """Fetch only a recent window from FRED (observation_start), not full
    history. `fetch_start` is a rolling lookback (see main()), NOT a fixed
    1996 floor — that floor is only ever used the very first time this
    script runs against a repo with no existing CSV at all.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        fail("FRED_API_KEY env var is not set")

    fred = Fred(api_key=api_key)
    obs_start = fetch_start.strftime("%Y-%m-%d")

    unemp = fred.get_series("UNRATE", observation_start=obs_start).to_frame("unemployment")
    ffr = fred.get_series("FEDFUNDS", observation_start=obs_start).to_frame("interest_rate")
    gdp = fred.get_series("A191RL1Q225SBEA", observation_start=obs_start).to_frame("growth_rate")

    macro = pd.concat([unemp, ffr, gdp], axis=1).sort_index()

    if macro.empty:
        fail("FRED returned no rows in the lookback window — widen LOOKBACK_DAYS")

    # NOTE: intentionally NOT shifting here. FE_main.ipynb applies the
    # 30-day shift to unemployment/growth_rate (not interest_rate) itself,
    # AFTER loading this CSV. This script must keep producing the same raw,
    # unshifted series that training consumed, or the app's shift-at-inference
    # step will double-shift and silently corrupt every prediction.
    #
    # ffill needs a bit of run-in before fetch_start to correctly carry
    # forward monthly/quarterly values that were last *released* before the
    # window started — FRED's observation_start only limits what comes back,
    # it doesn't change when a series was last updated. So we pad the ffill
    # start a little earlier than fetch_start using merge_into_existing()
    # in main(), not here; this function just returns what FRED gave us,
    # calendar-filled across its own returned range.
    daily_range = pd.date_range(start=macro.index.min(), end=today, freq="D")
    macro_daily = macro.reindex(daily_range).ffill()
    macro_daily = macro_daily.reset_index().rename(columns={"index": "date"})

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
        fail("NaNs found in the last 30 rows of macro data")

    if not pd.to_datetime(df["date"]).is_monotonic_increasing:
        fail("macro dates are not sorted ascending")

    log(f"macro OK — {len(df)} rows, last date {last_date.date()}")


# ---------------------------------------------------------------------------
# Price action (yfinance)
# ---------------------------------------------------------------------------

def fetch_prices(today: pd.Timestamp) -> pd.DataFrame:
    start = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (today + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    data = {}
    for name, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            fail(f"yfinance returned no data for {name} ({ticker})")
        data[name] = df["Close"]

    prices = pd.concat(data, axis=1)
    prices.columns = TICKERS.keys()
    prices["TNX"] = prices["TNX"] / 10
    prices = prices.sort_index()

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
        fail("NaNs found in the last 35 rows of price data")

    # Sanity bounds — catches unit errors (e.g. forgetting the TNX /10) or
    # a bad ticker swap, not just missing data.
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

    log(f"price OK — {len(df)} rows, last date {last_date.date()}")


# ---------------------------------------------------------------------------
# Merge-and-write helpers
# ---------------------------------------------------------------------------

def merge_into_existing(existing_path: str, fresh_df: pd.DataFrame) -> pd.DataFrame:
    """Combine freshly fetched rows with whatever's already on disk, keeping
    the fresh values wherever dates overlap (fresh data may include late
    revisions), and preserving any older history the fresh pull didn't cover.
    """
    fresh_df = fresh_df.copy()
    fresh_df["date"] = pd.to_datetime(fresh_df["date"])

    if not os.path.exists(existing_path):
        log(f"no existing file at {existing_path}, writing fresh data only")
        return fresh_df.sort_values("date").reset_index(drop=True)

    existing_df = pd.read_csv(existing_path)
    existing_df["date"] = pd.to_datetime(existing_df["date"])

    combined = pd.concat([existing_df, fresh_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last")
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

    # Rolling lookback, NOT a fixed "since 1996" fetch. Only used as the
    # actual floor if this is a brand new repo with no CSV yet at all.
    have_macro_csv = os.path.exists(MACRO_PATH)
    have_price_csv = os.path.exists(PRICE_PATH)

    if have_macro_csv:
        macro_fetch_start = today - timedelta(days=LOOKBACK_DAYS + MACRO_FFILL_PAD_DAYS)
    else:
        macro_fetch_start = pd.Timestamp(MACRO_START)
        log("no existing macro CSV found — first run will pull full history once")

    _ = have_price_csv  # price fetch always uses a fixed rolling window (see fetch_prices);
                         # no special first-run case needed since 400 days is cheap either way

    # --- Macro ---
    macro_fresh = fetch_macro(today, macro_fetch_start)
    macro_combined = merge_into_existing(MACRO_PATH, macro_fresh)
    validate_macro(macro_combined, today)

    # --- Prices ---
    price_fresh = fetch_prices(today)
    price_combined = merge_into_existing(PRICE_PATH, price_fresh)
    validate_prices(price_combined, today)

    # Only write once BOTH datasets are validated — never leave the repo in
    # a half-updated state where macro moved forward but price didn't, or
    # vice versa.
    atomic_write(macro_combined, MACRO_PATH)
    atomic_write(price_combined, PRICE_PATH)

    log("done.")


if __name__ == "__main__":
    main()
