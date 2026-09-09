"""
scripts/update_data_db.py

Runs daily via GitHub Actions. Refreshes macro_indicators and price_action
in Supabase so the deployed Streamlit app always has fresh trailing-window
data to build features from, without ever calling FRED or Yahoo Finance
itself.

Changes from the CSV version (update_data.py):
- State ('what's the last date we have?') comes from a MAX(date) query
  against each table instead of reading the CSV's last row. Everything
  downstream — the incremental fetch window, the self-heal buffer, the
  ffill anchor — is unchanged, since all of that logic only depends on
  knowing the last recorded date, not on where that date came from.
- No more merge_into_existing() reading the whole CSV into memory to
  combine with fresh data. Each fetch already returns a self-contained
  window (self-heal buffer through today); that window is upserted
  directly via INSERT ... ON CONFLICT (date) DO UPDATE, so overlapping
  dates get overwritten with the fresh (possibly revised) values and the
  database handles "fresh wins on overlap" itself — no manual
  concat/drop_duplicates needed.
- Validation runs on the freshly fetched window only, not the full
  historical table. The checks (no NaNs in the tail, monotonic dates, no
  internal duplicates, sane value ranges, recency) are about catching a
  bad fetch, not re-auditing 30 years of already-good history every day —
  so there's no need to pull the whole table back out of Postgres just to
  validate it.
- The "write only if both datasets pass validation" guarantee is now a
  single database transaction: both fetches are validated BEFORE either
  upsert runs, and both upserts commit together at the end. If anything
  raises before that final commit, nothing is written, same as the old
  atomic_write() pattern that never let macro update without price (or
  vice versa).

Requires the DB_URL environment variable — the Supabase "Transaction
pooler" connection string (port 6543). Set it as a GitHub Actions secret.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
import yfinance as yf
from fredapi import Fred
from psycopg2.extras import execute_values

DB_URL = os.environ["DB_URL"]

BOOTSTRAP_START = "1996-01-01"  # only used the very first time either table is empty

SELF_HEAL_DAYS = 7
MACRO_FFILL_ANCHOR_DAYS = 120

TICKERS = {
    "SPX": "^GSPC",
    "TNX": "^TNX",
    "GOLD": "GC=F",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
}


def log(msg: str) -> None:
    print(f"[update_data_db] {msg}", flush=True)


def fail(msg: str) -> None:
    log(f"VALIDATION FAILED: {msg}")
    sys.exit(msg)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_last_date(conn, table: str) -> "pd.Timestamp | None":
    assert table in ("macro_indicators", "price_action"), f"unexpected table: {table}"
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(date) FROM {table}")
        (last_date,) = cur.fetchone()
    return pd.Timestamp(last_date) if last_date is not None else None


def upsert_macro(conn, df: pd.DataFrame) -> None:
    rows = [
        (
            d.date(),
            None if pd.isna(u) else float(u),
            None if pd.isna(r) else float(r),
            None if pd.isna(g) else float(g),
        )
        for d, u, r, g in df[["date", "unemployment", "interest_rate", "growth_rate"]]
        .itertuples(index=False, name=None)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO macro_indicators (date, unemployment, interest_rate, growth_rate)
            VALUES %s
            ON CONFLICT (date) DO UPDATE SET
                unemployment = EXCLUDED.unemployment,
                interest_rate = EXCLUDED.interest_rate,
                growth_rate = EXCLUDED.growth_rate
            """,
            rows,
        )
    log(f"upserted {len(rows)} row(s) into macro_indicators")


def upsert_prices(conn, df: pd.DataFrame) -> None:
    rows = [
        (
            d.date(),
            None if pd.isna(spx) else float(spx),
            None if pd.isna(tnx) else float(tnx),
            None if pd.isna(gold) else float(gold),
            None if pd.isna(vix) else float(vix),
            None if pd.isna(dxy) else float(dxy),
        )
        for d, spx, tnx, gold, vix, dxy in df[["date", "SPX", "TNX", "GOLD", "VIX", "DXY"]]
        .itertuples(index=False, name=None)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO price_action (date, spx, tnx, gold, vix, dxy)
            VALUES %s
            ON CONFLICT (date) DO UPDATE SET
                spx = EXCLUDED.spx,
                tnx = EXCLUDED.tnx,
                gold = EXCLUDED.gold,
                vix = EXCLUDED.vix,
                dxy = EXCLUDED.dxy
            """,
            rows,
        )
    log(f"upserted {len(rows)} row(s) into price_action")


# ---------------------------------------------------------------------------
# Macro indicators (FRED) — fetch/validate logic unchanged from the CSV
# version; only *what it's compared against* (last_date's source) changed.
# ---------------------------------------------------------------------------

def fetch_macro(today: pd.Timestamp, last_date: "pd.Timestamp | None") -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        fail("FRED_API_KEY env var is not set")

    fred = Fred(api_key=api_key)

    if last_date is None:
        obs_start = BOOTSTRAP_START
        merge_floor = None
        log("no existing macro data — bootstrapping full history (one-time only)")
    else:
        obs_start = (last_date - timedelta(days=MACRO_FFILL_ANCHOR_DAYS)).strftime("%Y-%m-%d")
        merge_floor = last_date - timedelta(days=SELF_HEAL_DAYS)

    unemp = fred.get_series("UNRATE", observation_start=obs_start).to_frame("unemployment")
    ffr = fred.get_series("FEDFUNDS", observation_start=obs_start).to_frame("interest_rate")
    gdp = fred.get_series("A191RL1Q225SBEA", observation_start=obs_start).to_frame("growth_rate")

    macro = pd.concat([unemp, ffr, gdp], axis=1).sort_index()
    if macro.empty:
        fail("FRED returned no rows at all — check FRED_API_KEY / series IDs")

    daily_range = pd.date_range(start=macro.index.min(), end=today, freq="D")
    macro_daily = macro.reindex(daily_range).ffill()
    macro_daily = macro_daily.reset_index().rename(columns={"index": "date"})

    if merge_floor is not None:
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
        fail("duplicate dates found in this fetch's macro window")

    log(f"macro OK — {len(df)} row(s) in this fetch window, last date {last_date.date()}")


# ---------------------------------------------------------------------------
# Price action (yfinance) — same story as macro above.
# ---------------------------------------------------------------------------

def fetch_prices(today: pd.Timestamp, last_date: "pd.Timestamp | None") -> pd.DataFrame:
    if last_date is None:
        start = BOOTSTRAP_START
        log("no existing price data — bootstrapping full history (one-time only). "
            "NOTE: this will NOT replicate the pre-2000 XAUUSD gold patch from "
            "TOOLS_get_prices.ipynb — flagging in case this branch ever actually runs.")
    else:
        start = (last_date - timedelta(days=SELF_HEAL_DAYS)).strftime("%Y-%m-%d")

    end = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    data = {}
    for name, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            fail(f"yfinance returned no data for {name} ({ticker})")
        data[name] = df["Close"]

    prices = pd.concat(data, axis=1, sort=True)
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

    tail = df.tail(35)
    numeric_cols = ["SPX", "TNX", "GOLD", "VIX", "DXY"]
    if tail[numeric_cols].isna().any().any():
        fail("NaNs found in the last 35 rows of price data — every calendar day, "
             "including weekends/holidays, should have a forward-filled value")

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
        fail("duplicate dates found in this fetch's price window")

    log(f"price OK — {len(df)} row(s) in this fetch window, last date {last_date.date()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from pipeline_logging import start_run, finish_run  # add near the top, with the other imports


def main() -> None:
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    log(f"run date (UTC): {today.date()}")

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    run_id = start_run(conn, "market_data_updater")

    try:
        macro_last = get_last_date(conn, "macro_indicators")
        price_last = get_last_date(conn, "price_action")
        log(f"last recorded macro date: {macro_last.date() if macro_last is not None else 'none (bootstrap)'}")
        log(f"last recorded price date: {price_last.date() if price_last is not None else 'none (bootstrap)'}")

        macro_fresh = fetch_macro(today, macro_last)
        validate_macro(macro_fresh, today)

        price_fresh = fetch_prices(today, price_last)
        validate_prices(price_fresh, today)

        upsert_macro(conn, macro_fresh)
        upsert_prices(conn, price_fresh)

        conn.commit()
        log("done — both tables committed together.")

        finish_run(
            conn, run_id, status="success",
            rows_processed=len(macro_fresh) + len(price_fresh),
            message=f"Upserted {len(macro_fresh)} macro row(s), {len(price_fresh)} price row(s)",
            details={
                "macro_rows": len(macro_fresh),
                "price_rows": len(price_fresh),
                "macro_last_date": str(macro_fresh["date"].max()),
                "price_last_date": str(price_fresh["date"].max()),
            },
        )

    except SystemExit as e:
        conn.rollback()
        error_msg = str(e.code) if e.code else "validation failed (no message)"
        finish_run(conn, run_id, status="failed", error_message=error_msg)
        raise
    except Exception as e:
        conn.rollback()
        finish_run(conn, run_id, status="failed", error_message=str(e))
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
