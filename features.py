import pandas as pd
import numpy as np

from db import fetch_all_rows

# Same asset set + order the notebook engineered features for.
ASSET_COLS = ["SPX", "GOLD", "TNX", "DXY", "VIX"]

# Macro columns that got the 30-day publication-lag shift in training.
MACRO_LAG_COLS = ["unemployment", "growth_rate"]

PRICE_TABLE = "price_action"
MACRO_TABLE = "macro_indicators"


def _engineer_price_features() -> pd.DataFrame:
    rows = fetch_all_rows(PRICE_TABLE)
    if not rows:
        raise RuntimeError(
            f"Supabase table '{PRICE_TABLE}' returned no rows. "
            "Check the cron job that syncs it from yfinance."
        )

    prices = pd.DataFrame(rows)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)

    missing_assets = [c for c in ASSET_COLS if c not in prices.columns]
    if missing_assets:
        raise RuntimeError(
            f"'{PRICE_TABLE}' is missing expected columns: {missing_assets}"
        )

    engineered_cols = ["date"]

    for col in ASSET_COLS:
        log_ret = np.log(prices[col] / prices[col].shift(1))

        prices[f"{col}_mom_3"] = prices[col].shift(1) / prices[col].shift(4) - 1
        prices[f"{col}_mom_7"] = prices[col].shift(1) / prices[col].shift(8) - 1
        prices[f"{col}_mom_30"] = prices[col].shift(1) / prices[col].shift(31) - 1

        prices[f"{col}_t-3"] = log_ret.shift(1).rolling(3).mean()
        prices[f"{col}_t-7"] = log_ret.shift(1).rolling(7).mean()
        prices[f"{col}_t-30"] = log_ret.shift(1).rolling(30).mean()

        prices[f"{col}_vol_7"] = log_ret.shift(1).rolling(7).std()
        prices[f"{col}_vol_30"] = log_ret.shift(1).rolling(30).std()

        engineered_cols += [
            f"{col}_mom_3", f"{col}_mom_7", f"{col}_mom_30",
            f"{col}_t-3", f"{col}_t-7", f"{col}_t-30",
            f"{col}_vol_7", f"{col}_vol_30",
        ]

    return prices[engineered_cols]


def _engineer_macro_features() -> pd.DataFrame:
    rows = fetch_all_rows(MACRO_TABLE)
    if not rows:
        raise RuntimeError(
            f"Supabase table '{MACRO_TABLE}' returned no rows. "
            "Check the cron job that syncs it from FRED."
        )

    macro = pd.DataFrame(rows)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()

    missing = [c for c in MACRO_LAG_COLS + ["interest_rate"] if c not in macro.columns]
    if missing:
        raise RuntimeError(f"'{MACRO_TABLE}' is missing expected columns: {missing}")

    for col in MACRO_LAG_COLS:
        macro[col] = macro[col].shift(30)

    daily_index = pd.date_range(start=macro.index.min(), end=macro.index.max(), freq="D")
    macro_daily = macro.reindex(daily_index).ffill().reset_index()
    macro_daily = macro_daily.rename(columns={"index": "date"})

    return macro_daily[["date", "unemployment", "interest_rate", "growth_rate"]]
