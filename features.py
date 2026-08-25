"""
Feature assembly for inference.

IMPORTANT: This module no longer calls the FRED API or yfinance directly.
Streamlit Cloud has those two gateways blocked, so a GitHub Actions cron job
refreshes two local CSVs daily instead:

    dataset/price_action.csv      <- was yfinance
    dataset/macro_indicators.csv  <- was FRED

Everything below reads from those files and reproduces, row for row, the
same feature engineering that FE_main.ipynb applied at training time
(cell 4 for macro, cell 7 for price). If this ever drifts from the
notebook again, the model will silently see out-of-distribution inputs,
so keep the two in sync.
"""

import pandas as pd
import numpy as np
import os

PRICE_CSV = os.path.join("dataset", "price_action.csv")
MACRO_CSV = os.path.join("dataset", "macro_indicators.csv")

# Same asset set + order the notebook engineered features for.
ASSET_COLS = ["SPX", "GOLD", "TNX", "DXY", "VIX"]

# Macro columns that got the 30-day publication-lag shift in training.
# interest_rate (FEDFUNDS) was NOT shifted.
MACRO_LAG_COLS = ["unemployment", "growth_rate"]


def _engineer_price_features() -> pd.DataFrame:
    """
    Reproduces notebook cell 7 (the price-side half of it) on the full
    history in price_action.csv. Returns one row per date with the
    engineered columns only (no raw close prices - those weren't in
    X_train either).
    """
    if not os.path.exists(PRICE_CSV):
        raise FileNotFoundError(
            f"{PRICE_CSV} not found. Check the cron job that syncs it from yfinance."
        )

    prices = pd.read_csv(PRICE_CSV)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)

    missing_assets = [c for c in ASSET_COLS if c not in prices.columns]
    if missing_assets:
        raise RuntimeError(
            f"{PRICE_CSV} is missing expected columns: {missing_assets}"
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
    """
    Reproduces notebook cell 4: shift unemployment/growth_rate by 30 rows
    (publication lag), reindex to a daily calendar, forward-fill.
    """
    if not os.path.exists(MACRO_CSV):
        raise FileNotFoundError(
            f"{MACRO_CSV} not found. Check the cron job that syncs it from FRED."
        )

    macro = pd.read_csv(MACRO_CSV)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()

    missing = [c for c in MACRO_LAG_COLS + ["interest_rate"] if c not in macro.columns]
    if missing:
        raise RuntimeError(f"{MACRO_CSV} is missing expected columns: {missing}")

    for col in MACRO_LAG_COLS:
        macro[col] = macro[col].shift(30)

    daily_index = pd.date_range(start=macro.index.min(), end=macro.index.max(), freq="D")
    macro_daily = macro.reindex(daily_index).ffill().reset_index()
    macro_daily = macro_daily.rename(columns={"index": "date"})

    return macro_daily[["date", "unemployment", "interest_rate", "growth_rate"]]


def _last_row_on_or_before(df: pd.DataFrame, date: "pd.Timestamp | np.datetime64") -> pd.Series:
    subset = df[df["date"] <= pd.Timestamp(date)]
    if subset.empty:
        raise RuntimeError(
            f"No rows on or before {date} — CSV history doesn't reach back that far."
        )
    return subset.sort_values("date").iloc[-1]


def fetch_price_features(date) -> dict:
    """
    Engineered price features (mom/t-/vol, per asset) as of `date`,
    read from dataset/price_action.csv instead of yfinance.
    """
    engineered = _engineer_price_features()
    row = _last_row_on_or_before(engineered, date)
    feats = row.drop(labels=["date"]).to_dict()

    missing = [k for k, v in feats.items() if pd.isna(v)]
    if missing:
        raise RuntimeError(
            f"Not enough price history in {PRICE_CSV} to compute {missing} "
            f"for {date} (need up to 31 prior trading days)."
        )
    return feats


def fetch_macro_features(date) -> dict:
    """
    Macro indicators (with the same 30-day lag applied at training time)
    as of `date`, read from dataset/macro_indicators.csv instead of FRED.
    """
    engineered = _engineer_macro_features()
    row = _last_row_on_or_before(engineered, date)
    feats = row.drop(labels=["date"]).to_dict()

    missing = [k for k, v in feats.items() if pd.isna(v)]
    if missing:
        raise RuntimeError(
            f"Not enough macro history in {MACRO_CSV} to compute {missing} for {date}."
        )
    return feats


def build_feature_vector(
    embedding: np.ndarray,
    date,
    feature_columns: list,  # X_train.columns — must match exactly!
) -> pd.DataFrame:
    """
    Assemble the full feature vector in the same column order as X_train.
    """
    price_features = fetch_price_features(date)
    macro_features = fetch_macro_features(date)

    all_features = {**price_features, **macro_features}

    for i, val in enumerate(embedding):
        all_features[f"emb_{i}"] = val

    feature_df = pd.DataFrame([all_features])
    feature_df = feature_df.reindex(columns=feature_columns)

    missing = feature_df.columns[feature_df.isnull().any()].tolist()
    if missing:
        raise RuntimeError(f"Missing features after assembly: {missing}")

    return feature_df
