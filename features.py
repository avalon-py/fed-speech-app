import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

FRED_API_KEY = None  # Will be set from streamlit secrets

TICKERS = {
    "SPX": "^GSPC",
    "GOLD": "GC=F",
    "TNX": "^TNX",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
}

FRED_SERIES = {
    "unemployment": "UNRATE",
    "interest_rate": "FEDFUNDS",
    "growth_rate":   "A191RL1Q225SBEA",  # Real GDP growth
}


def fetch_price_features(date: datetime) -> dict:
    """
    Fetch t-3, t-7, t-30 log return features from Yahoo Finance.
    date = speech date (we use t-1 as baseline, consistent with training)
    """
    # Fetch 45 days of history to cover t-30 + buffer
    start = date - timedelta(days=60)
    end   = date + timedelta(days=2)  # +2 to ensure we get date itself

    features = {}

    for asset, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            df = df["Close"].dropna().sort_index()

            # Align to speech date: use last available price on/before date
            df = df[df.index <= pd.Timestamp(date)]

            if len(df) < 31:
                raise ValueError(f"Not enough price history for {ticker}")

            # Log returns
            log_ret = np.log(df / df.shift(1)).dropna()

            # t-3, t-7, t-30 rolling means (shift(1) then rolling, consistent with training)
            # At speech date, we use returns UP TO yesterday (shift(1) in training)
            features[f"{asset}_t-3"]  = log_ret.iloc[-4:-1].mean()   # 3 days before yesterday
            features[f"{asset}_t-7"]  = log_ret.iloc[-8:-1].mean()   # 7 days before yesterday
            features[f"{asset}_t-30"] = log_ret.iloc[-31:-1].mean()  # 30 days before yesterday

        except Exception as e:
            raise RuntimeError(f"Failed to fetch {ticker}: {e}")

    return features


def fetch_macro_features(date: datetime, fred_api_key: str) -> dict:
    """
    Fetch macro indicators from FRED API.
    Uses most recent available data before speech date,
    consistent with the shift(30) reporting lag in training.
    """
    features = {}

    for name, series_id in FRED_SERIES.items():
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&api_key={fred_api_key}"
            f"&file_type=json"
            f"&observation_start={(date - timedelta(days=90)).strftime('%Y-%m-%d')}"
            f"&observation_end={date.strftime('%Y-%m-%d')}"
            f"&sort_order=desc"
            f"&limit=1"
        )

        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations or observations[0]["value"] == ".":
            raise RuntimeError(f"No FRED data available for {series_id}")

        features[name] = float(observations[0]["value"])

    return features


def build_feature_vector(
    embedding: np.ndarray,
    date: datetime,
    fred_api_key: str,
    feature_columns: list  # X_train.columns — must match exactly!
) -> pd.DataFrame:
    """
    Assemble the full feature vector in the same column order as X_train.
    """
    price_features = fetch_price_features(date)
    macro_features = fetch_macro_features(date, fred_api_key)

    # Combine all features
    all_features = {**price_features, **macro_features}

    # Add embeddings
    for i, val in enumerate(embedding):
        all_features[f"emb_{i}"] = val

    # Build DataFrame with exact column order from training
    feature_df = pd.DataFrame([all_features])

    # Reindex to match training columns exactly (fills missing with NaN)
    feature_df = feature_df.reindex(columns=feature_columns)

    missing = feature_df.columns[feature_df.isnull().any()].tolist()
    if missing:
        raise RuntimeError(f"Missing features after assembly: {missing}")

    return feature_df