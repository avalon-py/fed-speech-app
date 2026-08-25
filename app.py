import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date
import os

from predict import load_models, load_finbert, predict

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fed Speech Analyzer",
    page_icon="🏦",
    layout="wide"
)

# ── Load everything once (cached) ──────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models("models")

@st.cache_resource
def get_finbert():
    # Returns (tokenizer, onnxruntime.InferenceSession) - both loaded
    # from models/finbert-onnx/ on disk, no network call.
    return load_finbert("models/finbert-onnx")

@st.cache_data
def get_feature_columns():
    return joblib.load("models/feature_columns.pkl")

@st.cache_data
def get_date_bounds():
    """
    Derives the selectable date range from what the CSVs actually cover,
    instead of relying on st.date_input's implicit default (10 years
    before `value` when min_value is omitted - that's what was producing
    the 2016/08/25 floor, not a real data-coverage check).

    earliest = the later of the two CSVs' start dates, since a date needs
    BOTH price and macro history available to build a full feature vector.
    latest   = the earlier of the two CSVs' end dates, same reasoning,
               capped at today since the app doesn't take future dates.
    """
    price = pd.read_csv(os.path.join("dataset", "price_action.csv"))
    macro = pd.read_csv(os.path.join("dataset", "macro_indicators.csv"))

    price_dates = pd.to_datetime(price["date"])
    macro_dates = pd.to_datetime(macro["date"])

    earliest = max(price_dates.min(), macro_dates.min())
    latest = min(price_dates.max(), macro_dates.max())

    return earliest.date(), min(latest.date(), date.today())

ml_models       = get_models()
tokenizer, finbert_session = get_finbert()
feature_columns = get_feature_columns()
min_date, max_date = get_date_bounds()

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🏦 Fed Speech Market Impact Analyzer")
st.caption("Predicts SPX, GOLD, VIX, and TNX direction following Federal Reserve speeches")

st.divider()

col1, col2 = st.columns([3, 1])

with col1:
    speech_text = st.text_area(
        "Fed Speech Text",
        height=300,
        placeholder="Paste the Federal Reserve speech here..."
    )

with col2:
    st.markdown("**Speech Date**")
    use_today = st.checkbox("Use today's date", value=True)

    speech_date = date.today()
    if not use_today:
        speech_date = st.date_input(
            "Select date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    st.caption(f"Using date: **{speech_date}**")
    st.caption(f"Data available from **{min_date}** to **{max_date}**")
    st.divider()
    st.markdown("**How it works**")
    st.caption("1. Speech → FinBERT embedding (local FP16 ONNX, CPU)")
    st.caption("2. Price history read from dataset/price_action.csv (synced daily via cron)")
    st.caption("3. Macro data read from dataset/macro_indicators.csv (synced daily via cron)")
    st.caption("4. Run through trained models")

st.divider()

run = st.button("🔍 Analyze Speech", type="primary", use_container_width=True)

if run:
    if not speech_text.strip():
        st.error("Please enter a speech.")
    elif not (min_date <= speech_date <= max_date):
        st.error(
            f"Selected date {speech_date} is outside available data coverage "
            f"({min_date} to {max_date})."
        )
    else:
        with st.spinner("Embedding speech and running predictions..."):
            try:
                results = predict(
                    text=speech_text,
                    date=datetime.combine(speech_date, datetime.min.time()),
                    tokenizer=tokenizer,
                    session=finbert_session,
                    ml_models=ml_models,
                    feature_columns=feature_columns,
                )

                st.success("Prediction complete!")
                st.divider()

                assets = {
                    "📈 S&P 500 (SPX)": ["SPX_t+3", "SPX_t+7", "SPX_t+30"],
                    "🥇 Gold (GOLD)":    ["GOLD_t+3", "GOLD_t+7", "GOLD_t+30"],
                    "😰 VIX":            ["VIX_t+3",  "VIX_t+7",  "VIX_t+30"],
                    "💵 10Y Treasury":   ["TNX_t+3",  "TNX_t+7",  "TNX_t+30"],
                }

                cols = st.columns(4)

                for i, (asset_name, horizons) in enumerate(assets.items()):
                    with cols[i]:
                        st.markdown(f"**{asset_name}**")
                        for horizon in horizons:
                            pred  = results[horizon]
                            label = horizon.split("_")[1]
                            pct   = pred * 100

                            if pred > 0:
                                st.metric(label, f"+{pct:.2f}%", delta="Bullish ↑")
                            else:
                                st.metric(label, f"{pct:.2f}%", delta="Bearish ↓", delta_color="inverse")

                with st.expander("Raw prediction values"):
                    results_df = pd.DataFrame([results]).T
                    results_df.columns = ["Predicted Return"]
                    results_df["Direction"] = results_df["Predicted Return"].apply(
                        lambda x: "🟢 Bullish" if x > 0 else "🔴 Bearish"
                    )
                    st.dataframe(results_df)

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
