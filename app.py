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
    return load_finbert()

@st.cache_data
def get_feature_columns():
    return joblib.load("models/feature_columns.pkl")

ml_models   = get_models()
tokenizer, bert_model = get_finbert()
feature_columns = get_feature_columns()
fred_api_key = st.secrets["FRED_API_KEY"]

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
            value=date.today(),
            max_value=date.today()
        )

    st.caption(f"Using date: **{speech_date}**")
    st.divider()
    st.markdown("**How it works**")
    st.caption("1. Speech → FinBERT embedding")
    st.caption("2. Fetch price history from Yahoo Finance")
    st.caption("3. Fetch macro data from FRED")
    st.caption("4. Run through trained models")

st.divider()

run = st.button("🔍 Analyze Speech", type="primary", use_container_width=True)

if run:
    if not speech_text.strip():
        st.error("Please enter a speech.")
    else:
        with st.spinner("Fetching market data and running predictions..."):
            try:
                results = predict(
                    text=speech_text,
                    date=datetime.combine(speech_date, datetime.min.time()),
                    tokenizer=tokenizer,
                    bert_model=bert_model,
                    ml_models=ml_models,
                    feature_columns=feature_columns,
                    fred_api_key=fred_api_key
                )

                st.success("Prediction complete!")
                st.divider()

                # ── Results display ────────────────────────────────────────
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
                        for col in horizons:
                            pred  = results[col]
                            label = col.split("_")[1]   # t+3, t+7, t+30
                            pct   = pred * 100

                            if pred > 0:
                                st.metric(label, f"+{pct:.2f}%", delta="Bullish ↑")
                            else:
                                st.metric(label, f"{pct:.2f}%", delta="Bearish ↓", delta_color="inverse")

                # ── Raw predictions ────────────────────────────────────────
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