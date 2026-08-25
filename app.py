import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date
import os

from predict import load_models, load_finbert, predict

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fed Speech Market Reader",
    page_icon="📊",
    layout="wide",
)

# ── Design tokens + chrome ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink:        #E7E4DA;
        --muted:      #8B92A0;
        --surface:    #141920;
        --line:       rgba(255,255,255,0.08);
        --gold:       #C9A227;
        --bull:       #4E9A6B;
        --bear:       #B84C3E;
    }

    .stApp, .stApp p, .stApp label, .stApp span { font-family: 'Inter', sans-serif; }
    [data-testid="stHeader"] { background: transparent; }

    /* ── Masthead ── */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.5rem;
        line-height: 1.08;
        color: var(--ink);
        margin: 0 0 0.6rem 0;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 0.95rem;
        max-width: 620px;
        line-height: 1.5;
        margin-bottom: 0.25rem;
    }

    /* ── Dispatch panel (right column) ── */
    .dispatch-box {
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 2px solid var(--gold);
        padding: 1rem 1.25rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .dispatch-label {
        color: var(--gold);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-size: 0.65rem;
        margin-bottom: 0.5rem;
    }
    .dispatch-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: var(--muted);
        padding: 0.3rem 0;
        border-bottom: 1px solid var(--line);
    }
    .dispatch-row:last-child { border-bottom: none; }
    .dispatch-row b { color: var(--ink); font-weight: 500; }

    .steps {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--muted);
        margin-top: 1.1rem;
    }
    .steps .step { display: flex; gap: 0.6rem; padding: 0.25rem 0; }
    .steps .step-n { color: var(--gold); }

    /* ── Ticker strip (signature element) ── */
    .ticker-strip {
        display: flex;
        gap: 1px;
        background: var(--line);
        border: 1px solid var(--line);
        margin: 1.6rem 0 0.5rem 0;
        overflow-x: auto;
    }
    .ticker-item {
        flex: 1;
        min-width: 150px;
        background: var(--surface);
        padding: 0.9rem 1.1rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    .ticker-symbol {
        display: block;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }
    .ticker-value { display: block; font-size: 1.35rem; font-weight: 600; }
    .ticker-horizon { font-size: 0.65rem; color: var(--muted); }
    .bullish { color: var(--bull); }
    .bearish { color: var(--bear); }

    /* ── Blotter table ── */
    table.blotter {
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        margin-top: 0.75rem;
    }
    table.blotter th {
        text-align: left;
        font-size: 0.66rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 500;
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid var(--line);
    }
    table.blotter td {
        padding: 0.6rem 0.75rem;
        border-bottom: 1px solid var(--line);
    }
    table.blotter td.asset {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
        font-weight: 500;
    }

    .disclaimer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 1.5rem;
        border-top: 1px solid var(--line);
        padding-top: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load everything once (cached) ──────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models("models")


@st.cache_resource
def get_finbert():
    return load_finbert("models/finbert-onnx")


@st.cache_data
def get_feature_columns():
    return joblib.load("models/feature_columns.pkl")


@st.cache_data
def get_date_bounds():
    """
    Derives the selectable date range from what the CSVs actually cover.
    earliest = the later of the two CSVs' start dates (need both price and
    macro history to build a full feature vector); latest = the earlier of
    the two end dates, capped at today.
    """
    price = pd.read_csv(os.path.join("dataset", "price_action.csv"))
    macro = pd.read_csv(os.path.join("dataset", "macro_indicators.csv"))

    price_dates = pd.to_datetime(price["date"])
    macro_dates = pd.to_datetime(macro["date"])

    earliest = max(price_dates.min(), macro_dates.min())
    latest = min(price_dates.max(), macro_dates.max())

    return earliest.date(), min(latest.date(), date.today())


ml_models = get_models()
tokenizer, finbert_session = get_finbert()
feature_columns = get_feature_columns()
min_date, max_date = get_date_bounds()

ASSETS = {
    "spx":  {"label": "S&P 500",     "symbol": "SPX"},
    "gold": {"label": "Gold",        "symbol": "GOLD"},
    "vix":  {"label": "VIX",         "symbol": "VIX"},
    "tnx":  {"label": "10Y Treasury", "symbol": "TNX"},
}
HORIZONS = ["t+3", "t+7", "t+30"]


def direction_class(v: float) -> str:
    return "bullish" if v > 0 else "bearish"


def arrow(v: float) -> str:
    return "▲" if v > 0 else "▼"


def fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


# ── Masthead ─────────────────────────────────────────────────────────────
st.markdown('<div class="eyebrow">Macro Research Desk · Model Output, Not Advice</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Fed Speech Market Reader</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Paste a Federal Reserve speech and the model reads it against '
    'price and macro history to project directional moves in equities, gold, volatility, '
    'and rates over the next 3, 7, and 30 days.</p>',
    unsafe_allow_html=True,
)

st.write("")

col1, col2 = st.columns([3, 1])

with col1:
    speech_text = st.text_area(
        "Speech transcript",
        height=300,
        placeholder="Paste the Federal Reserve speech text here…",
        label_visibility="visible",
    )

with col2:
    use_today = st.checkbox("Use today's date", value=True)

    speech_date = date.today()
    if not use_today:
        speech_date = st.date_input(
            "Speech date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    st.markdown(
        f"""
        <div class="dispatch-box">
            <div class="dispatch-label">Dispatch details</div>
            <div class="dispatch-row"><span>Speech date</span><b>{speech_date}</b></div>
            <div class="dispatch-row"><span>Data from</span><b>{min_date}</b></div>
            <div class="dispatch-row"><span>Data to</span><b>{max_date}</b></div>
        </div>
        <div class="steps">
            <div class="step"><span class="step-n">01</span><span>Speech → FinBERT embedding, local FP16 ONNX</span></div>
            <div class="step"><span class="step-n">02</span><span>Price history read from dataset/price_action.csv</span></div>
            <div class="step"><span class="step-n">03</span><span>Macro data read from dataset/macro_indicators.csv</span></div>
            <div class="step"><span class="step-n">04</span><span>Feature vector run through trained models</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
run = st.button("Run model", type="primary", use_container_width=True)

if run:
    if not speech_text.strip():
        st.error("Enter a speech before running the model.")
    elif not (min_date <= speech_date <= max_date):
        st.error(
            f"{speech_date} is outside available data coverage "
            f"({min_date} to {max_date})."
        )
    else:
        with st.spinner("Embedding speech and running predictions…"):
            try:
                results = predict(
                    text=speech_text,
                    date=datetime.combine(speech_date, datetime.min.time()),
                    tokenizer=tokenizer,
                    session=finbert_session,
                    ml_models=ml_models,
                    feature_columns=feature_columns,
                )

                # ── Ticker strip: headline (t+3) read per asset ──
                ticker_items = ""
                for key, meta in ASSETS.items():
                    v = results[f"{meta['symbol']}_t+3"]
                    ticker_items += f"""
                    <div class="ticker-item">
                        <span class="ticker-symbol">{meta['symbol']}</span>
                        <span class="ticker-value {direction_class(v)}">{arrow(v)} {fmt_pct(v)}</span>
                        <span class="ticker-horizon">t+3</span>
                    </div>
                    """
                st.markdown(f'<div class="ticker-strip">{ticker_items}</div>', unsafe_allow_html=True)

                # ── Blotter: full asset x horizon grid ──
                rows = ""
                for key, meta in ASSETS.items():
                    cells = ""
                    for h in HORIZONS:
                        v = results[f"{meta['symbol']}_{h}"]
                        cells += f'<td class="{direction_class(v)}">{arrow(v)} {fmt_pct(v)}</td>'
                    rows += f'<tr><td class="asset">{meta["label"]} · {meta["symbol"]}</td>{cells}</tr>'

                st.markdown(
                    f"""
                    <table class="blotter">
                        <thead><tr><th>Asset</th><th>t+3</th><th>t+7</th><th>t+30</th></tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="disclaimer">Predicted returns from a model trained on historical '
                    'speech and market data. Not investment advice.</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Raw prediction values"):
                    raw_rows = ""
                    for k, v in results.items():
                        raw_rows += (
                            f'<tr><td class="asset">{k}</td>'
                            f'<td class="{direction_class(v)}">{arrow(v)} {v:.6f}</td></tr>'
                        )
                    st.markdown(
                        f"""
                        <table class="blotter">
                            <thead><tr><th>Target</th><th>Predicted return</th></tr></thead>
                            <tbody>{raw_rows}</tbody>
                        </table>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
