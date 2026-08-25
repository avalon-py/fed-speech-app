import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from predict import load_models, load_finbert, predict

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech2Market Live Demo",
    page_icon="📊",
    layout="wide",
)

# ── Color tokens (kept in sync with the CSS custom properties below,
#    since Plotly can't read CSS vars) ──────────────────────────────────────
INK      = "#E7E4DA"
MUTED    = "#8B92A0"
SURFACE  = "#141920"
SURFACE2 = "#10141A"
LINE     = "rgba(255,255,255,0.08)"
GOLD     = "#C9A227"
BULL     = "#4E9A6B"
BEAR     = "#B84C3E"
MONO     = "IBM Plex Mono, monospace"


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── Design tokens + chrome ──────────────────────────────────────────────────
# NOTE ON THE HTML-RENDERING BUG: every fragment built below is assembled as
# a SINGLE-LINE string with no leading whitespace before being handed to
# st.markdown(..., unsafe_allow_html=True). Markdown treats any line that
# starts with 4+ spaces as a code block — that's what was causing raw HTML
# to appear as literal text on the page. Keep all injected HTML on one line.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink:        #E7E4DA;
        --muted:      #8B92A0;
        --surface:    #141920;
        --surface-2:  #10141A;
        --line:       rgba(255,255,255,0.08);
        --gold:       #C9A227;
        --bull:       #4E9A6B;
        --bear:       #B84C3E;
    }

    .stApp, .stApp p, .stApp label, .stApp span { font-family: 'Inter', sans-serif; }
    [data-testid="stHeader"] { background: transparent; height: 2.2rem; }
    .block-container { padding-top: 0.6rem !important; padding-bottom: 0.6rem !important; }

    [data-testid="stAppViewContainer"] {
        background-image: radial-gradient(rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 22px 22px;
    }

    /* ── Marquee ticker tape ── */
    .tape-wrap {
        overflow: hidden;
        border-bottom: 1px solid var(--line);
        background: var(--surface-2);
        white-space: nowrap;
        margin: -0.6rem -1rem 0.6rem -1rem;
        padding: 0;
    }
    .tape-track {
        display: inline-block;
        padding-left: 100%;
        animation: tape-scroll 42s linear infinite;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
    }
    .tape-track span { color: var(--gold); }
    @keyframes tape-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }
    @media (prefers-reduced-motion: reduce) {
        .tape-track { animation: none; padding-left: 1rem; }
    }

    /* ── Masthead ── */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .eyebrow .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--bull);
        box-shadow: 0 0 0 3px rgba(78,154,107,0.18);
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.55rem;
        line-height: 1.05;
        color: var(--ink);
        margin: 0 0 0.15rem 0;
        letter-spacing: -0.01em;
        display: inline-block;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.4;
        margin: 0 0 0.4rem 0;
    }
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: var(--gold);
        border-top: 1px solid var(--line);
        padding-top: 0.35rem;
        margin: 0.6rem 0 0.4rem 0;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
    }
    .section-label .coverage { color: var(--muted); text-transform: none; letter-spacing: 0; }

    /* ── Ticker strip (results, headline read) ── */
    .ticker-strip {
        display: flex;
        gap: 1px;
        background: var(--line);
        border: 1px solid var(--line);
        margin: 0.6rem 0 0.4rem 0;
        overflow-x: auto;
    }
    .ticker-item {
        flex: 1;
        min-width: 150px;
        background: var(--surface);
        padding: 0.6rem 0.9rem;
        font-family: 'IBM Plex Mono', monospace;
        transition: background 0.15s ease, transform 0.15s ease;
    }
    .ticker-item:hover { background: var(--surface-2); transform: translateY(-1px); }
    .ticker-symbol {
        display: block;
        font-size: 0.64rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }
    .ticker-value { display: block; font-size: 1.15rem; font-weight: 600; }
    .ticker-horizon { font-size: 0.62rem; color: var(--muted); }
    .bullish { color: var(--bull); }
    .bearish { color: var(--bear); }

    /* ── Context strip (small stat readouts above the charts) ──
       Revamped to a single horizontal row per card: ticker, price,
       and delta all sit inline (not stacked) to save vertical space. */
    .ctx-strip {
        display: flex;
        gap: 1px;
        background: var(--line);
        border: 1px solid var(--line);
        border-bottom: none;
    }
    .ctx-item {
        flex: 1;
        min-width: 170px;
        background: var(--surface-2);
        padding: 0.5rem 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        white-space: nowrap;
    }
    .ctx-symbol { font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
    .ctx-value { font-size: 0.92rem; font-weight: 600; color: var(--ink); }
    .ctx-delta { font-size: 0.7rem; }

    div[data-testid="stPlotlyChart"] {
        border: 1px solid var(--line);
        border-top: none;
        background: var(--surface);
        margin-bottom: 0.5rem;
    }

    /* ── Blotter table ── */
    table.blotter {
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    table.blotter th {
        text-align: left;
        font-size: 0.64rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 500;
        padding: 0.4rem 0.7rem;
        border-bottom: 1px solid var(--line);
    }
    table.blotter td {
        padding: 0.45rem 0.7rem;
        border-bottom: 1px solid var(--line);
    }
    table.blotter td.asset {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
        font-weight: 500;
    }

    /* ── Form controls ── */
    .stTextArea textarea {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    .stTextArea textarea:focus { border-color: var(--gold) !important; box-shadow: none !important; }
    .stCheckbox label, .stDateInput label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .stDateInput input {
        background: var(--surface) !important;
        border: 1px solid var(--line) !important;
        color: var(--ink) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    div.stButton > button {
        background: transparent;
        border: 1px solid var(--gold);
        color: var(--gold);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border-radius: 0;
        padding: 0.45rem 1rem;
        transition: background 0.15s ease, color 0.15s ease;
    }
    div.stButton > button:hover {
        background: var(--gold);
        color: #0B0E13;
        border-color: var(--gold);
    }
    div.stButton > button:focus:not(:active) {
        color: var(--gold);
    }

    .disclaimer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--muted);
        margin-top: 0.6rem;
        border-top: 1px solid var(--line);
        padding-top: 0.5rem;
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
def load_price_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join("dataset", "price_action.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


@st.cache_data
def load_macro_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join("dataset", "macro_indicators.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


@st.cache_data
def get_date_bounds():
    price_dates = load_price_data()["date"]
    macro_dates = load_macro_data()["date"]

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

MACRO_SERIES = {
    "unemployment": {"label": "Unemployment", "unit": "%"},
    "interest_rate": {"label": "Fed Funds Rate", "unit": "%"},
    "growth_rate": {"label": "GDP Growth", "unit": "%"},
}

PRICE_WINDOW_MONTHS = 3
MACRO_WINDOW_MONTHS = 12


def direction_class(v: float) -> str:
    return "bullish" if v > 0 else "bearish"


def arrow(v: float) -> str:
    return "▲" if v > 0 else "▼"


def fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


# ── Market context chart builders ───────────────────────────────────────────
def windowed(df: pd.DataFrame, end_date: date, months: int) -> pd.DataFrame:
    end_ts = pd.Timestamp(end_date)
    start_ts = end_ts - relativedelta(months=months)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]


def context_strip_html(items: list) -> str:
    """items: list of (symbol, value_str, delta, delta_str).
    Built as ONE line per div — no leading whitespace — so Streamlit's
    Markdown parser renders it as HTML instead of a code block.
    Layout: ticker, value, and delta all sit inline in one row
    (see .ctx-item flex styling) rather than stacked, to save
    vertical space."""
    cells = "".join(
        f'<div class="ctx-item"><span class="ctx-symbol">{symbol}</span>'
        f'<span class="ctx-value">{value_str}</span>'
        f'<span class="ctx-delta {"bullish" if delta >= 0 else "bearish"}">{arrow(delta)} {delta_str}</span></div>'
        for symbol, value_str, delta, delta_str in items
    )
    return f'<div class="ctx-strip">{cells}</div>'


def _date_axis(fig, row, col, nticks, tickformat, rangebreaks=None):
    fig.update_xaxes(
        showgrid=False,
        showticklabels=True,
        tickfont=dict(size=8.5, color=MUTED, family=MONO),
        tickformat=tickformat,
        nticks=nticks,
        ticks="outside",
        tickcolor=LINE,
        linecolor=LINE,
        rangebreaks=rangebreaks or [],
        row=row, col=col,
    )


def price_context_figure(df_window: pd.DataFrame) -> go.Figure:
    symbols = [meta["symbol"] for meta in ASSETS.values()]
    fig = make_subplots(rows=1, cols=len(symbols), horizontal_spacing=0.04)

    for i, symbol in enumerate(symbols, start=1):
        series = df_window[symbol].astype(float)
        up = series.iloc[-1] >= series.iloc[0] if len(series) > 1 else True
        color = BULL if up else BEAR
        fig.add_trace(
            go.Scatter(
                x=df_window["date"], y=series,
                mode="lines",
                line=dict(color=color, width=1.6),
                fill="tozeroy",
                fillcolor=_rgba(color, 0.10),
                hovertemplate=f"{symbol} · %{{x|%b %d, %Y}}<br>%{{y:,.2f}}<extra></extra>",
            ),
            row=1, col=i,
        )
        pad = (series.max() - series.min()) * 0.18 or series.mean() * 0.02 or 1
        fig.update_yaxes(
            range=[series.min() - pad, series.max() + pad],
            showgrid=False, zeroline=False, showticklabels=False,
            row=1, col=i,
        )
        # This is daily trading data, so Sat/Sun have no rows in the
        # dataframe. Without a rangebreak, Plotly still reserves those
        # two empty calendar days on the (continuous) date axis, which
        # throws off the spacing of every tick after the first weekend
        # gap — a tick label like "Jul 01" ends up positioned over the
        # nearest *trading* day rather than the actual date it names.
        # Collapsing weekends out of the axis fixes the alignment.
        _date_axis(
            fig, 1, i, nticks=3, tickformat="%b %d",
            rangebreaks=[dict(bounds=["sat", "mon"])],
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        showlegend=False,
        height=125,
        margin=dict(l=0, r=0, t=6, b=4),
        hoverlabel=dict(bgcolor=SURFACE2, font_family=MONO, font_color=INK, bordercolor=LINE),
    )
    return fig


def macro_context_figure(df_window: pd.DataFrame) -> go.Figure:
    keys = list(MACRO_SERIES.keys())
    fig = make_subplots(rows=1, cols=len(keys), horizontal_spacing=0.04)

    for i, key in enumerate(keys, start=1):
        series = df_window[key].astype(float)
        fig.add_trace(
            go.Scatter(
                x=df_window["date"], y=series,
                mode="lines",
                line=dict(color=GOLD, width=1.6, shape="hv"),
                fill="tozeroy",
                fillcolor=_rgba(GOLD, 0.08),
                hovertemplate=f"{MACRO_SERIES[key]['label']} · %{{x|%b %Y}}<br>%{{y:.2f}}%<extra></extra>",
            ),
            row=1, col=i,
        )
        pad = (series.max() - series.min()) * 0.25 or 0.1
        fig.update_yaxes(
            range=[series.min() - pad, series.max() + pad],
            showgrid=False, zeroline=False, showticklabels=False,
            row=1, col=i,
        )
        # Macro indicators are reported monthly with no weekly gaps,
        # so no rangebreak is needed here — the tick misalignment was
        # specific to the daily price series above.
        _date_axis(fig, 1, i, nticks=4, tickformat="%b '%y")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        showlegend=False,
        height=115,
        margin=dict(l=0, r=0, t=6, b=4),
        hoverlabel=dict(bgcolor=SURFACE2, font_family=MONO, font_color=INK, bordercolor=LINE),
    )
    return fig


def render_market_context(speech_date: date):
    price_df = load_price_data()
    macro_df = load_macro_data()

    price_win = windowed(price_df, speech_date, PRICE_WINDOW_MONTHS)
    macro_win = windowed(macro_df, speech_date, MACRO_WINDOW_MONTHS)

    price_start = (pd.Timestamp(speech_date) - relativedelta(months=PRICE_WINDOW_MONTHS)).date()
    macro_start = (pd.Timestamp(speech_date) - relativedelta(months=MACRO_WINDOW_MONTHS)).date()

    # ── Price / vol assets: trailing 3 months ──
    st.markdown(
        f'<div class="section-label"><span>Price &amp; Vol · Trailing 3 Months</span>'
        f'<span class="coverage">{price_start} → {speech_date}</span></div>',
        unsafe_allow_html=True,
    )

    if price_win.empty:
        st.info("No price data in the trailing 3-month window for this date.")
    else:
        price_items = []
        for meta in ASSETS.values():
            series = price_win[meta["symbol"]].astype(float)
            first, last = series.iloc[0], series.iloc[-1]
            delta = (last - first) / first if first else 0.0
            price_items.append((meta["symbol"], f"{last:,.2f}", delta, fmt_pct(delta)))

        st.markdown(context_strip_html(price_items), unsafe_allow_html=True)
        st.plotly_chart(price_context_figure(price_win), use_container_width=True, config={"displayModeBar": False})

    # ── Macro indicators: trailing 1 year ──
    st.markdown(
        f'<div class="section-label"><span>Macro Indicators · Trailing 1 Year</span>'
        f'<span class="coverage">{macro_start} → {speech_date}</span></div>',
        unsafe_allow_html=True,
    )

    if macro_win.empty:
        st.info("No macro data in the trailing 1-year window for this date.")
    else:
        macro_items = []
        for key, meta in MACRO_SERIES.items():
            series = macro_win[key].astype(float)
            first, last = series.iloc[0], series.iloc[-1]
            delta_pp = last - first
            macro_items.append((meta["label"], f"{last:.2f}{meta['unit']}", delta_pp, f"{delta_pp:+.2f}pp"))

        st.markdown(context_strip_html(macro_items), unsafe_allow_html=True)
        st.plotly_chart(macro_context_figure(macro_win), use_container_width=True, config={"displayModeBar": False})


# ── Ticker tape ──────────────────────────────────────────────────────────
tape_text = (
    "SPEECH2MARKET LIVE DEMO &nbsp; <span>·</span> &nbsp; MACRO RESEARCH DESK &nbsp; "
    "<span>·</span> &nbsp; MODEL OUTPUT, NOT ADVICE &nbsp; <span>·</span> &nbsp; "
    "S&amp;P 500 &nbsp; GOLD &nbsp; VIX &nbsp; 10Y TREASURY &nbsp; <span>·</span> &nbsp; "
)
st.markdown(
    f'<div class="tape-wrap"><div class="tape-track">{tape_text * 3}</div></div>',
    unsafe_allow_html=True,
)

# ── Masthead ─────────────────────────────────────────────────────────────
st.markdown('<div class="eyebrow"><span class="dot"></span>Macro Research Desk</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Speech2Market Live Demo</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Paste a Fed speech and project directional moves in equities, gold, '
    'volatility, and rates over the next 3, 7, and 30 days.</p>',
    unsafe_allow_html=True,
)

# ── Date controls (compact, drives the market context below) ───────────────
dc1, dc2, dc3 = st.columns([1, 1.4, 3])
with dc1:
    use_today = st.checkbox("Use today's date", value=True)
with dc2:
    speech_date = date.today()
    if not use_today:
        speech_date = st.date_input(
            "Speech date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )

# ── Market context (charts now sit above the input field) ──────────────────
render_market_context(speech_date)

# ── Speech input ─────────────────────────────────────────────────────────
# Height bumped up (100 -> 180) now that the more compact, single-row
# context cards above free up vertical space.
speech_text = st.text_area(
    "Speech transcript",
    height=180,
    placeholder="Paste the Federal Reserve speech text here…",
    label_visibility="collapsed",
)

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

                st.markdown(
                    '<div class="section-label"><span>Prediction · Headline Read (t+3)</span></div>',
                    unsafe_allow_html=True,
                )

                # ── Ticker strip: headline (t+3) read per asset ──
                ticker_items = "".join(
                    f'<div class="ticker-item"><span class="ticker-symbol">{meta["symbol"]}</span>'
                    f'<span class="ticker-value {direction_class(results[f"{meta["symbol"]}_t+3"])}">'
                    f'{arrow(results[f"{meta["symbol"]}_t+3"])} {fmt_pct(results[f"{meta["symbol"]}_t+3"])}</span>'
                    f'<span class="ticker-horizon">t+3</span></div>'
                    for meta in ASSETS.values()
                )
                st.markdown(f'<div class="ticker-strip">{ticker_items}</div>', unsafe_allow_html=True)

                # ── Blotter: full asset x horizon grid ──
                rows = "".join(
                    f'<tr><td class="asset">{meta["label"]} · {meta["symbol"]}</td>'
                    + "".join(
                        f'<td class="{direction_class(results[f"{meta["symbol"]}_{h}"])}">'
                        f'{arrow(results[f"{meta["symbol"]}_{h}"])} {fmt_pct(results[f"{meta["symbol"]}_{h}"])}</td>'
                        for h in HORIZONS
                    )
                    + "</tr>"
                    for meta in ASSETS.values()
                )

                st.markdown(
                    f'<table class="blotter"><thead><tr><th>Asset</th><th>t+3</th><th>t+7</th><th>t+30</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="disclaimer">Predicted returns from a model trained on historical '
                    'speech and market data. Not investment advice.</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Raw prediction values"):
                    raw_rows = "".join(
                        f'<tr><td class="asset">{k}</td>'
                        f'<td class="{direction_class(v)}">{arrow(v)} {v:.6f}</td></tr>'
                        for k, v in results.items()
                    )
                    st.markdown(
                        f'<table class="blotter"><thead><tr><th>Target</th><th>Predicted return</th></tr></thead>'
                        f'<tbody>{raw_rows}</tbody></table>',
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
