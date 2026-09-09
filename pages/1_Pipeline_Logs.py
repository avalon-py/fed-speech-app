import json
from datetime import timedelta

import pandas as pd
import streamlit as st

from db import fetch_all_rows

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech2Market · Pipeline Logs",
    page_icon="🗒️",
    layout="wide",
)

# ── Color tokens (same palette as app.py) ───────────────────────────────────
INK      = "#E7E4DA"
MUTED    = "#8B92A0"
SURFACE  = "#141920"
SURFACE2 = "#10141A"
LINE     = "rgba(255,255,255,0.08)"
GOLD     = "#C9A227"
BULL     = "#4E9A6B"
BEAR     = "#B84C3E"
MONO     = "IBM Plex Mono, monospace"

STATUS_STYLE = {
    "success":     {"color": BULL, "dot": "●", "label": "SUCCESS"},
    "completed":   {"color": BULL, "dot": "●", "label": "SUCCESS"},
    "failed":      {"color": BEAR, "dot": "●", "label": "FAILED"},
    "error":       {"color": BEAR, "dot": "●", "label": "FAILED"},
    "running":     {"color": GOLD, "dot": "◐", "label": "RUNNING"},
    "in_progress": {"color": GOLD, "dot": "◐", "label": "RUNNING"},
    "started":     {"color": GOLD, "dot": "◐", "label": "RUNNING"},
}
DEFAULT_STATUS_STYLE = {"color": MUTED, "dot": "○", "label": "UNKNOWN"}

# ── Chrome ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink: #E7E4DA; --muted: #8B92A0; --surface: #141920; --surface-2: #10141A;
        --line: rgba(255,255,255,0.08); --gold: #C9A227; --bull: #4E9A6B; --bear: #B84C3E;
    }

    .stApp, .stApp p, .stApp label, .stApp span { font-family: 'Inter', sans-serif; }
    [data-testid="stHeader"] { background: transparent; height: 2.2rem; }
    .block-container { padding-top: 0.3rem !important; padding-bottom: 0.6rem !important; }

    [data-testid="stAppViewContainer"] {
        background-image: radial-gradient(rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 22px 22px;
    }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; letter-spacing: 0.16em;
        text-transform: uppercase; color: var(--gold); margin-bottom: 0.25rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--bull); box-shadow: 0 0 0 3px rgba(78,154,107,0.18); }
    .hero-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.55rem; line-height: 1.05; color: var(--ink); margin: 0 0 0.15rem 0; letter-spacing: -0.01em; }
    .hero-sub { color: var(--muted); font-size: 0.78rem; line-height: 1.4; margin: 0 0 0.4rem 0; }

    .section-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.13em;
        text-transform: uppercase; color: var(--gold); border-top: 1px solid var(--line);
        padding-top: 0.35rem; margin: 0.6rem 0 0.4rem 0; display: flex; justify-content: space-between; gap: 1rem;
    }
    .section-label .coverage { color: var(--muted); text-transform: none; letter-spacing: 0; }

    .kpi-strip { display: flex; gap: 1px; background: var(--line); border: 1px solid var(--line); margin: 0.4rem 0 0.6rem 0; }
    .kpi-item { flex: 1; min-width: 140px; background: var(--surface); padding: 0.55rem 0.9rem; font-family: 'IBM Plex Mono', monospace; }
    .kpi-label { display: block; font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.25rem; }
    .kpi-value { display: block; font-size: 1.15rem; font-weight: 600; color: var(--ink); }

    .run-msg { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: var(--muted); }
    .run-err { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: var(--bear); background: rgba(184,76,62,0.08); border: 1px solid rgba(184,76,62,0.3); padding: 0.4rem 0.6rem; margin-top: 0.3rem; white-space: pre-wrap; word-break: break-word; }

    .badge {
        display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
        letter-spacing: 0.06em; text-transform: uppercase; padding: 0.1rem 0.45rem; border: 1px solid var(--line); margin: 0.1rem 0.25rem 0.1rem 0;
    }
    .badge.pass { color: var(--bull); border-color: var(--bull); }
    .badge.fail { color: var(--bear); border-color: var(--bear); }
    .badge.neutral { color: var(--muted); }

    div[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-top: none !important; background: var(--surface-2); }
    div[data-testid="stExpander"] summary { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.74rem !important; color: var(--muted) !important; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); }

    .disclaimer { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted); margin-top: 0.6rem; border-top: 1px solid var(--line); padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── NaN-safe helpers ─────────────────────────────────────────────────────
# Real Supabase/CSV exports leave blank cells as "", which pandas reads back
# as float('nan'). NaN is *truthy* in Python (`bool(float('nan')) is True`),
# so plain `if row.get("field"):` checks pass straight through for empty
# cells and either print the literal word "nan" or crash on the next
# operation (e.g. NaN.items(), NaN[:10]). Route every optional field through
# has_value()/clean() instead of a bare truthiness check.
def has_value(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    if isinstance(x, str) and x.strip() == "":
        return False
    return True


def clean_str(x) -> str:
    """Return a display-safe string, unwrapping accidental double-JSON-encoding
    (e.g. error_message stored as '"\\'SUPABASE_URL\\' not found..."')."""
    if not has_value(x):
        return ""
    s = str(x)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            s = json.loads(s)
        except (TypeError, json.JSONDecodeError):
            pass
    return s.strip("'\"") if s[:1] in "'\"" and s[-1:] in "'\"" else s


# ── Data loading ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_pipeline_runs() -> pd.DataFrame:
    rows = fetch_all_rows("pipeline_runs", order_col="started_at")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("started_at", "finished_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.sort_values("started_at", ascending=False)


@st.cache_data(ttl=60)
def load_training_details() -> pd.DataFrame:
    rows = fetch_all_rows("training_run_detail", order_col="id")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("train_start", "train_end", "eval_start", "eval_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _parse(value):
    """training_run_detail's jsonb columns come back either already
    parsed (dict/list), as JSON text, or as NaN for an empty cell —
    normalize all three into a dict/list-or-None."""
    if not has_value(value):
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
    return value


def fmt_duration(seconds) -> str:
    if not has_value(seconds):
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def status_meta(status: str) -> dict:
    return STATUS_STYLE.get(str(status).lower(), DEFAULT_STATUS_STYLE)


# ── Metrics / hyperparameter / sanity table builders ────────────────────────
def metrics_table(metrics: dict) -> pd.DataFrame:
    """One row per prediction target (asset+horizon), columns = metric."""
    records = []
    for target, vals in metrics.items():
        asset, horizon = (target.split("_", 1) + [""])[:2]
        row = {"target": target, "asset": asset, "horizon": horizon}
        row.update(vals)
        records.append(row)
    df = pd.DataFrame(records)
    horizon_order = {"t+3": 0, "t+7": 1, "t+30": 2}
    df["_h"] = df["horizon"].map(horizon_order).fillna(99)
    df = df.sort_values(["asset", "_h"]).drop(columns=["_h", "target"]).reset_index(drop=True)
    df = df.rename(columns={"asset": "Target", "horizon": "Horizon", "f1": "F1", "auc": "AUC", "rmse": "RMSE"})
    cols = [c for c in ["Target", "Horizon", "F1", "AUC", "RMSE"] if c in df.columns]
    return df[cols]


def hyperparams_table(hyperparams: dict, target: str) -> pd.DataFrame:
    """Sign-model vs magnitude-model params for one target, side by side."""
    sign = hyperparams.get(target, {}).get("sign", {}) or {}
    magnitude = hyperparams.get(target, {}).get("magnitude", {}) or {}
    keys = sorted(set(sign) | set(magnitude))
    df = pd.DataFrame(
        {"sign model": [sign.get(k, "—") for k in keys], "magnitude model": [magnitude.get(k, "—") for k in keys]},
        index=keys,
    )
    df.index.name = "param"
    return df


def sanity_badges(sanity: dict) -> str:
    badges = []
    for key, val in sanity.items():
        if key == "_inverted_targets":
            continue
        cls = "pass" if val else "fail"
        badges.append(f'<span class="badge {cls}">{"✓" if val else "✗"} {key.replace("_", " ")}</span>')
    inverted = sanity.get("_inverted_targets")
    if inverted:
        badges.append(f'<span class="badge fail">✗ inverted: {", ".join(inverted)}</span>')
    return "".join(badges)


def render_training_detail(td: pd.Series, row_key: str) -> None:
    metrics = _parse(td.get("metrics"))
    hyperparams = _parse(td.get("hyperparams"))
    sanity = _parse(td.get("sanity_details"))

    badge_row = []
    if has_value(td.get("sanity_passed")):
        badge_row.append(f'<span class="badge {"pass" if td["sanity_passed"] else "fail"}">'
                          f'{"✓ Sanity Passed" if td["sanity_passed"] else "✗ Sanity Failed"}</span>')
    if has_value(td.get("promoted")):
        badge_row.append(f'<span class="badge {"pass" if td["promoted"] else "neutral"}">'
                          f'{"✓ Promoted" if td["promoted"] else "Not Promoted"}</span>')
    if badge_row:
        st.markdown("".join(badge_row), unsafe_allow_html=True)

    train_range = eval_range = ""
    if has_value(td.get("train_start")) and has_value(td.get("train_end")):
        train_range = f'Train window: {td["train_start"].date()} → {td["train_end"].date()}'
    if has_value(td.get("eval_start")) and has_value(td.get("eval_end")):
        eval_range = f'Eval window: {td["eval_start"].date()} → {td["eval_end"].date()}'
    if train_range or eval_range:
        st.markdown(
            f'<span style="font-family:{MONO};color:{MUTED};font-size:0.75rem;">{train_range}&nbsp;&nbsp;&nbsp;{eval_range}</span>',
            unsafe_allow_html=True,
        )
    if has_value(td.get("git_commit")):
        commit = clean_str(td["git_commit"])
        st.markdown(f'<span style="font-family:{MONO};color:{MUTED};font-size:0.72rem;">commit `{commit[:10]}`</span>', unsafe_allow_html=True)

    if metrics:
        st.markdown('<div class="section-label"><span>Eval Metrics · by target</span></div>', unsafe_allow_html=True)
        m_df = metrics_table(metrics)
        st.dataframe(
            m_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "F1": st.column_config.ProgressColumn("F1", min_value=0, max_value=1, format="%.3f"),
                "AUC": st.column_config.ProgressColumn("AUC", min_value=0, max_value=1, format="%.3f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.4f"),
            },
        )

    if sanity:
        st.markdown('<div class="section-label"><span>Sanity Checks</span></div>', unsafe_allow_html=True)
        st.markdown(sanity_badges(sanity), unsafe_allow_html=True)

    if hyperparams:
        st.markdown('<div class="section-label"><span>Hyperparameters · sign vs. magnitude model</span></div>', unsafe_allow_html=True)
        targets = sorted(hyperparams.keys())
        picked = st.selectbox("Target", targets, key=f"hp-target-{row_key}", label_visibility="collapsed")
        hp_df = hyperparams_table(hyperparams, picked)
        st.dataframe(hp_df, use_container_width=True)


# ── Masthead ─────────────────────────────────────────────────────────────
st.markdown('<div class="eyebrow"><span class="dot"></span>Macro Research Desk</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Pipeline Logs</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Ingestion, embedding, and training job history — pulled straight from '
    '<code>pipeline_runs</code> and <code>training_run_detail</code>.</p>',
    unsafe_allow_html=True,
)

runs_df = load_pipeline_runs()
details_df = load_training_details()

if runs_df.empty:
    st.info("No pipeline runs logged yet.")
    st.stop()

# ── Filters ──────────────────────────────────────────────────────────────
fc1, fc2, fc3 = st.columns([1.4, 1.4, 1.6])
with fc1:
    job_options = sorted(runs_df["job_name"].dropna().unique().tolist())
    job_filter = st.multiselect("Job", job_options, default=job_options, label_visibility="collapsed", placeholder="Filter by job")
with fc2:
    status_options = sorted(runs_df["status"].dropna().unique().tolist())
    status_filter = st.multiselect("Status", status_options, default=status_options, label_visibility="collapsed", placeholder="Filter by status")
with fc3:
    lookback = st.selectbox(
        "Lookback", ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
        index=1, label_visibility="collapsed",
    )

filtered = runs_df[runs_df["job_name"].isin(job_filter) & runs_df["status"].isin(status_filter)]
if lookback != "All time" and not filtered.empty:
    hours = {"Last 24 hours": 24, "Last 7 days": 24 * 7, "Last 30 days": 24 * 30}[lookback]
    cutoff = pd.Timestamp.now(tz=filtered["started_at"].dt.tz) - timedelta(hours=hours)
    filtered = filtered[filtered["started_at"] >= cutoff]

# ── KPI strip ────────────────────────────────────────────────────────────
total_runs = len(filtered)
status_lower = filtered["status"].str.lower() if total_runs else pd.Series(dtype=str)
success_count = status_lower.isin(["success", "completed"]).sum()
failed_count = status_lower.isin(["failed", "error"]).sum()
running_count = status_lower.isin(["running", "in_progress", "started"]).sum()
finished = total_runs - running_count
success_rate = f"{(success_count / finished * 100):.0f}%" if finished else "—"
avg_duration = fmt_duration(filtered["duration_seconds"].mean()) if "duration_seconds" in filtered and finished else "—"

kpi_html = "".join(
    f'<div class="kpi-item"><span class="kpi-label">{label}</span><span class="kpi-value">{value}</span></div>'
    for label, value in [
        ("Runs in view", total_runs),
        ("Success rate", success_rate),
        ("Failed", failed_count),
        ("Running now", running_count),
        ("Avg duration", avg_duration),
    ]
)
st.markdown(f'<div class="kpi-strip">{kpi_html}</div>', unsafe_allow_html=True)

# ── Run table (compact, scannable) ──────────────────────────────────────
st.markdown(
    f'<div class="section-label"><span>Run History</span><span class="coverage">{lookback.lower()}</span></div>',
    unsafe_allow_html=True,
)

if filtered.empty:
    st.info("No runs match the current filters.")
else:
    table_df = filtered.copy()
    table_df["Status"] = table_df["status"].apply(lambda s: f'{status_meta(s)["dot"]} {status_meta(s)["label"]}')
    table_df["Started"] = table_df["started_at"].dt.strftime("%b %d, %H:%M:%S")
    table_df["Duration"] = table_df["duration_seconds"].apply(fmt_duration)
    table_df["Rows"] = table_df["rows_processed"].fillna(0).astype(int)
    table_df["Message"] = table_df.apply(
        lambda r: clean_str(r.get("error_message")) or clean_str(r.get("message")) or "", axis=1
    )
    display_df = table_df[["Status", "job_name", "Started", "Duration", "Rows", "Message"]].rename(
        columns={"job_name": "Job"}
    )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Message": st.column_config.TextColumn("Message", width="large"),
        },
    )

    # ── Drill-down: pick a run to see its full detail (error, links, training data) ──
    run_labels = {
        f'#{r["id"]} · {r["job_name"]} · {status_meta(r["status"])["label"]} · {r["started_at"].strftime("%b %d %H:%M:%S")}': r["id"]
        for _, r in filtered.iterrows()
    }
    st.markdown('<div class="section-label"><span>Run Detail</span></div>', unsafe_allow_html=True)
    picked_label = st.selectbox("Inspect run", list(run_labels.keys()), label_visibility="collapsed")
    picked_id = run_labels[picked_label]
    run = filtered[filtered["id"] == picked_id].iloc[0]

    meta = status_meta(run.get("status"))
    top = st.columns([1, 1, 1, 2])
    top[0].markdown(f'<span style="color:{meta["color"]};font-family:{MONO};">{meta["dot"]} {meta["label"]}</span>', unsafe_allow_html=True)
    top[1].markdown(f'<span style="font-family:{MONO};color:{MUTED};">{fmt_duration(run.get("duration_seconds"))}</span>', unsafe_allow_html=True)
    top[2].markdown(f'<span style="font-family:{MONO};color:{MUTED};">{int(run.get("rows_processed") or 0):,} rows</span>', unsafe_allow_html=True)
    if has_value(run.get("github_run_url")):
        top[3].markdown(f'[View on GitHub Actions ↗]({run["github_run_url"]})')

    err_text = clean_str(run.get("error_message"))
    msg_text = clean_str(run.get("message"))
    if err_text:
        st.markdown(f'<div class="run-err">{err_text}</div>', unsafe_allow_html=True)
    elif msg_text:
        st.markdown(f'<div class="run-msg">{msg_text}</div>', unsafe_allow_html=True)

    details = _parse(run.get("details"))
    if details:
        st.markdown('<div class="section-label"><span>Run Details</span></div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{"field": k, "value": v} for k, v in details.items()]),
            hide_index=True, use_container_width=True,
        )

    if not details_df.empty and "pipeline_run_id" in details_df.columns:
        matches = details_df[details_df["pipeline_run_id"] == picked_id]
        for _, td in matches.iterrows():
            render_training_detail(td, row_key=str(picked_id))

st.markdown(
    '<div class="disclaimer">Auto-refreshes every 60s from Supabase. '
    'Pulled via <code>fetch_all_rows</code> in db.py.</div>',
    unsafe_allow_html=True,
)
