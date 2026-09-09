#!/usr/bin/env python3
"""
Fed Speech Model Trainer — Rolling-Window Continuous Training (Postgres)
===========================================================================

Runs monthly. Retrains the two-stage sign+magnitude stack on a rolling
window of history using FIXED, per-target hyperparameters (sourced from
a one-time Optuna search run on 2026-09-09 — see HYPERPARAM_SOURCE below),
evaluates on a trailing, fully-realized eval window, and promotes the new
model unconditionally IF it passes a cheap sanity gate.

No Optuna runs here. Re-tuning on every rolling window adds selection
variance without adding signal. Hyperparameter search is a separate,
lower-frequency concern (quarterly/semi-annually) — when that script
exists, swap HYPERPARAMS below to read from wherever it writes output.

Windowing:
  - eval_end   = today - MAX_HORIZON days
  - eval_start = eval_end - EVAL_WINDOW_MONTHS
  - train_end  = eval_start - MAX_HORIZON days    (purge gap)
  - train_start = earliest available speech date

Requires only DB_URL — price_action/macro_indicators are read once each
via psycopg2, no REST/Supabase-client round trip needed.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta
from pgvector.psycopg2 import register_vector
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pipeline_logging import start_run, finish_run, log_training_detail  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = os.environ["DB_URL"]
PROD_DIR = os.path.join(REPO_ROOT, "models_el", "production")
HYPERPARAM_SOURCE = "optuna_2026-09-09"  # bump this string whenever HYPERPARAMS below changes

TARGET_ASSETS = ["SPX", "GOLD", "VIX", "TNX"]
HORIZONS = [3, 7, 30]
TARGET_COLS = [f"{a}_t+{h}" for a in TARGET_ASSETS for h in HORIZONS]

MAX_HORIZON = 30
EVAL_WINDOW_MONTHS = 4

MIN_TRAIN_ROWS = 200
MIN_EVAL_ROWS = 10
MIN_PRED_STD = 1e-6
MAX_ABS_PRED = 5.0
MIN_SIGN_AUC = 0.30

_COMMON_REG = {"max_bins": 255, "early_stopping": True, "n_iter_no_change": 50,
               "validation_fraction": 0.15, "random_state": 42}
_COMMON_CLF = {"max_bins": 255, "early_stopping": True, "n_iter_no_change": 30,
               "validation_fraction": 0.15, "random_state": 42}

# ---------------------------------------------------------------------------
# Fixed hyperparameters — per target, sourced from the 2026-09-09 Optuna run.
# ---------------------------------------------------------------------------

MAGNITUDE_HYPERPARAMS = {
    "SPX_t+3":   dict(max_depth=4, max_leaf_nodes=16, min_samples_leaf=30, l2_regularization=3.216738316130575,  learning_rate=0.014646107545610718, max_iter=500),
    "SPX_t+7":   dict(max_depth=3, max_leaf_nodes=14, min_samples_leaf=79, l2_regularization=5.857421224796318,  learning_rate=0.018390973937871168, max_iter=1400),
    "SPX_t+30":  dict(max_depth=4, max_leaf_nodes=19, min_samples_leaf=32, l2_regularization=3.229671632403372,  learning_rate=0.010189702162220416, max_iter=1200),
    "GOLD_t+3":  dict(max_depth=3, max_leaf_nodes=18, min_samples_leaf=54, l2_regularization=4.786845531508995,  learning_rate=0.00525471184201688,  max_iter=800),
    "GOLD_t+7":  dict(max_depth=3, max_leaf_nodes=13, min_samples_leaf=50, l2_regularization=4.019334453111255,  learning_rate=0.012586039096052513, max_iter=1200),
    "GOLD_t+30": dict(max_depth=3, max_leaf_nodes=13, min_samples_leaf=50, l2_regularization=4.02800842868932,   learning_rate=0.008186223091364419, max_iter=700),
    "VIX_t+3":   dict(max_depth=2, max_leaf_nodes=22, min_samples_leaf=41, l2_regularization=3.1848016110574657, learning_rate=0.014949151633306557, max_iter=500),
    "VIX_t+7":   dict(max_depth=2, max_leaf_nodes=22, min_samples_leaf=52, l2_regularization=5.863592359181043,  learning_rate=0.010741400147472746, max_iter=1200),
    "VIX_t+30":  dict(max_depth=3, max_leaf_nodes=20, min_samples_leaf=55, l2_regularization=2.951505217315425,  learning_rate=0.007061034117895645, max_iter=700),
    # TNX's search space fixed max_depth=2, max_leaf_nodes=10, learning_rate=0.01 — only min_samples_leaf/l2/max_iter were searched
    "TNX_t+3":   dict(max_depth=2, max_leaf_nodes=10, min_samples_leaf=115, l2_regularization=6.900724559253215, learning_rate=0.01, max_iter=800),
    "TNX_t+7":   dict(max_depth=2, max_leaf_nodes=10, min_samples_leaf=95,  l2_regularization=7.1212502249818534, learning_rate=0.01, max_iter=700),
    "TNX_t+30":  dict(max_depth=2, max_leaf_nodes=10, min_samples_leaf=80,  l2_regularization=6.953785098568782, learning_rate=0.01, max_iter=1300),
}

SIGN_HYPERPARAMS = {
    "SPX_t+3":   dict(max_iter=200, learning_rate=0.06697167353375243, max_depth=4, max_leaf_nodes=24, min_samples_leaf=21, l2_regularization=7.579479953348009),
    "SPX_t+7":   dict(max_iter=500, learning_rate=0.0947416943676608,  max_depth=2, max_leaf_nodes=15, min_samples_leaf=55, l2_regularization=0.004346338963924681),
    "SPX_t+30":  dict(max_iter=400, learning_rate=0.0862735828664018,  max_depth=4, max_leaf_nodes=22, min_samples_leaf=32, l2_regularization=0.004207053950287938),
    "GOLD_t+3":  dict(max_iter=400, learning_rate=0.0862735828664018,  max_depth=4, max_leaf_nodes=22, min_samples_leaf=32, l2_regularization=0.004207053950287938),
    "GOLD_t+7":  dict(max_iter=600, learning_rate=0.0340553900090262,  max_depth=3, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=0.25003876953319126),
    "GOLD_t+30": dict(max_iter=800, learning_rate=0.01774767828860134, max_depth=3, max_leaf_nodes=31, min_samples_leaf=88, l2_regularization=0.5895953702873469),
    "VIX_t+3":   dict(max_iter=400, learning_rate=0.0862735828664018,  max_depth=4, max_leaf_nodes=22, min_samples_leaf=32, l2_regularization=0.004207053950287938),
    "VIX_t+7":   dict(max_iter=300, learning_rate=0.04359848050541604, max_depth=3, max_leaf_nodes=27, min_samples_leaf=66, l2_regularization=9.590428665924259),
    "VIX_t+30":  dict(max_iter=200, learning_rate=0.06697167353375243, max_depth=4, max_leaf_nodes=24, min_samples_leaf=21, l2_regularization=7.579479953348009),
    "TNX_t+3":   dict(max_iter=650, learning_rate=0.09013710971500906, max_depth=4, max_leaf_nodes=21, min_samples_leaf=22, l2_regularization=0.4026351853278354),
    "TNX_t+7":   dict(max_iter=200, learning_rate=0.07621195864233186, max_depth=3, max_leaf_nodes=23, min_samples_leaf=45, l2_regularization=0.12030178871154672),
    "TNX_t+30":  dict(max_iter=650, learning_rate=0.02229573198727992, max_depth=3, max_leaf_nodes=8,  min_samples_leaf=60, l2_regularization=0.03743859414648188),
}


def get_magnitude_hyperparams(col: str) -> dict:
    return {**MAGNITUDE_HYPERPARAMS[col], **_COMMON_REG}


def get_sign_hyperparams(col: str) -> dict:
    return {**SIGN_HYPERPARAMS[col], **_COMMON_CLF}


# ---------------------------------------------------------------------------
# Dataset assembly — all three tables fetched exactly once, via psycopg2 only
# ---------------------------------------------------------------------------

def get_weight(speaker: str) -> float:
    speaker = (speaker or "").lower()
    if "chair" in speaker and "vice" not in speaker:
        return 3
    elif "vice chair" in speaker or "vice chairman" in speaker:
        return 2
    elif "governor" in speaker:
        return 1
    return 0.5


def fetch_speeches_with_embeddings(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, date, title, speaker, embedding FROM fed_speech "
            "WHERE embedding IS NOT NULL ORDER BY date"
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["id", "date", "title", "speaker", "embedding"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_raw_prices(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT date, spx, tnx, gold, vix, dxy FROM price_action ORDER BY date")
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "SPX", "TNX", "GOLD", "VIX", "DXY"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_raw_macro(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT date, unemployment, interest_rate, growth_rate FROM macro_indicators ORDER BY date")
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "unemployment", "interest_rate", "growth_rate"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_targets(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices[["date"]].copy()
    for col in TARGET_ASSETS:
        for h in HORIZONS:
            out[f"{col}_t+{h}"] = (prices[col].shift(-(h + 1)) / prices[col].shift(-1)) - 1
    return out


def engineer_price_features(raw_prices: pd.DataFrame) -> pd.DataFrame:
    prices = raw_prices.copy()
    engineered_cols = ["date"]
    for col in ["SPX", "GOLD", "TNX", "DXY", "VIX"]:
        log_ret = np.log(prices[col] / prices[col].shift(1))
        prices[f"{col}_mom_3"] = prices[col].shift(1) / prices[col].shift(4) - 1
        prices[f"{col}_mom_7"] = prices[col].shift(1) / prices[col].shift(8) - 1
        prices[f"{col}_mom_30"] = prices[col].shift(1) / prices[col].shift(31) - 1
        prices[f"{col}_t-3"] = log_ret.shift(1).rolling(3).mean()
        prices[f"{col}_t-7"] = log_ret.shift(1).rolling(7).mean()
        prices[f"{col}_t-30"] = log_ret.shift(1).rolling(30).mean()
        prices[f"{col}_vol_7"] = log_ret.shift(1).rolling(7).std()
        prices[f"{col}_vol_30"] = log_ret.shift(1).rolling(30).std()
        engineered_cols += [f"{col}_mom_3", f"{col}_mom_7", f"{col}_mom_30",
                             f"{col}_t-3", f"{col}_t-7", f"{col}_t-30",
                             f"{col}_vol_7", f"{col}_vol_30"]
    return prices[engineered_cols]


def engineer_macro_features(raw_macro: pd.DataFrame) -> pd.DataFrame:
    macro = raw_macro.set_index("date").sort_index()
    for col in ["unemployment", "growth_rate"]:
        macro[col] = macro[col].shift(30)
    daily_index = pd.date_range(start=macro.index.min(), end=macro.index.max(), freq="D")
    macro_daily = macro.reindex(daily_index).ffill().reset_index().rename(columns={"index": "date"})
    return macro_daily[["date", "unemployment", "interest_rate", "growth_rate"]]


def _vector_to_array(v) -> np.ndarray:
    if hasattr(v, "to_list"):
        return np.asarray(v.to_list(), dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


def build_dataset(conn) -> pd.DataFrame:
    speeches = fetch_speeches_with_embeddings(conn)
    if speeches.empty:
        raise RuntimeError("fed_speech has no embedded rows — nothing to train on.")

    raw_prices = fetch_raw_prices(conn)          # fetched once
    raw_macro = fetch_raw_macro(conn)             # fetched once
    price_engineered = engineer_price_features(raw_prices)
    macro_engineered = engineer_macro_features(raw_macro)
    targets = compute_targets(raw_prices)          # reuses the same raw_prices fetch

    merged = price_engineered.merge(macro_engineered, on="date", how="inner")
    merged = merged.merge(targets, on="date", how="left")
    df = speeches.merge(merged, on="date", how="left")
    df["sample_weight"] = df["speaker"].apply(get_weight)

    emb_matrix = np.vstack(df["embedding"].apply(_vector_to_array).values)
    assert emb_matrix.ndim == 2 and emb_matrix.shape[1] > 1, (
        f"Embedding matrix has unexpected shape {emb_matrix.shape}"
    )
    emb_df = pd.DataFrame(emb_matrix, index=df.index, columns=[f"emb_{i}" for i in range(emb_matrix.shape[1])])
    df = pd.concat([df.drop(columns=["embedding"]), emb_df], axis=1)

    return df.sort_values("date").reset_index(drop=True)


def purged_split(df: pd.DataFrame):
    today = datetime.now(timezone.utc).date()
    eval_end = today - timedelta(days=MAX_HORIZON)
    eval_start = eval_end - relativedelta(months=EVAL_WINDOW_MONTHS)
    train_end = eval_start - timedelta(days=MAX_HORIZON)
    train_start = df["date"].min().date()

    train_df = df[(df["date"].dt.date >= train_start) & (df["date"].dt.date <= train_end)]
    eval_df = df[(df["date"].dt.date >= eval_start) & (df["date"].dt.date <= eval_end)]

    windows = {"train_start": train_start, "train_end": train_end,
               "eval_start": eval_start, "eval_end": eval_end}
    return train_df, eval_df, windows


# ---------------------------------------------------------------------------
# Modeling — plain fit with fixed per-target hyperparams
# ---------------------------------------------------------------------------

def get_class_weight_multiplier(y_bin, neg_boost=1.0):
    n_pos, n_neg = y_bin.sum(), len(y_bin) - y_bin.sum()
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y_bin))
    w_pos = len(y_bin) / (2.0 * n_pos)
    w_neg = len(y_bin) / (2.0 * n_neg) * neg_boost
    return np.where(y_bin == 1, w_pos, w_neg)


def train_target(col, X_train, y_train, w_train):
    horizon = int(col.split("+")[1])

    y_col = y_train[col]
    valid = y_col.notna()
    X_full, y_full, w_full = X_train[valid], y_col[valid], w_train[valid]

    reg_params = get_magnitude_hyperparams(col)
    magnitude_model = HistGradientBoostingRegressor(**reg_params)
    magnitude_model.fit(X_full, y_full, sample_weight=w_full)

    clf_params = get_sign_hyperparams(col)
    y_bin_full = (y_full > 0).astype(int).values
    class_w = get_class_weight_multiplier(y_bin_full)
    w_clf_full = w_full.values * class_w

    tscv = TimeSeriesSplit(n_splits=5)
    oof_proba, oof_true = [], []
    for tr_idx, val_idx in tscv.split(X_full):
        purged = tr_idx[:-horizon] if horizon > 0 else tr_idx
        if len(purged) < 50 or len(np.unique(y_bin_full[purged])) < 2:
            continue
        m = HistGradientBoostingClassifier(**clf_params)
        m.fit(X_full.iloc[purged], y_bin_full[purged], sample_weight=w_clf_full[purged])
        oof_proba.append(m.predict_proba(X_full.iloc[val_idx])[:, 1])
        oof_true.append(y_bin_full[val_idx])
    oof_proba, oof_true = np.concatenate(oof_proba), np.concatenate(oof_true)

    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(oof_true, (oof_proba >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    sign_model = HistGradientBoostingClassifier(**clf_params)
    sign_model.fit(X_full, y_bin_full, sample_weight=w_clf_full)

    hyperparams = {"sign": clf_params, "magnitude": reg_params}
    return sign_model, best_t, magnitude_model, hyperparams


def combined_predict(sign_model, threshold, magnitude_model, X):
    proba = sign_model.predict_proba(X)[:, 1]
    sign = np.where(proba >= threshold, 1, -1)
    magnitude = np.abs(magnitude_model.predict(X))
    return sign * magnitude, proba


def evaluate_target(col, sign_model, threshold, magnitude_model, X_eval, y_eval):
    y_true = y_eval[col]
    valid = y_true.notna()
    if valid.sum() < 2:
        return None, None, None
    y_true_valid = y_true[valid]
    y_bin_true = (y_true_valid > 0).astype(int)

    pred, proba = combined_predict(sign_model, threshold, magnitude_model, X_eval[valid])
    pred_bin = (proba >= threshold).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_bin_true, proba)) if len(np.unique(y_bin_true)) == 2 else None,
        "f1": float(f1_score(y_bin_true, pred_bin, average="macro", zero_division=0)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_valid, pred))),
    }
    return metrics, pred, y_bin_true

MIN_CLASS_COUNT_FOR_AUC_CHECK = 8  # per class; below this, AUC is too noisy to judge inversion from

def run_sanity_checks(train_rows, eval_rows, all_metrics, all_preds, all_eval_labels):
    checks = {}
    checks["train_rows_ok"] = train_rows >= MIN_TRAIN_ROWS
    checks["eval_rows_ok"] = eval_rows >= MIN_EVAL_ROWS
    checks["no_nan_inf"] = all(np.isfinite(p).all() for p in all_preds.values() if p is not None and len(p) > 0)
    checks["variance_ok"] = all(np.std(p) > MIN_PRED_STD for p in all_preds.values() if p is not None and len(p) > 0)
    checks["magnitude_bounded"] = all(np.abs(p).max() < MAX_ABS_PRED for p in all_preds.values() if p is not None and len(p) > 0)

    inverted_targets = []
    for col, metrics in all_metrics.items():
        if not metrics or metrics["auc"] is None:
            continue
        y_bin = all_eval_labels.get(col)
        if y_bin is None:
            continue
        n_pos, n_neg = y_bin.sum(), len(y_bin) - y_bin.sum()
        if n_pos < MIN_CLASS_COUNT_FOR_AUC_CHECK or n_neg < MIN_CLASS_COUNT_FOR_AUC_CHECK:
            continue  # not enough data in this window to trust the AUC estimate
        if metrics["auc"] < MIN_SIGN_AUC:
            inverted_targets.append(col)
    checks["sign_not_inverted"] = len(inverted_targets) == 0
    checks["_inverted_targets"] = inverted_targets  # kept for visibility in sanity_details, not a bool gate itself

    passed = all(v for k, v in checks.items() if not k.startswith("_"))
    return passed, checks

def save_artifacts(sign_models, thresholds, magnitude_models, hyperparams, feature_columns, windows):
    os.makedirs(os.path.join(PROD_DIR, "magnitude"), exist_ok=True)
    os.makedirs(os.path.join(PROD_DIR, "sign"), exist_ok=True)
    for col, model in magnitude_models.items():
        joblib.dump(model, os.path.join(PROD_DIR, "magnitude", f"{col}.pkl"))
    for col, model in sign_models.items():
        joblib.dump(model, os.path.join(PROD_DIR, "sign", f"{col}.pkl"))
    with open(os.path.join(PROD_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    joblib.dump(feature_columns, os.path.join(PROD_DIR, "feature_columns.pkl"))
    with open(os.path.join(PROD_DIR, "metadata.json"), "w") as f:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            **{k: str(v) for k, v in windows.items()},
            "targets": TARGET_COLS,
            "hyperparam_source": HYPERPARAM_SOURCE,
        }, f, indent=2)


def write_github_output(promoted: bool):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as f:
            f.write(f"promoted={'true' if promoted else 'false'}\n")


def main():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    run_id = start_run(conn, "trainer")

    try:
        print("Building dataset...")
        df = build_dataset(conn)
        train_df, eval_df, windows = purged_split(df)
        print(f"Train: {windows['train_start']} -> {windows['train_end']} ({len(train_df)} rows)")
        print(f"Eval:  {windows['eval_start']} -> {windows['eval_end']} ({len(eval_df)} rows)")

        drop_cols = ["date", "title", "speaker", "sample_weight", "id"] + TARGET_COLS
        feature_columns = [c for c in df.columns if c not in drop_cols]

        X_train, y_train, w_train = train_df[feature_columns], train_df[TARGET_COLS], train_df["sample_weight"]
        X_eval, y_eval = eval_df[feature_columns], eval_df[TARGET_COLS]

        sign_models, thresholds, magnitude_models, hyperparams = {}, {}, {}, {}
        all_metrics, all_preds, all_eval_labels = {}, {}, {}

        for col in TARGET_COLS:
            print(f"Training {col}...")
            sign_model, threshold, magnitude_model, hp = train_target(col, X_train, y_train, w_train)
            sign_models[col], thresholds[col], magnitude_models[col], hyperparams[col] = (
                sign_model, threshold, magnitude_model, hp
            )
            metrics, pred, y_bin = evaluate_target(col, sign_model, threshold, magnitude_model, X_eval, y_eval)
            all_metrics[col], all_preds[col], all_eval_labels[col] = metrics, pred, y_bin
            print(f"  eval metrics: {metrics}")

        sanity_passed, sanity_details = run_sanity_checks(len(train_df), len(eval_df), all_metrics, all_preds, all_eval_labels)
        print(f"\nSanity gate: {'PASSED' if sanity_passed else 'FAILED'}")
        print(json.dumps(sanity_details, indent=2))

        if sanity_passed:
            save_artifacts(sign_models, thresholds, magnitude_models, hyperparams, feature_columns, windows)
            write_github_output(promoted=True)
            status, message = "success", "Trained and promoted new model"
        else:
            write_github_output(promoted=False)
            status, message = "failed", "Sanity gate failed — new model NOT promoted"

        finish_run(conn, run_id, status=status, rows_processed=len(TARGET_COLS), message=message)
        log_training_detail(
            conn, run_id,
            train_start=windows["train_start"], train_end=windows["train_end"],
            eval_start=windows["eval_start"], eval_end=windows["eval_end"],
            metrics=all_metrics, hyperparams=hyperparams,
            sanity_passed=sanity_passed, sanity_details=sanity_details,
            promoted=sanity_passed, git_commit=os.environ.get("GITHUB_SHA"),
        )

        if not sanity_passed:
            sys.exit(1)

    except Exception as e:
        finish_run(conn, run_id, status="failed", error_message=str(e))
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
