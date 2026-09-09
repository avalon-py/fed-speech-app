#!/usr/bin/env python3
"""
Fed Speech Model Trainer — Rolling-Window Continuous Training (Postgres)
===========================================================================

Runs monthly. Retrains the two-stage sign+magnitude stack on a rolling
window of history, evaluates on a trailing, fully-realized eval window,
and promotes the new model unconditionally IF it passes a cheap sanity
gate (not a performance-margin gate — see conversation history for why:
recency is preferred by design, sanity just catches pipeline breakage).

Windowing:
  - eval_end   = today - MAX_HORIZON days        (last date with a fully
                                                    realized t+30 label)
  - eval_start = eval_end - EVAL_WINDOW_MONTHS
  - train_end  = eval_start - MAX_HORIZON days    (purge gap, so no
                                                    training example's
                                                    label window bleeds
                                                    into the eval window)
  - train_start = earliest available speech date

Requires DB_URL. Writes model artifacts to models_el/production/ locally;
the calling workflow is responsible for committing/pushing them if this
script reports promoted=true via GITHUB_OUTPUT.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import optuna
import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta
from pgvector.psycopg2 import register_vector
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — features.py/db.py live at repo root, this script is in scripts/
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from features import _engineer_price_features, _engineer_macro_features  # noqa: E402
from pipeline_logging import start_run, finish_run, log_training_detail  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = os.environ["DB_URL"]
PROD_DIR = os.path.join(REPO_ROOT, "models_el", "production")

TARGET_ASSETS = ["SPX", "GOLD", "VIX", "TNX"]  # DXY is a feature only, never a target
HORIZONS = [3, 7, 30]
TARGET_COLS = [f"{a}_t+{h}" for a in TARGET_ASSETS for h in HORIZONS]

MAX_HORIZON = 30
EVAL_WINDOW_MONTHS = 4
N_TRIALS = int(os.environ.get("N_TRIALS", "20"))  # lower via env var for faster/cheaper runs

# Sanity-gate thresholds — deliberately loose. This is "not obviously
# broken," not "good." See conversation history for rationale.
MIN_TRAIN_ROWS = 200
MIN_EVAL_ROWS = 10
MIN_PRED_STD = 1e-6          # predictions collapsed to ~constant = broken
MAX_ABS_PRED = 5.0           # a >500% predicted move = something's wrong
MIN_SIGN_AUC = 0.30          # AUC this far below 0.5 suggests an inverted sign, not just noise


# ---------------------------------------------------------------------------
# Dataset assembly
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


def compute_targets(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward returns at t+3/t+7/t+30, SPX/GOLD/VIX/TNX only."""
    out = prices[["date"]].copy()
    for col in TARGET_ASSETS:
        for h in HORIZONS:
            out[f"{col}_t+{h}"] = (prices[col].shift(-(h + 1)) / prices[col].shift(-1)) - 1
    return out


def build_dataset(conn) -> pd.DataFrame:
    speeches = fetch_speeches_with_embeddings(conn)
    if speeches.empty:
        raise RuntimeError("fed_speech has no embedded rows — nothing to train on.")

    price_engineered = _engineer_price_features()
    macro_engineered = _engineer_macro_features()
    raw_prices = fetch_raw_prices(conn)
    targets = compute_targets(raw_prices)

    merged = price_engineered.merge(macro_engineered, on="date", how="inner")
    merged = merged.merge(targets, on="date", how="left")

    df = speeches.merge(merged, on="date", how="left")
    df["sample_weight"] = df["speaker"].apply(get_weight)

    emb_matrix = np.vstack(df["embedding"].apply(np.asarray).values)
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

    windows = {
        "train_start": train_start, "train_end": train_end,
        "eval_start": eval_start, "eval_end": eval_end,
    }
    return train_df, eval_df, windows


# ---------------------------------------------------------------------------
# Modeling — asset-aware Optuna search spaces, purged CV, sign+magnitude stack
# (mirrors EL_main.ipynb; see that notebook for the original derivation)
# ---------------------------------------------------------------------------

def regressor_search_space(trial, asset):
    if asset in ("SPX", "VIX"):
        return dict(
            max_depth=trial.suggest_int("max_depth", 2, 4),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 12, 25),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 30, 80),
            l2_regularization=trial.suggest_float("l2_regularization", 2.0, 8.0, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.02, log=True),
        )
    elif asset == "GOLD":
        return dict(
            max_depth=trial.suggest_int("max_depth", 2, 3),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 10, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 50, 100),
            l2_regularization=trial.suggest_float("l2_regularization", 4.0, 10.0, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.015, log=True),
        )
    else:  # TNX
        return dict(
            max_depth=2, max_leaf_nodes=10,
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 60, 120),
            l2_regularization=trial.suggest_float("l2_regularization", 5.0, 10.0, log=True),
            learning_rate=0.01,
        )


def regressor_objective(trial, X, y, sample_weights, asset, horizon):
    params = {
        **regressor_search_space(trial, asset),
        "max_iter": trial.suggest_int("max_iter", 500, 1500, step=100),
        "max_bins": 255, "early_stopping": True,
        "n_iter_no_change": 50, "validation_fraction": 0.15, "random_state": 42,
    }
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    for tr_idx, val_idx in tscv.split(X):
        purged = tr_idx[:-horizon] if horizon > 0 else tr_idx
        if len(purged) < 50:
            continue
        model = HistGradientBoostingRegressor(**params)
        model.fit(X.iloc[purged], y.iloc[purged], sample_weight=sample_weights.iloc[purged])
        pred = model.predict(X.iloc[val_idx])
        y_bin = (y.iloc[val_idx] > 0).astype(int)
        if len(np.unique(y_bin)) == 2:
            cv_aucs.append(roc_auc_score(y_bin, pred))
    return np.mean(cv_aucs) if cv_aucs else 0.0


def get_class_weight_multiplier(y_bin, neg_boost=1.0):
    n_pos, n_neg = y_bin.sum(), len(y_bin) - y_bin.sum()
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y_bin))
    w_pos = len(y_bin) / (2.0 * n_pos)
    w_neg = len(y_bin) / (2.0 * n_neg) * neg_boost
    return np.where(y_bin == 1, w_pos, w_neg)


def classifier_objective(trial, X, y_bin, sample_weights, horizon):
    params = {
        "max_iter": trial.suggest_int("max_iter", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 31),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_bins": 255, "early_stopping": True,
        "n_iter_no_change": 30, "validation_fraction": 0.15, "random_state": 42,
    }
    tscv = TimeSeriesSplit(n_splits=5)
    fold_f1s = []
    for tr_idx, val_idx in tscv.split(X):
        purged = tr_idx[:-horizon] if horizon > 0 else tr_idx
        if len(purged) < 50:
            continue
        y_tr, y_val = y_bin[purged], y_bin[val_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            continue
        w_tr = sample_weights[purged]
        model = HistGradientBoostingClassifier(**params)
        model.fit(X.iloc[purged], y_tr, sample_weight=w_tr)
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        best_f1 = max(
            f1_score(y_val, (proba >= t).astype(int), average="macro", zero_division=0)
            for t in np.linspace(0.1, 0.9, 17)
        )
        fold_f1s.append(best_f1)
    return np.mean(fold_f1s) if fold_f1s else 0.0


def train_target(col, X_train, y_train, w_train, feature_columns):
    asset = col.split("_")[0]
    horizon = int(col.split("+")[1])

    y_col = y_train[col]
    valid = y_col.notna()
    X_full, y_full, w_full = X_train[valid], y_col[valid], w_train[valid]

    # ── magnitude regressor ──
    study_r = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_r.optimize(lambda t: regressor_objective(t, X_full, y_full, w_full, asset, horizon),
                      n_trials=N_TRIALS, show_progress_bar=False)
    reg_params = {**study_r.best_params, "max_bins": 255, "early_stopping": True,
                  "n_iter_no_change": 50, "validation_fraction": 0.15, "random_state": 42}
    magnitude_model = HistGradientBoostingRegressor(**reg_params)
    magnitude_model.fit(X_full, y_full, sample_weight=w_full)

    # ── sign classifier ──
    y_bin_full = (y_full > 0).astype(int).values
    class_w = get_class_weight_multiplier(y_bin_full)
    w_clf_full = w_full.values * class_w

    study_c = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_c.optimize(lambda t: classifier_objective(t, X_full, y_bin_full, w_clf_full, horizon),
                      n_trials=N_TRIALS, show_progress_bar=False)
    clf_params = {**study_c.best_params, "max_bins": 255, "early_stopping": True,
                  "n_iter_no_change": 30, "validation_fraction": 0.15, "random_state": 42}

    # OOF threshold selection
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
        return None, None
    y_true_valid = y_true[valid]
    y_bin_true = (y_true_valid > 0).astype(int)

    pred, proba = combined_predict(sign_model, threshold, magnitude_model, X_eval[valid])
    pred_bin = (proba >= threshold).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_bin_true, proba)) if len(np.unique(y_bin_true)) == 2 else None,
        "f1": float(f1_score(y_bin_true, pred_bin, average="macro", zero_division=0)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_valid, pred))),
    }
    return metrics, pred


# ---------------------------------------------------------------------------
# Sanity gate — cheap "not obviously broken" checks, NOT a performance gate.
# ---------------------------------------------------------------------------

def run_sanity_checks(train_rows, eval_rows, all_metrics, all_preds):
    checks = {}
    checks["train_rows_ok"] = train_rows >= MIN_TRAIN_ROWS
    checks["eval_rows_ok"] = eval_rows >= MIN_EVAL_ROWS
    checks["no_nan_inf"] = all(
        np.isfinite(p).all() for p in all_preds.values() if p is not None and len(p) > 0
    )
    checks["variance_ok"] = all(
        np.std(p) > MIN_PRED_STD for p in all_preds.values() if p is not None and len(p) > 0
    )
    checks["magnitude_bounded"] = all(
        np.abs(p).max() < MAX_ABS_PRED for p in all_preds.values() if p is not None and len(p) > 0
    )
    aucs = [m["auc"] for m in all_metrics.values() if m and m["auc"] is not None]
    checks["sign_not_inverted"] = all(a >= MIN_SIGN_AUC for a in aucs) if aucs else True

    passed = all(checks.values())
    return passed, checks


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(sign_models, thresholds, magnitude_models, hyperparams,
                    feature_columns, windows):
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
        }, f, indent=2)


def write_github_output(promoted: bool):
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as f:
            f.write(f"promoted={'true' if promoted else 'false'}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        all_metrics, all_preds = {}, {}

        for col in TARGET_COLS:
            print(f"\n{'='*60}\nTraining {col}\n{'='*60}")
            sign_model, threshold, magnitude_model, hp = train_target(col, X_train, y_train, w_train, feature_columns)
            sign_models[col], thresholds[col], magnitude_models[col], hyperparams[col] = (
                sign_model, threshold, magnitude_model, hp
            )
            metrics, pred = evaluate_target(col, sign_model, threshold, magnitude_model, X_eval, y_eval)
            all_metrics[col], all_preds[col] = metrics, pred
            print(f"  eval metrics: {metrics}")

        sanity_passed, sanity_details = run_sanity_checks(len(train_df), len(eval_df), all_metrics, all_preds)
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
