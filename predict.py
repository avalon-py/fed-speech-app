import json
import numpy as np
import joblib
import onnxruntime as ort
from tokenizers import Tokenizer
from datetime import datetime
import os

TARGET_COLS = [
    "SPX_t+3",  "SPX_t+7",  "SPX_t+30",
    "GOLD_t+3", "GOLD_t+7", "GOLD_t+30",
    "VIX_t+3",  "VIX_t+7",  "VIX_t+30",
    "TNX_t+3",  "TNX_t+7",  "TNX_t+30",
]

FINBERT_ONNX_DIR = os.path.join("models", "finbert-onnx")
MAX_LENGTH = 512
STRIDE = 50


def load_models(production_dir: str = "models_el/production") -> dict:
    """
    Loads both halves of the sign+magnitude stack, plus the decision
    thresholds that go with the sign classifiers.

    Returns a dict with three keys:
      - "magnitude": {col: HistGradientBoostingRegressor}
      - "sign":      {col: HistGradientBoostingClassifier}
      - "thresholds": {col: float}
    """
    magnitude_models = {
        col: joblib.load(os.path.join(production_dir, "magnitude", f"{col}.pkl"))
        for col in TARGET_COLS
    }
    sign_models = {
        col: joblib.load(os.path.join(production_dir, "sign", f"{col}.pkl"))
        for col in TARGET_COLS
    }
    with open(os.path.join(production_dir, "thresholds.json")) as f:
        thresholds = json.load(f)

    return {
        "magnitude": magnitude_models,
        "sign": sign_models,
        "thresholds": thresholds,
    }


def load_finbert(onnx_dir: str = FINBERT_ONNX_DIR):
    """
    Loads the tokenizer + a local FP16 ONNX FinBERT session, both read
    straight from disk. No torch, no transformers, no network call to
    Hugging Face at runtime.
    """
    tokenizer = Tokenizer.from_file(os.path.join(onnx_dir, "tokenizer.json"))

    tokenizer.enable_truncation(max_length=MAX_LENGTH, stride=STRIDE, strategy="only_first")

    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0
    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]", length=MAX_LENGTH)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.enable_cpu_mem_arena = False

    session = ort.InferenceSession(
        os.path.join(onnx_dir, "model_fp16.onnx"),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    return tokenizer, session


def embed_speech(text: str, tokenizer, session) -> np.ndarray:
    encoding = tokenizer.encode(text)
    all_encodings = [encoding] + encoding.overflowing

    input_ids = np.array([e.ids for e in all_encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in all_encodings], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in all_encodings], dtype=np.int64)

    tokens = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    session_input_names = {inp.name for inp in session.get_inputs()}
    feed = {name: arr for name, arr in tokens.items() if name in session_input_names}

    outputs = session.run(None, feed)
    last_hidden_state = outputs[0]

    cls_embeddings = last_hidden_state[:, 0, :].astype(np.float32)
    return cls_embeddings.mean(axis=0)


def combined_predict(col: str, X, ml_models: dict) -> float:
    """
    sign x magnitude, mirroring the notebook's combined_predict:
      - sign comes from the classifier's up-probability vs. its tuned threshold
      - magnitude comes from the regressor's |raw output|
    """
    proba = ml_models["sign"][col].predict_proba(X)[:, 1][0]
    sign = 1 if proba >= ml_models["thresholds"][col] else -1
    magnitude = abs(ml_models["magnitude"][col].predict(X)[0])
    return sign * magnitude, proba


def predict(
    text: str,
    date: datetime,
    tokenizer,
    session,
    ml_models: dict,
    feature_columns: list,
) -> dict:
    """
    Returns, per target:
      {"pred": signed % move, "up_probability": raw classifier confidence}
    up_probability is included alongside pred since it's a genuinely
    different piece of information than the signed magnitude (confidence
    vs. size) — dropping it would throw away something the two-stage
    model actually computes.
    """
    from features import build_feature_vector

    embedding = embed_speech(text, tokenizer, session)
    X = build_feature_vector(embedding, date, feature_columns)

    results = {}
    for col in TARGET_COLS:
        pred, proba = combined_predict(col, X, ml_models)
        results[col] = {"pred": pred, "up_probability": proba}
    return results
