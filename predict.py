import numpy as np
import joblib
import onnxruntime as ort
from transformers import AutoTokenizer
from datetime import datetime
import os

TARGET_COLS = [
    "SPX_t+3",  "SPX_t+7",  "SPX_t+30",
    "GOLD_t+3", "GOLD_t+7", "GOLD_t+30",
    "VIX_t+3",  "VIX_t+7",  "VIX_t+30",
    "TNX_t+3",  "TNX_t+7",  "TNX_t+30",
]

FINBERT_ONNX_DIR = os.path.join("models", "finbert-onnx")


def load_models(models_dir: str = "models") -> dict:
    return {
        col: joblib.load(os.path.join(models_dir, f"{col}.pkl"))
        for col in TARGET_COLS
    }


def load_finbert(onnx_dir: str = FINBERT_ONNX_DIR):
    """
    Loads the tokenizer + a local FP16 ONNX FinBERT session, both read
    straight from disk. No torch, no network call to Hugging Face at
    runtime - see export_finbert_to_onnx.py for how models/finbert-onnx/
    was produced.
    """
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

    sess_options = ort.SessionOptions()
    # Small Streamlit instances only get 1 CPU core anyway; keeping thread
    # pools at 1 avoids onnxruntime spinning up extra threads that just
    # burn RAM without speeding anything up.
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        os.path.join(onnx_dir, "model_fp16.onnx"),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    return tokenizer, session


def embed_speech(text: str, tokenizer, session) -> np.ndarray:
    """
    Same 512-token sliding-window chunking as before, but run as a single
    local batched ONNX forward pass instead of one HTTP request per chunk.
    """
    tokens = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
        stride=50,
        return_overflowing_tokens=True,
        padding=True,
    )

    # Only pass along the inputs this particular ONNX graph actually
    # expects (e.g. some exports omit token_type_ids).
    session_input_names = {inp.name for inp in session.get_inputs()}
    feed = {
        name: tokens[name].astype(np.int64)
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in tokens and name in session_input_names
    }

    outputs = session.run(None, feed)
    last_hidden_state = outputs[0]  # (num_chunks, seq_len, hidden)

    cls_embeddings = last_hidden_state[:, 0, :].astype(np.float32)
    return cls_embeddings.mean(axis=0)


def predict(
    text: str,
    date: datetime,
    tokenizer,
    session,
    ml_models: dict,
    feature_columns: list,
) -> dict:
    from features import build_feature_vector

    embedding = embed_speech(text, tokenizer, session)
    X = build_feature_vector(embedding, date, feature_columns)
    return {col: model.predict(X)[0] for col, model in ml_models.items()}
