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


def load_models(models_dir: str = "models") -> dict:
    return {
        col: joblib.load(os.path.join(models_dir, f"{col}.pkl"))
        for col in TARGET_COLS
    }


def load_finbert(onnx_dir: str = FINBERT_ONNX_DIR):
    """
    Loads the tokenizer + a local FP16 ONNX FinBERT session, both read
    straight from disk. No torch, no transformers, no network call to
    Hugging Face at runtime.

    Uses tokenizers.Tokenizer.from_file() against tokenizer.json directly
    instead of transformers.AutoTokenizer - same vocab/merges, ~150-250MB
    less import overhead, and no dependency on the transformers package
    (huggingface-hub, safetensors, regex, tqdm, typer, etc.) at all.
    """
    tokenizer = Tokenizer.from_file(os.path.join(onnx_dir, "tokenizer.json"))

    # Match the old transformers call: truncation=True, max_length=512,
    # stride=50, return_overflowing_tokens=True, padding=True
    tokenizer.enable_truncation(max_length=MAX_LENGTH, stride=STRIDE, strategy="only_first")

    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0  # BERT-family vocabs put [PAD] at index 0 as a fallback
    # Pad every chunk to MAX_LENGTH rather than "longest in batch" - matches
    # the max_length cap already in play and keeps every ONNX input a fixed,
    # predictable shape.
    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]", length=MAX_LENGTH)

    sess_options = ort.SessionOptions()
    # Small Streamlit instances only get 1 CPU core anyway; keeping thread
    # pools at 1 avoids onnxruntime spinning up extra threads that just
    # burn RAM without speeding anything up.
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    # Default arena pre-allocates and grows greedily; disabling it trims
    # peak RSS at a small cost to per-call alloc speed - worth it here
    # given the 1GB ceiling.
    sess_options.enable_cpu_mem_arena = False

    session = ort.InferenceSession(
        os.path.join(onnx_dir, "model_fp16.onnx"),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    return tokenizer, session


def embed_speech(text: str, tokenizer, session) -> np.ndarray:
    """
    Same 512-token sliding-window chunking as before, run as a single
    local batched ONNX forward pass. With tokenizers, overflow chunks
    come back via Encoding.overflowing instead of a return_overflowing_tokens
    kwarg - the truncation/padding config set once in load_finbert()
    governs both.
    """
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

    # Only pass along the inputs this particular ONNX graph actually
    # expects (e.g. some exports omit token_type_ids).
    session_input_names = {inp.name for inp in session.get_inputs()}
    feed = {name: arr for name, arr in tokens.items() if name in session_input_names}

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
