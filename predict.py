import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import os

# Load models once at startup
TARGET_COLS = [
    "SPX_t+3",  "SPX_t+7",  "SPX_t+30",
    "GOLD_t+3", "GOLD_t+7", "GOLD_t+30",
    "VIX_t+3",  "VIX_t+7",  "VIX_t+30",
    "TNX_t+3",  "TNX_t+7",  "TNX_t+30",
]

def load_models(models_dir: str = "models") -> dict:
    return {
        col: joblib.load(os.path.join(models_dir, f"{col}.pkl"))
        for col in TARGET_COLS
    }


def load_finbert():
    """Load FinBERT tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModel.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tokenizer, model


def embed_speech(text: str, tokenizer, model) -> np.ndarray:
    """
    Embed speech text using FinBERT.
    Returns mean-pooled CLS embedding — must match how training embeddings were generated!
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling over token embeddings (verify this matches your training!)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding  # shape: (768,)


def predict(
    text: str,
    date: datetime,
    tokenizer,
    bert_model,
    ml_models: dict,
    feature_columns: list,
    fred_api_key: str
) -> dict:
    from features import build_feature_vector

    # 1. Embed speech
    embedding = embed_speech(text, tokenizer, bert_model)

    # 2. Build feature vector
    X = build_feature_vector(embedding, date, fred_api_key, feature_columns)

    # 3. Run predictions
    results = {}
    for col, model in ml_models.items():
        pred = model.predict(X)[0]
        results[col] = pred

    return results