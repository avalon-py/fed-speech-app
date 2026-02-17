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


import numpy as np
import joblib
import requests as req
from transformers import AutoTokenizer
from datetime import datetime
import os

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
    """Load tokenizer only — no model weights, no torch needed."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return tokenizer

def embed_speech(text: str, tokenizer, hf_token: str) -> np.ndarray:
    """
    Replicates training embedding exactly:
    - Chunks text with max_length=512, stride=50
    - Takes CLS token (index 0) from each chunk via HF Inference API
    - Mean pools CLS tokens across chunks
    """
    API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {hf_token}"}

    # Chunk text exactly as in training
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        stride=50,
        return_overflowing_tokens=True,
        padding=True
    )

    cls_embeddings = []

    for i in range(tokens["input_ids"].shape[0]):
        chunk_ids  = tokens["input_ids"][i]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        payload = {
            "inputs": chunk_text,
            "options": {"wait_for_model": True}
        }

        resp = req.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        # CLS token is index 0, matching training
        cls_embeddings.append(np.array(result[0][0]))

    # Mean pool across chunks — matches training exactly
    embedding = np.mean(cls_embeddings, axis=0)
    return embedding  # shape: (768,)

def predict(
    text: str,
    date: datetime,
    tokenizer,
    hf_token: str,       # ← replaces bert_model
    ml_models: dict,
    feature_columns: list,
    fred_api_key: str
) -> dict:
    from features import build_feature_vector

    # 1. Embed via HF API
    embedding = embed_speech(text, tokenizer, hf_token)

    # 2. Build feature vector
    X = build_feature_vector(embedding, date, fred_api_key, feature_columns)

    # 3. Run predictions
    results = {}
    for col, model in ml_models.items():
        results[col] = model.predict(X)[0]

    return results