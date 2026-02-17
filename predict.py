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
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    return tokenizer

def embed_speech(text: str, tokenizer, hf_token: str) -> np.ndarray:
    API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {hf_token}"}

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
        chunk_text = tokenizer.decode(tokens["input_ids"][i], skip_special_tokens=True)
        resp = req.post(API_URL, headers=headers, json={
            "inputs": chunk_text,
            "options": {"wait_for_model": True}
        }, timeout=30)
        resp.raise_for_status()
        cls_embeddings.append(np.array(resp.json()[0][0]))

    return np.mean(cls_embeddings, axis=0)

def predict(
    text: str,
    date: datetime,
    tokenizer,
    hf_token: str,
    ml_models: dict,
    feature_columns: list,
    fred_api_key: str
) -> dict:
    from features import build_feature_vector
    embedding = embed_speech(text, tokenizer, hf_token)
    X = build_feature_vector(embedding, date, fred_api_key, feature_columns)
    return {col: model.predict(X)[0] for col, model in ml_models.items()}