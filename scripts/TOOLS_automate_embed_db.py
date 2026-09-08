#!/usr/bin/env python3
"""
Fed Speech Embedder — Incremental Updater (Postgres)
======================================================

Finds rows in fed_speech with a NULL embedding and fills them in using
the local ONNX FinBERT model. Meant to run after the scraper adds new
rows, but is safe to run any time — it only ever touches rows where
embedding IS NULL, so running it twice in a row (or with nothing new
to do) is a no-op.

Requires the DB_URL environment variable (same Supabase pooler
connection string as the scraper).
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = os.environ["DB_URL"]
MODEL_DIR = os.environ.get("MODEL_DIR", "models/finbert-onnx")
BATCH_COMMIT_EVERY = 20  # commit periodically so a late failure doesn't lose earlier work

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_tokenizer = None
_session = None


def get_finbert_onnx(model_dir=MODEL_DIR):
    global _tokenizer, _session
    if _session is None:
        print(f"Loading ONNX FinBERT from {model_dir} ...")
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _session = ort.InferenceSession(
            f"{model_dir}/model_fp16.onnx",
            providers=["CPUExecutionProvider"],
        )
    return _tokenizer, _session


def speech_embedding(text: str, model_dir=MODEL_DIR, max_length=512, stride=50) -> np.ndarray:
    tokenizer, session = get_finbert_onnx(model_dir)

    tokens = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=True,
    )
    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)
    token_type_ids = tokens.get("token_type_ids")
    token_type_ids = (
        token_type_ids.astype(np.int64) if token_type_ids is not None else np.zeros_like(input_ids)
    )

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    last_hidden_state = outputs[0]
    cls_embeddings = last_hidden_state[:, 0, :].astype(np.float32)
    return cls_embeddings.mean(axis=0)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def fetch_unembedded_rows(conn) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content FROM fed_speech WHERE embedding IS NULL ORDER BY id"
        )
        return cur.fetchall()


def update_embedding(conn, row_id: int, embedding: np.ndarray) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE fed_speech SET embedding = %s WHERE id = %s",
            (embedding, row_id),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)

    try:
        rows = fetch_unembedded_rows(conn)
        print(f"Found {len(rows)} row(s) needing embeddings")

        if not rows:
            print("Nothing to do.")
            return

        # Load the model once, up front, so a failure on row 1 doesn't
        # leave you wondering whether the model itself is broken.
        get_finbert_onnx()

        computed = 0
        failed_ids = []

        for i, (row_id, content) in enumerate(rows):
            if not content or not content.strip():
                print(f"  ! Row {row_id} has empty content, skipping", file=sys.stderr)
                failed_ids.append(row_id)
                continue

            try:
                emb = speech_embedding(content)
                update_embedding(conn, row_id, emb)
                computed += 1
                print(f"[{i + 1}/{len(rows)}] Embedded row {row_id}")
            except Exception as e:
                print(f"  ! Failed to embed row {row_id}: {e}", file=sys.stderr)
                failed_ids.append(row_id)
                conn.rollback()  # clear the failed transaction before continuing
                continue

            if computed % BATCH_COMMIT_EVERY == 0:
                conn.commit()

        conn.commit()  # final commit for any remainder
        print(f"\nComputed and stored {computed} embedding(s)")

        if failed_ids:
            print(f"\n{len(failed_ids)} row(s) failed and are still NULL:", file=sys.stderr)
            for rid in failed_ids:
                print(f"  - id {rid}", file=sys.stderr)
            print("They're still NULL, so the next run will retry them.", file=sys.stderr)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
