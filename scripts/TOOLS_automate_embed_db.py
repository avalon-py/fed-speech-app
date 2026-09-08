#!/usr/bin/env python3
"""
Fed Speech Embedder — Incremental Updater (Postgres)
======================================================

Finds rows in fed_speech with a NULL embedding and fills them in using
the local ONNX FinBERT model. Only touches rows where embedding IS
NULL, so it's safe to run repeatedly / on a schedule.

Requires the DB_URL environment variable (Supabase pooler string).
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import psycopg2
from pgvector.psycopg2 import register_vector
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_URL = os.environ["DB_URL"]
MODEL_DIR = os.environ.get("MODEL_DIR", "models/finbert-onnx")
BATCH_COMMIT_EVERY = 20
MAX_LENGTH = 512
STRIDE = 50

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_tokenizer = None
_session = None


def get_finbert_onnx(model_dir=MODEL_DIR):
    global _tokenizer, _session
    if _session is None:
        print(f"Loading ONNX FinBERT from {model_dir} ...")
        _tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        _session = ort.InferenceSession(
            f"{model_dir}/model_fp16.onnx",
            providers=["CPUExecutionProvider"],
        )
    return _tokenizer, _session


def _chunk_ids(ids: list[int], max_length: int, stride: int) -> list[list[int]]:
    """Split content token ids into overlapping windows. Each window
    leaves room for a [CLS] and [SEP] to be added around it."""
    window = max_length - 2
    if len(ids) <= window:
        return [ids]

    chunks = []
    step = window - stride
    start = 0
    while start < len(ids):
        chunks.append(ids[start:start + window])
        if start + window >= len(ids):
            break
        start += step
    return chunks


def speech_embedding(text: str, model_dir=MODEL_DIR, max_length=MAX_LENGTH, stride=STRIDE) -> np.ndarray:
    tokenizer, session = get_finbert_onnx(model_dir)

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0

    base_ids = tokenizer.encode(text, add_special_tokens=False).ids
    chunks = _chunk_ids(base_ids, max_length=max_length, stride=stride)

    input_ids, attention_mask = [], []
    for chunk in chunks:
        ids = [cls_id] + chunk + [sep_id]
        mask = [1] * len(ids)
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids += [pad_id] * pad_len
            mask += [0] * pad_len
        input_ids.append(ids)
        attention_mask.append(mask)

    input_ids = np.array(input_ids, dtype=np.int64)
    attention_mask = np.array(attention_mask, dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

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

        get_finbert_onnx()  # load once, fail fast if the model itself is broken

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
                conn.rollback()
                continue

            if computed % BATCH_COMMIT_EVERY == 0:
                conn.commit()

        conn.commit()
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
