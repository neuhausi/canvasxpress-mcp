#!/usr/bin/env python3
"""
build_index.py
==============
One-time script to embed all few-shot examples and store them in a
sqlite-vec vector database for fast semantic retrieval.

Run this whenever you add or change few_shot_examples.json:
    python build_index.py

Options (all optional; with none, behaviour is unchanged):
    --examples PATH   input examples JSON  (default: data/few_shot_examples.json)
    --output PATH     output sqlite-vec DB (default: data/embeddings.db)
    --model NAME      embedding model      (default: $EMBEDDING_MODEL, else all-MiniLM-L6-v2)

Output: data/embeddings.db  (sqlite-vec database). The DB records the model,
dimension, example count and a hash of the examples file in a `meta` table so
the server can warn when the index and its settings disagree.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer


def _connect_with_sqlite_vec(path: str):
    """Open a connection with sqlite_vec loaded.

    Tries the standard sqlite3 path first (works on Linux).  Falls back to
    apsw on macOS / Python builds where enable_load_extension is unavailable.
    """
    try:
        db = sqlite3.connect(path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return db
    except AttributeError:
        db.close()

    import apsw as _apsw
    raw = _apsw.Connection(path)
    raw.enableloadextension(True)
    raw.loadextension(sqlite_vec.loadable_path())
    raw.enableloadextension(False)

    class _Compat:
        """Thin sqlite3-compatible wrapper around an apsw.Connection."""
        def __init__(self, conn):   self._c = conn
        def execute(self, sql, params=()):  return self._c.execute(sql, params)
        def commit(self):           pass    # apsw autocommits
        def close(self):            self._c.close()

    return _Compat(raw)

BASE_DIR = Path(__file__).parent
EXAMPLES_FILE = BASE_DIR / "data" / "few_shot_examples.json"
DB_FILE = BASE_DIR / "data" / "embeddings.db"

# Model: all-MiniLM-L6-v2 — 384 dimensions, ~22MB, fast on CPU
# Upgrade to all-mpnet-base-v2 (768d) for higher accuracy if needed. Read from the
# same EMBEDDING_MODEL setting as the server so the two cannot drift apart.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def serialize(vector: list[float]) -> bytes:
    """Serialize a float list to little-endian bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def main():
    parser = argparse.ArgumentParser(description="Build the sqlite-vec few-shot index.")
    parser.add_argument("--examples", default=str(EXAMPLES_FILE), help="input examples JSON")
    parser.add_argument("--output", default=str(DB_FILE), help="output sqlite-vec DB")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="sentence-transformers model")
    args = parser.parse_args()
    examples_file = Path(args.examples)
    db_file = Path(args.output)
    model_name = args.model

    print(f"Loading examples from {examples_file}...")
    raw = examples_file.read_bytes()
    examples = json.loads(raw)
    corpus_hash = hashlib.sha256(raw).hexdigest()
    print(f"  Found {len(examples)} examples")

    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    print("\nEmbedding descriptions...")
    descriptions = [ex["description"] for ex in examples]
    embeddings = model.encode(descriptions, show_progress_bar=True, normalize_embeddings=True)
    print(f"  Embedded {len(embeddings)} descriptions → {embedding_dim}d vectors")

    print(f"\nBuilding sqlite-vec index at {db_file}...")
    if db_file.exists():
        db_file.unlink()

    db = _connect_with_sqlite_vec(str(db_file))

    # Metadata table — stores the full example data
    db.execute("""
        CREATE TABLE examples (
            id      INTEGER PRIMARY KEY,
            type    TEXT,
            description TEXT,
            config  TEXT
        )
    """)

    # Vector table — stores the embeddings
    db.execute(f"""
        CREATE VIRTUAL TABLE vec_examples USING vec0(
            embedding float[{embedding_dim}]
        )
    """)

    for i, (ex, emb) in enumerate(zip(examples, embeddings)):
        db.execute(
            "INSERT INTO examples(id, type, description, config) VALUES (?, ?, ?, ?)",
            (i + 1, ex.get("type", ""), ex["description"], json.dumps(ex["config"]))
        )
        db.execute(
            "INSERT INTO vec_examples(rowid, embedding) VALUES (?, ?)",
            (i + 1, serialize(emb.tolist()))
        )

    # Provenance, so the server can warn when the index and its settings disagree.
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    for key, value in (
        ("model", model_name),
        ("dim", str(embedding_dim)),
        ("count", str(len(examples))),
        ("corpus_sha256", corpus_hash),
    ):
        db.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (key, value))

    db.commit()
    db.close()

    size_kb = db_file.stat().st_size // 1024
    print(f"\n✅ Index built: {db_file} ({size_kb} KB)")
    print(f"   {len(examples)} examples indexed with {embedding_dim}-dim embeddings")
    print("\nYou can now start the server: python src/server.py")


if __name__ == "__main__":
    main()
