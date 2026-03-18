#!/usr/bin/env python3
"""
build_index.py
==============
One-time script to embed all few-shot examples and store them in a
sqlite-vec vector database for fast semantic retrieval.

Run this whenever you add or change few_shot_examples.json:
    python build_index.py

Output: data/embeddings.db  (sqlite-vec database)
"""

import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
EXAMPLES_FILE = BASE_DIR / "data" / "few_shot_examples.json"
DB_FILE = BASE_DIR / "data" / "embeddings.db"

# Model: all-MiniLM-L6-v2 — 384 dimensions, ~22MB, fast on CPU
# Upgrade to all-mpnet-base-v2 (768d) for higher accuracy if needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def serialize(vector: list[float]) -> bytes:
    """Serialize a float list to little-endian bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def main():
    print(f"Loading examples from {EXAMPLES_FILE}...")
    with open(EXAMPLES_FILE) as f:
        examples = json.load(f)
    print(f"  Found {len(examples)} examples")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nEmbedding descriptions...")
    descriptions = [ex["description"] for ex in examples]
    embeddings = model.encode(descriptions, show_progress_bar=True, normalize_embeddings=True)
    print(f"  Embedded {len(embeddings)} descriptions → {EMBEDDING_DIM}d vectors")

    print(f"\nBuilding sqlite-vec index at {DB_FILE}...")
    if DB_FILE.exists():
        DB_FILE.unlink()

    db = sqlite3.connect(str(DB_FILE))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

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
            embedding float[{EMBEDDING_DIM}]
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

    db.commit()
    db.close()

    size_kb = DB_FILE.stat().st_size // 1024
    print(f"\n✅ Index built: {DB_FILE} ({size_kb} KB)")
    print(f"   {len(examples)} examples indexed with {EMBEDDING_DIM}-dim embeddings")
    print("\nYou can now start the server: python src/server.py")


if __name__ == "__main__":
    main()
