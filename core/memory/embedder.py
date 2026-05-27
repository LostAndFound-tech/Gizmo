"""
core/memory/embedder.py

Local embedding generation using sentence-transformers.
Runs on EC2, no external API calls, no cost, ~300MB RAM.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Fast inference (~5ms per text)
  - Good semantic similarity for retrieval

Embeddings are stored as raw float32 bytes in SQLite (BLOB).
Cosine similarity computed in Python — no sqlite-vec required,
though sqlite-vec can be added later for better performance at scale.
"""

from __future__ import annotations

import json
import struct
import time
from typing import Optional

from core.log import log, log_error

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class Embedder:

    def __init__(self):
        self._model     = None
        self._available = None   # None = not yet tried

    def _load(self) -> bool:
        """Lazy-load the model on first use."""
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer
            log("Embedder", f"loading {MODEL_NAME}...")
            t = time.monotonic()
            self._model     = SentenceTransformer(MODEL_NAME)
            self._available = True
            ms = round((time.monotonic() - t) * 1000)
            log("Embedder", f"loaded in {ms}ms")
        except Exception as e:
            log_error("Embedder", f"failed to load model: {e}", exc=None)
            self._available = False
        return self._available

    def embed(self, text: str) -> Optional[bytes]:
        """
        Generate an embedding for a text string.
        Returns raw float32 bytes suitable for SQLite BLOB storage.
        Returns None if model unavailable.
        """
        if not text or not text.strip():
            return None
        if not self._load():
            return None
        try:
            vec = self._model.encode(text, normalize_embeddings=True)
            return _floats_to_bytes(vec.tolist())
        except Exception as e:
            log_error("Embedder", f"embed failed: {e}", exc=None)
            return None

    def embed_batch(self, texts: list[str]) -> list[Optional[bytes]]:
        """Embed multiple texts at once — more efficient than one at a time."""
        if not texts:
            return []
        if not self._load():
            return [None] * len(texts)
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return [_floats_to_bytes(v.tolist()) for v in vecs]
        except Exception as e:
            log_error("Embedder", f"batch embed failed: {e}", exc=None)
            return [None] * len(texts)

    def similarity(self, a: bytes, b: bytes) -> float:
        """Cosine similarity between two stored embeddings."""
        try:
            va = _bytes_to_floats(a)
            vb = _bytes_to_floats(b)
            return _cosine(va, vb)
        except Exception:
            return 0.0

    def search(
        self,
        query_embedding: bytes,
        limit:           int   = 10,
        min_similarity:  float = 0.3,
        headmate:        Optional[str] = None,
        memory_type:     Optional[str] = None,
    ) -> list[dict]:
        """
        Find memories similar to a query embedding.
        Loads all active embeddings from the index and scores them.
        At current scale this is fast enough — revisit with sqlite-vec
        if the index grows beyond ~50k entries.
        """
        from core.memory.store import memory_store

        con    = memory_store._connect()
        wheres = ["active = 1", "embedding IS NOT NULL"]
        params = []

        if headmate:
            wheres.append("(headmate = ? OR headmate IS NULL)")
            params.append(headmate.lower())
        if memory_type:
            wheres.append("memory_type = ?")
            params.append(memory_type)

        sql  = f"SELECT id, file_path, memory_type, memory_subtype, headmate, embedding FROM memory_index WHERE {' AND '.join(wheres)}"
        rows = con.execute(sql, params).fetchall()
        con.close()

        query_vec = _bytes_to_floats(query_embedding)
        scored    = []

        for row in rows:
            if not row["embedding"]:
                continue
            try:
                sim = _cosine(query_vec, _bytes_to_floats(row["embedding"]))
                if sim >= min_similarity:
                    scored.append({
                        "id":            row["id"],
                        "file_path":     row["file_path"],
                        "memory_type":   row["memory_type"],
                        "memory_subtype": row["memory_subtype"],
                        "headmate":      row["headmate"],
                        "similarity":    sim,
                    })
            except Exception:
                continue

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:limit]

    @property
    def available(self) -> bool:
        return self._load()


# ── Float serialisation ───────────────────────────────────────────────────────

def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)

def _bytes_to_floats(b: bytes) -> list[float]:
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))

def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Singleton ─────────────────────────────────────────────────────────────────

embedder = Embedder()
