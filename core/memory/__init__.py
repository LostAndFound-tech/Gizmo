"""
core/memory

Gizmo's memory system.

  store.py    — file and SQLite operations
  embedder.py — local sentence-transformers embedding
  encoder.py  — async post-conversation encoding pass
  retriever.py — pull-based retrieval for response context (next)
"""

from core.memory.store     import memory_store
from core.memory.embedder  import embedder
from core.memory.encoder   import memory_encoder, build_transcript, write_daily_summary
from core.memory.retriever import memory_retriever, MemoryContext

__all__ = [
    "memory_store",
    "embedder",
    "memory_encoder",
    "memory_retriever",
    "MemoryContext",
    "build_transcript",
    "write_daily_summary",
]
