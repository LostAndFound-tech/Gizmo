"""
core/memory

Gizmo's memory system.

  store.py    — file and SQLite operations
  embedder.py — local sentence-transformers embedding
  encoder.py  — async post-conversation encoding pass
  retriever.py — pull-based retrieval for response context (next)
"""

from core.memory.store           import memory_store
from core.memory.embedder        import embedder
from core.memory.encoder         import memory_encoder, build_transcript, write_daily_summary
from core.memory.retriever       import memory_retriever, MemoryContext
from core.memory.session_context import session_context_manager, SessionContext
from core.memory.message         import Message, MessageSummary, Scene, SceneCharacter, scene_extractor
from core.memory.curiosity       import curiosity_engine
from core.memory.psychology      import psychology_engine, load_psychology_for_retrieval

__all__ = [
    "memory_store",
    "embedder",
    "memory_encoder",
    "memory_retriever",
    "session_context_manager",
    "scene_extractor",
    "curiosity_engine",
    "psychology_engine",
    "load_psychology_for_retrieval",
    "MemoryContext",
    "SessionContext",
    "Message",
    "MessageSummary",
    "Scene",
    "SceneCharacter",
    "build_transcript",
    "write_daily_summary",
]
