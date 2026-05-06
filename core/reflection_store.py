"""
core/reflection_store.py
ChromaDB-backed tensor for Gizmo's reflections — both surfaced and unsaid.

Each entry is a reflection event:
  - What was noticed
  - Who it was about
  - The emotional context when it was noticed
  - Whether it was surfaced or held
  - If held: what it would have said, and when it expires

Unsaid reflections are queryable by Mind — so when a topic comes up again,
the pending thread surfaces naturally rather than being lost.

Schema per entry:
  text:         what Gizmo noticed / would have said
  headmate:     who it's about (or "session" for whole-session observations)
  topic:        primary topic
  valence:      float
  intensity:    float
  chaos:        float
  surfaced:     "true" | "false"
  held_since:   ISO timestamp (if unsaid)
  session_id:   which session
  expires:      ISO timestamp or "never"
  outcome:      "surfaced" | "held" | "expired" | "pending"
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Optional

from core.log import log, log_event, log_error

REFLECTION_COLLECTION = "gizmo_reflections"

# How long an unsaid reflection stays alive before expiring (days)
_DEFAULT_EXPIRY_DAYS = 7


def _store():
    from core.rag import RAGStore
    return RAGStore(collection_name=REFLECTION_COLLECTION)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _expiry_iso(days: int = _DEFAULT_EXPIRY_DAYS) -> str:
    return (datetime.now() + timedelta(days=days)).isoformat(timespec="seconds")


def _is_expired(expiry_iso: str) -> bool:
    if not expiry_iso or expiry_iso == "never":
        return False
    try:
        return datetime.now() > datetime.fromisoformat(expiry_iso)
    except Exception:
        return False


# ── Write ─────────────────────────────────────────────────────────────────────

def store_reflection(
    text: str,
    headmate: Optional[str],
    topic: str,
    valence: float,
    intensity: float,
    chaos: float,
    session_id: str,
    surfaced: bool = False,
    expiry_days: int = _DEFAULT_EXPIRY_DAYS,
) -> str:
    """
    Store a reflection. Returns the ID.
    surfaced=True means Gizmo said it. False means it was held.
    """
    now = _now_iso()
    reflection_id = f"ref_{uuid.uuid4().hex[:12]}"

    try:
        store = _store()
        store.ingest_texts(
            [text],
            metadatas=[{
                "headmate":   (headmate or "session").lower(),
                "topic":      topic,
                "valence":    str(round(valence, 3)),
                "intensity":  str(round(intensity, 3)),
                "chaos":      str(round(chaos, 3)),
                "surfaced":   "true" if surfaced else "false",
                "held_since": now,
                "session_id": session_id,
                "expires":    _expiry_iso(expiry_days) if not surfaced else "never",
                "outcome":    "surfaced" if surfaced else "pending",
                "timestamp":  now,
            }],
            ids=[reflection_id],
        )
        log_event("ReflectionStore", "STORED",
            id=reflection_id[:8],
            headmate=headmate or "session",
            surfaced=surfaced,
            topic=topic,
            preview=text[:60],
        )
        return reflection_id
    except Exception as e:
        log_error("ReflectionStore", "store failed", exc=e)
        return ""


def mark_surfaced(reflection_id: str) -> None:
    """Mark a held reflection as surfaced."""
    try:
        store = _store()
        store.collection.update(
            ids=[reflection_id],
            metadatas=[{
                "surfaced": "true",
                "outcome":  "surfaced",
                "expires":  "never",
            }],
        )
        log_event("ReflectionStore", "MARKED_SURFACED", id=reflection_id[:8])
    except Exception as e:
        log_error("ReflectionStore", "mark_surfaced failed", exc=e)


def expire_old() -> int:
    """
    Scan for expired unsaid reflections and mark them expired.
    Returns count expired.
    Called periodically by the archiver.
    """
    try:
        store = _store()
        if store.count == 0:
            return 0

        result = store.collection.get(
            where={"outcome": {"$eq": "pending"}}
        )

        expired_ids = []
        for doc_id, meta in zip(result["ids"], result["metadatas"]):
            if _is_expired(meta.get("expires", "")):
                expired_ids.append(doc_id)

        if expired_ids:
            for doc_id in expired_ids:
                store.collection.update(
                    ids=[doc_id],
                    metadatas=[{"outcome": "expired", "surfaced": "false"}],
                )
            log_event("ReflectionStore", "EXPIRED", count=len(expired_ids))

        return len(expired_ids)
    except Exception as e:
        log_error("ReflectionStore", "expire_old failed", exc=e)
        return 0


# ── Query ─────────────────────────────────────────────────────────────────────

def get_pending(
    headmate: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """
    Get pending (unsaid) reflections, optionally filtered by headmate/topic.
    Used by Mind to surface old threads when relevant.
    """
    try:
        store = _store()
        if store.count == 0:
            return []

        where = {"outcome": {"$eq": "pending"}}

        result = store.collection.get(
            where=where,
            limit=limit * 3,  # fetch more to filter
        )

        reflections = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            if _is_expired(meta.get("expires", "")):
                continue
            if headmate and meta.get("headmate", "") != headmate.lower():
                continue
            if topic and topic.lower() not in meta.get("topic", "").lower():
                continue
            reflections.append({
                "text":      doc,
                "headmate":  meta.get("headmate"),
                "topic":     meta.get("topic"),
                "valence":   float(meta.get("valence", 0)),
                "intensity": float(meta.get("intensity", 0)),
                "chaos":     float(meta.get("chaos", 0)),
                "held_since": meta.get("held_since"),
                "id":        result["ids"][len(reflections)],
            })
            if len(reflections) >= limit:
                break

        return reflections
    except Exception as e:
        log_error("ReflectionStore", "get_pending failed", exc=e)
        return []


def query_relevant(
    query: str,
    headmate: Optional[str] = None,
    n_results: int = 3,
    include_surfaced: bool = False,
) -> list[dict]:
    """
    Semantic search over reflections — used by Mind when a topic comes up.
    Returns pending reflections relevant to the current query.
    """
    try:
        store = _store()
        if store.count == 0:
            return []

        where = None
        if not include_surfaced:
            where = {"outcome": {"$eq": "pending"}}
        if headmate:
            hm_filter = {"headmate": {"$eq": headmate.lower()}}
            where = {"$and": [where, hm_filter]} if where else hm_filter

        chunks = store.retrieve(query=query, n_results=n_results, where=where)

        results = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            if not include_surfaced and meta.get("outcome") != "pending":
                continue
            if _is_expired(meta.get("expires", "")):
                continue
            results.append({
                "text":      chunk["text"],
                "headmate":  meta.get("headmate"),
                "topic":     meta.get("topic"),
                "distance":  chunk.get("distance", 1.0),
                "held_since": meta.get("held_since"),
                "id":        None,  # not returned by retrieve()
            })

        return results
    except Exception as e:
        log_error("ReflectionStore", "query_relevant failed", exc=e)
        return []
