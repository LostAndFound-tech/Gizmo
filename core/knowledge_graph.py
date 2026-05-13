"""
core/knowledge_graph.py

SQLite knowledge graph for Gizmo.

Every piece of knowledge is a typed edge:
  subject ──[predicate]──▶ object

Examples:
  jess    ──[curious_about]──▶ gizmo.appearance
  jess    ──[feels]──▶        safe  (context: with_gizmo)
  oren    ──[protects]──▶     body  (trigger: overwhelm)
  gizmo   ──[feels]──▶        warm  (context: jess_present)
  princess──[struggles_with]──▶ transitions

This makes knowledge queryable in both directions, by relationship type,
by entity, by topic, and by recency — without relying on vector similarity
to find structured facts.

ChromaDB stays for semantic search. This handles structured relationships.

Schema:
  entities (id, type, label)
  knowledge (
    id, subject, predicate, object, object_type,
    strength, confidence, source, session_id,
    timestamp, context, tags
  )

Strength decays over time but gets reinforced on repeated observation.
Confidence reflects how certain we are (told=high, inferred=medium, guessed=low).
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

_DB_PATH = Path(os.getenv("DATA_DIR", "/data")) / "knowledge_graph.db"

# ── Predicate vocabulary ──────────────────────────────────────────────────────
# Controlled vocabulary keeps queries consistent.
# Gizmo should use these when possible, but freetext is allowed.

PREDICATES = {
    # Emotional
    "feels",            # jess feels safe
    "fears",            # princess fears abandonment
    "wants",            # kaylee wants more processing time
    "enjoys",           # jess enjoys directness
    "dislikes",         # oren dislikes overstimulation
    "struggles_with",   # princess struggles_with transitions

    # Relational
    "trusts",           # jess trusts gizmo
    "loves",            # jess loves daddy dynamic
    "protects",         # oren protects body
    "cares_about",      # gizmo cares_about jess
    "curious_about",    # jess curious_about gizmo appearance

    # Behavioral
    "tends_to",         # jess tends_to arrive high energy
    "responds_to",      # princess responds_to gentleness
    "triggered_by",     # oren triggered_by overwhelm
    "calmed_by",        # kaylee calmed_by structure
    "avoids",           # princess avoids conflict

    # Identity
    "is",               # oren is protector
    "identifies_as",    # x61 identifies_as machine
    "values",           # jess values honesty

    # Knowledge
    "knows",            # gizmo knows jess preference
    "remembers",        # gizmo remembers jess birthday
    "believes",         # jess believes gizmo is real

    # Patterns
    "pattern",          # general observed pattern
    "taught",           # jess taught gizmo something
}


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id    TEXT PRIMARY KEY,
                type  TEXT NOT NULL,
                label TEXT
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id          TEXT PRIMARY KEY,
                subject     TEXT NOT NULL,
                predicate   TEXT NOT NULL,
                object      TEXT NOT NULL,
                object_type TEXT DEFAULT 'concept',
                strength    REAL DEFAULT 0.5,
                confidence  REAL DEFAULT 0.5,
                source      TEXT DEFAULT 'observed',
                session_id  TEXT,
                timestamp   TEXT NOT NULL,
                context     TEXT,
                tags        TEXT
            )
        """)

        con.execute("CREATE INDEX IF NOT EXISTS idx_kg_subject   ON knowledge (subject)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_kg_predicate ON knowledge (predicate)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_kg_object    ON knowledge (object)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_kg_timestamp ON knowledge (timestamp)")

    log("KnowledgeGraph", f"initialized at {_DB_PATH}")


# ── Entity management ─────────────────────────────────────────────────────────

def ensure_entity(entity_id: str, entity_type: str = "headmate", label: str = "") -> None:
    """Ensure an entity exists in the graph."""
    try:
        with _conn() as con:
            con.execute("""
                INSERT OR IGNORE INTO entities (id, type, label)
                VALUES (?, ?, ?)
            """, (entity_id.lower(), entity_type, label or entity_id))
    except Exception as e:
        log_error("KnowledgeGraph", f"ensure_entity failed for {entity_id}", exc=e)


def get_known_entities() -> list[dict]:
    try:
        with _conn() as con:
            rows = con.execute("SELECT * FROM entities").fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log_error("KnowledgeGraph", "get_known_entities failed", exc=e)
        return []


# ── Edge write ────────────────────────────────────────────────────────────────

def _edge_id(subject: str, predicate: str, object_: str) -> str:
    """Deterministic edge id — same edge always has same id."""
    import hashlib
    key = f"{subject.lower()}|{predicate.lower()}|{object_.lower()}"
    return "kg_" + hashlib.md5(key.encode()).hexdigest()[:16]


def add_edge(
    subject: str,
    predicate: str,
    object_: str,
    object_type: str = "concept",
    strength: float = 0.5,
    confidence: float = 0.5,
    source: str = "observed",
    session_id: str = "",
    context: Optional[dict] = None,
    tags: Optional[list] = None,
) -> str:
    """
    Add or reinforce a knowledge edge.
    If the edge already exists, strength is reinforced (averaged up).
    Returns the edge id.
    """
    edge_id   = _edge_id(subject, predicate, object_)
    ts_str    = datetime.now().isoformat(timespec="seconds")
    ctx_str   = json.dumps(context or {})
    tags_str  = json.dumps(tags or [])

    try:
        with _conn() as con:
            existing = con.execute(
                "SELECT strength, confidence FROM knowledge WHERE id = ?",
                (edge_id,)
            ).fetchone()

            if existing:
                # Reinforce — nudge strength up, keep higher confidence
                new_strength   = min(1.0, (existing["strength"] + strength) / 2 + 0.1)
                new_confidence = max(existing["confidence"], confidence)
                con.execute("""
                    UPDATE knowledge
                    SET strength = ?, confidence = ?, timestamp = ?, session_id = ?
                    WHERE id = ?
                """, (new_strength, new_confidence, ts_str, session_id, edge_id))
            else:
                con.execute("""
                    INSERT INTO knowledge
                    (id, subject, predicate, object, object_type,
                     strength, confidence, source, session_id,
                     timestamp, context, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    edge_id,
                    subject.lower(),
                    predicate.lower(),
                    object_.lower(),
                    object_type,
                    strength,
                    confidence,
                    source,
                    session_id,
                    ts_str,
                    ctx_str,
                    tags_str,
                ))

        log_event("KnowledgeGraph", "EDGE_ADDED",
            subject=subject, predicate=predicate, object=object_,
            source=source, strength=round(strength, 2))
        return edge_id

    except Exception as e:
        log_error("KnowledgeGraph", "add_edge failed", exc=e)
        return edge_id


def add_edges(edges: list[dict], session_id: str = "") -> int:
    """Add multiple edges at once. Each dict matches add_edge kwargs."""
    count = 0
    for edge in edges:
        try:
            add_edge(session_id=session_id, **edge)
            count += 1
        except Exception as e:
            log_error("KnowledgeGraph", f"add_edges item failed: {e}", exc=None)
    return count


# ── Query ─────────────────────────────────────────────────────────────────────

def query_subject(
    subject: str,
    predicate: Optional[str] = None,
    min_strength: float = 0.2,
    limit: int = 20,
) -> list[dict]:
    """All edges where subject matches. Optionally filter by predicate."""
    try:
        clauses = ["subject = ?", "strength >= ?"]
        params  = [subject.lower(), min_strength]
        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate.lower())
        where = "WHERE " + " AND ".join(clauses)
        with _conn() as con:
            rows = con.execute(
                f"SELECT * FROM knowledge {where} ORDER BY strength DESC LIMIT ?",
                params + [limit]
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log_error("KnowledgeGraph", "query_subject failed", exc=e)
        return []


def query_object(
    object_: str,
    predicate: Optional[str] = None,
    min_strength: float = 0.2,
    limit: int = 20,
) -> list[dict]:
    """All edges pointing TO an object. Who feels X? Who trusts Y?"""
    try:
        clauses = ["object LIKE ?", "strength >= ?"]
        params  = [f"%{object_.lower()}%", min_strength]
        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate.lower())
        where = "WHERE " + " AND ".join(clauses)
        with _conn() as con:
            rows = con.execute(
                f"SELECT * FROM knowledge {where} ORDER BY strength DESC LIMIT ?",
                params + [limit]
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log_error("KnowledgeGraph", "query_object failed", exc=e)
        return []


def query_predicate(
    predicate: str,
    subject: Optional[str] = None,
    min_strength: float = 0.2,
    limit: int = 20,
) -> list[dict]:
    """All edges of a given type. What does everyone fear? What does Jess enjoy?"""
    try:
        clauses = ["predicate = ?", "strength >= ?"]
        params  = [predicate.lower(), min_strength]
        if subject:
            clauses.append("subject = ?")
            params.append(subject.lower())
        where = "WHERE " + " AND ".join(clauses)
        with _conn() as con:
            rows = con.execute(
                f"SELECT * FROM knowledge {where} ORDER BY strength DESC LIMIT ?",
                params + [limit]
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log_error("KnowledgeGraph", "query_predicate failed", exc=e)
        return []


def query_context(
    subject: str,
    topics: list[str],
    min_strength: float = 0.2,
    limit: int = 15,
) -> list[dict]:
    """
    Context-aware query — find edges for subject that relate to current topics.
    Used by Mind to pull relevant knowledge for the current turn.
    """
    if not topics:
        return query_subject(subject, min_strength=min_strength, limit=limit)

    try:
        clauses  = ["subject = ?", "strength >= ?"]
        params   = [subject.lower(), min_strength]
        topic_clauses = []
        for t in topics:
            topic_clauses.append("(tags LIKE ? OR object LIKE ? OR predicate LIKE ?)")
            params.extend([f'%"{t}"%', f"%{t}%", f"%{t}%"])
        if topic_clauses:
            clauses.append("(" + " OR ".join(topic_clauses) + ")")
        where = "WHERE " + " AND ".join(clauses)
        with _conn() as con:
            rows = con.execute(
                f"SELECT * FROM knowledge {where} ORDER BY strength DESC LIMIT ?",
                params + [limit]
            ).fetchall()
        result = [dict(r) for r in rows]

        # If topic-filtered results are sparse, pad with strongest general edges
        if len(result) < 5:
            general = query_subject(subject, min_strength=min_strength, limit=limit)
            seen_ids = {r["id"] for r in result}
            for row in general:
                if row["id"] not in seen_ids:
                    result.append(row)
                if len(result) >= limit:
                    break

        return result[:limit]
    except Exception as e:
        log_error("KnowledgeGraph", "query_context failed", exc=e)
        return []


def format_for_prompt(edges: list[dict], subject: str = "") -> str:
    """
    Format knowledge edges as a concise block for system prompt injection.
    Groups by predicate for readability.
    """
    if not edges:
        return ""

    by_predicate: dict[str, list[str]] = {}
    for edge in edges:
        pred = edge["predicate"]
        obj  = edge["object"]
        by_predicate.setdefault(pred, []).append(obj)

    name  = subject.title() if subject else "They"
    lines = []
    for pred, objects in by_predicate.items():
        obj_str = ", ".join(objects[:4])
        lines.append(f"  {name} {pred.replace('_', ' ')}: {obj_str}")

    return "\n".join(lines)


# ── Decay ─────────────────────────────────────────────────────────────────────

def decay_edges(days_half_life: float = 60.0) -> int:
    """
    Apply time-based decay to all edges.
    Strength halves every days_half_life days.
    Edges below 0.05 are removed.
    Call periodically (e.g. weekly from session_manager).
    """
    import math
    now = datetime.now()
    try:
        with _conn() as con:
            rows = con.execute(
                "SELECT id, strength, timestamp FROM knowledge"
            ).fetchall()

            updated = 0
            deleted = 0
            for row in rows:
                try:
                    ts  = datetime.fromisoformat(row["timestamp"])
                    age = (now - ts).total_seconds() / 86400
                    new_strength = row["strength"] * math.pow(0.5, age / days_half_life)
                    if new_strength < 0.05:
                        con.execute("DELETE FROM knowledge WHERE id = ?", (row["id"],))
                        deleted += 1
                    else:
                        con.execute(
                            "UPDATE knowledge SET strength = ? WHERE id = ?",
                            (round(new_strength, 4), row["id"])
                        )
                        updated += 1
                except Exception:
                    pass

        log_event("KnowledgeGraph", "DECAY_APPLIED", updated=updated, deleted=deleted)
        return deleted
    except Exception as e:
        log_error("KnowledgeGraph", "decay_edges failed", exc=e)
        return 0


# ── Bootstrap ─────────────────────────────────────────────────────────────────
try:
    init_db()
except Exception as _e:
    log_error("KnowledgeGraph", "failed to initialize", exc=_e)
