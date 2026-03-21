"""
core/entity_store.py
SQLite-backed entity graph. Everything that exists gets an entry.

Tables:
  entities        — every known thing: headmates, places, objects, concepts
  attributes      — growing key/value facts about an entity
  relations       — directed edges between entities with typed relationships
  memories        — dated events with entity refs, tags, emotional weights
  emotional_weights — per-memory emotion dimensions (open vocabulary, weighted)
  terms           — system-specific vocabulary with origin and definition

Design:
  - Every entity has a UUID. Relations reference UUIDs, not names.
  - Relation types are either free strings OR a term UUID + display label.
  - ChromaDB documents carry entity_uuid in metadata for fast lookup bridging.
  - Recent memories = 14-day window, PLUS anything with significance > 0.7 (kept permanently).
  - Emotional weights are open-vocabulary: the LLM deduces them from context.
    Multiple emotions per memory, each with its own 0.0-1.0 weight.
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = os.getenv("ENTITY_DB_PATH", "/data/entity_store.db")


# ── Connection management ─────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ── Schema init ───────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            uuid        TEXT PRIMARY KEY,
            type        TEXT NOT NULL,      -- headmate | place | object | concept | event | unknown
            name        TEXT NOT NULL,
            owner_uuid  TEXT,               -- for possessions: who owns this
            subtype     TEXT,               -- headspace | local | internal_object | etc.
            created     TEXT NOT NULL,
            updated     TEXT NOT NULL,
            notes       TEXT                -- freeform, for things that don't fit elsewhere
        );

        CREATE TABLE IF NOT EXISTS attributes (
            uuid        TEXT PRIMARY KEY,
            entity_uuid TEXT NOT NULL REFERENCES entities(uuid) ON DELETE CASCADE,
            key         TEXT NOT NULL,      -- e.g. "description", "location", "significance"
            value       TEXT NOT NULL,
            value_type  TEXT DEFAULT 'str', -- str | json | uuid_ref | number
            created     TEXT NOT NULL,
            updated     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relations (
            uuid            TEXT PRIMARY KEY,
            from_uuid       TEXT NOT NULL REFERENCES entities(uuid) ON DELETE CASCADE,
            to_uuid         TEXT NOT NULL REFERENCES entities(uuid) ON DELETE CASCADE,
            relation_type   TEXT NOT NULL,  -- display string e.g. "antsas", "owner", "has_space"
            relation_term_uuid TEXT,        -- UUID of the term entry if this is system vocabulary
            weight          REAL DEFAULT 1.0,
            created         TEXT NOT NULL,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS memories (
            uuid            TEXT PRIMARY KEY,
            owner_uuid      TEXT NOT NULL REFERENCES entities(uuid) ON DELETE CASCADE,
            description     TEXT NOT NULL,
            tags            TEXT NOT NULL DEFAULT '[]',   -- JSON array of tag strings
            entity_uuids    TEXT NOT NULL DEFAULT '[]',  -- JSON array of all involved entity UUIDs
            significance    REAL DEFAULT 0.5,            -- 0.0-1.0; > 0.7 = kept permanently
            session_id      TEXT,
            occurred_at     TEXT NOT NULL,
            created         TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS emotional_weights (
            uuid        TEXT PRIMARY KEY,
            memory_uuid TEXT NOT NULL REFERENCES memories(uuid) ON DELETE CASCADE,
            emotion     TEXT NOT NULL,      -- open vocabulary: "joy", "stress", "nostalgia", etc.
            weight      REAL NOT NULL       -- 0.0-1.0
        );

        CREATE TABLE IF NOT EXISTS terms (
            uuid        TEXT PRIMARY KEY,
            term        TEXT NOT NULL UNIQUE,
            definition  TEXT NOT NULL,
            origin      TEXT,               -- who coined it (entity UUID or name)
            example     TEXT,               -- usage example
            created     TEXT NOT NULL,
            updated     TEXT NOT NULL
        );

        -- Indexes for fast lookup
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
        CREATE INDEX IF NOT EXISTS idx_attributes_entity ON attributes(entity_uuid);
        CREATE INDEX IF NOT EXISTS idx_attributes_key ON attributes(key);
        CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_uuid);
        CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_uuid);
        CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_uuid);
        CREATE INDEX IF NOT EXISTS idx_memories_occurred ON memories(occurred_at);
        CREATE INDEX IF NOT EXISTS idx_memories_significance ON memories(significance);
        CREATE INDEX IF NOT EXISTS idx_emotional_weights_memory ON emotional_weights(memory_uuid);
        CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term);
        """)
    print(f"[EntityStore] DB initialized at {DB_PATH}")


# ── Entity CRUD ───────────────────────────────────────────────────────────────

def upsert_entity(
    name: str,
    entity_type: str,
    owner_uuid: Optional[str] = None,
    subtype: Optional[str] = None,
    notes: Optional[str] = None,
    entity_uuid: Optional[str] = None,
) -> str:
    """
    Create or update an entity. Returns the UUID.
    If entity_uuid is provided, updates that record.
    Otherwise searches by name+type — creates if not found.
    """
    now = _now_iso()

    with _conn() as con:
        if entity_uuid:
            existing = con.execute(
                "SELECT uuid FROM entities WHERE uuid = ?", (entity_uuid,)
            ).fetchone()
            if existing:
                con.execute(
                    "UPDATE entities SET name=?, type=?, owner_uuid=?, subtype=?, notes=?, updated=? WHERE uuid=?",
                    (name, entity_type, owner_uuid, subtype, notes, now, entity_uuid)
                )
                return entity_uuid

        # Search by name + type
        existing = con.execute(
            "SELECT uuid FROM entities WHERE name = ? AND type = ?",
            (name, entity_type)
        ).fetchone()

        if existing:
            eid = existing["uuid"]
            con.execute(
                "UPDATE entities SET owner_uuid=?, subtype=?, notes=?, updated=? WHERE uuid=?",
                (owner_uuid, subtype, notes, now, eid)
            )
            return eid

        # New entity
        eid = entity_uuid or _new_uuid()
        con.execute(
            "INSERT INTO entities (uuid, type, name, owner_uuid, subtype, created, updated, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (eid, entity_type, name, owner_uuid, subtype, now, now, notes)
        )
        print(f"[EntityStore] New entity: '{name}' ({entity_type}) → {eid[:8]}")
        return eid


def get_entity(name: str, entity_type: Optional[str] = None) -> Optional[dict]:
    """Look up an entity by name, optionally filtered by type."""
    with _conn() as con:
        if entity_type:
            row = con.execute(
                "SELECT * FROM entities WHERE name = ? AND type = ?",
                (name, entity_type)
            ).fetchone()
        else:
            row = con.execute(
                "SELECT * FROM entities WHERE name = ?", (name,)
            ).fetchone()
        return dict(row) if row else None


def get_entity_by_uuid(entity_uuid: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM entities WHERE uuid = ?", (entity_uuid,)
        ).fetchone()
        return dict(row) if row else None


def get_all_entities(entity_type: Optional[str] = None) -> list[dict]:
    with _conn() as con:
        if entity_type:
            rows = con.execute(
                "SELECT * FROM entities WHERE type = ? ORDER BY name", (entity_type,)
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM entities ORDER BY type, name").fetchall()
        return [dict(r) for r in rows]


# ── Attributes ────────────────────────────────────────────────────────────────

def set_attribute(entity_uuid: str, key: str, value: str, value_type: str = "str") -> str:
    """Set or update an attribute on an entity. Returns attribute UUID."""
    now = _now_iso()
    with _conn() as con:
        existing = con.execute(
            "SELECT uuid FROM attributes WHERE entity_uuid = ? AND key = ?",
            (entity_uuid, key)
        ).fetchone()

        if existing:
            con.execute(
                "UPDATE attributes SET value=?, value_type=?, updated=? WHERE uuid=?",
                (value, value_type, now, existing["uuid"])
            )
            return existing["uuid"]

        attr_uuid = _new_uuid()
        con.execute(
            "INSERT INTO attributes (uuid, entity_uuid, key, value, value_type, created, updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (attr_uuid, entity_uuid, key, value, value_type, now, now)
        )
        return attr_uuid


def get_attributes(entity_uuid: str) -> dict:
    """Return all attributes for an entity as a {key: value} dict."""
    with _conn() as con:
        rows = con.execute(
            "SELECT key, value, value_type FROM attributes WHERE entity_uuid = ?",
            (entity_uuid,)
        ).fetchall()
        result = {}
        for row in rows:
            val = row["value"]
            if row["value_type"] == "json":
                try:
                    val = json.loads(val)
                except Exception:
                    pass
            result[row["key"]] = val
        return result


# ── Relations ─────────────────────────────────────────────────────────────────

def add_relation(
    from_uuid: str,
    to_uuid: str,
    relation_type: str,
    relation_term_uuid: Optional[str] = None,
    weight: float = 1.0,
    notes: Optional[str] = None,
) -> str:
    """
    Add a directed relation between two entities.
    If the same from→to→type relation exists, updates weight.
    Returns relation UUID.
    """
    now = _now_iso()
    with _conn() as con:
        existing = con.execute(
            "SELECT uuid FROM relations WHERE from_uuid=? AND to_uuid=? AND relation_type=?",
            (from_uuid, to_uuid, relation_type)
        ).fetchone()

        if existing:
            con.execute(
                "UPDATE relations SET weight=?, notes=? WHERE uuid=?",
                (weight, notes, existing["uuid"])
            )
            return existing["uuid"]

        rel_uuid = _new_uuid()
        con.execute(
            "INSERT INTO relations (uuid, from_uuid, to_uuid, relation_type, relation_term_uuid, weight, created, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rel_uuid, from_uuid, to_uuid, relation_type, relation_term_uuid, weight, now, notes)
        )
        return rel_uuid


def get_relations(
    entity_uuid: str,
    direction: str = "both",   # "from" | "to" | "both"
    relation_type: Optional[str] = None,
) -> list[dict]:
    """Get all relations for an entity."""
    with _conn() as con:
        if direction == "from":
            query = "SELECT * FROM relations WHERE from_uuid = ?"
            params = (entity_uuid,)
        elif direction == "to":
            query = "SELECT * FROM relations WHERE to_uuid = ?"
            params = (entity_uuid,)
        else:
            query = "SELECT * FROM relations WHERE from_uuid = ? OR to_uuid = ?"
            params = (entity_uuid, entity_uuid)

        if relation_type:
            query += " AND relation_type = ?"
            params = params + (relation_type,)

        rows = con.execute(query, params).fetchall()
        return [dict(r) for r in rows]


# ── Memories ──────────────────────────────────────────────────────────────────

def add_memory(
    owner_uuid: str,
    description: str,
    tags: list[str],
    entity_uuids: list[str],
    emotions: dict[str, float],   # {"joy": 0.8, "stress": 0.3}
    significance: float = 0.5,
    session_id: str = "",
    occurred_at: Optional[str] = None,
) -> str:
    """
    Add a memory to an entity's record.
    emotions: open-vocabulary dict of {emotion_name: weight}
    Returns memory UUID.
    """
    now = _now_iso()
    occurred_at = occurred_at or now
    mem_uuid = _new_uuid()

    with _conn() as con:
        con.execute(
            "INSERT INTO memories (uuid, owner_uuid, description, tags, entity_uuids, "
            "significance, session_id, occurred_at, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mem_uuid, owner_uuid, description, json.dumps(tags),
             json.dumps(entity_uuids), significance, session_id, occurred_at, now)
        )

        for emotion, weight in emotions.items():
            con.execute(
                "INSERT INTO emotional_weights (uuid, memory_uuid, emotion, weight) VALUES (?, ?, ?, ?)",
                (_new_uuid(), mem_uuid, emotion.lower().strip(), float(weight))
            )

    return mem_uuid


def get_memories(
    owner_uuid: str,
    days: int = 14,
    include_significant: bool = True,
    significance_threshold: float = 0.7,
) -> list[dict]:
    """
    Get recent memories for an entity.
    Always includes the 14-day window.
    If include_significant=True, also returns anything above significance_threshold
    regardless of age.
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    with _conn() as con:
        if include_significant:
            rows = con.execute(
                "SELECT * FROM memories WHERE owner_uuid = ? "
                "AND (occurred_at >= ? OR significance >= ?) "
                "ORDER BY occurred_at DESC",
                (owner_uuid, cutoff, significance_threshold)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM memories WHERE owner_uuid = ? AND occurred_at >= ? "
                "ORDER BY occurred_at DESC",
                (owner_uuid, cutoff)
            ).fetchall()

        memories = []
        for row in rows:
            mem = dict(row)
            mem["tags"] = json.loads(mem["tags"])
            mem["entity_uuids"] = json.loads(mem["entity_uuids"])

            # Fetch emotional weights
            weights = con.execute(
                "SELECT emotion, weight FROM emotional_weights WHERE memory_uuid = ?",
                (mem["uuid"],)
            ).fetchall()
            mem["emotions"] = {w["emotion"]: w["weight"] for w in weights}
            memories.append(mem)

        return memories


def get_all_memories_for_entities(entity_uuids: list[str]) -> list[dict]:
    """Get all memories that involve any of the given entity UUIDs."""
    results = []
    with _conn() as con:
        for eid in entity_uuids:
            rows = con.execute(
                "SELECT * FROM memories WHERE entity_uuids LIKE ?",
                (f'%{eid}%',)
            ).fetchall()
            for row in rows:
                mem = dict(row)
                mem["tags"] = json.loads(mem["tags"])
                mem["entity_uuids"] = json.loads(mem["entity_uuids"])
                weights = con.execute(
                    "SELECT emotion, weight FROM emotional_weights WHERE memory_uuid = ?",
                    (mem["uuid"],)
                ).fetchall()
                mem["emotions"] = {w["emotion"]: w["weight"] for w in weights}
                if mem not in results:
                    results.append(mem)
    return results


# ── Terms/glossary ────────────────────────────────────────────────────────────

def upsert_term(
    term: str,
    definition: str,
    origin: Optional[str] = None,
    example: Optional[str] = None,
) -> str:
    """Add or update a system-specific term. Returns term UUID."""
    now = _now_iso()
    with _conn() as con:
        existing = con.execute(
            "SELECT uuid FROM terms WHERE term = ?", (term.lower().strip(),)
        ).fetchone()

        if existing:
            con.execute(
                "UPDATE terms SET definition=?, origin=?, example=?, updated=? WHERE uuid=?",
                (definition, origin, example, now, existing["uuid"])
            )
            return existing["uuid"]

        term_uuid = _new_uuid()
        con.execute(
            "INSERT INTO terms (uuid, term, definition, origin, example, created, updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (term_uuid, term.lower().strip(), definition, origin, example, now, now)
        )
        print(f"[EntityStore] New term: '{term}' → {term_uuid[:8]}")
        return term_uuid


def get_term(term: str) -> Optional[dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM terms WHERE term = ?", (term.lower().strip(),)
        ).fetchone()
        return dict(row) if row else None


def get_all_terms() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM terms ORDER BY term").fetchall()
        return [dict(r) for r in rows]


# ── Rich entity profile ───────────────────────────────────────────────────────

def get_entity_profile(name: str, entity_type: Optional[str] = None) -> Optional[dict]:
    """
    Return a full profile for an entity — entity record, all attributes,
    all relations (with resolved names), and recent memories.
    Used by synthesis to build context blocks.
    """
    entity = get_entity(name, entity_type)
    if not entity:
        return None

    eid = entity["uuid"]
    attrs = get_attributes(eid)
    relations = get_relations(eid)
    memories = get_memories(eid)

    # Resolve relation targets to names
    resolved_relations = []
    with _conn() as con:
        for rel in relations:
            other_uuid = rel["to_uuid"] if rel["from_uuid"] == eid else rel["from_uuid"]
            other = con.execute(
                "SELECT name, type FROM entities WHERE uuid = ?", (other_uuid,)
            ).fetchone()
            resolved_relations.append({
                "relation_type":  rel["relation_type"],
                "direction":      "to" if rel["from_uuid"] == eid else "from",
                "other_name":     other["name"] if other else other_uuid,
                "other_type":     other["type"] if other else "unknown",
                "other_uuid":     other_uuid,
                "weight":         rel["weight"],
                "notes":          rel["notes"],
            })

    return {
        "entity":     entity,
        "attributes": attrs,
        "relations":  resolved_relations,
        "memories":   memories,
    }


def get_headmate_profile(name: str) -> Optional[dict]:
    """Convenience wrapper for headmate entity profiles."""
    return get_entity_profile(name, entity_type="headmate")


# ── ChromaDB bridge ───────────────────────────────────────────────────────────

def get_uuid_for_chroma(name: str, entity_type: str) -> str:
    """
    Get or create an entity UUID for tagging ChromaDB documents.
    Ensures ChromaDB metadata can always reference back to the entity store.
    """
    entity = get_entity(name, entity_type)
    if entity:
        return entity["uuid"]
    return upsert_entity(name=name, entity_type=entity_type)


# ── Wipe (for factory reset) ──────────────────────────────────────────────────

def wipe_all() -> None:
    """Delete all data from all tables. Used by factory reset."""
    with _conn() as con:
        con.executescript("""
        DELETE FROM emotional_weights;
        DELETE FROM memories;
        DELETE FROM relations;
        DELETE FROM attributes;
        DELETE FROM terms;
        DELETE FROM entities;
        """)
    print("[EntityStore] Full wipe complete")
