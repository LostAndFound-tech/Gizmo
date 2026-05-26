"""
core/store.py
Gizmo's unified knowledge store.

Single SQLite database. Single interface. Everything goes through here.

No ChromaDB. No JSON entity files. No flat personality files.
No scattered SQLite databases. One file: /data/gizmo.db

Semantic search: SQLite FTS5 for full-text search.
Embeddings stored as BLOBs for vector similarity re-ranking.
Embedding model: sentence-transformers/all-MiniLM-L6-v2

Tables:
  entities          — headmates, externals, pets
  facts             — discrete facts about anyone/anything
  relationships     — freeform relationship edges (niifta, bestie, etc.)
  messages          — inbound message envelopes
  responses         — outbound response envelopes + reasoning trace
  sessions          — conversation session records
  reflections       — Gizmo's internal observations
  personality       — Gizmo's personality, global + per-headmate
  corrections       — hard behavioral rules. Never wiped. Never overridden.
  preferences       — soft preferences, context-weighted, Gizmo can override
  preference_qa     — questions Gizmo asked, answers received
  protocols         — rules and commitments from conversation
  knowledge_graph   — typed relationship edges
  wellbeing         — per-headmate wellbeing observations
  patterns          — discovered behavioral patterns
  pattern_instances — each time a pattern fires
  pattern_refinements — pattern learning history
  emotion_log       — per-message emotional arc data
  files             — index of files Gizmo has written
  entity_messages   — entities ↔ messages cross-reference
  interaction_prefs — per-headmate interaction preferences

Usage:
  from core.store import store

  store.get("entities", id="entity_jess")
  store.query("facts", headmate="jess", limit=10)
  store.write("facts", {"headmate": "jess", "fact": "loves pink buttons"})
  store.search("pink buttons", tables=["facts", "messages"], headmate="jess")
  store.delete("facts", id="fact_abc123")
"""

from __future__ import annotations

import json
import sqlite3
import struct
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional
import os

from core.log import log_event, log_error

# ── Config ────────────────────────────────────────────────────────────────────

_DB_PATH = Path(os.getenv("GIZMO_DB", "/data/gizmo.db"))

# Tables that support FTS5 full-text search
_FTS_TABLES = {
    "facts":       "fact",
    "messages":    "content",
    "reflections": "text",
    "personality": "text",
    "wellbeing":   "observation",
    "files":       "description",
    "responses":   "content",
}

# Tables that store embeddings
_EMBEDDING_TABLES = {
    "facts", "messages", "reflections",
    "personality", "wellbeing", "files", "responses",
}

# Universal columns present on every table
_UNIVERSAL = {
    "headmate", "source", "confidence", "active",
    "created_at", "updated_at", "tags", "session_id",
}


# ── Embedding ─────────────────────────────────────────────────────────────────

_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        from chromadb.utils import embedding_functions
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception:
        # Fallback: no embeddings, FTS only
        _embed_fn = None
    return _embed_fn


def _embed(text: str) -> Optional[bytes]:
    """Embed text to bytes blob. Returns None if embedding unavailable."""
    try:
        fn = _get_embed_fn()
        if fn is None:
            return None
        vectors = fn([text])
        if vectors and len(vectors) > 0:
            vec = vectors[0]
            return struct.pack(f"{len(vec)}f", *vec)
    except Exception:
        pass
    return None


def _cosine(blob_a: bytes, blob_b: bytes) -> float:
    """Cosine similarity between two packed float vectors."""
    try:
        n = len(blob_a) // 4
        a = struct.unpack(f"{n}f", blob_a)
        b = struct.unpack(f"{n}f", blob_b)
        dot  = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:
        return 0.0


# ── Connection ────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH), timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """

-- ── Entities ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    entity_type       TEXT NOT NULL,  -- headmate/external/pet
    pronouns          TEXT,
    age               TEXT,
    baseline_vibe     TEXT,           -- JSON array
    persona           TEXT,
    notes             TEXT,
    observation_count INTEGER DEFAULT 0,
    last_seen         REAL,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'system',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Facts ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS facts (
    id          TEXT PRIMARY KEY,
    fact        TEXT NOT NULL,
    fact_type   TEXT DEFAULT 'observation',  -- baseline/moment/preference/observation/inference
    entity_id   TEXT,
    context     TEXT,
    register    TEXT,
    embedding   BLOB,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'observer',
    confidence  REAL DEFAULT 0.7,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Relationships ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS relationships (
    id                      TEXT PRIMARY KEY,
    speaker                 TEXT NOT NULL,
    entity                  TEXT NOT NULL,
    entity_type             TEXT,
    owner                   TEXT,
    owner_type              TEXT,
    relationship_label      TEXT NOT NULL,
    relationship_category   TEXT,
    relationship_direction  TEXT,
    confidence_type         TEXT DEFAULT 'stated',  -- stated/inferred/hearsay/observed/speculative
    hearsay_source          TEXT,
    hearsay_about           TEXT,
    intimate                INTEGER DEFAULT 0,
    times_seen              INTEGER DEFAULT 1,
    first_seen              REAL,
    last_seen               REAL,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'extractor',
    confidence  REAL DEFAULT 0.7,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Messages ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id                  TEXT PRIMARY KEY,
    content             TEXT NOT NULL,
    fronters            TEXT,           -- JSON array
    subjects            TEXT,           -- JSON array of SubjectRef
    entities_referenced TEXT,           -- JSON array
    emotional_valence   REAL DEFAULT 0.0,
    register            TEXT DEFAULT 'neutral',
    topics              TEXT,           -- JSON array
    needs_active        TEXT,           -- JSON array
    new_facts           TEXT,           -- JSON array
    has_intimate        INTEGER DEFAULT 0,
    vibe                TEXT,           -- JSON array
    stress_level        TEXT,
    time_of_day         TEXT,
    session_momentum    TEXT,
    embedding           BLOB,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'user',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Responses ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS responses (
    id                       TEXT PRIMARY KEY,
    content                  TEXT NOT NULL,
    response_to              TEXT,       -- FK → messages.id
    approach                 TEXT,
    what_i_knew              TEXT,       -- JSON array
    what_i_looked_up         TEXT,       -- JSON array
    conversations_referenced TEXT,       -- JSON array
    graph_nodes_read         TEXT,       -- JSON array
    why                      TEXT,
    brief_snapshot           TEXT,
    outcome                  TEXT,       -- landed/cooled/dismissed/escalated/redirected
    outcome_signal           TEXT,
    outcome_filled_at        REAL,
    embedding                BLOB,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'gizmo',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Sessions ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    opened_at       REAL,
    closed_at       REAL,
    hosts           TEXT,       -- JSON array
    fronters        TEXT,       -- JSON array
    topics          TEXT,       -- JSON array
    message_count   INTEGER DEFAULT 0,
    mood            TEXT,
    summary         TEXT,
    notable         TEXT,       -- JSON array
    changes         TEXT,       -- JSON array
    unresolved      TEXT,
    transcript_path TEXT,
    arc             TEXT,       -- JSON: time-of-day buckets
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'archiver',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Reflections ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reflections (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    topic       TEXT,
    valence     REAL DEFAULT 0.0,
    intensity   REAL DEFAULT 0.2,
    chaos       REAL DEFAULT 0.0,
    surfaced    INTEGER DEFAULT 0,
    held_since  REAL,
    expires_at  REAL,
    outcome     TEXT DEFAULT 'pending',  -- surfaced/held/expired/pending
    embedding   BLOB,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'reflector',
    confidence  REAL DEFAULT 0.8,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Personality ──────────────────────────────────────────────────────────────
-- Gizmo's personality: global + per-headmate expression
-- aspect = seed / voice / values / tone / boundary / interest / observation
--          / with_headmate / change_request / drift_note
CREATE TABLE IF NOT EXISTS personality (
    id          TEXT PRIMARY KEY,
    aspect      TEXT NOT NULL,
    text        TEXT NOT NULL,
    subject     TEXT,           -- 'gizmo' or headmate name
    depth       TEXT,           -- surface/conversational/deep (for interests)
    adjacent    TEXT,           -- JSON array of related topics
    version     INTEGER DEFAULT 1,
    embedding   BLOB,
    -- universal
    headmate    TEXT,           -- null = global, headmate name = per-headmate
    source      TEXT DEFAULT 'system',
    confidence  REAL DEFAULT 0.9,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Corrections ──────────────────────────────────────────────────────────────
-- Hard stops. Never wiped. Never overridden. Full stops.
CREATE TABLE IF NOT EXISTS corrections (
    id              TEXT PRIMARY KEY,
    rule            TEXT NOT NULL,
    who_corrected   TEXT,
    times_violated  INTEGER DEFAULT 0,
    source          TEXT DEFAULT 'correction',  -- correction/reset (audit)
    -- universal
    headmate    TEXT,
    source_     TEXT DEFAULT 'user',    -- renamed to avoid conflict
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Preferences ──────────────────────────────────────────────────────────────
-- Soft preferences. Context-weighted. Gizmo can override.
CREATE TABLE IF NOT EXISTS preferences (
    id                  TEXT PRIMARY KEY,
    preference          TEXT NOT NULL,      -- what the preference is
    valence             TEXT DEFAULT 'positive',  -- positive/negative
    default_context     TEXT,               -- JSON array: contexts where this applies
    avoid_context       TEXT,               -- JSON array: contexts where this doesn't
    gizmo_override      INTEGER DEFAULT 1,  -- 1 = Gizmo can use judgment
    override_note       TEXT,               -- when/why override is appropriate
    times_used          INTEGER DEFAULT 0,
    times_landed        INTEGER DEFAULT 0,
    times_missed        INTEGER DEFAULT 0,
    last_used           REAL,
    source_message_id   TEXT,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'extractor',
    confidence  REAL DEFAULT 0.7,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Preference Q&A ───────────────────────────────────────────────────────────
-- Questions Gizmo asked. Answers received.
-- Non-intrusive but shameless. He asks when the moment is right.
CREATE TABLE IF NOT EXISTS preference_qa (
    id              TEXT PRIMARY KEY,
    question        TEXT NOT NULL,
    answer          TEXT,               -- null until answered
    answer_summary  TEXT,               -- distilled from answer
    asked_at        REAL,
    answered_at     REAL,
    context_at_ask  TEXT,               -- what was happening when asked
    gap_identified  TEXT,               -- what gap this was filling
    applied_to      TEXT,               -- preference/personality/pattern table + id
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'gizmo',
    confidence  REAL DEFAULT 0.8,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Protocols ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS protocols (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    content     TEXT NOT NULL,
    scope       TEXT DEFAULT 'global',  -- global/headmate-specific
    file_path   TEXT,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'gizmo',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Knowledge graph ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_graph (
    id          TEXT PRIMARY KEY,
    from_node   TEXT NOT NULL,
    from_type   TEXT,
    edge_type   TEXT NOT NULL,
    to_node     TEXT NOT NULL,
    to_type     TEXT,
    weight      REAL DEFAULT 1.0,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'extractor',
    confidence  REAL DEFAULT 0.7,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Wellbeing ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wellbeing (
    id          TEXT PRIMARY KEY,
    category    TEXT NOT NULL,  -- emotional_need/physical_need/works/pulled_away/pattern/limit
    observation TEXT NOT NULL,
    context     TEXT,
    register    TEXT,
    embedding   BLOB,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'extractor',
    confidence  REAL DEFAULT 0.6,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Patterns ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patterns (
    id                  TEXT PRIMARY KEY,
    pattern_type        TEXT,           -- temporal/trigger_response/frequency/absence/escalation/recovery
    trigger_conditions  TEXT,           -- JSON array
    trigger_confidence  REAL DEFAULT 0.0,
    false_positive_rate REAL DEFAULT 0.0,
    action              TEXT DEFAULT 'hold',  -- feed/break/hold/flag_therapy
    push_to             REAL,           -- intensity target if feeding
    approach            TEXT,
    reasoning           TEXT,
    watch_for           TEXT,           -- JSON array of edge indicators
    edge_intensity      REAL,
    outcome_quality_avg REAL DEFAULT 0.0,
    data_points         INTEGER DEFAULT 0,
    version             INTEGER DEFAULT 1,
    last_refined        REAL,
    therapy_flag        INTEGER DEFAULT 0,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'pattern_engine',
    confidence  REAL DEFAULT 0.2,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Pattern instances ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pattern_instances (
    id              TEXT PRIMARY KEY,
    pattern_id      TEXT NOT NULL,
    intensity_in    REAL DEFAULT 0.0,
    intensity_out   REAL DEFAULT 0.0,
    gizmo_pushed    INTEGER DEFAULT 0,
    push_type       TEXT,
    post_pattern    TEXT,
    outcome_quality REAL DEFAULT 0.5,
    notes           TEXT,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'pattern_engine',
    confidence  REAL DEFAULT 0.8,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Pattern refinements ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pattern_refinements (
    id              TEXT PRIMARY KEY,
    pattern_id      TEXT NOT NULL,
    version         INTEGER,
    change          TEXT NOT NULL,
    reason          TEXT,
    data_points_at  INTEGER,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'pattern_engine',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Emotion log ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS emotion_log (
    id          TEXT PRIMARY KEY,
    valence     REAL DEFAULT 0.0,
    intensity   REAL DEFAULT 0.2,
    chaos       REAL DEFAULT 0.0,
    register    TEXT DEFAULT 'neutral',
    topic       TEXT,
    word_count  INTEGER DEFAULT 0,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'emotion_tracker',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Files ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS files (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    description TEXT,
    file_type   TEXT,           -- protocol/note/observation/reflection/etc.
    content_ref TEXT,           -- "table:id" of where content is stored
    embedding   BLOB,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'gizmo',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── Entity ↔ message cross-reference ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entity_messages (
    entity_id   TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    role        TEXT,
    headmate    TEXT,
    timestamp   REAL,
    PRIMARY KEY (entity_id, message_id, role)
);

-- ── Interaction preferences ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS interaction_prefs (
    id          TEXT PRIMARY KEY,
    pref_type   TEXT NOT NULL,  -- persona/explicit/tone/format/boundary
    content     TEXT NOT NULL,
    -- universal
    headmate    TEXT,
    source      TEXT DEFAULT 'user',
    confidence  REAL DEFAULT 1.0,
    active      INTEGER DEFAULT 1,
    created_at  REAL,
    updated_at  REAL,
    tags        TEXT,
    session_id  TEXT
);

-- ── FTS5 virtual tables ──────────────────────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts
    USING fts5(fact, content='facts', content_rowid='rowid');

CREATE VIRTUAL TABLE IF NOT EXISTS fts_messages
    USING fts5(content, content='messages', content_rowid='rowid');

CREATE VIRTUAL TABLE IF NOT EXISTS fts_reflections
    USING fts5(text, content='reflections', content_rowid='rowid');

CREATE VIRTUAL TABLE IF NOT EXISTS fts_personality
    USING fts5(text, content='personality', content_rowid='rowid');

CREATE VIRTUAL TABLE IF NOT EXISTS fts_wellbeing
    USING fts5(observation, content='wellbeing', content_rowid='rowid');

CREATE VIRTUAL TABLE IF NOT EXISTS fts_responses
    USING fts5(content, content='responses', content_rowid='rowid');

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_facts_headmate        ON facts(headmate);
CREATE INDEX IF NOT EXISTS idx_facts_entity          ON facts(entity_id);
CREATE INDEX IF NOT EXISTS idx_facts_type            ON facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_messages_headmate     ON messages(headmate);
CREATE INDEX IF NOT EXISTS idx_messages_session      ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created      ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_register     ON messages(register);
CREATE INDEX IF NOT EXISTS idx_responses_session     ON responses(session_id);
CREATE INDEX IF NOT EXISTS idx_responses_to          ON responses(response_to);
CREATE INDEX IF NOT EXISTS idx_responses_outcome     ON responses(outcome);
CREATE INDEX IF NOT EXISTS idx_sessions_headmate     ON sessions(headmate);
CREATE INDEX IF NOT EXISTS idx_sessions_opened       ON sessions(opened_at);
CREATE INDEX IF NOT EXISTS idx_reflections_headmate  ON reflections(headmate);
CREATE INDEX IF NOT EXISTS idx_reflections_outcome   ON reflections(outcome);
CREATE INDEX IF NOT EXISTS idx_personality_headmate  ON personality(headmate);
CREATE INDEX IF NOT EXISTS idx_personality_aspect    ON personality(aspect);
CREATE INDEX IF NOT EXISTS idx_corrections_active    ON corrections(active);
CREATE INDEX IF NOT EXISTS idx_preferences_headmate  ON preferences(headmate);
CREATE INDEX IF NOT EXISTS idx_preference_qa_headmate ON preference_qa(headmate);
CREATE INDEX IF NOT EXISTS idx_preference_qa_answered ON preference_qa(answered_at);
CREATE INDEX IF NOT EXISTS idx_patterns_headmate     ON patterns(headmate);
CREATE INDEX IF NOT EXISTS idx_patterns_action       ON patterns(action);
CREATE INDEX IF NOT EXISTS idx_patterns_type         ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_pattern_instances_pid ON pattern_instances(pattern_id);
CREATE INDEX IF NOT EXISTS idx_emotion_log_headmate  ON emotion_log(headmate);
CREATE INDEX IF NOT EXISTS idx_emotion_log_session   ON emotion_log(session_id);
CREATE INDEX IF NOT EXISTS idx_emotion_log_created   ON emotion_log(created_at);
CREATE INDEX IF NOT EXISTS idx_wellbeing_headmate    ON wellbeing(headmate);
CREATE INDEX IF NOT EXISTS idx_wellbeing_category    ON wellbeing(category);
CREATE INDEX IF NOT EXISTS idx_entity_messages_eid   ON entity_messages(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_messages_mid   ON entity_messages(message_id);
CREATE INDEX IF NOT EXISTS idx_relationships_speaker ON relationships(speaker);
CREATE INDEX IF NOT EXISTS idx_relationships_entity  ON relationships(entity);
CREATE INDEX IF NOT EXISTS idx_relationships_label   ON relationships(relationship_label);
CREATE INDEX IF NOT EXISTS idx_knowledge_from        ON knowledge_graph(from_node);
CREATE INDEX IF NOT EXISTS idx_knowledge_to          ON knowledge_graph(to_node);
CREATE INDEX IF NOT EXISTS idx_knowledge_edge        ON knowledge_graph(edge_type);
CREATE INDEX IF NOT EXISTS idx_interaction_prefs_hm  ON interaction_prefs(headmate);
CREATE INDEX IF NOT EXISTS idx_files_headmate        ON files(headmate);
CREATE INDEX IF NOT EXISTS idx_files_path            ON files(path);
"""

# ── Table prefix map ──────────────────────────────────────────────────────────
_TABLE_PREFIXES = {
    "entities":           "ent",
    "facts":              "fact",
    "relationships":      "rel",
    "messages":           "msg",
    "responses":          "resp",
    "sessions":           "sess",
    "reflections":        "refl",
    "personality":        "pers",
    "corrections":        "corr",
    "preferences":        "pref",
    "preference_qa":      "qa",
    "protocols":          "prot",
    "knowledge_graph":    "kg",
    "wellbeing":          "wb",
    "patterns":           "pat",
    "pattern_instances":  "pi",
    "pattern_refinements":"pr",
    "emotion_log":        "emo",
    "files":              "file",
    "interaction_prefs":  "ipr",
}


def _new_id(table: str) -> str:
    prefix = _TABLE_PREFIXES.get(table, "rec")
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _now() -> float:
    return time.time()


# ── Store class ───────────────────────────────────────────────────────────────

class Store:
    """
    Single interface to gizmo.db.
    All reads and writes go through here.
    """

    def __init__(self):
        self._init_db()
        log_event("Store", "INIT", path=str(_DB_PATH))

    def _init_db(self) -> None:
        with _conn() as con:
            con.executescript(_SCHEMA)

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        table:     str,
        data:      dict,
        embed_field: Optional[str] = None,
    ) -> str:
        """
        Insert or update a record.
        If data contains 'id' and that ID exists, updates the record.
        Otherwise inserts a new record.
        Returns the record ID.
        """
        ts  = _now()
        row = dict(data)

        # Inject universal defaults
        if "id" not in row:
            row["id"] = _new_id(table)
        if "created_at" not in row:
            row["created_at"] = ts
        row["updated_at"] = ts
        if "active" not in row:
            row["active"] = 1

        # Serialize lists/dicts to JSON
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v)

        # Generate embedding if applicable
        if table in _EMBEDDING_TABLES:
            field = embed_field or _FTS_TABLES.get(table)
            text  = row.get(field, "")
            if text and isinstance(text, str) and "embedding" not in row:
                row["embedding"] = _embed(text)

        with _conn() as con:
            # Check if exists
            existing = con.execute(
                f"SELECT id FROM {table} WHERE id = ?",
                (row["id"],)
            ).fetchone()

            if existing:
                # Update
                sets    = ", ".join(f"{k} = ?" for k in row if k != "id")
                values  = [row[k] for k in row if k != "id"]
                values.append(row["id"])
                con.execute(f"UPDATE {table} SET {sets} WHERE id = ?", values)
            else:
                # Insert
                cols   = ", ".join(row.keys())
                placeholders = ", ".join("?" for _ in row)
                con.execute(
                    f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                    list(row.values()),
                )

            # Update FTS index
            if table in _FTS_TABLES:
                fts_table = f"fts_{table}"
                fts_col   = _FTS_TABLES[table]
                text_val  = row.get(fts_col, "")
                if text_val and isinstance(text_val, str):
                    try:
                        con.execute(
                            f"INSERT OR REPLACE INTO {fts_table}(rowid, {fts_col}) "
                            f"SELECT rowid, {fts_col} FROM {table} WHERE id = ?",
                            (row["id"],),
                        )
                    except Exception:
                        pass

        return row["id"]

    def write_many(self, table: str, rows: list[dict]) -> list[str]:
        """Write multiple records. Returns list of IDs."""
        return [self.write(table, row) for row in rows]

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, table: str, id: str) -> Optional[dict]:
        """Fetch a single record by ID."""
        with _conn() as con:
            row = con.execute(
                f"SELECT * FROM {table} WHERE id = ?", (id,)
            ).fetchone()
        return self._deserialize(dict(row)) if row else None

    def query(
        self,
        table:    str,
        limit:    int = 20,
        order_by: str = "created_at DESC",
        **filters,
    ) -> list[dict]:
        """
        Flexible filter query.
        Any column can be passed as a keyword filter.
        Special values:
          active=True  → active = 1
          active=False → active = 0
          headmate=None → headmate IS NULL
        """
        clauses = []
        params  = []

        for k, v in filters.items():
            if v is None:
                clauses.append(f"{k} IS NULL")
            elif isinstance(v, bool):
                clauses.append(f"{k} = ?")
                params.append(1 if v else 0)
            else:
                clauses.append(f"{k} = ?")
                params.append(v)

        where  = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        with _conn() as con:
            rows = con.execute(
                f"SELECT * FROM {table} {where} ORDER BY {order_by} LIMIT ?",
                params,
            ).fetchall()

        return [self._deserialize(dict(r)) for r in rows]

    def get_active(self, table: str, **filters) -> list[dict]:
        """Query active records only."""
        return self.query(table, active=1, **filters)

    def count(self, table: str, **filters) -> int:
        """Count matching records."""
        clauses = []
        params  = []

        for k, v in filters.items():
            if v is None:
                clauses.append(f"{k} IS NULL")
            elif isinstance(v, bool):
                clauses.append(f"{k} = ?")
                params.append(1 if v else 0)
            else:
                clauses.append(f"{k} = ?")
                params.append(v)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with _conn() as con:
            row = con.execute(
                f"SELECT COUNT(*) FROM {table} {where}", params
            ).fetchone()
        return row[0] if row else 0

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, table: str, id: str, **fields) -> None:
        """Update specific fields on a record."""
        fields["updated_at"] = _now()
        sets   = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values())
        values.append(id)
        with _conn() as con:
            con.execute(f"UPDATE {table} SET {sets} WHERE id = ?", values)

    def touch(self, table: str, id: str) -> None:
        """Update updated_at timestamp."""
        self.update(table, id, updated_at=_now())

    def delete(self, table: str, id: str, hard: bool = False) -> None:
        """
        Soft delete by default (sets active=0).
        Hard delete removes the row entirely.
        """
        if hard:
            with _conn() as con:
                con.execute(f"DELETE FROM {table} WHERE id = ?", (id,))
        else:
            self.update(table, id, active=0)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:    str,
        tables:   Optional[list[str]] = None,
        headmate: Optional[str]       = None,
        limit:    int                 = 10,
        semantic: bool                = True,
    ) -> list[dict]:
        """
        Full-text search across one or more tables.
        If semantic=True and embeddings are available, re-ranks by
        vector similarity after FTS retrieval.

        Returns results sorted by relevance, each with a _score field
        and a _table field indicating which table it came from.
        """
        search_tables = tables or list(_FTS_TABLES.keys())
        results       = []

        query_embedding = _embed(query) if semantic else None

        for table in search_tables:
            if table not in _FTS_TABLES:
                continue
            fts_table = f"fts_{table}"
            fts_col   = _FTS_TABLES[table]

            try:
                headmate_clause = "AND t.headmate = ?" if headmate else ""
                headmate_param  = [headmate] if headmate else []

                with _conn() as con:
                    rows = con.execute(f"""
                        SELECT t.*, fts.rank AS fts_rank
                        FROM {fts_table} fts
                        JOIN {table} t ON t.rowid = fts.rowid
                        WHERE fts_{table} MATCH ?
                          AND t.active = 1
                          {headmate_clause}
                        ORDER BY fts.rank
                        LIMIT ?
                    """, [query, *headmate_param, limit * 2]).fetchall()

                for row in rows:
                    d = self._deserialize(dict(row))
                    d["_table"]     = table
                    d["_fts_score"] = row["fts_rank"]
                    d["_score"]     = abs(row["fts_rank"])  # FTS rank is negative

                    # Vector re-rank
                    if semantic and query_embedding and d.get("embedding"):
                        sim = _cosine(query_embedding, d["embedding"])
                        d["_score"] = d["_score"] * 0.4 + sim * 0.6
                    results.append(d)

            except Exception as e:
                log_error("Store", f"search failed on {table}: {e}", exc=None)

        # Sort by score descending, return top N
        results.sort(key=lambda x: x.get("_score", 0), reverse=True)
        return results[:limit]

    def search_semantic(
        self,
        query:    str,
        table:    str,
        headmate: Optional[str] = None,
        limit:    int           = 10,
        threshold: float        = 0.3,
    ) -> list[dict]:
        """
        Pure vector similarity search on a specific table.
        Loads all embeddings for the table (or headmate subset) and
        ranks by cosine similarity.
        Only returns results above threshold.
        """
        query_embedding = _embed(query)
        if query_embedding is None:
            return []

        headmate_clause = "WHERE headmate = ? AND active = 1" if headmate else "WHERE active = 1"
        params          = [headmate] if headmate else []

        with _conn() as con:
            rows = con.execute(
                f"SELECT * FROM {table} {headmate_clause}",
                params,
            ).fetchall()

        scored = []
        for row in rows:
            d = self._deserialize(dict(row))
            if d.get("embedding"):
                sim = _cosine(query_embedding, d["embedding"])
                if sim >= threshold:
                    d["_score"] = sim
                    d["_table"] = table
                    scored.append(d)

        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:limit]

    # ── Specialized queries ───────────────────────────────────────────────────

    def get_corrections(self) -> list[str]:
        """Return all active correction rules as strings."""
        rows = self.query("corrections", active=1, limit=200)
        return [r["rule"] for r in rows if r.get("rule")]

    def get_personality(
        self,
        headmate: Optional[str] = None,
        aspect:   Optional[str] = None,
    ) -> list[dict]:
        """
        Get personality records.
        headmate=None returns global (Gizmo's own personality).
        headmate='jess' returns Jess-specific personality notes.
        """
        filters: dict = {"active": 1}
        if headmate is not None:
            filters["headmate"] = headmate.lower()
        if aspect is not None:
            filters["aspect"] = aspect
        return self.query("personality", **filters, limit=50)

    def get_preferences(
        self,
        headmate: str,
        context:  Optional[str] = None,
    ) -> list[dict]:
        """
        Get active preferences for a headmate.
        If context provided, filters to preferences applicable in that context.
        """
        rows = self.query("preferences", headmate=headmate.lower(), active=1, limit=100)
        if not context:
            return rows

        # Filter by context applicability
        applicable = []
        for row in rows:
            default_ctx = row.get("default_context") or []
            avoid_ctx   = row.get("avoid_context") or []

            if isinstance(default_ctx, str):
                try: default_ctx = json.loads(default_ctx)
                except: default_ctx = []
            if isinstance(avoid_ctx, str):
                try: avoid_ctx = json.loads(avoid_ctx)
                except: avoid_ctx = []

            # Skip if explicitly avoided in this context
            if any(c.lower() in context.lower() for c in avoid_ctx):
                continue
            applicable.append(row)

        return applicable

    def get_pending_questions(self, headmate: str) -> list[dict]:
        """Questions Gizmo has queued but not yet asked."""
        return self.query(
            "preference_qa",
            headmate=headmate.lower(),
            active=1,
            order_by="created_at ASC",
            limit=5,
        )

    def get_patterns(
        self,
        headmate: str,
        action:   Optional[str] = None,
        min_confidence: float   = 0.4,
    ) -> list[dict]:
        """
        Get active patterns for a headmate above confidence threshold.
        action: feed / break / hold / flag_therapy
        """
        rows = self.query("patterns", headmate=headmate.lower(), active=1, limit=50)
        filtered = [
            r for r in rows
            if r.get("confidence", 0) >= min_confidence
            and (action is None or r.get("action") == action)
        ]
        # Sort by confidence descending
        filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return filtered

    def get_wellbeing(
        self,
        headmate: str,
        category: Optional[str] = None,
        limit:    int           = 20,
    ) -> list[dict]:
        """Get wellbeing observations for a headmate."""
        filters: dict = {"headmate": headmate.lower(), "active": 1}
        if category:
            filters["category"] = category
        return self.query("wellbeing", **filters, limit=limit)

    def get_entity(self, name: str) -> Optional[dict]:
        """Get entity by name (case-insensitive)."""
        with _conn() as con:
            row = con.execute(
                "SELECT * FROM entities WHERE LOWER(name) = ? AND active = 1",
                (name.lower(),)
            ).fetchone()
        return self._deserialize(dict(row)) if row else None

    def get_recent_messages(
        self,
        headmate:   Optional[str] = None,
        session_id: Optional[str] = None,
        limit:      int           = 20,
        since:      Optional[float] = None,
    ) -> list[dict]:
        """Recent messages with optional filters."""
        filters: dict = {"active": 1}
        if headmate:
            filters["headmate"] = headmate.lower()
        if session_id:
            filters["session_id"] = session_id

        rows = self.query("messages", **filters,
                         order_by="created_at DESC", limit=limit)

        if since:
            rows = [r for r in rows if r.get("created_at", 0) >= since]

        return rows

    def get_last_response(self, session_id: str) -> Optional[dict]:
        """Most recent unfilled response for a session."""
        rows = self.query(
            "responses",
            session_id=session_id,
            active=1,
            order_by="created_at DESC",
            limit=1,
        )
        # Prefer one without outcome filled yet
        for r in rows:
            if not r.get("outcome"):
                return r
        return rows[0] if rows else None

    def get_today_sessions(self, headmate: Optional[str] = None) -> list[dict]:
        """Sessions opened today."""
        import time as _time
        today_start = _time.mktime(
            __import__("datetime").datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timetuple()
        )
        with _conn() as con:
            if headmate:
                rows = con.execute("""
                    SELECT * FROM sessions
                    WHERE opened_at >= ?
                      AND active = 1
                      AND (
                        headmate = ?
                        OR hosts LIKE ?
                      )
                    ORDER BY opened_at ASC
                """, (today_start, headmate.lower(), f"%{headmate.lower()}%")).fetchall()
            else:
                rows = con.execute("""
                    SELECT * FROM sessions
                    WHERE opened_at >= ? AND active = 1
                    ORDER BY opened_at ASC
                """, (today_start,)).fetchall()
        return [self._deserialize(dict(r)) for r in rows]

    # ── Preference management ─────────────────────────────────────────────────

    def add_preference(
        self,
        headmate:       str,
        preference:     str,
        valence:        str            = "positive",
        default_context: Optional[list] = None,
        avoid_context:  Optional[list]  = None,
        gizmo_override: bool            = True,
        override_note:  Optional[str]   = None,
        source_message_id: Optional[str] = None,
        session_id:     Optional[str]   = None,
    ) -> str:
        """Add a soft preference for a headmate."""
        return self.write("preferences", {
            "headmate":          headmate.lower(),
            "preference":        preference,
            "valence":           valence,
            "default_context":   json.dumps(default_context or []),
            "avoid_context":     json.dumps(avoid_context or []),
            "gizmo_override":    1 if gizmo_override else 0,
            "override_note":     override_note,
            "source_message_id": source_message_id,
            "session_id":        session_id,
            "source":            "extractor",
            "tags":              f"preference,{headmate.lower()},{valence}",
        })

    def record_preference_outcome(
        self,
        preference_id: str,
        landed:        bool,
    ) -> None:
        """Update times_used and times_landed/missed for a preference."""
        with _conn() as con:
            row = con.execute(
                "SELECT times_used, times_landed, times_missed FROM preferences WHERE id = ?",
                (preference_id,)
            ).fetchone()
            if row:
                con.execute("""
                    UPDATE preferences
                    SET times_used = ?,
                        times_landed = ?,
                        times_missed = ?,
                        last_used = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    row["times_used"] + 1,
                    row["times_landed"] + (1 if landed else 0),
                    row["times_missed"] + (0 if landed else 1),
                    _now(),
                    _now(),
                    preference_id,
                ))

    def queue_question(
        self,
        headmate:       str,
        question:       str,
        gap_identified: str,
        context:        Optional[str] = None,
        session_id:     Optional[str] = None,
    ) -> str:
        """Queue a follow-up question for Gizmo to ask when the moment is right."""
        return self.write("preference_qa", {
            "headmate":       headmate.lower(),
            "question":       question,
            "gap_identified": gap_identified,
            "context_at_ask": context,
            "session_id":     session_id,
            "source":         "gizmo",
            "tags":           f"question,{headmate.lower()},pending",
        })

    def answer_question(
        self,
        qa_id:          str,
        answer:         str,
        answer_summary: str,
        applied_to:     Optional[str] = None,
    ) -> None:
        """Record the answer to a queued question."""
        self.update("preference_qa", qa_id,
            answer=answer,
            answer_summary=answer_summary,
            answered_at=_now(),
            applied_to=applied_to,
            tags=f"question,answered",
        )

    # ── Correction management ─────────────────────────────────────────────────

    def add_correction(
        self,
        rule:          str,
        headmate:      Optional[str] = None,
        who_corrected: Optional[str] = None,
        session_id:    Optional[str] = None,
    ) -> str:
        """Add a hard behavioral rule. Never wiped."""
        return self.write("corrections", {
            "rule":          rule,
            "headmate":      headmate.lower() if headmate else None,
            "who_corrected": who_corrected,
            "session_id":    session_id,
            "source":        "correction",
            "active":        1,
            "tags":          f"correction,{headmate.lower() if headmate else 'global'}",
        })

    def increment_violation(self, correction_id: str) -> None:
        """Track how many times a rule has been violated."""
        with _conn() as con:
            con.execute("""
                UPDATE corrections
                SET times_violated = times_violated + 1, updated_at = ?
                WHERE id = ?
            """, (_now(), correction_id))

    # ── Pattern management ────────────────────────────────────────────────────

    def upsert_pattern(self, pattern_data: dict) -> str:
        """
        Insert or update a pattern.
        If a pattern with the same headmate + pattern_type + trigger_conditions
        already exists, updates it and bumps version.
        """
        existing = None
        if pattern_data.get("headmate") and pattern_data.get("pattern_type"):
            rows = self.query(
                "patterns",
                headmate=pattern_data["headmate"].lower(),
                pattern_type=pattern_data["pattern_type"],
                active=1,
                limit=10,
            )
            # Match on trigger_conditions hash if available
            for r in rows:
                if r.get("trigger_conditions") == pattern_data.get("trigger_conditions"):
                    existing = r
                    break

        if existing:
            old_version = existing.get("version", 1)
            pattern_data["id"]      = existing["id"]
            pattern_data["version"] = old_version + 1
            pattern_data["last_refined"] = _now()

        return self.write("patterns", pattern_data)

    def log_pattern_instance(
        self,
        pattern_id:      str,
        headmate:        str,
        session_id:      str,
        intensity_in:    float,
        intensity_out:   float,
        gizmo_pushed:    bool,
        push_type:       Optional[str],
        post_pattern:    Optional[str],
        outcome_quality: float,
        notes:           Optional[str] = None,
    ) -> str:
        """Log a pattern firing event."""
        inst_id = self.write("pattern_instances", {
            "pattern_id":      pattern_id,
            "headmate":        headmate.lower(),
            "session_id":      session_id,
            "intensity_in":    intensity_in,
            "intensity_out":   intensity_out,
            "gizmo_pushed":    1 if gizmo_pushed else 0,
            "push_type":       push_type,
            "post_pattern":    post_pattern,
            "outcome_quality": outcome_quality,
            "notes":           notes,
            "source":          "pattern_engine",
            "tags":            f"pattern_instance,{headmate.lower()},{pattern_id}",
        })

        # Update pattern's rolling average and data_points
        instances = self.query(
            "pattern_instances",
            pattern_id=pattern_id,
            active=1,
            limit=100,
        )
        if instances:
            avg = sum(i.get("outcome_quality", 0) for i in instances) / len(instances)
            self.update("patterns", pattern_id,
                outcome_quality_avg=round(avg, 3),
                data_points=len(instances),
            )

        return inst_id

    def log_pattern_refinement(
        self,
        pattern_id:     str,
        version:        int,
        change:         str,
        reason:         str,
        data_points_at: int,
        headmate:       Optional[str] = None,
    ) -> str:
        """Log a pattern refinement event."""
        return self.write("pattern_refinements", {
            "pattern_id":     pattern_id,
            "version":        version,
            "change":         change,
            "reason":         reason,
            "data_points_at": data_points_at,
            "headmate":       headmate.lower() if headmate else None,
            "source":         "pattern_engine",
            "tags":           f"refinement,{pattern_id}",
        })

    # ── Serialization ─────────────────────────────────────────────────────────

    def _deserialize(self, row: dict) -> dict:
        """Deserialize JSON fields and clean up the row."""
        _JSON_FIELDS = {
            "fronters", "subjects", "entities_referenced", "topics",
            "needs_active", "new_facts", "vibe", "what_i_knew",
            "what_i_looked_up", "conversations_referenced", "graph_nodes_read",
            "graph_nodes", "baseline_vibe", "adjacent", "hosts",
            "notable", "changes", "trigger_conditions", "watch_for",
            "default_context", "avoid_context",
        }
        for field in _JSON_FIELDS:
            if field in row and isinstance(row[field], str):
                try:
                    row[field] = json.loads(row[field])
                except Exception:
                    pass

        # Don't expose raw embedding bytes to callers
        if "embedding" in row:
            row["has_embedding"] = row["embedding"] is not None
            del row["embedding"]

        return row


# ── Singleton ─────────────────────────────────────────────────────────────────

store = Store()
