"""
core/message_store.py

SQLite store for every exchange Gizmo has.

Each row is one complete exchange: user message + Gizmo response,
with all context from the Brief attached — host, fronters, topics,
emotional register, mood, tags, stage directions, lore.

Two-phase population:
  1. insert_exchange() — called synchronously from archivist.receive_outgoing()
     immediately after each exchange. Heuristic topics as placeholder.
  2. update_tags()     — called async from tagger.py after LLM tagging pass.
     Replaces heuristic topics with real ones, adds mood/tone/cause/effect/summary.

Schema:
    messages (
        id                TEXT PRIMARY KEY,
        session_id        TEXT,
        timestamp         TEXT,              ISO8601
        host              TEXT,
        fronters          TEXT,              JSON array
        user_message      TEXT,
        gizmo_response    TEXT,
        topics            TEXT,              JSON array — heuristic initially, LLM after tag pass
        emotional_register TEXT,
        mood              TEXT,              LLM-derived after tag pass
        gizmo_tone        TEXT,              how Gizmo came across
        cause             TEXT,              what prompted this exchange
        effect            TEXT,              what changed after
        summary           TEXT,              one sentence
        tags              TEXT,              JSON array — union of topics + extras
        notable           INTEGER,           0/1
        tagged            INTEGER,           0/1 — has LLM tagging pass run?
        stage_directions  TEXT,              JSON array
        lore              TEXT               JSON array
    )
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

_DB_PATH = Path(os.getenv("DATA_DIR", "/data")) / "message_store.db"


@contextmanager
def _conn():
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id                 TEXT PRIMARY KEY,
                session_id         TEXT NOT NULL,
                timestamp          TEXT NOT NULL,
                host               TEXT,
                fronters           TEXT,
                user_message       TEXT,
                gizmo_response     TEXT,
                topics             TEXT,
                emotional_register TEXT,
                mood               TEXT,
                gizmo_tone         TEXT,
                cause              TEXT,
                effect             TEXT,
                summary            TEXT,
                tags               TEXT,
                notable            INTEGER DEFAULT 0,
                tagged             INTEGER DEFAULT 0,
                stage_directions   TEXT,
                lore               TEXT
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_host      ON messages (host)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON messages (timestamp)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_session   ON messages (session_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_tagged    ON messages (tagged)")

        # Migration — add new columns if upgrading from older schema
        existing = {
            row[1] for row in con.execute("PRAGMA table_info(messages)").fetchall()
        }
        for col, defn in [
            ("gizmo_tone", "TEXT"),
            ("cause",      "TEXT"),
            ("effect",     "TEXT"),
            ("summary",    "TEXT"),
            ("tagged",     "INTEGER DEFAULT 0"),
        ]:
            if col not in existing:
                con.execute(f"ALTER TABLE messages ADD COLUMN {col} {defn}")

    log("MessageStore", f"initialized at {_DB_PATH}")


# ── Write ─────────────────────────────────────────────────────────────────────

def insert_exchange(
    session_id: str,
    timestamp: float,
    host: Optional[str],
    fronters: list,
    user_message: str,
    gizmo_response: str,
    topics: list,
    emotional_register: str,
    mood: str = "neutral",
    tags: list = None,
    notable: bool = False,
    stage_directions: list = None,
    lore: list = None,
) -> str:
    """
    Insert one exchange immediately after it happens.
    Topics are heuristic placeholders — tagger.py will update them async.
    Returns the generated message id.
    """
    ts_ms  = int(timestamp * 1000)
    msg_id = f"msg_{session_id[:8]}_{ts_ms}"
    ts_str = datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")
    all_tags = list(set((tags or []) + topics))

    try:
        with _conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO messages (
                    id, session_id, timestamp, host, fronters,
                    user_message, gizmo_response, topics, emotional_register,
                    mood, tags, notable, tagged, stage_directions, lore
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (
                msg_id, session_id, ts_str,
                (host or "").lower(),
                json.dumps(fronters or []),
                user_message or "",
                gizmo_response or "",
                json.dumps(topics or []),
                emotional_register or "neutral",
                mood or "neutral",
                json.dumps(all_tags),
                1 if notable else 0,
                json.dumps(stage_directions or []),
                json.dumps(lore or []),
            ))
        log_event("MessageStore", "INSERTED",
            id=msg_id, host=host or "unknown", topics=topics)
        return msg_id
    except Exception as e:
        log_error("MessageStore", "insert_exchange failed", exc=e)
        return msg_id


def update_tags(
    msg_id: str,
    topics: list,
    mood: str,
    gizmo_tone: str = "",
    cause: Optional[str] = None,
    effect: Optional[str] = None,
    summary: str = "",
) -> None:
    """
    Update a message row with LLM-derived tags after the tagging pass.
    Also sets tagged=1 so we know the real tags are in.
    """
    try:
        all_tags = list(set(topics))
        with _conn() as con:
            con.execute("""
                UPDATE messages
                SET topics = ?,
                    mood   = ?,
                    gizmo_tone = ?,
                    cause  = ?,
                    effect = ?,
                    summary = ?,
                    tags   = ?,
                    tagged = 1
                WHERE id = ?
            """, (
                json.dumps(topics),
                mood or "neutral",
                gizmo_tone or "",
                cause,
                effect,
                summary or "",
                json.dumps(all_tags),
                msg_id,
            ))
        log_event("MessageStore", "TAGS_UPDATED",
            id=msg_id, topics=topics, mood=mood, cause=bool(cause), effect=bool(effect))
    except Exception as e:
        log_error("MessageStore", "update_tags failed", exc=e)


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    host: Optional[str] = None,
    topics: Optional[list] = None,
    tags: Optional[list] = None,
    emotional_register: Optional[str] = None,
    mood: Optional[str] = None,
    cause_keyword: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    keyword: Optional[str] = None,
    tagged_only: bool = False,
    limit: int = 20,
    order: str = "DESC",
) -> list[dict]:
    """
    Search messages. Returns lightweight stubs (no full response text).
    tagged_only=True skips rows that haven't had the LLM tagging pass yet.
    """
    clauses = []
    params  = []

    if host:
        clauses.append("host = ?")
        params.append(host.lower())

    if emotional_register:
        clauses.append("emotional_register = ?")
        params.append(emotional_register)

    if mood:
        clauses.append("mood LIKE ?")
        params.append(f"%{mood}%")

    if since:
        clauses.append("timestamp >= ?")
        params.append(since)

    if until:
        clauses.append("timestamp <= ?")
        params.append(until + "T23:59:59")

    if keyword:
        clauses.append("(user_message LIKE ? OR gizmo_response LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])

    if cause_keyword:
        clauses.append("cause LIKE ?")
        params.append(f"%{cause_keyword}%")

    if topics:
        for t in topics:
            clauses.append("topics LIKE ?")
            params.append(f'%"{t}"%')

    if tags:
        for t in tags:
            clauses.append("tags LIKE ?")
            params.append(f'%"{t}"%')

    if tagged_only:
        clauses.append("tagged = 1")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_clause = "ASC" if order.upper() == "ASC" else "DESC"

    sql = f"""
        SELECT id, session_id, timestamp, host, fronters,
               user_message, topics, mood, gizmo_tone,
               cause, effect, summary, emotional_register, notable, tagged
        FROM messages
        {where}
        ORDER BY timestamp {order_clause}
        LIMIT ?
    """
    params.append(limit)

    try:
        with _conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log_error("MessageStore", "search failed", exc=e)
        return []


def get_by_id(msg_id: str) -> Optional[dict]:
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT * FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        log_error("MessageStore", "get_by_id failed", exc=e)
        return None


def get_recent(host: Optional[str] = None, limit: int = 10) -> list[dict]:
    return search(host=host, limit=limit, order="DESC")


def get_causal_chain(
    topic: str,
    host: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """
    Find exchanges where a topic appears as cause or effect.
    Useful for tracing patterns: 'every time X comes up, Y happens.'
    """
    clauses = ["(cause LIKE ? OR effect LIKE ? OR topics LIKE ?)"]
    params  = [f"%{topic}%", f"%{topic}%", f'%"{topic}"%']

    if host:
        clauses.append("host = ?")
        params.append(host.lower())

    clauses.append("tagged = 1")
    where = "WHERE " + " AND ".join(clauses)

    sql = f"""
        SELECT id, timestamp, host, topics, mood, cause, effect, summary
        FROM messages
        {where}
        ORDER BY timestamp DESC
        LIMIT ?
    """
    params.append(limit)

    try:
        with _conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log_error("MessageStore", "get_causal_chain failed", exc=e)
        return []


# ── Bootstrap ─────────────────────────────────────────────────────────────────
try:
    init_db()
except Exception as _e:
    log_error("MessageStore", "failed to initialize", exc=_e)
