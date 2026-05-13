"""
core/message_store.py

SQLite store for every exchange Gizmo has.

Each row is one complete exchange: user message + Gizmo response,
with all the context from the Brief attached — host, fronters, topics,
emotional register, mood, tags, stage directions, lore.

This makes the past searchable at the message level, not just the
session level. Gizmo can find "the last time Princess mentioned dresses"
or "what we talked about around 3pm" without scanning transcript files.

Schema:
    messages (
        id                TEXT PRIMARY KEY,  -- msg_{session_id[:8]}_{timestamp_ms}
        session_id        TEXT,
        timestamp         TEXT,              -- ISO8601
        host              TEXT,
        fronters          TEXT,              -- JSON array
        user_message      TEXT,
        gizmo_response    TEXT,
        topics            TEXT,              -- JSON array
        emotional_register TEXT,
        mood              TEXT,
        tags              TEXT,              -- JSON array
        notable           INTEGER,           -- 0/1
        stage_directions  TEXT,              -- JSON array
        lore              TEXT               -- JSON array
    )

Indexes: host, timestamp, topics (text search), tags (text search)
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
    """Create table and indexes if they don't exist."""
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
                tags               TEXT,
                notable            INTEGER DEFAULT 0,
                stage_directions   TEXT,
                lore               TEXT
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_host      ON messages (host)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON messages (timestamp)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_session   ON messages (session_id)")
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
    Insert one exchange. Returns the generated message id.
    Safe to call from async context via asyncio.ensure_future.
    """
    ts_ms = int(timestamp * 1000)
    msg_id = f"msg_{session_id[:8]}_{ts_ms}"
    ts_str = datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")

    # Tags = union of topics + any extras passed in
    all_tags = list(set((tags or []) + topics))

    try:
        with _conn() as con:
            con.execute("""
                INSERT OR REPLACE INTO messages (
                    id, session_id, timestamp, host, fronters,
                    user_message, gizmo_response, topics, emotional_register,
                    mood, tags, notable, stage_directions, lore
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                msg_id,
                session_id,
                ts_str,
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
            id=msg_id,
            host=host or "unknown",
            topics=topics,
        )
        return msg_id
    except Exception as e:
        log_error("MessageStore", "insert_exchange failed", exc=e)
        return msg_id


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    host: Optional[str] = None,
    topics: Optional[list] = None,
    tags: Optional[list] = None,
    emotional_register: Optional[str] = None,
    mood: Optional[str] = None,
    since: Optional[str] = None,     # ISO8601 date string e.g. "2026-05-12"
    until: Optional[str] = None,     # ISO8601 date string
    keyword: Optional[str] = None,   # freetext search in user_message
    limit: int = 20,
    order: str = "DESC",
) -> list[dict]:
    """
    Search messages by any combination of filters.
    Returns list of lightweight stub dicts (no full response text).
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

    # Topic/tag filters — JSON array text search
    if topics:
        for t in topics:
            clauses.append("topics LIKE ?")
            params.append(f'%"{t}"%')

    if tags:
        for t in tags:
            clauses.append("tags LIKE ?")
            params.append(f'%"{t}"%')

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_clause = "ASC" if order.upper() == "ASC" else "DESC"

    sql = f"""
        SELECT id, session_id, timestamp, host, fronters,
               user_message, topics, emotional_register, mood, tags, notable
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
    """Fetch a single message by id, including full response text."""
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT * FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        log_error("MessageStore", "get_by_id failed", exc=e)
        return None


def get_recent(
    host: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Return the N most recent messages, optionally filtered by host."""
    return search(host=host, limit=limit, order="DESC")


# ── Bootstrap ─────────────────────────────────────────────────────────────────
try:
    init_db()
except Exception as _e:
    log_error("MessageStore", "failed to initialize", exc=_e)
