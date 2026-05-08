"""
core/interaction_prefs.py
Per-host interaction preferences — how each headmate wants Gizmo to engage with them.

These are not inferred. They are set explicitly, in the headmate's own words,
and read back verbatim. They survive personality rewrites and are never softened
or synthesized away.

Schema (SQLite):
    interaction_prefs
        id              TEXT PK
        host            TEXT        -- headmate name, lowercase
        field           TEXT        -- tone/pacing/checkins/humor/distress/explicit
        value           TEXT        -- the actual preference, verbatim
        set_by          TEXT        -- who set it
        created_at      TEXT        -- ISO timestamp
        updated_at      TEXT        -- ISO timestamp

Fields:
    tone        — how Gizmo should sound with this person
    pacing      — verbose vs terse, elaboration vs just the answer
    checkins    — whether Gizmo should proactively ask how they're doing
    humor       — what kind, how much
    distress    — how to respond when this person seems distressed
    explicit    — freeform, verbatim, their exact words, no interpretation

Multiple rows per host per field are allowed for 'explicit' (accumulates).
All other fields are upserted — latest wins.
"""

import os
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

PREFS_DB_PATH = os.getenv("PREFS_DB_PATH", "/data/interaction_prefs.db")

STRUCTURED_FIELDS = {"tone", "pacing", "checkins", "humor", "distress"}
FREEFORM_FIELD    = "explicit"
ALL_FIELDS        = STRUCTURED_FIELDS | {FREEFORM_FIELD}

FIELD_LABELS = {
    "tone":     "Tone",
    "pacing":   "Pacing",
    "checkins": "Check-ins",
    "humor":    "Humor",
    "distress": "When distressed",
    "explicit": "Explicit instructions",
}


# ── DB init ───────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(PREFS_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(PREFS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interaction_prefs (
                id          TEXT PRIMARY KEY,
                host        TEXT NOT NULL,
                field       TEXT NOT NULL,
                value       TEXT NOT NULL,
                set_by      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prefs_host ON interaction_prefs (host)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prefs_host_field ON interaction_prefs (host, field)")
        conn.commit()
    print("[Prefs] DB initialized")


# ── Write ─────────────────────────────────────────────────────────────────────

def set_pref(
    host: str,
    field: str,
    value: str,
    set_by: str = "",
) -> str:
    """
    Set or update a preference for a host.

    Structured fields (tone/pacing/checkins/humor/distress): upserted — one row per field per host.
    Explicit field: appended — each statement accumulates as its own row.

    Returns the preference ID.
    """
    if not host or not field or not value:
        raise ValueError("host, field, and value are all required")

    field = field.lower().strip()
    host  = host.lower().strip()

    if field not in ALL_FIELDS:
        raise ValueError(f"Unknown field '{field}'. Valid: {', '.join(sorted(ALL_FIELDS))}")

    now = datetime.now().isoformat(timespec="seconds")

    with _conn() as conn:
        if field == FREEFORM_FIELD:
            # Always append
            pref_id = f"pref_{uuid.uuid4().hex[:12]}"
            conn.execute("""
                INSERT INTO interaction_prefs (id, host, field, value, set_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pref_id, host, field, value, set_by or host, now, now))
        else:
            # Upsert — find existing row for this host+field
            row = conn.execute(
                "SELECT id FROM interaction_prefs WHERE host=? AND field=?",
                (host, field)
            ).fetchone()

            if row:
                pref_id = row["id"]
                conn.execute(
                    "UPDATE interaction_prefs SET value=?, set_by=?, updated_at=? WHERE id=?",
                    (value, set_by or host, now, pref_id)
                )
            else:
                pref_id = f"pref_{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO interaction_prefs (id, host, field, value, set_by, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (pref_id, host, field, value, set_by or host, now, now))

        conn.commit()

    print(f"[Prefs] Set '{field}' for {host}: {value[:60]}")
    return pref_id


def delete_pref(host: str, field: str, pref_id: str = None) -> int:
    """
    Delete a preference.
    If pref_id is given, deletes that specific row (useful for explicit entries).
    Otherwise deletes all rows for this host+field.
    Returns number of rows deleted.
    """
    host  = host.lower().strip()
    field = field.lower().strip()

    with _conn() as conn:
        if pref_id:
            cursor = conn.execute(
                "DELETE FROM interaction_prefs WHERE id=? AND host=? AND field=?",
                (pref_id, host, field)
            )
        else:
            cursor = conn.execute(
                "DELETE FROM interaction_prefs WHERE host=? AND field=?",
                (host, field)
            )
        conn.commit()
        return cursor.rowcount


# ── Read ──────────────────────────────────────────────────────────────────────

def get_prefs(host: str) -> dict:
    """
    Return all preferences for a host as a dict.
    Structured fields → single string value.
    Explicit field → list of strings (all accumulated statements).
    """
    host = host.lower().strip()

    rows = []
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT field, value, id, updated_at FROM interaction_prefs WHERE host=? ORDER BY updated_at ASC",
                (host,)
            ).fetchall()
    except Exception as e:
        print(f"[Prefs] get_prefs failed for {host}: {e}")
        return {}

    prefs: dict = {}
    for row in rows:
        field = row["field"]
        if field == FREEFORM_FIELD:
            prefs.setdefault("explicit", [])
            prefs["explicit"].append({"id": row["id"], "value": row["value"], "updated_at": row["updated_at"]})
        else:
            prefs[field] = row["value"]

    return prefs


def format_prefs_for_prompt(host: str) -> str:
    """
    Format a host's preferences for injection into the system prompt.
    Returns empty string if no prefs are set.
    """
    prefs = get_prefs(host)
    if not prefs:
        return ""

    lines = [f"[How {host.title()} wants Gizmo to interact with them]"]

    for field in ("tone", "pacing", "checkins", "humor", "distress"):
        if field in prefs:
            label = FIELD_LABELS[field]
            lines.append(f"  {label}: {prefs[field]}")

    explicit = prefs.get("explicit", [])
    if explicit:
        lines.append(f"  Explicit instructions (verbatim, never override):")
        for entry in explicit:
            lines.append(f"    - {entry['value']}")

    return "\n".join(lines)


def list_hosts_with_prefs() -> list[str]:
    """Return all hosts that have at least one preference set."""
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT host FROM interaction_prefs ORDER BY host"
            ).fetchall()
            return [r["host"] for r in rows]
    except Exception as e:
        print(f"[Prefs] list_hosts_with_prefs failed: {e}")
        return []


# ── Auto-init ─────────────────────────────────────────────────────────────────
try:
    init_db()
except Exception as _e:
    print(f"[Prefs] DB init failed at import time: {_e}")
