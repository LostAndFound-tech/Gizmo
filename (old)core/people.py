"""
core/people.py
Unified people model — headmates and external contacts in one schema.

Everyone is a person. Headmates have external=False. External contacts
have external=True and a note that they don't share the body. Beyond
that the model is identical — just sparser for people Gizmo knows less about.

Tables:
    people                  — core identity and psychological model
    people_patterns         — behavioral patterns, earned from action log over time
    people_relationships    — the shared core of a relationship between two people
    relationship_perspectives — each person's side of a relationship
    encounters              — timestamped references to real events involving a relationship

All timestamps via tz_now(). Never datetime.now().

Usage:
    from core.people import people_store

    # Get or create a person
    person = people_store.get_or_create("jess", external=False)

    # Log an encounter
    people_store.log_encounter(
        relationship_id=rel_id,
        reported_by="jess",
        session_file="2026-05-31_jess.txt",
        message_id="msg_abc123",
        summary="Jess mentioned Dave being difficult at work",
        emotional_tone="frustrated",
        flagged=True,
        follow_up="What did Dave do specifically?",
    )
"""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/data")
DB_PATH  = DATA_DIR / "people.db"


# ── Schema ────────────────────────────────────────────────────────────────────

def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:

        # ── People ────────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS people (
                person_id       TEXT PRIMARY KEY,
                name            TEXT NOT NULL UNIQUE,
                external        INTEGER NOT NULL DEFAULT 0,  -- 0=headmate, 1=external

                -- Identity
                pronouns        TEXT,
                role            TEXT,    -- role in system or relationship to system
                first_seen      TEXT,    -- tz_now() iso
                last_seen       TEXT,

                -- Nature
                nature          TEXT,    -- fundamental orientation, plain language
                direction       TEXT,    -- what forward looks like for them specifically

                -- Relational model with Gizmo
                gizmo_dynamic   TEXT,    -- texture of their relationship with Gizmo
                gizmo_needs     TEXT,    -- what they need from Gizmo
                gizmo_trust     REAL,    -- 0.0–1.0
                gizmo_confidence REAL,   -- how confident Gizmo is in his read of them

                -- Emotional baseline
                default_register TEXT,   -- neutral/positive/elevated/distress/subdued
                dysregulation   TEXT,    -- what dysregulation looks like for them
                recovery        TEXT,    -- what recovery looks like

                -- Unknowns — honest gaps
                unknowns        TEXT,    -- JSON array of plain language unknowns

                -- Meta
                notes           TEXT,    -- anything that doesn't fit elsewhere
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_people_name ON people(name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_people_external ON people(external)"
        )

        # ── Patterns — earned from action log, not assumed ────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS people_patterns (
                pattern_id      TEXT PRIMARY KEY,
                person_id       TEXT NOT NULL REFERENCES people(person_id),
                pattern         TEXT NOT NULL,   -- plain language description
                first_observed  TEXT NOT NULL,   -- tz_now() iso
                last_observed   TEXT NOT NULL,
                occurrence_count INTEGER DEFAULT 1,
                confidence      REAL DEFAULT 0.5,
                tags            TEXT,            -- JSON array
                notes           TEXT,
                FOREIGN KEY (person_id) REFERENCES people(person_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_person "
            "ON people_patterns(person_id)"
        )

        # ── Relationships — shared core ───────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                party_a         TEXT NOT NULL,   -- person name, lowercase
                party_b         TEXT NOT NULL,   -- person name, lowercase
                type            TEXT NOT NULL,   -- "intra-system" | "external"

                dynamic         TEXT,   -- plain language texture of the relationship
                history         TEXT,   -- what Gizmo knows about how this developed
                shared_patterns TEXT,   -- JSON array of observed shared behaviors
                tension_points  TEXT,   -- JSON array of known friction points
                unknowns        TEXT,   -- JSON array of what Gizmo doesn't know

                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,

                UNIQUE(party_a, party_b)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_party_a "
            "ON relationships(party_a)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_party_b "
            "ON relationships(party_b)"
        )

        # ── Relationship perspectives — each person's side ────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationship_perspectives (
                perspective_id  TEXT PRIMARY KEY,
                relationship_id TEXT NOT NULL REFERENCES relationships(relationship_id),
                subject         TEXT NOT NULL,   -- whose perspective this is

                feeling_toward  TEXT,   -- how subject feels about the other party
                needs_from      TEXT,   -- what subject needs from this relationship
                gives_to        TEXT,   -- what subject contributes to it
                comfort_level   REAL,   -- 0.0–1.0
                trust_level     REAL,   -- 0.0–1.0
                unresolved      TEXT,   -- JSON array of unresolved things
                gizmo_confidence REAL,  -- how confident Gizmo is in this read

                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,

                UNIQUE(relationship_id, subject),
                FOREIGN KEY (relationship_id) REFERENCES relationships(relationship_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_perspectives_relationship "
            "ON relationship_perspectives(relationship_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_perspectives_subject "
            "ON relationship_perspectives(subject)"
        )

        # ── Encounters — timestamped references to real events ────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS encounters (
                encounter_id    TEXT PRIMARY KEY,
                relationship_id TEXT NOT NULL REFERENCES relationships(relationship_id),
                reported_by     TEXT NOT NULL,   -- which headmate mentioned it
                source          TEXT NOT NULL,   -- "direct" | "ambient"

                session_file    TEXT NOT NULL,   -- filename of source conversation
                message_id      TEXT,            -- message ID within that session

                timestamp       TEXT NOT NULL,   -- tz_now() when logged
                summary         TEXT NOT NULL,   -- brief plain language description
                emotional_tone  TEXT,            -- how reporter seemed recounting it
                confidence      REAL DEFAULT 1.0, -- lower for ambient-sourced

                flagged         INTEGER DEFAULT 0,  -- worth following up on
                follow_up       TEXT,               -- if flagged, what Gizmo wants to know

                created_at      TEXT NOT NULL,
                FOREIGN KEY (relationship_id) REFERENCES relationships(relationship_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_encounters_relationship "
            "ON encounters(relationship_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_encounters_reported_by "
            "ON encounters(reported_by)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_encounters_flagged "
            "ON encounters(flagged)"
        )

        conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json(value: Optional[list]) -> Optional[str]:
    return json.dumps(value) if value is not None else None


def _from_json(value: Optional[str]) -> list:
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Deserialize JSON fields
    for field in ("unknowns", "shared_patterns", "tension_points", "unresolved", "tags"):
        if field in d:
            d[field] = _from_json(d[field])
    return d


# ── People Store ──────────────────────────────────────────────────────────────

class PeopleStore:
    """
    Singleton. Unified store for headmates and external contacts.
    """

    def __init__(self):
        _init_db()

    # ── People ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[dict]:
        """Fetch a person by name. Returns None if not found."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM people WHERE name = ?",
                (name.lower(),)
            ).fetchone()
            return _row_to_dict(row) if row else None

    def get_or_create(self, name: str, external: bool = False) -> dict:
        """
        Fetch a person by name, creating a stub if they don't exist yet.
        Stub creation fires on first encounter — name from context window is enough.
        """
        existing = self.get(name)
        if existing:
            return existing

        now = tz_now().isoformat()
        person_id = str(uuid.uuid4())

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO people (
                    person_id, name, external,
                    unknowns,
                    created_at, updated_at, first_seen, last_seen,
                    gizmo_trust, gizmo_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                person_id,
                name.lower(),
                1 if external else 0,
                json.dumps([f"Everything — first encounter with {name}"]),
                now, now, now, now,
                0.0,   # trust starts at zero, earned
                0.0,   # confidence starts at zero
            ))
            conn.commit()

        log_event("PeopleStore", "PERSON_CREATED",
            name=name,
            external=external,
        )

        return self.get(name)

    def update(self, name: str, **fields) -> None:
        """Update fields on a person record."""
        if not fields:
            return

        now = tz_now().isoformat()
        fields["updated_at"] = now

        # Serialize list fields
        for key in ("unknowns",):
            if key in fields and isinstance(fields[key], list):
                fields[key] = json.dumps(fields[key])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [name.lower()]

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                f"UPDATE people SET {set_clause} WHERE name = ?",
                values
            )
            conn.commit()

    def touch(self, name: str) -> None:
        """Update last_seen timestamp."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE people SET last_seen = ?, updated_at = ? WHERE name = ?",
                (tz_now().isoformat(), tz_now().isoformat(), name.lower())
            )
            conn.commit()

    def all_headmates(self) -> list[dict]:
        """Return all non-external people."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM people WHERE external = 0 ORDER BY name"
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    # ── Patterns ──────────────────────────────────────────────────────────────

    def add_pattern(
        self,
        name:       str,
        pattern:    str,
        tags:       Optional[list] = None,
        confidence: float = 0.5,
        notes:      Optional[str] = None,
    ) -> None:
        """Add a behavioral pattern for a person."""
        person = self.get(name)
        if not person:
            return

        now = tz_now().isoformat()

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO people_patterns (
                    pattern_id, person_id, pattern,
                    first_observed, last_observed,
                    confidence, tags, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                person["person_id"],
                pattern,
                now, now,
                confidence,
                _json(tags),
                notes,
            ))
            conn.commit()

    def get_patterns(self, name: str) -> list[dict]:
        """Get all patterns for a person, most frequent first."""
        person = self.get(name)
        if not person:
            return []

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM people_patterns WHERE person_id = ? "
                "ORDER BY occurrence_count DESC, confidence DESC",
                (person["person_id"],)
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    # ── Relationships ─────────────────────────────────────────────────────────

    def get_relationship(self, party_a: str, party_b: str) -> Optional[dict]:
        """Fetch relationship between two people. Order-insensitive."""
        a, b = sorted([party_a.lower(), party_b.lower()])
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM relationships WHERE party_a = ? AND party_b = ?",
                (a, b)
            ).fetchone()
            return _row_to_dict(row) if row else None

    def get_or_create_relationship(
        self,
        party_a: str,
        party_b: str,
        rel_type: str = "intra-system",
    ) -> dict:
        """Fetch or create a relationship between two people."""
        existing = self.get_relationship(party_a, party_b)
        if existing:
            return existing

        a, b = sorted([party_a.lower(), party_b.lower()])
        now = tz_now().isoformat()
        rel_id = str(uuid.uuid4())

        # Ensure both people exist
        self.get_or_create(party_a, external=(rel_type == "external"))
        self.get_or_create(party_b, external=(rel_type == "external"))

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO relationships (
                    relationship_id, party_a, party_b, type,
                    unknowns, shared_patterns, tension_points,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rel_id, a, b, rel_type,
                json.dumps([f"Nature of relationship between {a} and {b}"]),
                json.dumps([]),
                json.dumps([]),
                now, now,
            ))

            # Create perspective stubs for both parties
            for subject in [a, b]:
                conn.execute("""
                    INSERT INTO relationship_perspectives (
                        perspective_id, relationship_id, subject,
                        unresolved,
                        comfort_level, trust_level, gizmo_confidence,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    rel_id, subject,
                    json.dumps([f"How {subject} feels about this relationship"]),
                    0.0, 0.0, 0.0,
                    now, now,
                ))

            conn.commit()

        log_event("PeopleStore", "RELATIONSHIP_CREATED",
            party_a=a, party_b=b, type=rel_type,
        )

        return self.get_relationship(a, b)

    def update_relationship(self, party_a: str, party_b: str, **fields) -> None:
        """Update fields on a relationship record."""
        rel = self.get_relationship(party_a, party_b)
        if not rel:
            return

        now = tz_now().isoformat()
        fields["updated_at"] = now

        for key in ("shared_patterns", "tension_points", "unknowns"):
            if key in fields and isinstance(fields[key], list):
                fields[key] = json.dumps(fields[key])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [rel["relationship_id"]]

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                f"UPDATE relationships SET {set_clause} WHERE relationship_id = ?",
                values
            )
            conn.commit()

    def update_perspective(
        self,
        party_a:  str,
        party_b:  str,
        subject:  str,
        **fields,
    ) -> None:
        """Update one perspective on a relationship."""
        rel = self.get_relationship(party_a, party_b)
        if not rel:
            return

        now = tz_now().isoformat()
        fields["updated_at"] = now

        for key in ("unresolved",):
            if key in fields and isinstance(fields[key], list):
                fields[key] = json.dumps(fields[key])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [rel["relationship_id"], subject.lower()]

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                f"UPDATE relationship_perspectives SET {set_clause} "
                f"WHERE relationship_id = ? AND subject = ?",
                values
            )
            conn.commit()

    def get_perspective(
        self,
        party_a: str,
        party_b: str,
        subject: str,
    ) -> Optional[dict]:
        """Get one person's perspective on their relationship with another."""
        rel = self.get_relationship(party_a, party_b)
        if not rel:
            return None

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM relationship_perspectives "
                "WHERE relationship_id = ? AND subject = ?",
                (rel["relationship_id"], subject.lower())
            ).fetchone()
            return _row_to_dict(row) if row else None

    def get_all_relationships(self, name: str) -> list[dict]:
        """Get all relationships involving a person."""
        n = name.lower()
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM relationships WHERE party_a = ? OR party_b = ? "
                "ORDER BY updated_at DESC",
                (n, n)
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    # ── Encounters ────────────────────────────────────────────────────────────

    def log_encounter(
        self,
        relationship_id: str,
        reported_by:     str,
        session_file:    str,
        summary:         str,
        source:          str = "direct",   # "direct" | "ambient"
        message_id:      Optional[str] = None,
        emotional_tone:  Optional[str] = None,
        confidence:      float = 1.0,
        flagged:         bool = False,
        follow_up:       Optional[str] = None,
    ) -> str:
        """Log a real-world encounter. Returns the encounter_id."""
        now = tz_now().isoformat()
        encounter_id = str(uuid.uuid4())

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO encounters (
                    encounter_id, relationship_id, reported_by, source,
                    session_file, message_id,
                    timestamp, summary, emotional_tone, confidence,
                    flagged, follow_up,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                encounter_id,
                relationship_id,
                reported_by.lower(),
                source,
                session_file,
                message_id,
                now,
                summary,
                emotional_tone,
                confidence,
                1 if flagged else 0,
                follow_up,
                now,
            ))
            conn.commit()

        log_event("PeopleStore", "ENCOUNTER_LOGGED",
            relationship=relationship_id[:8],
            reported_by=reported_by,
            source=source,
            flagged=flagged,
        )

        return encounter_id

    def get_encounters(
        self,
        relationship_id: Optional[str] = None,
        reported_by:     Optional[str] = None,
        flagged_only:    bool = False,
        limit:           int = 50,
    ) -> list[dict]:
        """Fetch encounters, optionally filtered."""
        query  = "SELECT * FROM encounters WHERE 1=1"
        params: list = []

        if relationship_id:
            query += " AND relationship_id = ?"
            params.append(relationship_id)

        if reported_by:
            query += " AND reported_by = ?"
            params.append(reported_by.lower())

        if flagged_only:
            query += " AND flagged = 1"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_flagged_encounters(self, reported_by: Optional[str] = None) -> list[dict]:
        """Convenience — get all flagged encounters, optionally for one headmate."""
        return self.get_encounters(reported_by=reported_by, flagged_only=True)


people_store = PeopleStore()
