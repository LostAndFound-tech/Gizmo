"""
core/question_bank.py
The intent architecture — questions, says, dos, goals, and temperatures.

Not a queue. A bank. Items are retrieved by context match, not by order.
Order is determined at retrieval time based on what's happening right now.

Four intent types:
    questions   — things Gizmo wants to understand
    says        — things Gizmo wants to express unprompted  
    dos         — actions and tasks Gizmo wants to take
    goals       — long horizon relational intentions (~3 months)
                  goals decompose into real sub-items as evidence accumulates

Shared tagging system across all four types.
Different activation logic per type:
    questions   — context match (topic tags align with conversation)
    says        — register match (emotional tone is right)
    dos         — trigger match (condition becomes true)
    goals       — always active, shapes behavior, evaluated in daily psych pass

Temperature system:
    Per-headmate dimensions controlling Gizmo's behavioral register.
    Core dimensions: mood_match, silliness
    Headmate-specific: dominance, aggression, warmth, etc. — added as needed.
    Some dimensions auto-adjust from context. Others are fixed until changed.

Question perspectives:
    Questions don't close when one person answers — they close per perspective.
    Each headmate has their own resolution status on every question.
    "No opinion" is a valid closed state and doesn't get re-asked.

Self-cleaning:
    Before surfacing any question, Gizmo checks his knowledge base.
    If the answer is already there, the question closes silently.
    The bank stays lean automatically.

Usage:
    from core.question_bank import question_bank

    # Add a question
    q_id = question_bank.add_question(
        question="What does Oren think humanity's timeline to Mars looks like?",
        resolution_query="Oren's belief about Mars colonization timeline",
        tags=["space", "futurism", "science"],
        importance=0.6,
    )

    # Surface questions for current context
    questions = question_bank.surface_questions(
        tags=["space", "science"],
        headmate="oren",
        limit=3,
    )

    # Close a perspective
    question_bank.close_perspective(q_id, headmate="oren", status="resolved",
                                    response="He thinks 50 years minimum")
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
DB_PATH  = DATA_DIR / "question_bank.db"

# ── Schema ────────────────────────────────────────────────────────────────────

def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:

        # ── Tags ──────────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                tag_id      TEXT PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                category    TEXT,   -- "topic" | "person" | "emotion" | "context" | etc.
                created_at  TEXT NOT NULL
            )
        """)

        # ── Item tags — shared junction table for all intent types ────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS item_tags (
                item_id     TEXT NOT NULL,
                item_type   TEXT NOT NULL,   -- "question" | "say" | "do" | "goal"
                tag         TEXT NOT NULL,
                PRIMARY KEY (item_id, tag)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_item_tags_tag ON item_tags(tag)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_item_tags_item ON item_tags(item_id)"
        )

        # ── Questions ─────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                question_id         TEXT PRIMARY KEY,
                question            TEXT NOT NULL,
                resolution_query    TEXT NOT NULL,  -- what would constitute an answer
                for_headmate        TEXT,           -- NULL until explicitly assigned
                importance          REAL DEFAULT 0.5,
                importance_rationale TEXT,
                readiness           TEXT DEFAULT 'any',  -- "any"|"intimate"|"headmate_initiates"
                match_mood          INTEGER DEFAULT 0,   -- bypasses mood filter if 1
                source_event        TEXT,           -- event_id if generated from action log
                source_session      TEXT,           -- session file if generated from conversation
                clarification_type  INTEGER DEFAULT 0,  -- 1 = entity type clarification
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL,
                resolved_at         TEXT            -- set when ALL perspectives resolved
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_questions_for "
            "ON questions(for_headmate)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_questions_resolved "
            "ON questions(resolved_at)"
        )

        # ── Question perspectives — one row per headmate per question ─────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS question_perspectives (
                perspective_id  TEXT PRIMARY KEY,
                question_id     TEXT NOT NULL REFERENCES questions(question_id),
                headmate        TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'open',
                    -- open | asked | resolved | none | deflected | dormant
                response        TEXT,           -- what they said
                response_quality TEXT,          -- psychology engine classification
                    -- "answered" | "deflected:humor" | "deflected:self_degradation" |
                    -- "avoided" | "partial" | "none_stated"
                reason_open     TEXT,           -- why still open after response
                asked_at        TEXT,
                resolved_at     TEXT,
                session_file    TEXT,           -- session where it was asked/answered
                UNIQUE(question_id, headmate),
                FOREIGN KEY (question_id) REFERENCES questions(question_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_perspectives_question "
            "ON question_perspectives(question_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_perspectives_headmate "
            "ON question_perspectives(headmate)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_perspectives_status "
            "ON question_perspectives(status)"
        )

        # ── Says ──────────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS says (
                say_id          TEXT PRIMARY KEY,
                content         TEXT NOT NULL,      -- what Gizmo wants to express
                for_headmate    TEXT,               -- NULL = any headmate
                readiness       TEXT DEFAULT 'any', -- register condition
                importance      REAL DEFAULT 0.5,
                match_mood      INTEGER DEFAULT 0,
                source_goal     TEXT,               -- goal_id if spawned by a goal
                source_event    TEXT,
                expires_at      TEXT,               -- say becomes irrelevant after this
                delivered       INTEGER DEFAULT 0,
                delivered_at    TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_says_for ON says(for_headmate)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_says_delivered ON says(delivered)"
        )

        # ── Dos ───────────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dos (
                do_id           TEXT PRIMARY KEY,
                action          TEXT NOT NULL,      -- what Gizmo wants to do
                for_headmate    TEXT,
                trigger_type    TEXT NOT NULL,
                    -- "time" | "event" | "threshold" | "manual"
                trigger_condition TEXT,             -- plain language description
                trigger_value   TEXT,               -- JSON — specific condition data
                importance      REAL DEFAULT 0.5,
                source_goal     TEXT,
                source_event    TEXT,
                status          TEXT DEFAULT 'pending',
                    -- pending | triggered | done | cancelled
                triggered_at    TEXT,
                completed_at    TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dos_status ON dos(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dos_trigger ON dos(trigger_type)"
        )

        # ── Goals ─────────────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                goal_id         TEXT PRIMARY KEY,
                description     TEXT NOT NULL,
                purpose         TEXT NOT NULL,      -- why this serves the headmate
                subject         TEXT,               -- headmate this concerns
                status          TEXT DEFAULT 'active',
                    -- active | achieved | abandoned
                origin          TEXT DEFAULT 'autonomous',
                    -- autonomous | explicit
                target_date     TEXT,               -- ~3 months out by default
                progress_notes  TEXT,               -- running summary
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                resolved_at     TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_goals_subject ON goals(subject)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)"
        )

        # ── Goal intentions — plain language sub-orientations ─────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goal_intentions (
                intention_id    TEXT PRIMARY KEY,
                goal_id         TEXT NOT NULL REFERENCES goals(goal_id),
                intention       TEXT NOT NULL,      -- plain language orientation
                status          TEXT DEFAULT 'active',
                    -- active | progressing | achieved | abandoned
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_intentions_goal "
            "ON goal_intentions(goal_id)"
        )

        # ── Goal psych report — daily pass entries ────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goal_psych_report (
                entry_id        TEXT PRIMARY KEY,
                goal_id         TEXT NOT NULL REFERENCES goals(goal_id),
                intention_id    TEXT REFERENCES goal_intentions(intention_id),
                date            TEXT NOT NULL,      -- YYYY-MM-DD in local time
                progress        INTEGER NOT NULL,   -- 1 = yes, 0 = no
                what            TEXT,               -- what happened or was attempted
                how             TEXT,               -- how it was done
                outcome         TEXT,               -- how it landed
                blocked_by      TEXT,               -- if no progress, why
                created_at      TEXT NOT NULL,
                FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_psych_report_goal "
            "ON goal_psych_report(goal_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_psych_report_date "
            "ON goal_psych_report(date)"
        )

        # ── Goal sub-item links — questions/says/dos spawned by goals ─────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goal_sub_items (
                link_id         TEXT PRIMARY KEY,
                goal_id         TEXT NOT NULL REFERENCES goals(goal_id),
                item_type       TEXT NOT NULL,  -- "question" | "say" | "do"
                item_id         TEXT NOT NULL,
                spawned_at      TEXT NOT NULL,
                FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sub_items_goal "
            "ON goal_sub_items(goal_id)"
        )

        # ── Temperatures ──────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS temperatures (
                temp_id         TEXT PRIMARY KEY,
                headmate        TEXT,               -- NULL = applies to all
                dimension       TEXT NOT NULL,      -- "mood_match"|"silliness"|custom
                value           REAL NOT NULL DEFAULT 0.5,
                default_value   REAL NOT NULL DEFAULT 0.5,
                auto_adjust     INTEGER DEFAULT 0,  -- 1 = Gizmo can regulate himself
                note            TEXT,               -- why this dimension exists
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                UNIQUE(headmate, dimension)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_temperatures_headmate "
            "ON temperatures(headmate)"
        )

        conn.commit()

    # Seed core temperature dimensions
    _seed_core_temperatures()


def _seed_core_temperatures() -> None:
    """Ensure core temperature dimensions exist for the global profile."""
    core = [
        ("mood_match", 0.5, True,
         "How strictly Gizmo filters questions to the current register"),
        ("silliness",  0.3, True,
         "How willing Gizmo is to introduce levity"),
    ]
    now = tz_now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        for dimension, default, auto, note in core:
            conn.execute("""
                INSERT OR IGNORE INTO temperatures (
                    temp_id, headmate, dimension, value, default_value,
                    auto_adjust, note, created_at, updated_at
                ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), dimension, default, default,
                  1 if auto else 0, note, now, now))
        conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json(value) -> Optional[str]:
    return json.dumps(value) if value is not None else None


def _tags_for(item_id: str, conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT tag FROM item_tags WHERE item_id = ?", (item_id,)
    ).fetchall()
    return [r[0] for r in rows]


def _insert_tags(item_id: str, item_type: str, tags: list[str],
                 conn: sqlite3.Connection) -> None:
    for tag in tags:
        conn.execute(
            "INSERT OR IGNORE INTO item_tags (item_id, item_type, tag) "
            "VALUES (?, ?, ?)",
            (item_id, item_type, tag.lower().strip())
        )


# ── Question Bank ─────────────────────────────────────────────────────────────

class QuestionBank:
    """
    Singleton. Full intent architecture.
    """

    def __init__(self):
        _init_db()

    # ── Questions ─────────────────────────────────────────────────────────────

    def add_question(
        self,
        question:           str,
        resolution_query:   str,
        tags:               list[str],
        importance:         float = 0.5,
        importance_rationale: Optional[str] = None,
        readiness:          str = "any",
        match_mood:         bool = False,
        for_headmate:       Optional[str] = None,
        source_event:       Optional[str] = None,
        source_session:     Optional[str] = None,
        clarification_type: bool = False,
    ) -> str:
        """Add a question to the bank. Returns question_id."""
        now = tz_now().isoformat()
        q_id = str(uuid.uuid4())

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO questions (
                    question_id, question, resolution_query,
                    for_headmate, importance, importance_rationale,
                    readiness, match_mood,
                    source_event, source_session,
                    clarification_type,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                q_id, question, resolution_query,
                for_headmate.lower() if for_headmate else None,
                importance, importance_rationale,
                readiness, 1 if match_mood else 0,
                source_event, source_session,
                1 if clarification_type else 0,
                now, now,
            ))
            _insert_tags(q_id, "question", tags, conn)
            conn.commit()

        log_event("QuestionBank", "QUESTION_ADDED",
            question=question[:60],
            tags=tags,
            importance=importance,
            clarification=clarification_type,
        )
        return q_id

    def add_clarification_question(
        self,
        name:           str,
        source_session: Optional[str] = None,
        source_event:   Optional[str] = None,
    ) -> str:
        """
        Convenience — add an entity type clarification question for an unknown name.
        High importance, any readiness, fires as soon as naturally possible.
        """
        return self.add_question(
            question=f"Is {name} a headmate or someone external to the system?",
            resolution_query=f"{name} entity type — headmate or external",
            tags=["clarification", "entity_type", name.lower()],
            importance=0.9,
            importance_rationale="Entity type determines how all downstream data is filed",
            readiness="any",
            match_mood=False,
            source_session=source_session,
            source_event=source_event,
            clarification_type=True,
        )

    def get_question(self, question_id: str) -> Optional[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM questions WHERE question_id = ?",
                (question_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["tags"] = _tags_for(question_id, conn)
            return d

    def surface_questions(
        self,
        tags:       list[str],
        headmate:   Optional[str] = None,
        limit:      int = 5,
        include_match_mood: bool = True,
    ) -> list[dict]:
        """
        Retrieve questions relevant to current context.
        Matches on tag intersection. Returns highest importance first.
        Filters to questions where this headmate's perspective is open or unasked.
        """
        if not tags:
            return []

        placeholders = ",".join("?" * len(tags))
        params: list = tags + [limit * 3]  # fetch extra, filter down

        # Questions with at least one matching tag, not yet resolved overall
        query = f"""
            SELECT DISTINCT q.*,
                COUNT(it.tag) as tag_matches
            FROM questions q
            JOIN item_tags it ON it.item_id = q.question_id
            WHERE it.tag IN ({placeholders})
              AND q.resolved_at IS NULL
              AND (q.for_headmate IS NULL
                   OR q.for_headmate = ?)
            GROUP BY q.question_id
            ORDER BY tag_matches DESC, q.importance DESC
            LIMIT ?
        """
        params = tags + [headmate.lower() if headmate else None, limit * 3]
        # Handle NULL for_headmate properly
        query = f"""
            SELECT DISTINCT q.*,
                COUNT(it.tag) as tag_matches
            FROM questions q
            JOIN item_tags it ON it.item_id = q.question_id
            WHERE it.tag IN ({placeholders})
              AND q.resolved_at IS NULL
              AND (q.for_headmate IS NULL
                   OR q.for_headmate = ?)
            GROUP BY q.question_id
            ORDER BY tag_matches DESC, q.importance DESC
            LIMIT ?
        """

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["tags"] = _tags_for(d["question_id"], conn)

                # Filter to questions where this headmate's perspective is open
                if headmate:
                    perspective = conn.execute(
                        "SELECT status FROM question_perspectives "
                        "WHERE question_id = ? AND headmate = ?",
                        (d["question_id"], headmate.lower())
                    ).fetchone()

                    if perspective:
                        status = perspective["status"]
                        # Skip if already resolved, stated no opinion, or dormant
                        if status in ("resolved", "none", "dormant"):
                            continue
                    # No perspective row = never surfaced to this headmate = open

                results.append(d)
                if len(results) >= limit:
                    break

            return results

    def open_perspective(
        self,
        question_id: str,
        headmate:    str,
    ) -> None:
        """Create or reset a perspective row for a headmate on a question."""
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO question_perspectives (
                    perspective_id, question_id, headmate, status, created_at
                ) VALUES (?, ?, ?, 'open', ?)
            """, (str(uuid.uuid4()), question_id, headmate.lower(), now))
            conn.commit()

    def mark_asked(
        self,
        question_id:  str,
        headmate:     str,
        session_file: Optional[str] = None,
    ) -> None:
        """Mark a question as having been asked to a headmate."""
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO question_perspectives (
                    perspective_id, question_id, headmate,
                    status, asked_at, session_file, created_at
                ) VALUES (?, ?, ?, 'asked', ?, ?, ?)
                ON CONFLICT(question_id, headmate) DO UPDATE SET
                    status = 'asked',
                    asked_at = excluded.asked_at,
                    session_file = excluded.session_file
            """, (
                str(uuid.uuid4()), question_id, headmate.lower(),
                now, session_file, now,
            ))
            conn.commit()

    def close_perspective(
        self,
        question_id:      str,
        headmate:         str,
        status:           str,   # resolved|none|deflected|dormant
        response:         Optional[str] = None,
        response_quality: Optional[str] = None,
        reason_open:      Optional[str] = None,
        session_file:     Optional[str] = None,
    ) -> None:
        """
        Close a headmate's perspective on a question.
        Checks if all perspectives are now resolved — closes the question if so.
        """
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO question_perspectives (
                    perspective_id, question_id, headmate,
                    status, response, response_quality, reason_open,
                    resolved_at, session_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(question_id, headmate) DO UPDATE SET
                    status           = excluded.status,
                    response         = excluded.response,
                    response_quality = excluded.response_quality,
                    reason_open      = excluded.reason_open,
                    resolved_at      = excluded.resolved_at,
                    session_file     = excluded.session_file
            """, (
                str(uuid.uuid4()), question_id, headmate.lower(),
                status, response, response_quality, reason_open,
                now, session_file, now,
            ))

            # Check if any perspectives are still open
            open_count = conn.execute("""
                SELECT COUNT(*) FROM question_perspectives
                WHERE question_id = ?
                  AND status IN ('open', 'asked', 'deflected')
            """, (question_id,)).fetchone()[0]

            if open_count == 0:
                conn.execute(
                    "UPDATE questions SET resolved_at = ?, updated_at = ? "
                    "WHERE question_id = ?",
                    (now, now, question_id)
                )
                log_event("QuestionBank", "QUESTION_RESOLVED",
                    question_id=question_id[:8]
                )

            conn.commit()

    def reassess_importance(
        self,
        question_id:  str,
        importance:   float,
        rationale:    str,
    ) -> None:
        """
        Re-score a question's importance. Called by psych engine batch pass
        when new data changes the significance of an old question.
        """
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE questions SET
                    importance = ?,
                    importance_rationale = ?,
                    updated_at = ?
                WHERE question_id = ?
            """, (importance, rationale, now, question_id))
            conn.commit()

    def assign_to_headmate(self, question_id: str, headmate: str) -> None:
        """Assign a question to a specific headmate. Migration from unowned."""
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE questions SET for_headmate = ?, updated_at = ? "
                "WHERE question_id = ?",
                (headmate.lower(), now, question_id)
            )
            conn.commit()

    # ── Says ──────────────────────────────────────────────────────────────────

    def add_say(
        self,
        content:        str,
        tags:           list[str],
        for_headmate:   Optional[str] = None,
        readiness:      str = "any",
        importance:     float = 0.5,
        match_mood:     bool = False,
        source_goal:    Optional[str] = None,
        source_event:   Optional[str] = None,
        expires_at:     Optional[str] = None,
    ) -> str:
        """Add something Gizmo wants to express. Returns say_id."""
        now = tz_now().isoformat()
        say_id = str(uuid.uuid4())

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO says (
                    say_id, content, for_headmate,
                    readiness, importance, match_mood,
                    source_goal, source_event, expires_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                say_id, content,
                for_headmate.lower() if for_headmate else None,
                readiness, importance, 1 if match_mood else 0,
                source_goal, source_event, expires_at,
                now, now,
            ))
            _insert_tags(say_id, "say", tags, conn)
            conn.commit()

        return say_id

    def deliver_say(self, say_id: str) -> None:
        """Mark a say as delivered."""
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE says SET delivered = 1, delivered_at = ?, updated_at = ? "
                "WHERE say_id = ?",
                (now, now, say_id)
            )
            conn.commit()

    def surface_says(
        self,
        tags:       list[str],
        headmate:   Optional[str] = None,
        limit:      int = 3,
    ) -> list[dict]:
        """Retrieve undelivered says relevant to current context."""
        if not tags:
            return []

        placeholders = ",".join("?" * len(tags))
        params = tags + [headmate.lower() if headmate else None, limit]

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"""
                SELECT DISTINCT s.*
                FROM says s
                JOIN item_tags it ON it.item_id = s.say_id
                WHERE it.tag IN ({placeholders})
                  AND s.delivered = 0
                  AND (s.expires_at IS NULL OR s.expires_at > ?)
                  AND (s.for_headmate IS NULL OR s.for_headmate = ?)
                ORDER BY s.importance DESC
                LIMIT ?
            """, tags + [tz_now().isoformat(),
                         headmate.lower() if headmate else None,
                         limit]).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["tags"] = _tags_for(d["say_id"], conn)
                results.append(d)
            return results

    # ── Dos ───────────────────────────────────────────────────────────────────

    def add_do(
        self,
        action:             str,
        trigger_type:       str,
        tags:               list[str],
        for_headmate:       Optional[str] = None,
        trigger_condition:  Optional[str] = None,
        trigger_value:      Optional[dict] = None,
        importance:         float = 0.5,
        source_goal:        Optional[str] = None,
        source_event:       Optional[str] = None,
    ) -> str:
        """Add an action Gizmo wants to take. Returns do_id."""
        now = tz_now().isoformat()
        do_id = str(uuid.uuid4())

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO dos (
                    do_id, action, for_headmate,
                    trigger_type, trigger_condition, trigger_value,
                    importance, source_goal, source_event,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                do_id, action,
                for_headmate.lower() if for_headmate else None,
                trigger_type, trigger_condition,
                json.dumps(trigger_value) if trigger_value else None,
                importance, source_goal, source_event,
                now, now,
            ))
            _insert_tags(do_id, "do", tags, conn)
            conn.commit()

        return do_id

    def complete_do(self, do_id: str) -> None:
        """Mark a do as completed."""
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE dos SET status = 'done', completed_at = ?, updated_at = ? "
                "WHERE do_id = ?",
                (now, now, do_id)
            )
            conn.commit()

    def get_pending_dos(
        self,
        trigger_type: Optional[str] = None,
        headmate:     Optional[str] = None,
    ) -> list[dict]:
        """Fetch pending dos, optionally filtered by trigger type or headmate."""
        query  = "SELECT * FROM dos WHERE status = 'pending'"
        params: list = []

        if trigger_type:
            query += " AND trigger_type = ?"
            params.append(trigger_type)

        if headmate:
            query += " AND (for_headmate IS NULL OR for_headmate = ?)"
            params.append(headmate.lower())

        query += " ORDER BY importance DESC"

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["tags"] = _tags_for(d["do_id"], conn)
                results.append(d)
            return results

    # ── Goals ─────────────────────────────────────────────────────────────────

    def add_goal(
        self,
        description:    str,
        purpose:        str,
        tags:           list[str],
        intentions:     list[str],
        subject:        Optional[str] = None,
        origin:         str = "autonomous",
        target_date:    Optional[str] = None,
    ) -> str:
        """
        Add a long-horizon goal. Intentions are plain language orientations —
        not pre-populated tasks. Sub-items accumulate from real evidence.
        Returns goal_id.
        """
        now = tz_now().isoformat()
        goal_id = str(uuid.uuid4())

        # Default target: 3 months out
        if not target_date:
            from datetime import timedelta
            target_date = (tz_now() + timedelta(days=90)).strftime("%Y-%m-%d")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO goals (
                    goal_id, description, purpose, subject,
                    origin, target_date,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                goal_id, description, purpose,
                subject.lower() if subject else None,
                origin, target_date,
                now, now,
            ))

            # Add intentions
            for intention in intentions:
                conn.execute("""
                    INSERT INTO goal_intentions (
                        intention_id, goal_id, intention, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (str(uuid.uuid4()), goal_id, intention, now, now))

            _insert_tags(goal_id, "goal", tags, conn)
            conn.commit()

        log_event("QuestionBank", "GOAL_CREATED",
            description=description[:60],
            subject=subject,
            origin=origin,
            intentions=len(intentions),
        )

        return goal_id

    def link_sub_item(
        self,
        goal_id:   str,
        item_type: str,
        item_id:   str,
    ) -> None:
        """Link a question, say, or do to a goal as an earned sub-item."""
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO goal_sub_items (
                    link_id, goal_id, item_type, item_id, spawned_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), goal_id, item_type, item_id, now))
            conn.commit()

    def add_psych_report_entry(
        self,
        goal_id:        str,
        progress:       bool,
        what:           Optional[str] = None,
        how:            Optional[str] = None,
        outcome:        Optional[str] = None,
        blocked_by:     Optional[str] = None,
        intention_id:   Optional[str] = None,
    ) -> None:
        """Add a daily psych pass entry for a goal."""
        now = tz_now().isoformat()
        date = tz_now().strftime("%Y-%m-%d")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO goal_psych_report (
                    entry_id, goal_id, intention_id, date,
                    progress, what, how, outcome, blocked_by,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), goal_id, intention_id, date,
                1 if progress else 0,
                what, how, outcome, blocked_by,
                now,
            ))

            # Update progress notes on goal
            if what:
                conn.execute("""
                    UPDATE goals SET
                        progress_notes = COALESCE(progress_notes || char(10), '') || ?,
                        updated_at = ?
                    WHERE goal_id = ?
                """, (f"[{date}] {what}", now, goal_id))

            conn.commit()

    def get_active_goals(self, subject: Optional[str] = None) -> list[dict]:
        """Get all active goals, optionally for one headmate."""
        query  = "SELECT * FROM goals WHERE status = 'active'"
        params: list = []

        if subject:
            query += " AND (subject IS NULL OR subject = ?)"
            params.append(subject.lower())

        query += " ORDER BY created_at ASC"

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["tags"] = _tags_for(d["goal_id"], conn)
                # Attach intentions
                intentions = conn.execute(
                    "SELECT * FROM goal_intentions WHERE goal_id = ? "
                    "ORDER BY created_at",
                    (d["goal_id"],)
                ).fetchall()
                d["intentions"] = [dict(i) for i in intentions]
                results.append(d)
            return results

    def get_goal_report(self, goal_id: str) -> list[dict]:
        """Get the full psych report for a goal."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM goal_psych_report WHERE goal_id = ? "
                "ORDER BY date DESC",
                (goal_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Temperatures ──────────────────────────────────────────────────────────

    def get_temperature(
        self,
        dimension: str,
        headmate:  Optional[str] = None,
    ) -> float:
        """
        Get current temperature for a dimension.
        Headmate-specific overrides global. Falls back to 0.5 if not found.
        """
        with sqlite3.connect(DB_PATH) as conn:
            # Try headmate-specific first
            if headmate:
                row = conn.execute(
                    "SELECT value FROM temperatures "
                    "WHERE dimension = ? AND headmate = ?",
                    (dimension, headmate.lower())
                ).fetchone()
                if row:
                    return row[0]

            # Fall back to global
            row = conn.execute(
                "SELECT value FROM temperatures "
                "WHERE dimension = ? AND headmate IS NULL",
                (dimension,)
            ).fetchone()
            return row[0] if row else 0.5

    def set_temperature(
        self,
        dimension:  str,
        value:      float,
        headmate:   Optional[str] = None,
        note:       Optional[str] = None,
        auto_adjust: bool = False,
    ) -> None:
        """Set or create a temperature dimension."""
        now = tz_now().isoformat()
        value = max(0.0, min(1.0, value))  # clamp to 0–1

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO temperatures (
                    temp_id, headmate, dimension, value, default_value,
                    auto_adjust, note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(headmate, dimension) DO UPDATE SET
                    value      = excluded.value,
                    auto_adjust = excluded.auto_adjust,
                    note       = COALESCE(excluded.note, temperatures.note),
                    updated_at = excluded.updated_at
            """, (
                str(uuid.uuid4()),
                headmate.lower() if headmate else None,
                dimension, value, value,
                1 if auto_adjust else 0,
                note, now, now,
            ))
            conn.commit()

    def adjust_temperature(
        self,
        dimension: str,
        delta:     float,
        headmate:  Optional[str] = None,
    ) -> float:
        """
        Nudge a temperature by delta. Only fires if auto_adjust is True.
        Returns new value.
        """
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT value, auto_adjust FROM temperatures "
                "WHERE dimension = ? AND headmate IS ?",
                (dimension, headmate.lower() if headmate else None)
            ).fetchone()

            if not row or not row[1]:  # not found or auto_adjust = False
                return self.get_temperature(dimension, headmate)

            new_value = max(0.0, min(1.0, row[0] + delta))
            now = tz_now().isoformat()
            conn.execute(
                "UPDATE temperatures SET value = ?, updated_at = ? "
                "WHERE dimension = ? AND headmate IS ?",
                (new_value, now, dimension,
                 headmate.lower() if headmate else None)
            )
            conn.commit()
            return new_value

    def reset_temperatures(self, headmate: Optional[str] = None) -> None:
        """Reset all temperatures to their defaults. Called between sessions."""
        now = tz_now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            if headmate:
                conn.execute(
                    "UPDATE temperatures SET value = default_value, updated_at = ? "
                    "WHERE headmate = ?",
                    (now, headmate.lower())
                )
            else:
                conn.execute(
                    "UPDATE temperatures SET value = default_value, updated_at = ?",
                    (now,)
                )
            conn.commit()

    def get_all_temperatures(self, headmate: Optional[str] = None) -> list[dict]:
        """Get all temperature dimensions, merged global + headmate-specific."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            if headmate:
                rows = conn.execute(
                    "SELECT * FROM temperatures "
                    "WHERE headmate IS NULL OR headmate = ? "
                    "ORDER BY headmate NULLS FIRST, dimension",
                    (headmate.lower(),)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM temperatures WHERE headmate IS NULL "
                    "ORDER BY dimension"
                ).fetchall()
            return [dict(r) for r in rows]


question_bank = QuestionBank()
