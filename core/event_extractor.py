"""
core/event_extractor.py
Atomic event extraction from conversational exchanges.

Runs on every exchange (user message + Gizmo response) as a fire-and-forget
async task. Produces a list of atomic event records — one per discrete action
or statement — and appends them to a SQLite action log.

No psychology here. No scoring. No interpretation beyond what the narrative
plainly shows. That comes later, in the psychology engine batch pass.

Each record captures:
  - Who did it (subject)
  - Who received it (recipient, if any)
  - What they did (action, as described — full and uncut)
  - Why the narrative suggests they did it (intent, single word or short phrase)
  - Whether it was physical, verbal, postural, or expressed emotion
  - The source exchange for traceback

LLM used: cheap, fast model. This is not a reasoning task.
Mistral Small 3 or Ministral 3B via OpenRouter both work well here.
The prompt is tight and the output is structured JSON — low token cost.

Usage (fire-and-forget from archivist):
    asyncio.create_task(
        event_extractor.extract(
            user_message=brief["message"],
            gizmo_response=response,
            subject=brief["host"],
            session_file=session_file,
            timestamp=tz_now().isoformat(),
        )
    )
"""

import asyncio
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = Path("/data")
DB_PATH     = DATA_DIR / "action_log.db"

# Cheap model — this is classification, not reasoning
EXTRACTOR_MODEL = "mistralai/mistral-small-3"

# ── DB setup ──────────────────────────────────────────────────────────────────

def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS action_log (
                event_id        TEXT PRIMARY KEY,
                session_file    TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                sequence        INTEGER NOT NULL,

                subject         TEXT NOT NULL,
                recipient       TEXT,
                recipient_type  TEXT,

                action_type     TEXT NOT NULL,
                action          TEXT NOT NULL,
                intent          TEXT,

                verbatim        TEXT,
                topic           TEXT,
                directed_at     TEXT,

                narrative_ref   TEXT,

                -- psychology engine fields, populated later
                intent_scores   TEXT,
                intent_rationale TEXT,
                analyzed_at     TEXT,

                created_at      TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_action_log_subject "
            "ON action_log(subject)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_action_log_session "
            "ON action_log(session_file)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_action_log_analyzed "
            "ON action_log(analyzed_at)"
        )
        conn.commit()


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """
You extract atomic events from conversational exchanges.

For each discrete action or statement, produce one event record.
Return ONLY a JSON array. No preamble, no explanation, no markdown fences.

Each record must have exactly these fields:
  subject       — who performed the action (use their name exactly as given)
  recipient     — who or what received it, or null
  recipient_type — "person", "object", "self", "room", or null
  action_type   — one of: "physical", "verbal", "postural", "expressed_emotion"
  action        — the full action as described, uncut, preserving how it was performed
  intent        — single word or short phrase: what the narrative plainly suggests they wanted
                  use null only if genuinely unknowable from the text
  verbatim      — exact words spoken, or null if not verbal
  topic         — what speech was about, or null if not verbal
  directed_at   — who speech was addressed to, or null if not verbal
  narrative_ref — the fragment of source text this event came from

Rules:
- Preserve adverbs and qualifiers in action. "knelt gracefully" stays whole.
- One action per record. "stood and walked to the door" is two records.
- intent is what the narrative shows, not what you infer. Keep it simple.
- Do not add fields. Do not omit fields. Return valid JSON only.
""".strip()


def _build_prompt(
    user_message: str,
    gizmo_response: str,
    subject: str,
) -> str:
    return (
        f"Extract all atomic events from this exchange.\n\n"
        f"The person speaking to Gizmo is: {subject}\n\n"
        f"{subject} said/did:\n{user_message}\n\n"
        f"Gizmo responded:\n{gizmo_response}"
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[list[dict]]:
    """
    Call the cheap extractor model. Returns parsed event list or None on failure.
    Uses OpenRouter directly — separate from the main LLM client so we can
    specify a different model without touching the main config.
    """
    try:
        import os
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30,
        )

        response = await client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.0,   # deterministic — this is extraction not generation
        )

        raw = response.choices[0].message.content or ""
        if not raw.strip():
            log_event("EventExtractor", "EMPTY_RESPONSE", model=EXTRACTOR_MODEL)
            return None

        # Strip markdown fences if the model added them anyway
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)

        if not isinstance(parsed, list):
            log_event("EventExtractor", "BAD_SHAPE", raw=raw[:200])
            return None

        return parsed

    except json.JSONDecodeError as e:
        log_error("EventExtractor", "JSON parse failed", exc=e)
        return None
    except Exception as e:
        log_error("EventExtractor", "LLM call failed", exc=e)
        return None


# ── Validation ────────────────────────────────────────────────────────────────

_REQUIRED = {
    "subject", "recipient", "recipient_type", "action_type",
    "action", "intent", "verbatim", "topic", "directed_at", "narrative_ref",
}

_VALID_ACTION_TYPES = {"physical", "verbal", "postural", "expressed_emotion"}
_VALID_RECIPIENT_TYPES = {"person", "object", "self", "room", None}


def _validate(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    if not _REQUIRED.issubset(record.keys()):
        return False
    if record.get("action_type") not in _VALID_ACTION_TYPES:
        return False
    if record.get("recipient_type") not in _VALID_RECIPIENT_TYPES:
        return False
    if not record.get("subject") or not record.get("action"):
        return False
    return True


# ── File write ────────────────────────────────────────────────────────────────

def _actions_file(subject: str) -> Path:
    """
    Lazy path resolution — called at write time, never at import.
    /data/headmates/{subject}/actions/{YYYY-MM-DD}.jsonl
    """
    date_str = tz_now().strftime("%Y-%m-%d")
    path = DATA_DIR / "headmates" / subject.lower() / "actions" / f"{date_str}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_to_file(record: dict, subject: str) -> None:
    """Append one event record as a JSON line to the headmate's daily action file."""
    try:
        path = _actions_file(subject)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log_error("EventExtractor", "File write failed", exc=e)


# ── DB write ──────────────────────────────────────────────────────────────────

def _write_events(
    events:       list[dict],
    session_file: str,
) -> int:
    """
    Write validated events to both SQLite and JSONL file.
    SQLite is the query layer. JSONL is the canonical human-readable record.
    Returns count written.
    """
    now = tz_now().isoformat()
    written = 0

    with sqlite3.connect(DB_PATH) as conn:
        for i, ev in enumerate(events, start=1):
            if not _validate(ev):
                log_event("EventExtractor", "INVALID_RECORD", record=ev)
                continue

            event_id = str(uuid.uuid4())

            try:
                conn.execute("""
                    INSERT INTO action_log (
                        event_id, session_file, timestamp, sequence,
                        subject, recipient, recipient_type,
                        action_type, action, intent,
                        verbatim, topic, directed_at,
                        narrative_ref,
                        intent_scores, intent_rationale, analyzed_at,
                        created_at
                    ) VALUES (
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?,
                        NULL, NULL, NULL,
                        ?
                    )
                """, (
                    event_id,
                    session_file,
                    now,
                    i,
                    ev["subject"],
                    ev.get("recipient"),
                    ev.get("recipient_type"),
                    ev["action_type"],
                    ev["action"],
                    ev.get("intent"),
                    ev.get("verbatim"),
                    ev.get("topic"),
                    ev.get("directed_at"),
                    ev.get("narrative_ref"),
                    now,
                ))
                written += 1

                # Mirror to JSONL — same record, event_id and timestamps included
                _append_to_file({
                    "event_id":      event_id,
                    "session_file":  session_file,
                    "timestamp":     now,
                    "sequence":      i,
                    **ev,
                }, subject=ev["subject"])

            except sqlite3.Error as e:
                log_error("EventExtractor", "DB write failed", exc=e)

        conn.commit()

    return written


# ── Public API ────────────────────────────────────────────────────────────────

class EventExtractor:
    """
    Singleton. Call extract() as a fire-and-forget task from the archivist.
    """

    def __init__(self):
        _init_db()

    async def extract(
        self,
        user_message:   str,
        gizmo_response: str,
        subject:        str,
        session_file:   str,
    ) -> None:
        """
        Extract atomic events from one exchange and append to action_log.
        Designed to run as asyncio.create_task() — never blocks the main flow.
        Timestamp is generated internally via tz_now() — never trusted from caller.
        """
        if not user_message.strip() and not gizmo_response.strip():
            return

        prompt = _build_prompt(user_message, gizmo_response, subject)
        events = await _call_llm(prompt)

        if not events:
            log_event("EventExtractor", "NO_EVENTS_EXTRACTED",
                subject=subject,
                session=session_file,
            )
            return

        written = _write_events(events, session_file)

        log_event("EventExtractor", "EXTRACTED",
            subject=subject,
            session=session_file,
            events_extracted=len(events),
            events_written=written,
        )

    def get_events_for_subject(
        self,
        subject:   str,
        limit:     int = 200,
        unanalyzed_only: bool = False,
    ) -> list[dict]:
        """
        Fetch action log entries for a subject.
        Used by the psychology engine batch pass.
        """
        query = "SELECT * FROM action_log WHERE subject = ?"
        params: list = [subject]

        if unanalyzed_only:
            query += " AND analyzed_at IS NULL"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_events_for_session(self, session_file: str) -> list[dict]:
        """Fetch all events for a session. Used by post-session analysis."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM action_log WHERE session_file = ? ORDER BY sequence",
                (session_file,)
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_analyzed(
        self,
        event_id:         str,
        intent_scores:    dict,
        intent_rationale: str,
    ) -> None:
        """
        Called by the psychology engine after it scores an event.
        Writes scores back into the record.
        """
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE action_log
                SET intent_scores    = ?,
                    intent_rationale = ?,
                    analyzed_at      = ?
                WHERE event_id = ?
            """, (
                json.dumps(intent_scores),
                intent_rationale,
                tz_now().isoformat(),
                event_id,
            ))
            conn.commit()


event_extractor = EventExtractor()
