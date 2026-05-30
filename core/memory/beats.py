"""
core/memory/beats.py

The beat system. A beat is the fundamental unit of a scene.

Every message — from user or Gizmo — gets parsed into an ordered
sequence of beats. A beat is either dialogue or an action.

One message can contain multiple beats in sequence:
  "*wiggles ass* I win!" → [action, dialogue]
  "*steps back* Oh you think so? *raises eyebrow*" → [action, dialogue, action]

Gizmo's beats get enriched with WHY — reasoning behind each choice.
User beats get type, content, speaker, register.

The beat sequence for a session is a screenplay. You can read it
and understand exactly what happened, in what order, and why.

Beat sequences feed:
  - Action tracker (reads actions with why + what came after)
  - Psychology passes (reads dialogue patterns)
  - Encoding pass (reads beats instead of raw transcript)
  - Scene context (understands current scene state from recent beats)
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

from core.log import log_event, log_error


# ── Beat dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Beat:
    """One unit of a scene — either dialogue or action."""
    id:          str
    session_id:  str
    headmate:    str           # fronting headmate when this beat happened
    speaker:     str           # who said/did this
    type:        str           # "dialogue" | "action"
    content:     str           # what was said or done
    register:    str           # emotional register
    directed_at: Optional[str] # who it's directed at
    timestamp:   float

    # Gizmo's beats only
    why:         str   = ""    # why this choice was made
    
    # Filled in after next beat is known
    landed:      Optional[str] = None   # yes|no|unclear|yes_playing_up|real_shift
    next_beat_id: Optional[str] = None


# ── Beat parser ────────────────────────────────────────────────────────────────

# Matches *action text* — including multiword, punctuation
_ACTION_RE  = re.compile(r'\*([^*]+)\*')
# Matches inline stage directions in parentheses
_STAGE_RE   = re.compile(r'\(([^)]{3,80})\)')


def parse_to_beats(
    raw:        str,
    speaker:    str,
    session_id: str,
    headmate:   str,
    register:   str        = "neutral",
    directed_at: Optional[str] = None,
    timestamp:  Optional[float] = None,
) -> list[Beat]:
    """
    Parse a raw message into an ordered list of beats.

    "*wiggles ass* I win!" →
      [Beat(action, "wiggles ass"), Beat(dialogue, "I win!")]

    "*steps back* Oh you think so? *raises eyebrow*" →
      [Beat(action, "steps back"),
       Beat(dialogue, "Oh you think so?"),
       Beat(action, "raises eyebrow")]
    """
    ts    = timestamp or time.time()
    beats = []

    if not raw or not raw.strip():
        return beats

    # Find all actions and their positions
    # Build a sequence of (start, end, type, content)
    segments = []

    last_end = 0
    for match in _ACTION_RE.finditer(raw):
        start, end = match.span()
        action_text = match.group(1).strip()

        # Text before this action is dialogue
        before = raw[last_end:start].strip()
        if before:
            segments.append(("dialogue", before))

        if action_text:
            segments.append(("action", action_text))

        last_end = end

    # Remaining text after last action is dialogue
    remaining = raw[last_end:].strip()
    if remaining:
        segments.append(("dialogue", remaining))

    # If no actions found, whole thing is dialogue
    if not segments:
        segments = [("dialogue", raw.strip())]

    for i, (seg_type, content) in enumerate(segments):
        if not content:
            continue

        beat_id = _make_beat_id(session_id, speaker, ts, i)
        beat    = Beat(
            id          = beat_id,
            session_id  = session_id,
            headmate    = headmate,
            speaker     = speaker.lower(),
            type        = seg_type,
            content     = content,
            register    = register,
            directed_at = directed_at,
            timestamp   = ts + (i * 0.001),  # tiny offset to preserve order
        )
        beats.append(beat)

    return beats


def beats_to_transcript(beats: list[Beat]) -> str:
    """
    Render a beat sequence as a readable transcript.
    Used as input to encoding and psychology passes.
    """
    lines = []
    for b in beats:
        speaker = b.speaker.title()
        if b.type == "action":
            why_str = f" [{b.why}]" if b.why else ""
            lines.append(f"*{speaker} {b.content}*{why_str}")
        else:
            lines.append(f"{speaker}: {b.content}")
    return "\n".join(lines)


def beats_to_action_summary(beats: list[Beat], speaker: str = "gizmo") -> str:
    """
    Extract action beats from a speaker with their context.
    Used by action tracker.
    """
    lines    = []
    beat_list = list(beats)

    for i, b in enumerate(beat_list):
        if b.speaker.lower() != speaker.lower() or b.type != "action":
            continue

        # Find the next beat for context
        next_beat = beat_list[i + 1] if i + 1 < len(beat_list) else None
        next_str  = ""
        if next_beat:
            next_speaker = next_beat.speaker.title()
            if next_beat.type == "action":
                next_str = f"→ *{next_speaker} {next_beat.content[:80]}*"
            else:
                next_str = f"→ {next_speaker}: {next_beat.content[:80]}"

        why_str = f" (why: {b.why})" if b.why else ""
        lines.append(f"*{b.content}*{why_str} {next_str}".strip())

    return "\n".join(lines) if lines else "(no actions)"


# ── Beat store ─────────────────────────────────────────────────────────────────

class BeatStore:
    """
    SQLite storage for beat sequences.
    Separate from the main memory store — beats are operational data,
    not long-term memory. They feed the encoding pass and action tracker.
    """

    def __init__(self):
        self._db: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        from core.memory.store import memory_store
        db_path = memory_store.root / "beats.db"
        con     = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init(self) -> None:
        con = self._connect()
        con.executescript("""
            CREATE TABLE IF NOT EXISTS beats (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                headmate     TEXT,
                speaker      TEXT NOT NULL,
                type         TEXT NOT NULL,
                content      TEXT NOT NULL,
                register     TEXT DEFAULT 'neutral',
                directed_at  TEXT,
                why          TEXT DEFAULT '',
                landed       TEXT,
                next_beat_id TEXT,
                timestamp    REAL,
                active       INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_beats_session  ON beats(session_id);
            CREATE INDEX IF NOT EXISTS idx_beats_speaker  ON beats(speaker);
            CREATE INDEX IF NOT EXISTS idx_beats_type     ON beats(type);
            CREATE INDEX IF NOT EXISTS idx_beats_ts       ON beats(timestamp);
        """)
        con.commit()
        con.close()

    def save_beats(self, beats: list[Beat]) -> None:
        """Save a list of beats to the store."""
        if not beats:
            return
        try:
            self._init()
            con = self._connect()
            for b in beats:
                con.execute("""
                    INSERT OR REPLACE INTO beats
                      (id, session_id, headmate, speaker, type, content,
                       register, directed_at, why, landed, next_beat_id,
                       timestamp, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (
                    b.id, b.session_id, b.headmate, b.speaker,
                    b.type, b.content, b.register, b.directed_at,
                    b.why, b.landed, b.next_beat_id, b.timestamp,
                ))
            con.commit()
            con.close()
        except Exception as e:
            log_error("BeatStore", f"save failed: {e}", exc=None)

    def update_beat(self, beat_id: str, **kwargs) -> None:
        """Update fields on an existing beat."""
        if not kwargs:
            return
        try:
            con  = self._connect()
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            con.execute(
                f"UPDATE beats SET {sets} WHERE id = ?",
                list(kwargs.values()) + [beat_id],
            )
            con.commit()
            con.close()
        except Exception as e:
            log_error("BeatStore", f"update failed: {e}", exc=None)

    def get_session_beats(
        self,
        session_id: str,
        limit:      int = 200,
    ) -> list[Beat]:
        """Get all beats for a session in order."""
        try:
            self._init()
            con  = self._connect()
            rows = con.execute(
                "SELECT * FROM beats WHERE session_id = ? AND active = 1 "
                "ORDER BY timestamp ASC LIMIT ?",
                (session_id, limit)
            ).fetchall()
            con.close()
            return [_row_to_beat(r) for r in rows]
        except Exception as e:
            log_error("BeatStore", f"get_session_beats failed: {e}", exc=None)
            return []

    def get_recent_beats(
        self,
        headmate: str,
        limit:    int = 50,
        type:     Optional[str] = None,
    ) -> list[Beat]:
        """Get recent beats for a headmate across sessions."""
        try:
            self._init()
            con    = self._connect()
            wheres = ["headmate = ?", "active = 1"]
            params = [headmate.lower()]
            if type:
                wheres.append("type = ?")
                params.append(type)
            rows = con.execute(
                f"SELECT * FROM beats WHERE {' AND '.join(wheres)} "
                f"ORDER BY timestamp DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            con.close()
            return [_row_to_beat(r) for r in rows]
        except Exception as e:
            log_error("BeatStore", f"get_recent_beats failed: {e}", exc=None)
            return []

    def get_action_beats_with_context(
        self,
        headmate:  str,
        speaker:   str = "gizmo",
        limit:     int = 100,
    ) -> list[tuple[Beat, Optional[Beat]]]:
        """
        Get action beats paired with the beat that followed.
        Used by action tracker to assess what landed.
        """
        try:
            self._init()
            con  = self._connect()
            rows = con.execute(
                "SELECT * FROM beats WHERE headmate = ? AND speaker = ? "
                "AND type = 'action' AND active = 1 "
                "ORDER BY timestamp DESC LIMIT ?",
                (headmate.lower(), speaker.lower(), limit)
            ).fetchall()
            con.close()

            result = []
            for row in rows:
                beat = _row_to_beat(row)
                # Get the next beat
                next_rows = self._connect().execute(
                    "SELECT * FROM beats WHERE session_id = ? "
                    "AND timestamp > ? AND active = 1 "
                    "ORDER BY timestamp ASC LIMIT 1",
                    (beat.session_id, beat.timestamp)
                ).fetchall()
                next_beat = _row_to_beat(next_rows[0]) if next_rows else None
                result.append((beat, next_beat))

            return result
        except Exception as e:
            log_error("BeatStore", f"get_action_beats failed: {e}", exc=None)
            return []

    def link_beats(self, beats: list[Beat]) -> None:
        """Set next_beat_id on each beat to point to the following beat."""
        for i, beat in enumerate(beats[:-1]):
            self.update_beat(beat.id, next_beat_id=beats[i + 1].id)


# ── Why extraction ─────────────────────────────────────────────────────────────

async def extract_why(
    beats:      list[Beat],
    session_id: str,
    headmate:   str,
    llm,
) -> list[Beat]:
    """
    Enrich Gizmo's action beats with WHY reasoning.
    Fast pass — reads the beat sequence and adds brief reasoning
    for each of Gizmo's actions.
    Returns the enriched beat list.
    """
    gizmo_actions = [b for b in beats if b.speaker == "gizmo" and b.type == "action"]
    if not gizmo_actions:
        return beats

    # Build a compact scene view for context
    scene = beats_to_transcript(beats[-20:])  # last 20 beats for context

    actions_text = "\n".join(
        f"[{i}] *{b.content}*"
        for i, b in enumerate(gizmo_actions)
    )

    prompt = f"""You are Gizmo. You just responded in a scene.

Recent scene:
---
{scene}
---

Your actions:
{actions_text}

For each action, what was the reasoning? Why that choice, right then?
Keep it brief — one short phrase per action.

Return one JSON object per line:
{{"index": N, "why": "brief reasoning"}}

JSON only, one per line."""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo explaining your action choices briefly. "
                "JSON only, one per line. Short phrases."
            ),
            max_new_tokens=300,
            temperature=0.3,
        )
    except Exception as e:
        log_error("BeatStore", f"why extraction failed: {e}", exc=None)
        return beats

    if not raw or not raw.strip():
        return beats

    # Apply why reasoning to action beats
    why_map: dict[int, str] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
            idx = d.get("index")
            why = d.get("why", "").strip()
            if idx is not None and why:
                why_map[int(idx)] = why
        except Exception:
            continue

    for i, beat in enumerate(gizmo_actions):
        if i in why_map:
            beat.why = why_map[i]

    return beats


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_beat_id(session_id: str, speaker: str, ts: float, idx: int) -> str:
    import hashlib
    raw = f"{session_id}:{speaker}:{ts}:{idx}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _row_to_beat(row: sqlite3.Row) -> Beat:
    return Beat(
        id          = row["id"],
        session_id  = row["session_id"],
        headmate    = row["headmate"] or "",
        speaker     = row["speaker"],
        type        = row["type"],
        content     = row["content"],
        register    = row["register"] or "neutral",
        directed_at = row["directed_at"],
        why         = row["why"] or "",
        landed      = row["landed"],
        next_beat_id = row["next_beat_id"],
        timestamp   = row["timestamp"] or 0.0,
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

beat_store = BeatStore()
