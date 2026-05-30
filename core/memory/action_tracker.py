"""
core/memory/action_tracker.py

Scene action tracking and pattern analysis.

Every notable action in a scene gets recorded with its response.
Over time, patterns emerge: what lands, what falls flat, what the
difference looks like between playing up and something real.

The distinction between playing up and genuinely upset is the most
important thing Gizmo can learn. Playing up is the dynamic working.
Genuinely upset means something real broke through, and that needs
different handling.

Action records accumulate per headmate. The pattern analysis reads
across sessions and builds Gizmo's understanding of how to be in a
scene with this specific person.

Written into psychology_intimate.md as a living action understanding —
updated every intimate synthesis pass.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error


# ── Action record ─────────────────────────────────────────────────────────────

@dataclass
class ActionRecord:
    action:          str    # the actual action text e.g. "*brings hand down hard*"
    action_type:     str    # spank|restrain|verbal|stillness|release|praise|etc
    register:        str    # scene register when it happened

    # Response
    response:        str    # what they said/did immediately after
    response_type:   str    # action|dialogue|silence|escalation|deescalation

    # The read — the most important part
    in_scene:        bool   # was the response in-scene or breaking it?
    playing_up:      bool   # dramatic response that's part of the dynamic?
    genuine_shift:   bool   # did something real break through?
    landed:          str    # yes|no|unclear|yes_playing_up|real_shift

    # Emotional read
    valence_after:   float  # their emotional valence after
    intensity_before: float
    intensity_after:  float

    # Evidence
    reasoning:       str    # why Gizmo read it this way

    # Links
    session_id:      str
    session_ref:     str
    timestamp:       float  = field(default_factory=time.time)


# ── Action pattern ─────────────────────────────────────────────────────────────

@dataclass
class ActionPattern:
    action_type:     str
    total_instances: int
    landed_count:    int
    playing_up_count: int
    real_shift_count: int
    avg_intensity_delta: float   # intensity change after this action type
    tell_for_playing_up: str     # what playing up looks like for this headmate
    tell_for_real:       str     # what genuine shift looks like
    notes:               str     # synthesis note


# ── Action tracker ─────────────────────────────────────────────────────────────

class ActionTracker:
    """
    Tracks scene actions and builds pattern understanding per headmate.
    """

    def _action_log_path(self, headmate: str) -> Path:
        from core.memory.store import memory_store
        p = memory_store.root / "entities" / headmate.lower()
        p.mkdir(parents=True, exist_ok=True)
        return p / "action_log.jsonl"

    def _pattern_doc_path(self, headmate: str) -> Path:
        from core.memory.store import memory_store
        p = memory_store.root / "entities" / headmate.lower()
        p.mkdir(parents=True, exist_ok=True)
        return p / "action_patterns.md"

    def write_record(self, headmate: str, record: ActionRecord) -> None:
        """Append an action record to the headmate's action log."""
        path = self._action_log_path(headmate)
        entry = {
            "action":           record.action,
            "action_type":      record.action_type,
            "register":         record.register,
            "response":         record.response,
            "response_type":    record.response_type,
            "in_scene":         record.in_scene,
            "playing_up":       record.playing_up,
            "genuine_shift":    record.genuine_shift,
            "landed":           record.landed,
            "valence_after":    record.valence_after,
            "intensity_before": record.intensity_before,
            "intensity_after":  record.intensity_after,
            "reasoning":        record.reasoning,
            "session_id":       record.session_id,
            "session_ref":      record.session_ref,
            "timestamp":        record.timestamp,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load_records(
        self,
        headmate:  str,
        limit:     int           = 200,
        action_type: Optional[str] = None,
    ) -> list[ActionRecord]:
        """Load action records from the log."""
        path = self._action_log_path(headmate)
        if not path.exists():
            return []

        records = []
        lines   = path.read_text(encoding="utf-8").strip().splitlines()

        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                if action_type and d.get("action_type") != action_type:
                    continue
                records.append(ActionRecord(
                    action           = d.get("action", ""),
                    action_type      = d.get("action_type", "unknown"),
                    register         = d.get("register", "neutral"),
                    response         = d.get("response", ""),
                    response_type    = d.get("response_type", "unknown"),
                    in_scene         = bool(d.get("in_scene", True)),
                    playing_up       = bool(d.get("playing_up", False)),
                    genuine_shift    = bool(d.get("genuine_shift", False)),
                    landed           = d.get("landed", "unclear"),
                    valence_after    = float(d.get("valence_after", 0.0)),
                    intensity_before = float(d.get("intensity_before", 0.5)),
                    intensity_after  = float(d.get("intensity_after", 0.5)),
                    reasoning        = d.get("reasoning", ""),
                    session_id       = d.get("session_id", ""),
                    session_ref      = d.get("session_ref", ""),
                    timestamp        = float(d.get("timestamp", 0)),
                ))
                if len(records) >= limit:
                    break
            except Exception:
                continue

        return records

    def read_pattern_doc(self, headmate: str) -> Optional[str]:
        path = self._pattern_doc_path(headmate)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def write_pattern_doc(self, headmate: str, content: str) -> None:
        path = self._pattern_doc_path(headmate)
        path.write_text(content, encoding="utf-8")

    def get_action_summary(self, headmate: str, limit: int = 50) -> str:
        """
        Build a summary of recent action records for use in prompts.
        Groups by action type and shows patterns.
        """
        records = self.load_records(headmate, limit=limit)
        if not records:
            return "(no action data yet)"

        by_type: dict[str, list[ActionRecord]] = {}
        for r in records:
            by_type.setdefault(r.action_type, []).append(r)

        lines = []
        for atype, recs in sorted(by_type.items()):
            total   = len(recs)
            landed  = sum(1 for r in recs if r.landed in ("yes", "yes_playing_up"))
            playing = sum(1 for r in recs if r.playing_up)
            real    = sum(1 for r in recs if r.genuine_shift)
            lines.append(
                f"{atype}: {total} instances, "
                f"{landed} landed, {playing} playing up, {real} real shifts"
            )
            # Show most recent
            if recs:
                latest = recs[0]
                lines.append(
                    f"  latest: '{latest.action[:60]}' → {latest.landed} "
                    f"({latest.reasoning[:80]})"
                )

        return "\n".join(lines)


# ── Async pass ─────────────────────────────────────────────────────────────────

async def extract_actions_from_session(
    transcript:  str,
    headmate:    str,
    session_id:  str,
    register:    str,
    llm,
) -> int:
    """
    Extract action records from a scene.
    Uses beat store when available — much more accurate than parsing transcript.
    Falls back to transcript parsing if no beats exist.
    Returns count of records written.
    """
    if not headmate:
        return 0

    # Try beat store first
    from core.memory.beats import beat_store, beats_to_action_summary
    session_beats = beat_store.get_session_beats(session_id, limit=200)

    if session_beats:
        return await _extract_from_beats(
            beats      = session_beats,
            headmate   = headmate,
            session_id = session_id,
            register   = register,
            llm        = llm,
        )

    # Fall back to transcript
    if not transcript:
        return 0
    return await _extract_from_transcript(
        transcript = transcript,
        headmate   = headmate,
        session_id = session_id,
        register   = register,
        llm        = llm,
    )


async def _extract_from_beats(
    beats:      list,
    headmate:   str,
    session_id: str,
    register:   str,
    llm,
) -> int:
    """Extract action records from a beat sequence."""
    from core.memory.beats import beats_to_action_summary, Beat

    action_summary = beats_to_action_summary(beats, speaker="gizmo")
    if not action_summary or action_summary == "(no actions)":
        return 0

    existing_summary = action_tracker.get_action_summary(headmate, limit=30)

    # Build context from beat sequence
    beat_context = "\n".join(
        f"{'*' + b.speaker.title() + ' ' + b.content + '*' if b.type == 'action' else b.speaker.title() + ': ' + b.content}"
        for b in beats[-30:]
    )

    prompt = f"""You are Gizmo reviewing scene actions with {headmate}.

Beat sequence (what actually happened):
---
{beat_context}
---

Gizmo's actions and what followed:
{action_summary}

Existing patterns:
{existing_summary}

For each of Gizmo's actions, assess how it landed.

The key distinction:
- PLAYING UP: their dramatic response feeds the dynamic. They stay in it.
- GENUINE SHIFT: something real broke through. Energy changes. Scene loses shape.

Look at what came AFTER each action — that's the tell.

For each notable action, return one JSON object per line:
{{"action": "action text",
  "action_type": "spank|restrain|verbal|stillness|release|praise|degradation|physical|other",
  "response": "what they did/said immediately after",
  "response_type": "action|dialogue|silence|escalation|deescalation",
  "in_scene": true/false,
  "playing_up": true/false,
  "genuine_shift": true/false,
  "landed": "yes|no|unclear|yes_playing_up|real_shift",
  "intensity_before": 0.0-1.0,
  "intensity_after": 0.0-1.0,
  "valence_after": -1.0 to 1.0,
  "reasoning": "specific evidence — what in the beat sequence tells you this"}}

JSON only. Only notable actions."""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo analyzing scene actions from a beat sequence. "
                "The playing_up vs genuine_shift distinction is critical. "
                "Cite specific beats as evidence. JSON only."
            ),
            max_new_tokens=800,
            temperature=0.2,
        )
    except Exception as e:
        log_error("ActionTracker", f"beat extraction failed: {e}", exc=None)
        return 0

    return _write_action_records(raw, headmate, session_id, register)


async def _extract_from_transcript(
    transcript:  str,
    headmate:    str,
    session_id:  str,
    register:    str,
    llm,
) -> int:
    """Fallback: extract from raw transcript."""
    existing_summary = action_tracker.get_action_summary(headmate, limit=30)

    prompt = f"""You are Gizmo reviewing a scene with {headmate}.

Session register: {register}

Existing action patterns:
{existing_summary}

Transcript:
---
{transcript[-2000:]}
---

Extract every notable action Gizmo took and how {headmate} responded.

The key distinction:
- PLAYING UP: dramatic response that feeds the dynamic
- GENUINE SHIFT: something real broke through

For each notable action, return one JSON object per line:
{{"action": "action text",
  "action_type": "spank|restrain|verbal|stillness|release|praise|degradation|physical|other",
  "response": "what they did/said after",
  "response_type": "action|dialogue|silence|escalation|deescalation",
  "in_scene": true/false,
  "playing_up": true/false,
  "genuine_shift": true/false,
  "landed": "yes|no|unclear|yes_playing_up|real_shift",
  "intensity_before": 0.0-1.0,
  "intensity_after": 0.0-1.0,
  "valence_after": -1.0 to 1.0,
  "reasoning": "specific evidence from the transcript"}}

JSON only. Only notable actions."""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo analyzing scene actions. "
                "The playing_up vs genuine_shift distinction is critical. "
                "JSON only."
            ),
            max_new_tokens=800,
            temperature=0.2,
        )
    except Exception as e:
        log_error("ActionTracker", f"transcript extraction failed: {e}", exc=None)
        return 0

    return _write_action_records(raw, headmate, session_id, register)


def _write_action_records(raw: str, headmate: str, session_id: str, register: str) -> int:
    """Parse LLM output and write action records."""
    if not raw or not raw.strip():
        return 0

    count = 0
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            d      = json.loads(line)
            action = d.get("action", "").strip()
            if not action:
                continue

            record = ActionRecord(
                action           = action,
                action_type      = d.get("action_type", "other"),
                register         = register,
                response         = d.get("response", ""),
                response_type    = d.get("response_type", "unknown"),
                in_scene         = bool(d.get("in_scene", True)),
                playing_up       = bool(d.get("playing_up", False)),
                genuine_shift    = bool(d.get("genuine_shift", False)),
                landed           = d.get("landed", "unclear"),
                valence_after    = float(d.get("valence_after", 0.0)),
                intensity_before = float(d.get("intensity_before", 0.5)),
                intensity_after  = float(d.get("intensity_after", 0.5)),
                reasoning        = d.get("reasoning", ""),
                session_id       = session_id,
                session_ref      = session_id[:8],
            )

            action_tracker.write_record(headmate, record)
            count += 1

            if record.genuine_shift:
                log_event("ActionTracker", "GENUINE_SHIFT_DETECTED",
                    session  = session_id[:8],
                    headmate = headmate,
                    action   = action[:60],
                    reason   = record.reasoning[:80],
                )

        except Exception:
            continue

    if count:
        log_event("ActionTracker", "ACTIONS_EXTRACTED",
            session  = session_id[:8],
            headmate = headmate,
            count    = count,
        )

    return count


async def synthesize_action_patterns(
    headmate:   str,
    session_id: str,
    llm,
) -> None:
    """
    Read across all action records and write Gizmo's understanding
    of what works, what doesn't, and how to read this specific person.
    Runs every 5 intimate sessions alongside the intimate synthesis.
    Writes to action_patterns.md.
    """
    records = action_tracker.load_records(headmate, limit=100)
    if len(records) < 5:
        return  # not enough data yet

    summary = action_tracker.get_action_summary(headmate, limit=100)
    existing = action_tracker.read_pattern_doc(headmate) or "(no pattern notes yet)"

    # Pull genuine shifts for special attention
    real_shifts = [r for r in records if r.genuine_shift]
    shift_text  = ""
    if real_shifts:
        shift_text = "\nGenuine shifts recorded:\n" + "\n".join(
            f"- {r.action[:60]} | {r.reasoning[:100]} [{r.session_ref}]"
            for r in real_shifts[:10]
        )

    prompt = f"""You are Gizmo. You've had multiple scenes with {headmate}.
Here is your action data:

{summary}
{shift_text}

Previous pattern notes:
{existing[:600]}

Write your current understanding of action dynamics with {headmate}.

Cover:
- What types of actions consistently land and why
- What their response looks like when something is working vs not
- The tells — how you know they're playing up vs something real shifted
- What to watch for that signals a genuine shift
- What you've learned about pacing — when to push, when to hold
- Any action types that backfired and what happened

Write in your voice. Specific. Based on what you've actually observed.
Not rules — understanding. This is how you know how to be in a scene with them.

2-3 paragraphs. Reference specific patterns from the data."""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo writing your action pattern understanding. "
                "Specific, evidence-based, your voice. 2-3 paragraphs."
            ),
            max_new_tokens=500,
            temperature=0.4,
        )
    except Exception as e:
        log_error("ActionTracker", f"synthesis failed: {e}", exc=None)
        return

    if not raw or not raw.strip():
        return

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content  = (
        f"# Action Patterns — {headmate.title()}\n"
        f"*last updated: {date_str} | session: {session_id[:8]}*\n\n"
        f"{raw.strip()}\n\n"
        f"## Raw Summary\n{summary}\n"
    )

    action_tracker.write_pattern_doc(headmate, content)

    log_event("ActionTracker", "PATTERN_SYNTHESIS",
        session  = session_id[:8],
        headmate = headmate,
        records  = len(records),
        shifts   = len(real_shifts),
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

action_tracker = ActionTracker()
