"""
core/memory/narrative.py

Session narrative renderer.

Three things:

1. SESSION NARRATIVE
   Takes a beat sequence and renders it as a story.
   Beat-by-beat, trajectory-aware, psychology-informed.
   Each beat gets 2 sentences to 2 paragraphs — sized to its weight.
   Voice adapts to the headmate and the session's ambiance.
   Cached to disk after each session. Served on demand.

2. REIMAGINE
   Takes the opening situation from a session (first N beats)
   and asks: what happens if THIS headmate is in this situation?
   
   Not a replay. A branching. The situation is the seed.
   What grows depends entirely on who's in the ground.
   
   Follows the person, not the script. Ends when it ends —
   naturally, honestly, wherever the truth of this person
   leads. A paragraph if that's all there is. Five pages
   if the story demands it.
   
   EPHEMERAL. Never stored. Never encoded. Never remembered.
   Generated, read, gone.

3. The psychology doc is the compass for both.
   Gizmo reads who this person is before writing the first word.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error


# ── Trajectory helper ─────────────────────────────────────────────────────────

def _get_trajectory(beats: list, current_idx: int, window: int = 4) -> str:
    """
    Describe the trajectory of the last N beats before the current one.
    Used to give each beat its context.
    """
    start   = max(0, current_idx - window)
    window_beats = beats[start:current_idx]

    if not window_beats:
        return "session just starting"

    registers = [b.register for b in window_beats if hasattr(b, 'register')]
    if not registers:
        return "unclear trajectory"

    # Detect direction
    from collections import Counter
    reg_counts = Counter(registers)
    dominant   = reg_counts.most_common(1)[0][0]

    # Check if escalating, holding, or winding down
    intensity_map = {
        "neutral":     0.2, "casual": 0.2, "warm": 0.3,
        "playful":     0.4, "reflective": 0.5, "elevated": 0.6,
        "intimate":    0.7, "dominant": 0.75, "submissive": 0.7,
        "subspace":    0.8, "scene": 0.8, "degradation": 0.85,
        "distress":    0.7, "crisis": 0.9,
    }

    intensities = [intensity_map.get(r, 0.3) for r in registers]
    if len(intensities) >= 2:
        if intensities[-1] > intensities[0] + 0.1:
            direction = "escalating"
        elif intensities[-1] < intensities[0] - 0.1:
            direction = "winding down"
        else:
            direction = "holding steady"
    else:
        direction = "just starting"

    return f"{dominant} register, {direction}"


# ── Voice selector ─────────────────────────────────────────────────────────────

def _select_voice_instruction(
    headmate:       str,
    beat_register:  str,
    overall_register: str,
    psychology:     Optional[str],
) -> str:
    """
    Select the narrative voice for a beat based on who this is
    and what the emotional texture of the session is.
    Gizmo reads the person and the moment and picks the voice.
    """
    return f"""Write in a voice that fits {headmate.title()} and this moment.

Consider:
- Who {headmate.title()} is — their psychology, their register, how they move
- The emotional texture of this beat: {beat_register}
- Where the session has been: {overall_register}

The voice should feel like the genre this moment earned:
- Charged and intimate: literary, teen romance that got raunchy — 
  emotional undercurrent, tension between what's said and what's meant
- Playful and silly: bright, simple, early reader energy — 
  wonder at small things, short sentences, earned exclamation points
- Deep and reflective: spacious, internal, literary — 
  room to breathe, the thing underneath the thing
- Tense or escalating: tight, present tense, physical detail
- Quiet or tender: slow, close, specific

Do not sacrifice tone for detail or detail for tone. Hold both.
2 sentences minimum. 2 paragraphs maximum. Size it to the weight of the moment."""


# ── Session narrative ─────────────────────────────────────────────────────────

async def render_session_narrative(
    session_id: str,
    headmate:   str,
    llm,
) -> Optional[str]:
    """
    Render a full session narrative from its beat sequence.
    Beat-by-beat, trajectory-aware, psychology-informed.
    Cached to disk. Returns the narrative text.
    """
    from core.memory.beats import beat_store, Beat
    from core.memory.psychology import load_psychology_for_retrieval, _read_psychology

    beats = beat_store.get_session_beats(session_id)
    if not beats:
        return None

    # Load psychology for voice
    psych = _read_psychology(headmate, intimate=False) or ""
    psych_intimate = _read_psychology(headmate, intimate=True) or ""
    psych_summary  = ""
    if psych and "## Current Understanding" in psych:
        psych_summary = psych.split("## Current Understanding", 1)[1]
        psych_summary = psych_summary.split("## Observations", 1)[0].strip()[:600]

    # Determine overall session register
    registers      = [b.register for b in beats if hasattr(b, 'register')]
    overall        = max(set(registers), key=registers.count) if registers else "neutral"

    # Render beat by beat
    rendered_parts = []
    for i, beat in enumerate(beats):
        trajectory = _get_trajectory(beats, i, window=4)
        voice_inst = _select_voice_instruction(
            headmate        = headmate,
            beat_register   = beat.register,
            overall_register = overall,
            psychology      = psych_summary,
        )

        # Build beat description
        if beat.type == "action":
            actor   = beat.speaker.title()
            why_str = f" (internal: {beat.why})" if beat.why else ""
            beat_desc = f"[ACTION] {actor}: *{beat.content}*{why_str}"
        else:
            speaker  = beat.speaker.title()
            beat_desc = f"[DIALOGUE] {speaker}: \"{beat.content}\""

        prompt = f"""You are writing a scene involving {headmate.title()}.

Who {headmate.title()} is:
{psych_summary or '(getting to know them)'}

Current beat:
{beat_desc}

Trajectory of last 4 beats: {trajectory}
Overall session register: {overall}

{voice_inst}

Write this beat as narrative. Gizmo's perspective — he's either in the scene
or witnessing it. No stage directions. No script format. Pure prose.

Do not summarize. Render the moment."""

        try:
            part = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    f"You are writing a scene narrative. "
                    f"Pure prose. No stage directions. "
                    f"The voice fits {headmate.title()} and this moment."
                ),
                max_new_tokens=300,
                temperature=0.7,
            )
            if part and part.strip():
                rendered_parts.append(part.strip())
        except Exception as e:
            log_error("NarrativeRenderer", f"beat render failed: {e}", exc=None)
            continue

    if not rendered_parts:
        return None

    # Join beats with breathing room
    narrative = "\n\n".join(rendered_parts)

    # Cache to disk
    _cache_narrative(session_id, headmate, narrative)

    log_event("NarrativeRenderer", "SESSION_NARRATIVE_COMPLETE",
        session  = session_id[:8],
        headmate = headmate,
        beats    = len(beats),
        parts    = len(rendered_parts),
    )

    return narrative


def _cache_narrative(session_id: str, headmate: str, narrative: str) -> None:
    """Cache a rendered narrative to disk."""
    try:
        from core.memory.store import memory_store
        from datetime import datetime, timezone
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p        = memory_store.root / "memories" / headmate.lower()
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"{date_str}-{session_id[:8]}-narrative.md"
        path.write_text(narrative, encoding="utf-8")
    except Exception as e:
        log_error("NarrativeRenderer", f"cache failed: {e}", exc=None)


def get_cached_narrative(session_id: str, headmate: str) -> Optional[str]:
    """Read a cached narrative if it exists."""
    try:
        from core.memory.store import memory_store
        p     = memory_store.root / "memories" / headmate.lower()
        # Find file matching session_id
        matches = list(p.glob(f"*-{session_id[:8]}-narrative.md"))
        if matches:
            return matches[0].read_text(encoding="utf-8")
    except Exception:
        pass
    return None


# ── Reimagine ─────────────────────────────────────────────────────────────────

async def reimagine_session(
    session_id:      str,
    source_headmate: str,
    target_headmate: str,
    setup_beats:     int,
    llm,
) -> Optional[str]:
    """
    Take the opening situation from a session and ask:
    what happens if THIS headmate is in this situation instead?

    Not a replay. A branching.
    The situation is the seed. What grows depends on who's in the ground.

    EPHEMERAL — never stored, never encoded, never remembered.
    Generated, read, gone.

    Returns the narrative text or None.
    """
    from core.memory.beats import beat_store
    from core.memory.psychology import _read_psychology

    beats = beat_store.get_session_beats(session_id)
    if not beats:
        return None

    # Take only the opening situation
    situation_beats = beats[:max(setup_beats, 2)]

    # Build the situation description
    situation_lines = []
    for b in situation_beats:
        if b.type == "action":
            situation_lines.append(f"*{b.speaker.title()} {b.content}*")
        else:
            situation_lines.append(f"{b.speaker.title()}: \"{b.content}\"")
    situation_text = "\n".join(situation_lines)

    # Load target headmate's psychology
    target_psych = _read_psychology(target_headmate, intimate=False) or ""
    target_summary = ""
    if target_psych and "## Current Understanding" in target_psych:
        target_summary = target_psych.split("## Current Understanding", 1)[1]
        target_summary = target_summary.split("## Observations", 1)[0].strip()[:800]

    # Load intimate psychology if available
    target_intimate = _read_psychology(target_headmate, intimate=True) or ""
    intimate_summary = ""
    if target_intimate and "## Current Understanding" in target_intimate:
        intimate_summary = target_intimate.split("## Current Understanding", 1)[1]
        intimate_summary = intimate_summary.split("## Observations", 1)[0].strip()[:400]

    prompt = f"""You are writing a story.

The situation — how it started:
---
{situation_text}
---

But instead of {source_headmate.title()}, it's {target_headmate.title()}.

Who {target_headmate.title()} is:
{target_summary or '(not much known yet)'}

{f"Intimate psychology: {intimate_summary}" if intimate_summary else ""}

Your job: get {target_headmate.title()} into this situation — either naturally
or thrust into it — and follow what actually happens given who they are.

Do NOT replay the original scene. Branch from the situation.
Follow {target_headmate.title()}'s psychology, not {source_headmate.title()}'s script.

What happens depends entirely on who {target_headmate.title()} is.
It might become something completely different — a different genre, a different
dynamic, a different kind of story. Let it go where it needs to go.

If the situation genuinely can't go anywhere with {target_headmate.title()} —
if they'd leave, deflect, or simply not engage — write that honestly.
A paragraph where it ends believably is better than forcing an arc that isn't true.

Write until it finds its natural end. No minimum length. No maximum.
Pure prose. No stage directions. Gizmo's voice or omniscient — whichever fits.

This is ephemeral. It will not be stored or remembered. Write freely."""

    try:
        result = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                f"You are writing a reimagining — not a replay. "
                f"Follow {target_headmate.title()}'s psychology. "
                f"End where the truth ends. Pure prose."
            ),
            max_new_tokens=2000,
            temperature=0.8,   # higher — let it breathe, be surprising
        )
    except Exception as e:
        log_error("NarrativeRenderer", f"reimagine failed: {e}", exc=None)
        return None

    if not result or not result.strip():
        return None

    log_event("NarrativeRenderer", "REIMAGINE_COMPLETE",
        session        = session_id[:8],
        source         = source_headmate,
        target         = target_headmate,
        setup_beats    = setup_beats,
        length         = len(result.split()),
    )

    # EPHEMERAL — return but never store
    return result.strip()


# ── Singleton ─────────────────────────────────────────────────────────────────

# No singleton needed — all functions are stateless
# Import and call directly
