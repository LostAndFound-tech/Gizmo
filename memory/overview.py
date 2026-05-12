"""
memory/overview.py
Auto-generates a conversational overview from recent history turns.
Kicks in after OVERVIEW_AFTER_TURNS and refreshes every OVERVIEW_REFRESH_TURNS.

Persona-aware: uses persona_prefix_multi() to build the system prompt
from the headmate JSON files directly — no LLM call, no ChromaDB query.
The overview is always written through the correct relational lens.

Time-aware: time_context_block() injected so the LLM knows when this
conversation is happening.

Failure diagnostics: errors are classified and logged with full context
so "summary failed to generate" is never the only information you get.
"""

from __future__ import annotations
import traceback
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from memory.history import ConversationHistory

OVERVIEW_AFTER_TURNS  = 3   # start generating after this many user turns
OVERVIEW_REFRESH_TURNS = 4  # regenerate every N turns after that

_overviews:     dict[str, str] = {}   # session_id -> current overview text
_overview_turn: dict[str, int] = {}   # session_id -> turn count at last generation


async def get_overview(
    session_id: str,
    history: "ConversationHistory",
    llm,
    speaker: Optional[str] = None,
    fronters: Optional[list] = None,
) -> str:
    """
    Return the current overview for a session, regenerating if needed.
    Returns empty string if not enough history yet.

    speaker:  current_host — whose lens to write through
    fronters: everyone else present — included in persona context
    """
    turn_count = len(history) // 2

    if turn_count < OVERVIEW_AFTER_TURNS:
        return ""

    last_generated_at = _overview_turn.get(session_id, 0)
    needs_refresh     = (turn_count - last_generated_at) >= OVERVIEW_REFRESH_TURNS
    first_time        = session_id not in _overviews

    if first_time or needs_refresh:
        _overviews[session_id]     = await _generate_overview(
            session_id, history, llm, speaker, fronters
        )
        _overview_turn[session_id] = turn_count

    return _overviews.get(session_id, "")


async def _generate_overview(
    session_id: str,
    history: "ConversationHistory",
    llm,
    speaker: Optional[str] = None,
    fronters: Optional[list] = None,
) -> str:
    """
    Ask the LLM to summarize the conversation so far in 2-3 sentences,
    written through Gizmo's persona + whoever is present.

    On failure, classifies and logs:
      CONTENT_FILTER  — provider rejected the call
      EMPTY_RESPONSE  — model returned nothing
      PARSE_ERROR     — response came back malformed
      TIMEOUT         — request timed out
      LLM_ERROR       — anything else
      HISTORY_EMPTY   — nothing to summarize
    """
    from core.persona import persona_prefix_multi
    from core.temporal import time_context_block

    recent = history.as_list()[-10:]
    if not recent:
        print(f"[Overview:{session_id[:8]}] HISTORY_EMPTY — nothing to summarize")
        return ""

    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content']}"
        for m in recent
    )

    all_present = list({s for s in ([speaker] + (fronters or [])) if s})
    persona     = persona_prefix_multi(all_present, include_gizmo_seed=True)
    time_ctx    = time_context_block()

    system_prompt = (
        f"{persona}\n\n"
        f"You are reflecting on your own ongoing conversation. "
        f"Write a brief 2-3 sentence summary in your own voice — "
        f"what's being discussed, what matters, what you're paying attention to. "
        f"Let your knowledge of who you're talking to color what feels significant. "
        f"No preamble. Write as yourself."
    )

    prompt = [{
        "role": "user",
        "content": (
            f"{time_ctx}\n\n"
            f"Summarize this conversation from your perspective in 2-3 sentences.\n\n"
            f"{transcript}"
        ),
    }]

    raw = None
    try:
        raw = await llm.generate(
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=150,
            temperature=0.3,
        )

        if not raw or not raw.strip():
            print(
                f"[Overview:{session_id[:8]}] EMPTY_RESPONSE — "
                f"model returned nothing (raw={repr(raw)}) "
                f"| speaker={speaker} | turns={len(recent)}"
            )
            return ""

        return raw.strip()

    except Exception as e:
        tb     = traceback.format_exc()
        e_str  = str(e).lower()
        e_type = type(e).__name__

        if any(k in e_str for k in ("content", "filter", "policy", "moderat", "safety", "400", "451")):
            label = "CONTENT_FILTER"
            hint  = "provider rejected the summary call — try adjusting the system prompt framing"
        elif any(k in e_str for k in ("timeout", "timed out", "read timeout", "connect timeout")):
            label = "TIMEOUT"
            hint  = "LLM call timed out — transient or overloaded provider"
        elif any(k in e_str for k in ("json", "parse", "decode", "unexpected")):
            label = "PARSE_ERROR"
            hint  = f"response came back malformed (raw={repr(raw)})"
        elif any(k in e_str for k in ("502", "503", "504", "unavailable", "connection")):
            label = "LLM_ERROR"
            hint  = "provider returned a server error — transient"
        else:
            label = "LLM_ERROR"
            hint  = "unclassified exception — see traceback below"

        print(
            f"[Overview:{session_id[:8]}] {label} — {hint}\n"
            f"  exception type : {e_type}\n"
            f"  exception msg  : {e}\n"
            f"  speaker        : {speaker}\n"
            f"  turns in window: {len(recent)}\n"
            f"  transcript len : {len(transcript)} chars\n"
            f"  traceback      :\n{tb}"
        )
        return ""


def clear_overview(session_id: str) -> None:
    _overviews.pop(session_id, None)
    _overview_turn.pop(session_id, None)
