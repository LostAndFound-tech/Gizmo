"""
memory/overview.py
Auto-generates a conversational overview from recent history turns.
Kicks in after OVERVIEW_AFTER_TURNS and refreshes every OVERVIEW_REFRESH_TURNS.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.history import ConversationHistory

OVERVIEW_AFTER_TURNS = 3       # start generating after this many user turns
OVERVIEW_REFRESH_TURNS = 4     # regenerate every N turns after that

_overviews: dict[str, str] = {}        # session_id -> current overview text
_overview_turn: dict[str, int] = {}    # session_id -> turn count at last generation


async def get_overview(
    session_id: str,
    history: "ConversationHistory",
    llm,
) -> str:
    """
    Return the current overview for a session, regenerating if needed.
    Returns empty string if not enough history yet.
    """
    turn_count = len(history) // 2  # each turn = 1 user + 1 assistant message

    if turn_count < OVERVIEW_AFTER_TURNS:
        return ""

    last_generated_at = _overview_turn.get(session_id, 0)
    needs_refresh = (turn_count - last_generated_at) >= OVERVIEW_REFRESH_TURNS
    first_time = session_id not in _overviews

    if first_time or needs_refresh:
        _overviews[session_id] = await _generate_overview(history, llm)
        _overview_turn[session_id] = turn_count

    return _overviews.get(session_id, "")


async def _generate_overview(history: "ConversationHistory", llm) -> str:
    """
    Ask the LLM to summarize the conversation so far in 2-3 sentences.
    Uses a separate lightweight call — doesn't affect main response.
    """
    # Take last 10 messages max for the summary
    recent = history.as_list()[-10:]
    if not recent:
        return ""

    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )

    prompt = [
        {
            "role": "user",
            "content": (
                f"Summarize this conversation in 2-3 sentences. "
                f"Focus on what's being discussed and any key context established. "
                f"Be concise and factual.\n\n{transcript}"
            )
        }
    ]

    try:
        summary = await llm.generate(
            prompt,
            system_prompt="You summarize conversations concisely. No preamble, just the summary.",
            max_new_tokens=150,
            temperature=0.3,
        )
        return summary.strip()
    except Exception as e:
        print(f"[Overview] Generation failed: {e}")
        return ""


def clear_overview(session_id: str) -> None:
    _overviews.pop(session_id, None)
    _overview_turn.pop(session_id, None)