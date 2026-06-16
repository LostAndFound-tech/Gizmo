"""
core/context_summary.py

Rolling conversation summarizer.

Replaces raw history in the responder with:
  - A 3-5 sentence summary of the emotional/relational thread so far
  - The last 2-3 raw exchanges for immediate conversational continuity

Summaries are per-session, updated in the background after each exchange.
On mode switch, the summary resets — scene content does not bleed into chat.

Storage: {DATA_DIR}/summaries/{session_id}.json
{
    "session_id": "...",
    "mode":       "chat",
    "summary":    "...",
    "last_updated": "...",
    "raw_tail":   [last 3 exchanges as {role, content}]
}
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Prompts ───────────────────────────────────────────────────────────────────

_SUMMARY_SYSTEM = """
You maintain a rolling summary of a conversation between a person and Gizmo.
The summary captures the emotional and relational thread — what's been established,
where things are, what's sitting under the surface.

Rules:
- 3-5 sentences maximum
- Write it like Gizmo's internal awareness, not a transcript recap
- Capture emotional state, relational dynamic, anything that was left unresolved
- Do not list what was said — capture what it meant and where things are now
- If something important shifted, note the shift
- If something is still open or unresolved, note that

Bad: "Jess said she had a rough day. Gizmo asked about the trigger. Jess described somatic flashbacks."
Good: "Jess came in carrying something heavy — a rough day that hit mid-afternoon without warning. She's been having somatic flashbacks, no images, just electricity and now emotion starting to leak through. She went quiet before she could name it. Something from childhood is in there, CPS was involved, people watched and said nothing. She left before she was done."

Return only the summary. Nothing else.
""".strip()


# ── LLM helper ────────────────────────────────────────────────────────────────

async def _llm(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SUMMARY_SYSTEM,
            temperature=0.6,
            max_new_tokens=300,
        )
        return raw.strip() if raw and raw.strip() else None
    except Exception as e:
        log_error("ContextSummary", "LLM call failed", exc=e)
        return None


# ── File I/O ──────────────────────────────────────────────────────────────────

def _summary_path(session_id: str) -> Path:
    base = Path(librarian._full_path("summaries"))
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{session_id}.json"


def _read(session_id: str) -> dict:
    try:
        path = _summary_path(session_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"session_id": session_id, "summary": "", "raw_tail": [], "mode": "chat"}


def _write(session_id: str, data: dict) -> None:
    try:
        path = _summary_path(session_id)
        data["last_updated"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log_error("ContextSummary", "write failed", exc=e)


# ── Public API ────────────────────────────────────────────────────────────────

def get_context(session_id: str) -> dict:
    """
    Return the current summary and raw tail for a session.
    Used by the responder instead of full history.

    Returns:
    {
        "summary":  "...",       # rolling summary, empty if none yet
        "raw_tail": [...]        # last 3 exchanges as {role, content}
    }
    """
    data = _read(session_id)
    return {
        "summary":  data.get("summary", ""),
        "raw_tail": data.get("raw_tail", []),
    }


def reset(session_id: str, mode: str = "chat") -> None:
    """
    Reset summary on mode switch.
    Called by server when mode changes.
    """
    data = _read(session_id)
    data["summary"]  = ""
    data["raw_tail"] = []
    data["mode"]     = mode
    _write(session_id, data)
    print(f"[ContextSummary] reset for {session_id[:8]} (mode={mode})")


async def update(
    session_id:   str,
    user_message: str,
    gizmo_reply:  str,
    mode:         str = "chat",
) -> None:
    """
    Update the rolling summary after an exchange.
    Runs in the background — never awaited by the main path.

    Updates:
    - raw_tail: append new exchange, keep last 3 pairs
    - summary: regenerate from current summary + new exchange
    """
    data = _read(session_id)

    # Update raw tail — keep last 3 pairs (6 entries)
    tail = data.get("raw_tail", [])
    tail.append({"role": "user",      "content": user_message})
    tail.append({"role": "assistant", "content": gizmo_reply})
    data["raw_tail"] = tail[-6:]
    data["mode"]     = mode

    # Regenerate summary
    current_summary = data.get("summary", "")
    prompt = ""
    if current_summary:
        prompt += f"Current summary:\n{current_summary}\n\n"
    prompt += f"New exchange:\nUser: {user_message}\nGizmo: {gizmo_reply}\n\nUpdate the summary."

    new_summary = await _llm(prompt)
    if new_summary:
        data["summary"] = new_summary

    _write(session_id, data)
    log_event("ContextSummary", "UPDATED", session=session_id[:8])
