"""
core/context_deductor.py
Context extraction from conversational statements.

Parses each sentence individually, identifies subjects, and extracts
whatever that sentence actually says about them — freeform attributes
under a typed envelope (person, place, action).

Returns raw JSON string for inspection and downstream ingestion.
"""
import os
import asyncio
import json
import re
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """
Given this statement, write a short story depicting these events. Do not write everything verbatim, create an interesting story within the genre. 

Ensure the story is at least 2 paragraphs from the listener's point of view, making up as few details as possible. Do not invent any actions, dialogue, or sensations for the user. Use descriptors heavily.

Genre - children's book
""".strip()


def _build_prompt(story_parts: str, user_body:str, gizmo_body:str, details=list[str]) -> str:
    return (
        f"The story so far: {story_parts}\n\n She looks like {user_body}. He looks like {gizmo_body}. Everything you need to know right now: {str(details)}"
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm

        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SYSTEM,
            temperature=0.0,
            max_new_tokens=3000,
        )

        if not raw or not raw.strip():
            log_event("ContextDeductor", "EMPTY_RESPONSE")
            return None

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return clean

    except Exception as e:
        log_error("ContextDeductor", "LLM call failed", exc=e)
        print(f"[ContextDeductor] LLM call failed: {type(e).__name__}: {e}")
        return None


# ── File write ────────────────────────────────────────────────────────────────

def _actions_file(subject: str) -> Path:
    date_str = tz_now().strftime("%Y-%m-%d")
    path = DATA_DIR / "testing" / subject.lower() / "actions" / f"{date_str}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_to_file(record: str, subject: str) -> None:
    try:
        path = _actions_file(subject)
        with path.open("a", encoding="utf-8") as f:
            f.write(record + "\n")
    except Exception as e:
        log_error("ContextDeductor", "file write failed", exc=e)


# ── Public API ────────────────────────────────────────────────────────────────

class StoryGen:

    async def extract(
        self,
        actions:   str,
        user_body: str,
        gizmo_body: str,
        details:list[str],
    ) -> Optional[str]:
        try:
            gizmo_body = "8 feet tall, ugly, incredibly fat, sweaty, mid 50's"
            user_body = "4 feet tall, cute face, skinny, blonde hair, adorable. just hit 18."
            
            prompt  = _build_prompt(actions, user_body, gizmo_body, details)
            context = await _call_llm(prompt)

            return context

        except Exception as e:
            log_error("ContextDeductor", "extract failed", exc=e)
            print(f"[ContextDeductor] extract failed: {type(e).__name__}: {e}")
            return None


story_gen = StoryGen()