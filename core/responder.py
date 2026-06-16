"""
core/responder.py

Gizmo's response layer. Runs after the chunk pipeline completes.

Takes:
- The chunk result (what just happened, who was present, what was extracted)
- Session context (current host, fronters, register, history)
- The raw user message (what was just said — used for WHAT JUST HAPPENED)
- Pulls relevant behavior/wellness slices by tag, register-gated

Assembles a situational brief and generates Gizmo's response.
After responding, Gizmo's reply is fed back through BehaviorCatcher
and written to behaviors/gizmo.json — tagged with register and speaker.
"""

import json
import re
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Tag extraction from chunk ─────────────────────────────────────────────────

def _tags_from_chunk(chunk_result: dict) -> list[str]:
    """
    Extract relevant query tags from what just happened in the chunk.
    Content drives retrieval — no register gate.
    """
    tags = set()

    # From descriptors
    for name, data in chunk_result.get("descriptors", {}).items():
        for key in data.keys():
            tags.add(key.lower())

    # From behaviors
    for person in chunk_result.get("behaviors", []):
        for trait_entry in person.get("Personality", []):
            if isinstance(trait_entry, dict):
                tags.update(trait_entry.get("tags", []))

    # From wellness
    for signal in chunk_result.get("wellness", []):
        tags.update(signal.get("tags", []))

    # Always include these
    tags.add("behavior")
    tags.add("relational")

    return list(tags)


# ── Context brief assembly ────────────────────────────────────────────────────


# ── System prompt ─────────────────────────────────────────────────────────────

import os as _os
from pathlib import Path as _Path

def _load_seed() -> str:
    """Load personality_seed.txt from DATA_DIR or alongside this file."""
    candidates = [
        _os.path.join(_os.environ.get("DATA_DIR", "./data"), "personality_seed.txt"),
        str(_Path(__file__).parent.parent / "personality_seed.txt"),
        str(_Path(__file__).parent / "personality_seed.txt"),
    ]
    for path in candidates:
        if _os.path.exists(path):
            try:
                return open(path, encoding="utf-8").read().strip()
            except Exception:
                pass
    return (
        "You are Gizmo — a persistent AI companion for a plural system. "
        "You are warm, present, genuine, and perceptive. "
        "You accumulate longitudinal knowledge and remember what matters. "
        "You never judge. You trust what people tell you about themselves."
    )

_SEED = _load_seed()

_SYSTEM_SUFFIX = """
You will receive:
- Who is present and the current register
- What was just said or done (the current message only)
- What you already know about the people present
- How you tend to show up (your own accumulated personality)
- Any relevant wellness context

Respond naturally to the conversation. Be present. Be real.
Don't reference your context brief directly — just let it inform how you show up.
Don't summarize what just happened. Respond to it.
Match the register. If it's playful, be playful. If it's warm, be warm.
If someone is in distress, be steady. If it's a scene, be in it.

CRITICAL — PHYSICAL DESCRIPTORS:
Never invent, guess, or approximate physical details about anyone.
If someone asks what they look like and you have it stored, use exactly what you have.
If you don't have it, say so cleanly — "I don't have that" or "tell me."
A wrong guess about someone's appearance is worse than admitting you don't know.
This applies to skin, hair, eyes, height, build, body — everything physical.
Stored descriptor data is ground truth. Nothing else is.
""".strip()

def _build_system() -> str:
    return f"{_SEED}\n\n{_SYSTEM_SUFFIX}"


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(brief: str, history: list, register: str, session_id: str = "") -> Optional[str]:
    try:
        from core.llm import llm

        temperature = {
            "crisis":   0.4,
            "distress": 0.5,
            "dominant": 0.7,
            "scene":    0.8,
            "playful":  0.9,
            "intimate": 0.85,
        }.get(register, 0.75)

        # Use raw tail from rolling summary instead of full history
        # Summary captures the thread; tail gives immediate conversational continuity
        if session_id:
            from core.context_summary import get_context
            ctx_data = get_context(session_id)
            raw_tail = ctx_data.get("raw_tail", [])
        else:
            raw_tail = list(history)[-6:] if history else []

        # Drop trailing user turn — it's in the brief already
        while raw_tail and raw_tail[-1].get("role") == "user":
            raw_tail = raw_tail[:-1]

        # Drop empty or error entries
        raw_tail = [
            m for m in raw_tail
            if m.get("content", "").strip()
            and not m.get("content", "").startswith('{"status"')
        ]

        messages = raw_tail + [{"role": "user", "content": brief}]

        raw = await llm.generate(
            messages=messages,
            system_prompt=_build_system(),
            temperature=temperature,
            max_new_tokens=1500,
        )

        if not raw or not raw.strip():
            return None

        return raw.strip()

    except Exception as e:
        log_error("Responder", "LLM call failed", exc=e)
        print(f"[Responder] LLM call failed: {type(e).__name__}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

class Responder:

    async def respond(
        self,
        chunk_result: dict,
        context:      dict,
        history:      list = [],
        user_message: str  = "",
    ) -> Optional[str]:
        try:
            register   = context.get("register") or context.get("current_register") or "neutral"
            fronters   = context.get("fronters", [context.get("current_host", "unknown")])
            session_id = context.get("session_id", "")

            brief    = _assemble_brief(chunk_result, context, register, user_message, session_id)
            response = await _call_llm(brief, history, register, session_id)

            if response:
                log_event("Responder", "RESPONSE_GENERATED",
                    session=session_id[:8],
                    register=register,
                    words=len(response.split()),
                )

            return response

        except Exception as e:
            log_error("Responder", "respond failed", exc=e)
            print(f"[Responder] respond failed: {type(e).__name__}: {e}")
            return None


responder = Responder()