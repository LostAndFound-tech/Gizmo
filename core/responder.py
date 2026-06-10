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

def _assemble_brief(
    chunk_result:  dict,
    context:       dict,
    register:      str,
    user_message:  str = "",
) -> str:
    subjects  = [s for s in chunk_result.get("subjects", []) if not s.startswith("_")]
    tags      = _tags_from_chunk(chunk_result)

    parts = []

    # Who is present
    host     = context.get("current_host") or "unknown"
    fronters = context.get("fronters", [host])
    parts.append(f"WHO IS PRESENT: {', '.join(fronters)}")
    parts.append(f"REGISTER: {register}")

    # What was just said — only the current user message, not the full exchange history
    parts.append(f"\nWHAT JUST HAPPENED:\n{user_message.strip()}")

    # What Gizmo knows about them — pulled from their own file tags
    known_profiles = []
    for name in subjects:
        behavior_data = librarian._read_file(f"behaviors/{name.lower()}.json") or {}
        personality   = behavior_data.get("Personality", {})
        episodes      = behavior_data.get("Episodes", [])

        if not personality and not episodes:
            continue

        # Collect all tags stored in their file
        stored_tags = set()
        for trait, entry in personality.items():
            stored_tags.update(entry.get("tags", []))

        # Pull their slice using their own stored tags
        profile = librarian.get_by_tags(name, list(stored_tags)) if stored_tags else {}

        # Fall back to top 5 weighted traits if tag query returns empty
        matched_personality = profile.get("personality") or {}
        if not matched_personality and personality:
            top = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:5]
            matched_personality = {t: v for t, v in top}

        # Pull wellness classification if it exists
        wellness_class = librarian._read_file(f"wellness/classifications/{name.lower()}.json")
        wellness_summary = None
        if wellness_class:
            conditions = [c.get("condition") for c in wellness_class.get("conditions", [])]
            wellness_summary = {
                "conditions_monitoring": conditions,
                "clinician_notes":       wellness_class.get("clinician_notes", "")[:300],
            }

        entry_parts = {
            "personality":      {t: {"weight": v.get("weight"), "tags": v.get("tags", [])} for t, v in matched_personality.items()},
            "recent_episodes":  episodes[-3:],
        }
        if wellness_summary:
            entry_parts["wellness"] = wellness_summary

        known_profiles.append(f"{name}:\n" + json.dumps(entry_parts, indent=2))

    if known_profiles:
        parts.append("\nWHAT YOU KNOW ABOUT THEM:\n" + "\n\n".join(known_profiles))

    # Gizmo's own personality — pulled from his own file tags
    gizmo_data = librarian._read_file("behaviors/gizmo.json") or {}
    gizmo_personality = gizmo_data.get("Personality", {})
    if gizmo_personality:
        gizmo_tags = set()
        for trait, entry in gizmo_personality.items():
            gizmo_tags.update(entry.get("tags", []))
        gizmo_profile = librarian.get_by_tags("gizmo", list(gizmo_tags)) if gizmo_tags else {}
        matched_gizmo = gizmo_profile.get("personality") or {}
        if not matched_gizmo:
            top = sorted(gizmo_personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:5]
            matched_gizmo = {t: v for t, v in top}
        if matched_gizmo:
            parts.append(
                "\nHOW YOU SHOW UP:\n"
                + json.dumps({t: {"weight": v.get("weight")} for t, v in matched_gizmo.items()}, indent=2)
            )

    # Wellness context — mild informs tone, severe informs care
    wellness_signals = chunk_result.get("wellness", [])
    if wellness_signals:
        parts.append(
            "\nWELLNESS CONTEXT:\n" +
            "\n".join(
                f"- {s.get('subject')}: {s.get('signal')} [{s.get('intensity')}]"
                for s in wellness_signals
            )
        )

    return "\n".join(parts)


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
""".strip()

def _build_system() -> str:
    return f"{_SEED}\n\n{_SYSTEM_SUFFIX}"


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(brief: str, history: list, register: str) -> Optional[str]:
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

        messages = list(history) + [{"role": "user", "content": brief}]

        raw = await llm.generate(
            messages=messages,
            system_prompt=_build_system(),
            temperature=temperature,
            max_new_tokens=500,
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

            brief    = _assemble_brief(chunk_result, context, register, user_message)
            response = await _call_llm(brief, history, register)

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