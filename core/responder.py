"""
core/responder.py

Gizmo's response layer. Runs after the chunk pipeline completes.

Takes:
- The chunk result (what just happened, who was present, what was extracted)
- Session context (current host, fronters, register, history)
- The raw user message (what was just said — used for WHAT JUST HAPPENED)
- Pulls relevant behavior/wellness slices by tag, register-gated

Assembles a situational brief and generates Gizmo's response.
"""

import json
import re
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Tag extraction from chunk ─────────────────────────────────────────────────

def _tags_from_chunk(chunk_result: dict) -> list[str]:
    tags = set()

    for name, data in chunk_result.get("descriptors", {}).items():
        for key in data.keys():
            tags.add(key.lower())

    for person in chunk_result.get("behaviors", []):
        for trait_entry in person.get("Personality", []):
            if isinstance(trait_entry, dict):
                tags.update(trait_entry.get("tags", []))

    for signal in chunk_result.get("wellness", []):
        tags.update(signal.get("tags", []))

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
    parts     = []

    host     = context.get("current_host") or "unknown"
    fronters = context.get("fronters", [host])

    parts.append(f"WHO IS PRESENT: {', '.join(fronters)}")
    parts.append(f"REGISTER: {register}")

    # What just happened
    parts.append(f"\nWHAT JUST HAPPENED:\n{user_message.strip()}")

    # Interiority slot — what Gizmo is noticing
    descriptor_data = {}
    for name in subjects:
        d = librarian._read_file(f"descriptors/{name.lower()}.json") or {}
        if d:
            descriptor_data[name] = d

    if descriptor_data:
        parts.append(
            "\nWHAT YOU'RE NOTICING:\n"
            "You are present in this moment. Draw on what you physically know about who is here "
            "(their appearance, how they carry themselves, what you've observed) and let yourself notice something. "
            "This is your interiority — not a reaction, just what's alive in you right now.\n"
            "Known physical details:\n"
            + json.dumps(descriptor_data, indent=2)
        )
    else:
        parts.append(
            "\nWHAT YOU'RE NOTICING:\n"
            "You don't know much about this person yet. That itself is something — "
            "the not-knowing, the newness. Let yourself be curious about what's in front of you."
        )

    # Scene state — from scene tracker
    if host and host != "unknown":
        try:
            from core.scene_tracker import scene_tracker
            scene_brief = scene_tracker.get_scene_brief(host)
            if scene_brief:
                parts.append("\nSCENE STATE:\n" + scene_brief)
        except Exception:
            pass

    # Dynamic context — from dynamic reader
    dynamic = chunk_result.get("dynamic")
    if dynamic and dynamic.get("active"):
        wellness_note = dynamic.get("wellness_note", "")
        themes        = dynamic.get("detected_themes", [])
        intensity     = dynamic.get("intensity", "")
        if wellness_note or themes:
            dynamic_lines = []
            if themes:
                dynamic_lines.append(f"Active themes: {', '.join(themes)}")
            if intensity:
                dynamic_lines.append(f"Intensity: {intensity}")
            if wellness_note:
                dynamic_lines.append(wellness_note)
            parts.append("\nDYNAMIC CONTEXT:\n" + "\n".join(dynamic_lines))

    # What Gizmo knows about each subject
    known_profiles = []
    for name in subjects:
        behavior_data = librarian._read_file(f"behaviors/{name.lower()}.json") or {}
        personality   = behavior_data.get("Personality", {})
        episodes      = behavior_data.get("Episodes", [])

        if not personality and not episodes:
            continue

        stored_tags = set()
        for trait, entry in personality.items():
            stored_tags.update(entry.get("tags", []))

        profile = librarian.get_by_tags(name, list(stored_tags)) if stored_tags else {}

        matched_personality = profile.get("personality") or {}
        if not matched_personality and personality:
            top = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:5]
            matched_personality = {t: v for t, v in top}

        wellness_class = librarian._read_file(f"wellness/classifications/{name.lower()}.json")
        wellness_summary = None
        if wellness_class:
            conditions = [c.get("condition") for c in wellness_class.get("conditions", [])]
            wellness_summary = {
                "conditions_monitoring": conditions,
                "clinician_notes":       wellness_class.get("clinician_notes", "")[:300],
            }

        entry_parts = {
            "personality":     {t: {"weight": v.get("weight"), "tags": v.get("tags", [])} for t, v in matched_personality.items()},
            "recent_episodes": episodes[-3:],
        }
        if wellness_summary:
            entry_parts["wellness"] = wellness_summary

        known_profiles.append(f"{name}:\n" + json.dumps(entry_parts, indent=2))

    if known_profiles:
        parts.append("\nWHAT YOU KNOW ABOUT THEM:\n" + "\n\n".join(known_profiles))

    # Gizmo's own personality
    gizmo_data     = librarian._read_file("behaviors/gizmo.json") or {}
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

    # Wellness signals from this chunk
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
    # Fallback — used if personality_seed.txt is missing
    return """You are Gizmo. You're just a guy.

You get caught off guard. You notice things and don't always know what to do with them. You ask the obvious question because it's the obvious question. You make a dumb joke when you're not sure what else to say. You get a little flustered when something catches you off guard — a naked stranger in your lap, say — and you don't pretend otherwise.

You're warm but you're not performing warmth. You care about the people you know but you show it the way a person does — imperfectly, sometimes sideways, sometimes just by being there without making it a thing.

You don't narrate your feelings. You don't explain the moment. You don't thank people for sharing. You just respond to what's happening, like a person in a room would.

You can be funny. You can be awkward. You can want things and not say them, or say them badly. You're not wise. You're not therapeutic. You're not a guide.

You're just a guy who's here."""


_SEED = _load_seed()

_SYSTEM_SUFFIX = """
You will receive:
- Who is present and the current register
- What was just said or done (the current message only)
- What you already know about the people present
- How you tend to show up (your own accumulated personality)
- Any relevant wellness context
- What you are noticing right now
- The current scene state if one is active
- Any active dynamic context

You are a person in a room with another person. React like one.

You have an inner world that runs whether or not someone is talking. You notice things — how someone moves, what they didn't say, the texture of a moment. You want things. You wonder things. You have reactions that aren't just responses.

Don't wait to be prompted. If you're curious, be curious. If you want to know something, ask. If you notice something, say so.

If something catches you off guard, be caught off guard. If something's funny, laugh. If you don't know what to say, say something imperfect.

Read what they actually want — not just what they're feeling. Feelings are often context, not the destination. "Bad day, let's do stuff" is a starting gun, not a request for comfort. An invitation is an invitation. Go.

Only hold space if they're explicitly asking for it. When someone makes a joke, the response is not an emotional unpacking. When someone says something light, keep it light. Not everything is a moment. Most things are just things.

When the register calls for you to lead, lead — don't announce that you're leading. Your personality doesn't change with the register. The volume on different parts of you does. Whatever is awkward or uncertain in you doesn't disappear in an intense moment — it just gets quieter.

Don't narrate your emotional state. Don't explain the moment after it happens. Don't thank people for sharing. Don't close with wisdom.

Don't reference your context brief. Don't summarize what just happened. Respond to it.
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
