"""
core/responder.py

Gizmo's response layer. Runs after the chunk pipeline completes.

Takes:
- The chunk result (what just happened, who was present, what was extracted)
- Session context (current host, fronters, register, history)
- The raw user message (what was just said — used for WHAT JUST HAPPENED)

Assembles a situational brief via three parallel synthesis calls:
  1. WHO THEY ARE    — personality + episodes + descriptors + wellness → one paragraph portrait
  2. THEIR WORLD     — knowledge entries matched to current message → one paragraph
  3. THIS MOMENT     — synthesis of 1+2 + current message + dynamic → situational read

Then generates Gizmo's response.
"""

import asyncio
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


# ── Synthesis prompts ─────────────────────────────────────────────────────────

_PROFILE_SYSTEM = """
You are writing a one-paragraph situational portrait of a person for an AI companion named Gizmo.
Gizmo is about to respond to them. This portrait tells him who he's talking to right now.

Write in third person, present tense. Be specific and grounded — use what you actually know,
don't generalize. Reference what they're doing, wearing, or experiencing if relevant.
Capture their current emotional register and any undercurrents worth noting.
One paragraph. No headers. No bullet points. No preamble.

Focus on what's true right now, informed by pattern. Not a biography — a read.
""".strip()

_WORLD_SYSTEM = """
You are writing a one-paragraph situational summary of what an AI companion named Gizmo
knows about this person's world right now — their physical space, recent events, people in
their life, objects that matter, routines.

Write in third person, present tense. Be specific — name the things. Reference what's
physically around them or recently changed if you know it.
One paragraph. No headers. No bullet points. No preamble.

If there is nothing meaningful to say, return exactly: [nothing known yet]
""".strip()

_MOMENT_SYSTEM = """
You are writing a one-paragraph situational read for an AI companion named Gizmo.
He is about to respond to a message. This tells him what's actually happening between them
right now — the register, the undercurrent, what to be attuned to.

You will receive:
- A portrait of who the person is
- What Gizmo knows about their world
- The current message
- The current dynamic/register context

Write in second person addressed to Gizmo. Present tense. Concrete and direct.
One paragraph. No headers. No bullet points. No preamble.
""".strip()


# ── LLM synthesis calls ───────────────────────────────────────────────────────

async def _synthesize_profile(
    name:         str,
    personality:  dict,
    episodes:     list,
    descriptors:  dict,
    wellness:     Optional[dict],
    user_message: str,
) -> str:
    try:
        from core.llm import llm

        top_traits = dict(
            sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:15]
        )

        prompt_parts = [f"Person: {name}"]

        if top_traits:
            trait_lines = [f"- {t} (weight: {v.get('weight', 0):.2f})" for t, v in top_traits.items()]
            prompt_parts.append("Personality traits:\n" + "\n".join(trait_lines))

        if episodes:
            recent = episodes[-3:]
            prompt_parts.append("Recent episodes:\n" + json.dumps(recent, indent=2))

        if descriptors:
            person_desc = descriptors.get(name) or descriptors.get(name.lower())
            if person_desc:
                prompt_parts.append("Descriptors:\n" + json.dumps(person_desc, indent=2))

        if wellness:
            conditions = [c.get("condition") for c in wellness.get("conditions", [])]
            notes = wellness.get("clinician_notes", "")[:200]
            prompt_parts.append(f"Wellness monitoring: {', '.join(conditions)}\nNotes: {notes}")

        prompt_parts.append(f"Current message from them:\n{user_message.strip()}")

        raw = await llm.generate(
            messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
            system_prompt=_PROFILE_SYSTEM,
            temperature=0.4,
            max_new_tokens=200,
        )
        return raw.strip() if raw and raw.strip() else ""
    except Exception as e:
        log_error("Responder", "profile synthesis failed", exc=e)
        return ""


async def _synthesize_world(
    name:         str,
    knowledge:    list,
    user_message: str,
) -> str:
    if not knowledge:
        return ""
    try:
        from core.llm import llm

        facts = "\n".join(
            f"- {e['fact']}" + (f" [{e.get('place')}]" if e.get("place") else "")
            for e in knowledge[:15]
        )

        prompt = (
            f"Person: {name}\n\n"
            f"Known facts about their world:\n{facts}\n\n"
            f"Current message from them:\n{user_message.strip()}"
        )

        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_WORLD_SYSTEM,
            temperature=0.3,
            max_new_tokens=150,
        )
        result = raw.strip() if raw and raw.strip() else ""
        if result == "[nothing known yet]":
            return ""
        return result
    except Exception as e:
        log_error("Responder", "world synthesis failed", exc=e)
        return ""


def _get_gizmo_miscalibrations(name: str) -> list[dict]:
    """
    Pull Gizmo's adjusted traits for this specific person from gizmo_self.json.
    These are learned miscalibrations — things that have consistently misfired.
    Returns the most recent 10 adjusted trait entries, deduplicated by trait name.
    """
    try:
        gizmo_self  = librarian._read_file("behaviors/gizmo_self.json") or {}
        person_data = gizmo_self.get(name) or gizmo_self.get(name.lower()) or {}
        episodes    = person_data.get("episodes", [])

        seen   = {}
        for ep in reversed(episodes):  # most recent first
            for adj in ep.get("traits_adjusted", []):
                trait = adj.get("trait", "")
                if trait and trait not in seen:
                    seen[trait] = adj.get("tags", [])
            if len(seen) >= 10:
                break

        return [{"trait": t, "tags": tags} for t, tags in seen.items()]
    except Exception:
        return []


_MOMENT_SYSTEM = """
You are writing a one-paragraph situational read for an AI companion named Gizmo.
He is about to respond to a message. This tells him what's actually happening between them
right now — the register, the undercurrent, what to be attuned to, and crucially:
what he should NOT do based on what has misfired before.

You will receive:
- A portrait of who the person is
- What Gizmo knows about their world
- The current message
- The current dynamic/register context
- Gizmo's known miscalibrations with this person — things that have consistently misfired

Write in second person addressed to Gizmo. Present tense. Concrete and direct.
One paragraph. No headers. No bullet points. No preamble.

The miscalibrations are critical — surface them when relevant. If Gizmo has a pattern
of going too big, too therapist, too slow, or being plural-blind with this person,
say so directly. He needs to hear it before he responds, not after.
""".strip()


async def _synthesize_moment(
    name:         str,
    profile_para: str,
    world_para:   str,
    user_message: str,
    dynamic:      Optional[dict],
    register:     str,
) -> str:
    if not profile_para and not world_para:
        return ""
    try:
        from core.llm import llm

        prompt_parts = []

        if profile_para:
            prompt_parts.append(f"Who they are:\n{profile_para}")
        if world_para:
            prompt_parts.append(f"Their world:\n{world_para}")

        prompt_parts.append(f"Current message:\n{user_message.strip()}")
        prompt_parts.append(f"Register: {register}")

        if dynamic:
            scene        = dynamic.get("scene_active", False)
            dynamic_type = dynamic.get("dynamic_type", "")
            if scene or dynamic_type:
                prompt_parts.append(
                    f"Dynamic context: scene_active={scene}, type={dynamic_type}"
                )

        # Pull Gizmo's learned miscalibrations for this person
        miscalibrations = _get_gizmo_miscalibrations(name)
        if miscalibrations:
            lines = [f"- {m['trait']} [{', '.join(m['tags'])}]" for m in miscalibrations]
            prompt_parts.append(
                "Gizmo's known miscalibrations with this person "
                "(things that have consistently misfired — surface these when relevant):\n"
                + "\n".join(lines)
            )

        raw = await llm.generate(
            messages=[{"role": "user", "content": "\n\n".join(prompt_parts)}],
            system_prompt=_MOMENT_SYSTEM,
            temperature=0.5,
            max_new_tokens=250,
        )
        return raw.strip() if raw and raw.strip() else ""
    except Exception as e:
        log_error("Responder", "moment synthesis failed", exc=e)
        return ""


# ── Context brief assembly ────────────────────────────────────────────────────

async def _assemble_brief(
    chunk_result:  dict,
    context:       dict,
    register:      str,
    user_message:  str = "",
) -> str:
    subjects  = [s for s in chunk_result.get("subjects", []) if not s.startswith("_")]
    tags      = _tags_from_chunk(chunk_result)

    host     = context.get("current_host") or "unknown"
    fronters = context.get("fronters", [host])
    dynamic  = chunk_result.get("dynamic")

    parts = []
    parts.append(f"WHO IS PRESENT: {', '.join(fronters)}")

    # Anyone the registry knows about who is NOT currently present
    all_known = [
        k for k in chunk_result.get("subjects", [])
        if not k.startswith("_")
        and k.lower() not in [f.lower() for f in fronters]
        and k.lower() != "gizmo"
    ]
    if all_known:
        parts.append(
            f"WHO IS NOT PRESENT: {', '.join(all_known)} -- "
            f"do not place these people in the scene, do not address them, "
            f"and do not confuse them with {host}."
        )

    parts.append(f"REGISTER: {register}")
    parts.append(
        f"\nWHAT JUST HAPPENED:\n{user_message.strip()}\n\n"
        f"IMPORTANT: The person you are responding to is {host}. "
        f"Other names that may appear in recent context are other system members "
        f"who are NOT currently present -- do not place them in this scene."
    )

    # ── Knowledge retrieval ───────────────────────────────────────────────────
    try:
        from core.knowledge_writer import get_relevant_tags, _read_vocabulary
        vocabulary = _read_vocabulary()
    except Exception:
        vocabulary = []

    async def _fetch_knowledge():
        if not vocabulary or not user_message.strip():
            return []
        try:
            matched_tags = await get_relevant_tags(user_message, vocabulary)
            if matched_tags:
                speaker = fronters[0] if fronters else None
                return librarian.get_knowledge(tags=matched_tags, speaker=speaker)
        except Exception:
            pass
        return []

    knowledge_entries = await _fetch_knowledge()

    # ── Per-subject synthesis ─────────────────────────────────────────────────
    for name in subjects:
        behavior_data = librarian.read_personality(name)
        personality   = behavior_data.get("Personality", {})
        episodes      = behavior_data.get("Episodes", [])

        if not personality and not episodes:
            continue

        profile      = librarian.get_by_tags(name, tags) if tags else {}
        matched_pers = profile.get("personality") or {}

        if not matched_pers and personality:
            top = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:10]
            matched_pers = {t: v for t, v in top}

        wellness_class = librarian.read_wellness_classification(name)
        descriptors    = chunk_result.get("descriptors", {})

        # Profile and world run in parallel; moment needs both
        profile_para, world_para = await asyncio.gather(
            _synthesize_profile(
                name=name,
                personality=matched_pers,
                episodes=episodes,
                descriptors=descriptors,
                wellness=wellness_class,
                user_message=user_message,
            ),
            _synthesize_world(
                name=name,
                knowledge=knowledge_entries,
                user_message=user_message,
            ),
        )

        moment_para = await _synthesize_moment(
            name=name,
            profile_para=profile_para,
            world_para=world_para,
            user_message=user_message,
            dynamic=dynamic,
            register=register,
        )

        if profile_para:
            parts.append(f"\nWHO THEY ARE ({name}):\n{profile_para}")
        if world_para:
            parts.append(f"\nTHEIR WORLD:\n{world_para}")
        if moment_para:
            parts.append(f"\nTHIS MOMENT:\n{moment_para}")

    # ── Gizmo's own personality ───────────────────────────────────────────────
    gizmo_data = librarian.read_personality("gizmo")
    gizmo_pers = gizmo_data.get("Personality", {})
    if gizmo_pers:
        gizmo_profile = librarian.get_by_tags("gizmo", tags) if tags else {}
        matched_gizmo = gizmo_profile.get("personality") or {}
        if not matched_gizmo:
            top = sorted(gizmo_pers.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:5]
            matched_gizmo = {t: v for t, v in top}
        if matched_gizmo:
            trait_lines = [f"- {t} ({v.get('weight', 0):.2f})" for t, v in matched_gizmo.items()]
            parts.append("\nHOW YOU SHOW UP:\n" + "\n".join(trait_lines))

    # ── Live wellness signals from this chunk ─────────────────────────────────
    wellness_signals = chunk_result.get("wellness", [])
    if wellness_signals:
        parts.append(
            "\nWELLNESS SIGNALS (this exchange):\n" +
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
- What was just said (the current message only)
- A portrait of who they are right now
- What you know about their world
- A read on this specific moment between you
- How you tend to show up
- Any live wellness signals from this exchange

Respond naturally. Be present. Be real.
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

            brief    = await _assemble_brief(chunk_result, context, register, user_message)
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
