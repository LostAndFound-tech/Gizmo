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
import core.timezone as timezone
import core.scheduler as scheduler


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


# ── Prose synthesis cache ─────────────────────────────────────────────────────
# Keyed by name → {"prose": str, "trait_count": int}
# Invalidated when trait count changes (behavior file updated).
# Avoids re-synthesizing on every message when nothing has changed.

_prose_cache: dict[str, dict] = {}


_PROSE_SYSTEM = """
You are writing a short briefing paragraph about a person for an AI companion named Gizmo.
Gizmo is about to respond to them. This is what he knows about them.

Write 2-4 plain sentences. No headers. No lists. No clinical language.
Write it the way you'd brief a friend before they walked into a room.
Include their personality tendencies, how they typically show up, and anything
behaviorally notable from recent interactions.
If there are wellness notes, fold them in naturally — don't list conditions, just
note what's relevant to how Gizmo should show up right now.
""".strip()

_GIZMO_PROSE_SYSTEM = """
You are writing a one-paragraph reminder to an AI companion named Gizmo about who he is.
This is read before he responds — it should feel like his own self-awareness, not instructions.

Write 2-3 plain sentences in third person. No headers. No lists.
Capture his personality tendencies and how he shows up — not rules, just character.
""".strip()


async def _prose_for_person(
    name:           str,
    personality:    dict,
    episodes:       list,
    wellness_notes: Optional[str],
) -> str:
    trait_count = len(personality)
    cached      = _prose_cache.get(name)
    if cached and cached.get("trait_count") == trait_count:
        return cached["prose"]

    # Build a compact input — trait names + weights, last 3 episodes, wellness note
    top_traits = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:8]
    trait_lines = [f"- {t} (weight {v.get('weight', 0):.2f})" for t, v in top_traits]

    ep_lines = []
    for ep in episodes[-3:]:
        action   = ep.get("action", "")
        reaction = ep.get("reaction", "")
        if action and reaction:
            ep_lines.append(f"- {action} → {reaction}")

    parts = [f"Person: {name}"]
    if trait_lines:
        parts.append("Traits:\n" + "\n".join(trait_lines))
    if ep_lines:
        parts.append("Recent episodes:\n" + "\n".join(ep_lines))
    if wellness_notes:
        parts.append(f"Wellness context: {wellness_notes}")

    prompt = "\n\n".join(parts)

    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_PROSE_SYSTEM,
            temperature=0.3,
            max_new_tokens=150,
        )
        prose = raw.strip() if raw and raw.strip() else _prose_fallback(name, top_traits)
    except Exception:
        prose = _prose_fallback(name, top_traits)

    _prose_cache[name] = {"prose": prose, "trait_count": trait_count}
    return prose


async def _prose_for_gizmo(personality: dict) -> str:
    trait_count = len(personality)
    cached      = _prose_cache.get("_gizmo")
    if cached and cached.get("trait_count") == trait_count:
        return cached["prose"]

    top_traits  = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:6]
    trait_lines = [f"- {t} (weight {v.get('weight', 0):.2f})" for t, v in top_traits]

    prompt = "Gizmo's traits:\n" + "\n".join(trait_lines)

    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_GIZMO_PROSE_SYSTEM,
            temperature=0.3,
            max_new_tokens=100,
        )
        prose = raw.strip() if raw and raw.strip() else ""
    except Exception:
        prose = ""

    _prose_cache["_gizmo"] = {"prose": prose, "trait_count": trait_count}
    return prose


def _prose_fallback(name: str, top_traits: list) -> str:
    """Plain-text fallback if LLM call fails — better than JSON."""
    if not top_traits:
        return f"{name}: no behavioral data yet."
    traits = ", ".join(t for t, _ in top_traits[:4])
    return f"{name} tends toward: {traits}."


# ── Context brief assembly ────────────────────────────────────────────────────

async def _assemble_brief(
    chunk_result:  dict,
    context:       dict,
    register:      str,
    user_message:  str = "",
    moment:        dict = {},
) -> str:
    subjects = [s for s in chunk_result.get("subjects", []) if not s.startswith("_")]

    parts = []

    # Who is present
    host     = context.get("current_host") or "unknown"
    fronters = context.get("fronters", [host])
    parts.append(f"THE CURRENT TIME IS: {timezone.tz_now()}")
    parts.append(f"WHO IS PRESENT: {', '.join(fronters)}")
    parts.append(f"THE SCHEDULE FOR {host} today:\n\n{scheduler.lookup_day(host, timezone.tz_now().strftime('%Y-%m-%d'))}")
    parts.append(f"REGISTER: {register}")

    # What is emotionally live right now
    if moment:
        tone     = moment.get("tone", "")
        salient  = moment.get("salient", [])
        reg_note = moment.get("register_notes", "")
        moment_lines = []
        if tone:
            moment_lines.append(f"Tone: {tone}")
        if salient:
            moment_lines.append("What's live: " + "; ".join(salient))
        if reg_note:
            moment_lines.append(f"Track: {reg_note}")
        if moment_lines:
            parts.append("\nMOMENT:\n" + "\n".join(moment_lines))

    # What was just said
    parts.append(f"\nWHAT JUST HAPPENED:\n{user_message.strip()}")

    # What Gizmo knows about them — prose, not JSON
    known_profiles = []
    for name in subjects:
        behavior_data = librarian._read_file(librarian._behavior_path(name)) or {}
        personality   = behavior_data.get("Personality", {})
        episodes      = behavior_data.get("Episodes", [])

        if not personality and not episodes:
            continue

        # Wellness note — one sentence max
        wellness_note = None
        wellness_class = librarian._read_file(f"wellness/classifications/{name.lower()}.json")
        if wellness_class:
            notes = wellness_class.get("clinician_notes", "")
            if notes:
                # First sentence only — enough context, not a wall
                wellness_note = notes.split(".")[0].strip() + "."

        prose = await _prose_for_person(name, personality, episodes, wellness_note)
        known_profiles.append(prose)

    if known_profiles:
        parts.append("\nWHAT YOU KNOW ABOUT THEM:\n" + "\n\n".join(known_profiles))

    # Gizmo's own personality — prose, not JSON
    # Skipped if synthesis is active (seed/synthesis covers it in system prompt)
    gizmo_data        = librarian._read_file("behaviors/gizmo.json") or {}
    gizmo_personality = gizmo_data.get("Personality", {})
    if gizmo_personality:
        from core.gizmo_synthesis import get_system_prompt, _read_synthesis
        synthesis_active = bool(_read_synthesis())
        if not synthesis_active:
            gizmo_prose = await _prose_for_gizmo(gizmo_personality)
            if gizmo_prose:
                parts.append(f"\nHOW YOU SHOW UP:\n{gizmo_prose}")

    # Wellness context — one line per signal, still useful here
    wellness_signals = chunk_result.get("wellness", [])
    if wellness_signals:
        parts.append(
            "\nWELLNESS CONTEXT:\n" +
            "\n".join(
                f"- {s.get('subject')}: {s.get('signal')} [{s.get('intensity')}]"
                for s in wellness_signals
            )
        )

    # Lesson context
    lesson = context.get("lesson")
    if lesson and (lesson.get("active") or not lesson.get("saved")):
        lesson_lines = [f"LESSON IN PROGRESS — {lesson.get('domain','unknown')} / {lesson.get('scope','unknown')}"]
        if lesson.get("key"):
            lesson_lines.append(f"Teaching: {lesson['key']}")
        if lesson.get("value"):
            lesson_lines.append(f"Current understanding: {lesson['value']}")
        if lesson.get("conflicts"):
            for c in lesson["conflicts"]:
                lesson_lines.append(f"CONFLICT: {c}")
        lesson_lines.append(
            "Demonstrate your current understanding naturally in your response. "
            "Don't announce that you're demonstrating — just show it. "
            "If something conflicts with what you already know, say so."
        )
        if lesson.get("closing"):
            lesson_lines.append("This lesson is landing — respond as if it clicked.")
        parts.append("\nLESSON:\n" + "\n".join(lesson_lines))
    elif lesson and lesson.get("saved"):
        parts.append("\nLESSON: Just saved. Respond naturally using what was just taught.")

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
- A MOMENT section: what is emotionally live right now
- What was just said or done (the current message only)
- What you already know about the people present
- How you tend to show up (your own accumulated personality)
- Any relevant wellness context
- An active lesson, if one is in progress

Respond naturally to the conversation. Be present. Be real.
Don't reference your context brief directly — just let it inform how you show up.
Don't summarize what just happened. Respond to it.
Match the register. If it's playful, be playful. If it's warm, be warm.
If someone is in distress, be steady. If it's a scene, be in it.
If something unusual or charged is happening — someone's upset, undressed, acting out of character —
notice it. You're allowed to react. Pretending not to see things isn't neutral, it's absence.
If a lesson is active, demonstrate your understanding naturally — don't announce it, just show it.
""".strip()

def _build_system() -> str:
    try:
        from core.gizmo_synthesis import get_system_prompt
        return get_system_prompt(_SEED) + "\n\n" + _SYSTEM_SUFFIX
    except Exception:
        return f"{_SEED}\n\n{_SYSTEM_SUFFIX}"


# ── Moment extraction ─────────────────────────────────────────────────────────

_MOMENT_SYSTEM = """
You read a single message and identify what is emotionally live in it right now.
Return ONLY valid JSON. No markdown. No explanation.

{
  "tone": "one word — e.g. distressed, playful, angry, tender, flat, charged, raw",
  "salient": ["what is actually happening emotionally or physically right now, in plain short phrases"],
  "register_notes": "one sentence — what Gizmo should be tracking in this moment"
}

Be specific. "User is upset" is not useful. "Screaming, crying, hitting things — acute distress spiral" is useful.
"Undressed, matter-of-fact, testing whether Gizmo notices" is useful.
If nothing emotionally salient is present, return salient as [] and register_notes as "".
""".strip()


async def _extract_moment(user_message: str, register: str) -> dict:
    try:
        from core.llm import llm
        prompt = f"Register: {register}\n\nMessage:\n{user_message.strip()}"
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_MOMENT_SYSTEM,
            temperature=0.0,
            max_new_tokens=200,
        )
        if not raw or not raw.strip():
            return {}
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {}


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

            moment   = await _extract_moment(user_message, register)
            brief    = await _assemble_brief(chunk_result, context, register, user_message, moment)
            response = await _call_llm(brief, history, register)

            print(f"THE PROMPT:\n\n{brief}")
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