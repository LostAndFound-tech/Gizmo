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

Brief sections (in order):
  WHO IS PRESENT
  REGISTER
  WHAT JUST HAPPENED
  WHAT YOU KNOW ABOUT THEM        ← from behaviors/{name}.json
  HOW YOU KNOW THIS PERSON        ← from behaviors/gizmo_self.json (new)
  HOW YOU SHOW UP                 ← from behaviors/gizmo.json
  WELLNESS CONTEXT
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


# ── Tone read ─────────────────────────────────────────────────────────────────

_TONE_SYSTEM = """
You read the mood and situational texture of a short exchange.
Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "tags": ["couch-mode", "tired-but-warm", "just-chatting"]
}

Rules:
- 1 to 5 tags
- Single words or hyphenated phrases, lowercase
- Capture texture, not category — "tired-but-warm" not "emotional"
- If the message is too short or neutral to read, return {"tags": []}
""".strip()


async def _read_tone(user_message: str) -> list[str]:
    """
    Quick LLM pass to get situational tags for the current message.
    Used to match against gizmo_self.json episodes.
    Returns empty list if signal is too thin or call fails.
    """
    if not user_message or len(user_message.split()) < 4:
        return []
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=_TONE_SYSTEM,
            temperature=0.0,
            max_new_tokens=100,
        )
        if not raw or not raw.strip():
            return []
        clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        return parsed.get("tags", [])
    except Exception as e:
        log_error("Responder", "tone read failed", exc=e)
        print(f"[Responder] tone read failed: {type(e).__name__}: {e}")
        return []


# ── Gizmo self-knowledge assembly ─────────────────────────────────────────────

def _closeness_label(weight: float) -> str:
    """Convert closeness float to a human-readable label for the brief."""
    if weight >= 0.5:  return "very close"
    if weight >= 0.3:  return "close"
    if weight >= 0.15: return "familiar"
    if weight >= 0.05: return "acquaintance"
    return "just met"


def _closeness(name: str, data: dict) -> float:
    total = sum(
        v.get("episode_count", 0)
        for k, v in data.items()
        if isinstance(v, dict) and k != "tag_vocabulary"
    )
    if total == 0:
        return 1.0
    person_count = data.get(name, {}).get("episode_count", 0)
    return round(person_count / total, 4)


async def _assemble_gizmo_knowledge(name: str, user_message: str) -> Optional[str]:
    """
    Build the HOW YOU KNOW THIS PERSON section of the brief.

    1. Load gizmo_self.json
    2. Compute closeness for this person
    3. Read tone of current message
    4. Match tone tags against episode situational_tags
    5. Surface what landed, what didn't, punch bowl moments
    6. Return None if no prior experience — Gizmo goes in fresh
    """
    if not name or name == "unknown":
        return None

    self_data   = librarian._read_file("behaviors/gizmo_self.json") or {}
    person_data = self_data.get(name)

    if not person_data:
        return None
    
    if not self_data:
        with open("behaviors/gizmo_self.json", "w") as file:
            file.write("I am Gizmo.")

    episodes = person_data.get("episodes", [])
    if not episodes:
        return None
    
    

    closeness       = _closeness(name, self_data)
    closeness_label = _closeness_label(closeness)
    current_tags    = await _read_tone(user_message)

    # ── Match episodes by situational tag overlap ─────────────────────────────
    matched_episodes = []
    if current_tags:
        current_set = set(current_tags)
        for ep in episodes:
            ep_tags = set(ep.get("situational_tags", []))
            if ep_tags & current_set:
                matched_episodes.append(ep)

    # Fall back to most recent if no tag match
    if not matched_episodes:
        matched_episodes = episodes[-3:]

    # ── Aggregate what landed / what didn't ───────────────────────────────────
    landed   = []
    missed   = []
    punches  = []

    for ep in matched_episodes:
        if ep.get("punch_bowl"):
            punches.append(ep.get("what_missed", ""))
        for t in ep.get("traits_reinforced", []):
            trait = t.get("trait") if isinstance(t, dict) else t
            if trait and trait not in landed:
                landed.append(trait)
        for t in ep.get("traits_adjusted", []):
            trait = t.get("trait") if isinstance(t, dict) else t
            if trait and trait not in missed:
                missed.append(trait)

    # ── Build section ─────────────────────────────────────────────────────────
    lines = [f"HOW YOU KNOW THIS PERSON ({name}):"]
    lines.append(f"Closeness: {closeness_label} ({closeness})")

    if current_tags:
        lines.append(f"Vibe right now: {', '.join(current_tags)}")

    if landed:
        lines.append(f"\nWhat works with them in moments like this:")
        for t in landed[:4]:
            lines.append(f"  - {t}")

    if missed:
        lines.append(f"\nWhat doesn't:")
        for t in missed[:4]:
            lines.append(f"  - {t}")

    if punches:
        lines.append(f"\nPunch bowl — don't do this:")
        for p in punches[:2]:
            if p:
                lines.append(f"  - {p}")

    # Surface 2 most relevant recent episodes raw
    recent_relevant = matched_episodes[-2:]
    if recent_relevant:
        lines.append("\nRecent moments that rhyme with this:")
        for ep in recent_relevant:
            lines.append(
                f"  [{', '.join(ep.get('situational_tags', [])[:3])}] "
                f"{ep.get('assessment', '')}"
            )

    return "\n".join(lines)


# ── Context brief assembly ────────────────────────────────────────────────────

async def _assemble_brief(
    chunk_result:  dict,
    context:       dict,
    register:      str,
    user_message:  str = "",
) -> str:
    subjects = [s for s in chunk_result.get("subjects", []) if not s.startswith("_")]
    tags     = _tags_from_chunk(chunk_result)

    print(f"For subjects:{subjects}, I have the following tags: \n{tags}\n")
    parts = []

    host     = context.get("current_host") or "unknown"
    fronters = context.get("fronters", [host])
    parts.append(f"[WHO IS PRESENT] {', '.join(fronters)}")
    parts.append(f"[REGISTER] {register}")

    parts.append(f"\n[WHAT JUST HAPPENED]\n{user_message.strip()}")

    # ── What Gizmo knows about them ───────────────────────────────────────────
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
        print(f"The Profile Information I have is:\n{profile}")
        matched_personality = profile.get("personality") or {}
        print(f"My personality:{matched_personality}")
        if not matched_personality and personality:
            top = sorted(personality.items(), key=lambda x: x[1].get("weight", 0), reverse=True)[:5]
            matched_personality = {t: v for t, v in top}

        wellness_class   = librarian._read_file(f"wellness/classifications/{name.lower()}.json")
        wellness_summary = None
        if wellness_class:
            conditions = [c.get("condition") for c in wellness_class.get("conditions", [])]
            wellness_summary = {
                "conditions_monitoring": conditions,
                "clinician_notes":       wellness_class.get("clinician_notes", "")[:300],
            }
        
        print(f"What I'm looking at: {wellness_summary}")

        entry_parts = {
            "personality":     {t: {"weight": v.get("weight"), "tags": v.get("tags", [])} for t, v in matched_personality.items()},
            "recent_episodes": episodes[-3:],
        }
        if wellness_summary:
            entry_parts["wellness"] = wellness_summary

        known_profiles.append(f"{name}:\n" + json.dumps(entry_parts, indent=2))

    if known_profiles:
        parts.append("\nWHAT YOU KNOW ABOUT THEM:\n" + "\n\n".join(known_profiles))

    # ── How Gizmo knows this person — self-reflection data ────────────────────
    gizmo_knowledge = await _assemble_gizmo_knowledge(host, user_message)
    if gizmo_knowledge:
        parts.append(f"\n{gizmo_knowledge}")
    print(f"The knowledge I'm going in with:\n{gizmo_knowledge}")
    # If None — no section, Gizmo goes in fresh. Intentional.

    # ── How Gizmo shows up — his own accumulated personality ──────────────────
    gizmo_data        = librarian._read_file("behaviors/gizmo.json") or {}
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
                "\n[HOW YOU SHOW UP]\n"
                + json.dumps({t: {"weight": v.get("weight")} for t, v in matched_gizmo.items()}, indent=2)
            )
        print(f"My personality from here:\n{gizmo_personality}")

    # ── Wellness context ──────────────────────────────────────────────────────
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
Use all this information to respond to the user. You are a friend of theirs. They trust you, and like you. They are 
hanging out with you. This is information know you about them, so you can respond to them as they are. Take only this information
in mind as you consider your response. They are a fully realized person, and think of you the same way. You are on
equal footing with this person.
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
            max_new_tokens=2500,
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

            # _assemble_brief is now async — it calls _read_tone internally
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
