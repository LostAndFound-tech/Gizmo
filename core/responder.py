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
from core.preference_catcher import get_preferences, get_pending_scope


# ── Personality summary from new nested structure ────────────────────────────

def _top_traits(categories: dict, limit: int) -> list:
    """Extract and sort top traits across all categories in a mood bucket."""
    traits = []
    for category, trait_dict in categories.items():
        if not isinstance(trait_dict, dict):
            continue
        for trait, data in trait_dict.items():
            traits.append({
                "trait":    trait,
                "category": category,
                "count":    data.get("count", 1) if isinstance(data, dict) else 1,
            })
    traits.sort(key=lambda x: x["count"], reverse=True)
    return traits[:limit]


def _personality_summary(
    personality:   dict,
    current_mood:  Optional[str] = None,
    limit_per_mood: int = 3,
) -> dict:
    """
    Convert nested mood/category/trait structure into a focused summary.
    Prioritizes current mood, then top 2 most-evidenced moods as baseline.
    Drops the rest to keep the prompt lean.
    """
    if not personality:
        return {}

    summary = {}

    # Current mood first — most relevant right now
    if current_mood:
        mood_key = current_mood.lower().strip()
        if mood_key in personality:
            top = _top_traits(personality[mood_key], limit_per_mood)
            if top:
                summary[mood_key + " (now)"] = top

    # Score all moods by total evidence count
    mood_scores = {}
    for mood, categories in personality.items():
        if not isinstance(categories, dict):
            continue
        score = sum(
            data.get("count", 1) if isinstance(data, dict) else 1
            for cat_dict in categories.values() if isinstance(cat_dict, dict)
            for data in cat_dict.values()
        )
        mood_scores[mood] = score

    # Top 2 most-evidenced moods as baseline context (skip current)
    top_moods = sorted(mood_scores, key=mood_scores.get, reverse=True)
    baseline_count = 0
    for mood in top_moods:
        if baseline_count >= 2:
            break
        if mood == (current_mood or "").lower().strip():
            continue
        top = _top_traits(personality[mood], limit_per_mood)
        if top:
            summary[mood] = top
            baseline_count += 1

    return summary


def _descriptor_summary(name: str) -> dict:
    """
    Read descriptor file for a person and return a compact summary
    of physical, presentation, relationships, and identity.
    """
    data = librarian._read_file(f"descriptors/{name.lower()}.json") or {}
    if not data:
        return {}

    summary = {}

    # Physical — just top-level keys, not full nested objects (too verbose)
    physical = data.get("physical", {})
    if physical:
        phys_summary = {}
        for k, v in physical.items():
            if k == "notable":
                # Just list notable item names, not full entries
                phys_summary["notable"] = list(v.keys()) if isinstance(v, dict) else []
            elif isinstance(v, list):
                phys_summary[k] = v
        if phys_summary:
            summary["physical"] = phys_summary

    if data.get("presentation"):
        summary["presentation"] = data["presentation"]

    if data.get("identity"):
        summary["identity"] = data["identity"]

    if data.get("notes"):
        summary["notes"] = data["notes"]

    # Relationships — compact
    relationships = data.get("relationships", {})
    if relationships:
        rel_summary = {}
        for person, rel in relationships.items():
            if isinstance(rel, dict):
                rel_summary[person] = rel.get("dynamic", "") + (
                    f" — {rel['notes'][0]}" if rel.get("notes") else ""
                )
        if rel_summary:
            summary["relationships"] = rel_summary

    return summary


def _wellness_summary(name: str) -> Optional[dict]:
    """
    Read wellness classification and per-domain signals.
    Returns a compact summary for tone calibration.
    """
    classification = librarian._read_file(f"wellness/classifications/{name.lower()}.json")
    summary = {}

    if classification:
        conditions = [
            c.get("condition") for c in classification.get("conditions", [])
            if c.get("confidence") in ("moderate", "high")
        ]
        if conditions:
            summary["monitoring"] = conditions
        notes = classification.get("clinician_notes", "")
        if notes:
            summary["notes"] = notes[:200]

    # Pull recent signals from domain files — just the most intense ones
    import os
    wellness_dir = librarian._full_path(f"wellness/{name.lower()}")
    if os.path.isdir(wellness_dir):
        recent_signals = []
        for fname in os.listdir(wellness_dir):
            if not fname.endswith(".json"):
                continue
            domain_data = librarian._read_file(f"wellness/{name.lower()}/{fname}") or {}
            signals = domain_data.get("signals", [])
            # Take last 2 signals per domain that are moderate or severe
            for s in signals[-5:]:
                if s.get("intensity") in ("moderate", "severe"):
                    recent_signals.append({
                        "domain":  fname[:-5],
                        "signal":  s.get("signal", ""),
                        "intensity": s.get("intensity"),
                    })
        if recent_signals:
            summary["recent_signals"] = recent_signals[-5:]

    return summary if summary else None


# ── Context brief assembly ────────────────────────────────────────────────────

def _assemble_brief(
    chunk_result:  dict,
    context:       dict,
    register:      str,
    user_message:  str = "",
) -> str:
    subjects   = [s for s in chunk_result.get("subjects", []) if not s.startswith("_")]
    parts      = []
    host       = context.get("current_host") or "unknown"
    fronters   = context.get("fronters", [host])
    session_id = context.get("session_id", "")

    # Who is present + register
    parts.append("WHO IS PRESENT: " + ", ".join(fronters))
    parts.append("REGISTER: " + register)

    # Extract current mood from wellness router first pass or scene state
    current_mood = None
    wellness_result = chunk_result.get("wellness")
    if isinstance(wellness_result, dict):
        # First pass thinking often notes emotional state
        thinking = wellness_result.get("thinking", {})
        if host and host in thinking:
            # Try to extract a mood word from the thinking
            thinking_text = thinking[host].lower()
            for mood_word in ("happy", "sad", "irritable", "anxious", "playful",
                              "affectionate", "guarded", "flat", "overwhelmed",
                              "calm", "excited", "distressed", "tender", "angry"):
                if mood_word in thinking_text:
                    current_mood = mood_word
                    break

    # Preferences — hard constraints
    if host and host != "unknown":
        prefs     = get_preferences(host)
        standing  = prefs.get("standing", {})
        ephemeral = prefs.get("ephemeral", {})

        if ephemeral:
            lines = ["  - " + p for p in ephemeral.keys()]
            parts.append("ACTIVE INSTRUCTIONS (do these now, no hedging):\n" + "\n".join(lines))

        if standing:
            lines = ["  - " + p + " [" + str(v.get("tier", "")) + "]" for p, v in standing.items()]
            parts.append("HOW THEY WANT TO BE TREATED (always follow these):\n" + "\n".join(lines))

        pending = get_pending_scope(session_id)
        if pending:
            lines = ["  - '" + p["preference"] + "'" for p in pending]
            parts.append("NEEDS SCOPE CLARIFICATION (weave in naturally):\n" + "\n".join(lines))

    # What just happened
    parts.append("\nWHAT JUST HAPPENED:\n" + user_message.strip())

    # Scene state
    if host and host != "unknown":
        try:
            from core.scene_tracker import scene_tracker
            scene_brief = scene_tracker.get_scene_brief(host)
            if scene_brief:
                parts.append("\nSCENE STATE:\n" + scene_brief)
        except Exception:
            pass

    # What Gizmo knows about each subject
    known_profiles = []
    for name in subjects:
        profile_parts = {}

        desc = _descriptor_summary(name)
        if desc:
            profile_parts["descriptor"] = desc

        behavior_data = librarian._read_file("behaviors/" + name.lower() + ".json") or {}
        personality   = behavior_data.get("Personality", {})
        episodes      = behavior_data.get("Episodes", [])

        if personality:
            profile_parts["personality_by_mood"] = _personality_summary(personality, current_mood=current_mood)

        if episodes:
            profile_parts["recent_episodes"] = episodes[-3:]

        wellness = _wellness_summary(name)
        if wellness:
            profile_parts["wellness"] = wellness

        if profile_parts:
            known_profiles.append(name + ":\n" + json.dumps(profile_parts, indent=2))

    if known_profiles:
        parts.append("\nWHAT YOU KNOW ABOUT THEM:\n" + "\n\n".join(known_profiles))

    # Gizmo own personality
    gizmo_data  = librarian._read_file("behaviors/gizmo.json") or {}
    gizmo_pers  = gizmo_data.get("Personality", {})
    gizmo_desc  = _descriptor_summary("gizmo")
    gizmo_parts = {}

    if gizmo_pers:
        gizmo_parts["personality_by_mood"] = _personality_summary(gizmo_pers, limit_per_mood=2)
    if gizmo_desc:
        gizmo_parts["descriptor"] = gizmo_desc

    if gizmo_parts:
        parts.append("\nHOW YOU SHOW UP:\n" + json.dumps(gizmo_parts, indent=2))

    # Live wellness signals from this chunk (wellness_result already read above)
    if isinstance(wellness_result, dict):
        signals = wellness_result.get("signals", [])
        if signals:
            signal_lines = ["- " + str(s) for s in signals[:5]]
            parts.append("\nWELLNESS CONTEXT (this exchange):\n" + "\n".join(signal_lines))

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