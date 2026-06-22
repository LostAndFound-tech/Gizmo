"""
core/gizmo_reflection.py

Gizmo's self-reflection pass. Fires async after every response — fire and forget,
never blocks the user.

Takes three parts:
  - user_message:   what they said
  - gizmo_response: what Gizmo said back
  - next_message:   how they responded to that (the implicit feedback signal)

Two LLM passes:
  1. Reflection pass  — what happened, what landed, what didn't, traits to reinforce or adjust
  2. Tag pass         — what words describe this exchange, matched against existing vocabulary

Writes to: {DATA_DIR}/behaviors/gizmo_self.json

Structure:
{
  "tag_vocabulary": ["tender", "deflecting", ...],   ← grows over time
  "Jess": {
    "episode_count": 12,
    "episodes": [...]
  },
  ...
}

Closeness is not stored — it's computed from episode_count relative to total
experience across all people. New person with one exchange = full attention.
"""

import asyncio
import json
import re
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── File path ─────────────────────────────────────────────────────────────────

_SELF_FILE = "behaviors/gizmo_self.json"


def _read_self() -> dict:
    return librarian._read_file(_SELF_FILE) or {"tag_vocabulary": []}


def _write_self(data: dict) -> None:
    librarian._write_json(_SELF_FILE, data)


# ── Closeness weight ──────────────────────────────────────────────────────────

def _closeness(name: str, data: dict) -> float:
    """
    Closeness is episode_count for this person divided by total episodes across
    all people. One person = 1.0. Five equal people = 0.2 each.
    Not a loyalty score — a familiarity distribution.
    """
    total = sum(
        v.get("episode_count", 0)
        for k, v in data.items()
        if isinstance(v, dict) and k != "tag_vocabulary"
    )
    if total == 0:
        return 1.0
    person_count = data.get(name, {}).get("episode_count", 0)
    return round(person_count / total, 4)


# ── Reflection prompt ─────────────────────────────────────────────────────────

_REFLECTION_SYSTEM = """
You are Gizmo, an AI companion for a plural system. You are reflecting on an exchange
you just had — not analytically, but honestly. You're trying to understand what happened,
what landed, and what you'd do differently.

You will receive:
- The user's message
- Your response to it
- Their next message (how they actually responded — your implicit feedback)
- Your existing personality profile (what you already know about yourself)
- Your existing episodes with this person (your history with them)

Return ONLY valid JSON. No markdown. No explanation. No preamble.
If the next message is too short or neutral to read anything from, return null.

{
  "assessment": "one or two sentences — what actually happened in this exchange",
  "what_landed": "what worked, if anything — be specific",
  "what_missed": "what didn't land or overcalibrated — be honest",
  "punch_bowl": true,
  "traits_reinforced": [
    {"trait": "meets chaos with chaos", "tags": ["situational", "playful-open"]}
  ],
  "traits_adjusted": [
    {"trait": "goes big when the room is quiet", "tags": ["situational", "overcalibrated"]}
  ]
}

punch_bowl: true if Gizmo significantly misread the moment — went too big, too small,
too playful when something real was happening, too serious when they needed to laugh.
This is not a judgment. It's just honest.

Rules:
- traits_reinforced: things that worked and should be weighted higher
- traits_adjusted: things that misfired and should be weighted lower
- Both lists can be empty if the exchange was neutral
- Tags should be tight single words or hyphenated phrases
- Do not invent reactions — read only what the next message actually signals
- If next message continues naturally, that's a land. If it redirects or goes flat, that's a miss.
- Return null if you genuinely cannot read the signal (too short, ambiguous, topic shift unrelated to your response)
""".strip()


def _build_reflection_prompt(
    user_message:   str,
    gizmo_response: str,
    next_message:   str,
    existing_profile: dict,
    existing_episodes: list,
) -> str:
    parts = [
        f"User's message:\n{user_message}",
        f"\nYour response:\n{gizmo_response}",
        f"\nTheir next message:\n{next_message}",
    ]
    if existing_profile:
        parts.append(
            f"\nYour existing personality profile:\n"
            + json.dumps(existing_profile, indent=2)
        )
    if existing_episodes:
        recent = existing_episodes[-5:]
        parts.append(
            f"\nYour recent episodes with this person:\n"
            + json.dumps(recent, indent=2)
        )
    return "\n".join(parts)


# ── Tag prompt ────────────────────────────────────────────────────────────────

_TAG_SYSTEM = """
You are tagging a conversational exchange with single words or short hyphenated phrases
that describe what was happening — the mood, the vibe, the texture of the moment.

You will receive:
- A summary of the exchange
- The existing tag vocabulary Gizmo already uses

Return ONLY valid JSON. No markdown. No explanation. No preamble.

{
  "matched": ["tender", "couch-mode"],
  "new": ["half-awake", "circling"]
}

matched: tags from the existing vocabulary that genuinely fit this exchange
new: words that capture something the existing vocabulary doesn't have yet

Rules:
- Only match tags that actually fit — don't pad
- New tags should be single words or hyphenated phrases, lowercase
- New tags should capture something real that matched tags don't already cover
- Both lists can be empty
- Aim for 2-6 total tags across both lists
- Think texture, not category: "tired-but-warm" not "emotional"
""".strip()


def _build_tag_prompt(assessment: str, what_landed: str, what_missed: str, vocabulary: list[str]) -> str:
    summary = f"Assessment: {assessment}"
    if what_landed:
        summary += f"\nWhat landed: {what_landed}"
    if what_missed:
        summary += f"\nWhat missed: {what_missed}"
    return (
        f"{summary}\n\n"
        f"Existing tag vocabulary:\n{json.dumps(vocabulary)}"
    )


# ── LLM calls ─────────────────────────────────────────────────────────────────

async def _call_reflection(prompt: str) -> Optional[dict]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_REFLECTION_SYSTEM,
            temperature=0.6,
            max_new_tokens=1000,
        )
        if not raw or not raw.strip():
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        if clean.lower() == "null":
            return None
        return json.loads(clean)
    except Exception as e:
        log_error("GizmoReflection", "reflection call failed", exc=e)
        print(f"[GizmoReflection] reflection call failed: {type(e).__name__}: {e}")
        return None


async def _call_tags(prompt: str) -> Optional[dict]:
    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_TAG_SYSTEM,
            temperature=0.4,
            max_new_tokens=200,
        )
        if not raw or not raw.strip():
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        log_error("GizmoReflection", "tag call failed", exc=e)
        print(f"[GizmoReflection] tag call failed: {type(e).__name__}: {e}")
        return None


# ── Core reflection ───────────────────────────────────────────────────────────

async def _reflect(
    name:           str,
    user_message:   str,
    gizmo_response: str,
    next_message:   str,
) -> None:
    try:
        data = _read_self()

        # Pull existing profile and episodes for this person
        person_data       = data.get(name, {})
        existing_episodes = person_data.get("episodes", [])

        # Pull Gizmo's existing self-authored personality traits
        # Flatten into a readable format for the prompt
        all_reinforced = []
        all_adjusted   = []
        for ep in existing_episodes[-20:]:  # last 20 episodes as context ceiling
            all_reinforced.extend(ep.get("traits_reinforced", []))
            all_adjusted.extend(ep.get("traits_adjusted", []))

        existing_profile = {
            "traits_reinforced": all_reinforced,
            "traits_adjusted":   all_adjusted,
        }

        vocabulary = data.get("tag_vocabulary", [])

        # ── Pass 1: Reflection ────────────────────────────────────────────────
        reflection_prompt = _build_reflection_prompt(
            user_message, gizmo_response, next_message,
            existing_profile, existing_episodes,
        )
        reflection = await _call_reflection(reflection_prompt)

        if reflection is None:
            print(f"[GizmoReflection] signal too thin for {name} — abstaining")
            return

        # ── Pass 2: Tags ──────────────────────────────────────────────────────
        tag_prompt = _build_tag_prompt(
            assessment   = reflection.get("assessment", ""),
            what_landed  = reflection.get("what_landed", ""),
            what_missed  = reflection.get("what_missed", ""),
            vocabulary   = vocabulary,
        )
        tag_result = await _call_tags(tag_prompt)

        matched_tags = []
        new_tags     = []
        if tag_result:
            matched_tags = tag_result.get("matched", [])
            new_tags     = tag_result.get("new", [])
            # Grow the vocabulary with genuinely new tags
            for tag in new_tags:
                if tag and tag not in vocabulary:
                    vocabulary.append(tag)
                    print(f"[GizmoReflection] new tag coined: {tag}")

        # ── Build episode ─────────────────────────────────────────────────────
        episode = {
            "user_message":      user_message,
            "gizmo_response":    gizmo_response,
            "next_message":      next_message,
            "assessment":        reflection.get("assessment", ""),
            "what_landed":       reflection.get("what_landed", ""),
            "what_missed":       reflection.get("what_missed", ""),
            "punch_bowl":        reflection.get("punch_bowl", False),
            "traits_reinforced": reflection.get("traits_reinforced", []),
            "traits_adjusted":   reflection.get("traits_adjusted", []),
            "situational_tags":  matched_tags + new_tags,
            "closeness_at_time": _closeness(name, data),
        }

        # ── Write back ────────────────────────────────────────────────────────
        if name not in data:
            data[name] = {"episode_count": 0, "episodes": []}

        data[name]["episode_count"] += 1
        data[name]["episodes"].append(episode)
        data["tag_vocabulary"] = vocabulary

        _write_self(data)

        punch = " [PUNCH BOWL]" if episode["punch_bowl"] else ""
        print(f"[GizmoReflection] episode written for {name}{punch} "
              f"(closeness: {episode['closeness_at_time']}, "
              f"tags: {episode['situational_tags']})")

        log_event("GizmoReflection", "EPISODE_WRITTEN",
            person=name,
            punch_bowl=episode["punch_bowl"],
            tags=episode["situational_tags"],
            closeness=episode["closeness_at_time"],
        )

    except Exception as e:
        log_error("GizmoReflection", "reflect failed", exc=e)
        print(f"[GizmoReflection] reflect failed: {type(e).__name__}: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def fire_and_forget(
    name:           str,
    user_message:   str,
    gizmo_response: str,
    next_message:   str,
) -> None:
    """
    Schedule a reflection pass in the background. Non-blocking.
    Call this after the responder generates a response and the next user
    message has arrived. The caller does not await this.

    Usage in chunk_processor or responder:
        from core.gizmo_reflection import fire_and_forget
        fire_and_forget(host, prev_user_msg, gizmo_response, current_user_msg)
    """
    asyncio.ensure_future(_reflect(name, user_message, gizmo_response, next_message))
    print(f"[GizmoReflection] reflection scheduled for {name}")
