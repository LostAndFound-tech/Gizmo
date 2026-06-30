"""
core/gizmo_self_read.py

Pulls from Gizmo's own self-reflection log (headmates/gizmo/self.json) to inform
HOW he responds — not what he knows about the other person, but what he's learned
about his own calibration in moments like this.

Same tag → retrieve → interpret pattern as situational_read, but:
  - Source is Gizmo's own episode log per headmate, not their personality file
  - Retrieval prioritizes punch_bowl episodes (where a lesson was learned) over
    clean hits, since those are the corrections that matter most
  - Output is calibration guidance for Gizmo's own behavior, not a read on the
    other person's state

Usage:
    calibration = await gizmo_self_read.build(name="jess", message=user_message, dynamic=register)
"""

import json
import re
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Stage 1: Tagger ───────────────────────────────────────────────────────────

_TAG_SYSTEM = """
You extract relevant tags from a single conversational moment, for the purpose of
finding past instances where an AI companion either succeeded or miscalibrated in
a similar moment.

You will receive the current message, the dynamic/register, and the existing tag vocabulary.

Return ONLY valid JSON. No markdown. No explanation.

{
  "tags": ["tag1", "tag2", "tag3"],
  "new_tags": ["any tag not in the existing vocabulary that is genuinely needed"]
}

Rules:
- Prefer reusing existing vocabulary tags over inventing new ones
- Only propose a new tag if nothing in the existing vocabulary fits
- 3-6 tags total, tightest fit only
- Think about emotional register, intensity, and what KIND of moment this is
  (e.g. raw distress, playful tease, vulnerable disclosure, scene/roleplay, system switch)
""".strip()

async def _read_dynamic(message: str, vocabulary: list[str]) -> Optional[str]:
    try:
        from core.llm import llm
        dynamic_prompt = """
            Based on the users message, provide a one work descriptor that describes the dynamic between the user and Gizmo.

            Examples include 'friends', 'power-play', 'pet-play', 'colleagues', 'besties', 'GBF', 'catty'... Boil down to the
            simplest form of how to two interact.
        """.strip()
        dynamic = await llm.generate(
            messages = [{"role": "user", "content": message.strip()}],
            system_prompt=dynamic_prompt,
            temperature=0,
            max_new_tokens=100,
        )
        print(f"I think the dynamic is:\n{dynamic}")
        return dynamic.strip()
    except Exception as e:
        return e

async def _tag_moment(message: str, dynamic: str, vocabulary: list[str]) -> dict:
    try:
        from core.llm import llm
        prompt = (
            f"Dynamic/register: {dynamic}\n\n"
            f"Existing vocabulary:\n{json.dumps(vocabulary)}\n\n"
            f"Current message:\n{message.strip()}"
        )
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_TAG_SYSTEM,
            temperature=0.0,
            max_new_tokens=200,
        )
        if not raw or not raw.strip():
            return {"tags": [], "new_tags": []}
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(clean)
        result["tags"].append(dynamic)
        return {
            "tags":     result.get("tags", []),
            "new_tags": result.get("new_tags", []),
        }
    except Exception as e:
        log_error("GizmoSelfRead", "tag_moment failed", exc=e)
        return {"tags": [], "new_tags": []}


def _reconcile_vocabulary(new_tags: list[str]) -> None:
    """
    Extend the gizmo self-reflection vocabulary (separate namespace from the
    general librarian vocabulary, since these tags describe response-calibration
    patterns specifically — 'overcalibrated', 'missed-cue', etc).
    """
    if not new_tags:
        return
    self_data  = librarian.read_gizmo_self()
    vocabulary = self_data.get("tag_vocabulary", [])
    added = False
    for tag in new_tags:
        tag_clean = tag.lower().strip()
        if tag_clean and tag_clean not in vocabulary:
            vocabulary.append(tag_clean)
            added = True
    if added:
        self_data["tag_vocabulary"] = vocabulary
        librarian._write_json("headmates/gizmo/self.json", self_data)
        print(f"[GizmoSelfRead] vocabulary extended: {new_tags}")


# ── Stage 2: Retrieve ──────────────────────────────────────────────────────────

def _retrieve_episodes(name: str, tags: list[str], limit: int = 6) -> list[dict]:
    """
    Pull matching episodes from Gizmo's self-reflection log for this headmate.
    Prioritizes punch_bowl (lesson-learned) episodes over clean hits — those
    are the corrections most worth remembering.
    """
    self_data = librarian.read_gizmo_self()
    person    = self_data.get(name.lower(), {})
    episodes  = person.get("episodes", [])

    if not episodes or not tags:
        return []

    tag_set = set(t.lower() for t in tags)

    scored = []
    for ep in episodes:
        ep_tags = set(t.lower() for t in ep.get("situational_tags", []))
        overlap = len(ep_tags & tag_set)
        if overlap == 0:
            continue
        # Weight punch_bowl episodes higher — they're corrections, not confirmations
        score = overlap + (2 if ep.get("punch_bowl") else 0)
        scored.append((score, ep))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:limit]]


# ── Stage 3: Calibration read ──────────────────────────────────────────────────

_CALIBRATION_SYSTEM = """
You've spoken with this person in the past, and these are how you came across with them previously. They still care about
you, because you're friends, so you don't have to worry or be upset by anything you see here. 

Based on your previous reactions, how best can you respond to them in this moment? 

Return ONLY valid JSON. No markdown. No explanation.

{
  "pattern": "What works well, given how they are acting?",
  "guidance": "Include guidance framework",
  "confidence": 0.0 to 1.0,
  "avoid": "include things you ought to avoid"
}

If you think more than one response may be appropriate, include both, and a small statement as to why either would work.

Rules:
- Ground this entirely in the episodes provided — do not invent patterns not supported by evidence
- If multiple episodes show the same miscalibration, say so directly — repetition is the signal
- If episodes are mixed or inconclusive, lower the confidence score accordingly
- If no episodes are provided, return pattern as null and guidance as "No matching precedent — respond to what's in front of you."
""".strip()


async def _build_calibration(name: str, message: str, dynamic: str, episodes: list[dict]) -> Optional[dict]:
    if not episodes:
        return None

    # Trim episodes to the fields that matter for calibration — drop full response text
    trimmed = [
        {
            "user_message":  ep.get("user_message", "")[:150],
            "assessment":    ep.get("assessment", ""),
            "what_landed":   ep.get("what_landed", ""),
            "what_missed":   ep.get("what_missed", ""),
            "punch_bowl":    ep.get("punch_bowl", False),
            "traits_adjusted": [t.get("trait") for t in ep.get("traits_adjusted", [])],
            "situational_tags": ep.get("situational_tags", []),
        }
        for ep in episodes
    ]

    prompt = (
        f"Current moment with: {name}\n"
        f"Dynamic/register: {dynamic}\n\n"
        f"Current message:\n{message.strip()}\n\n"
        f"Relevant past episodes (most relevant first):\n"
        + json.dumps(trimmed, indent=2)
    )

    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_CALIBRATION_SYSTEM,
            temperature=0.2,
            max_new_tokens=300,
        )
        if not raw or not raw.strip():
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        log_error("GizmoSelfRead", "build_calibration failed", exc=e)
        print(f"[GizmoSelfRead] build_calibration failed: {type(e).__name__}: {e}")
        return None


def _calibration_to_prose(calibration: dict) -> str:
    if not calibration or not calibration.get("pattern"):
        return ""

    lines = [f"Pattern: {calibration['pattern']} (confidence {calibration.get('confidence', 0):.2f})"]

    guidance = calibration.get("guidance", "")
    if guidance:
        lines.append(f"Guidance: {guidance}")

    avoid = calibration.get("avoid")
    if avoid:
        lines.append(f"Avoid: {avoid}")

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

class GizmoSelfRead:

    async def build(self, name: str, message: str, dynamic: str = "neutral") -> str:
        """
        Full pipeline: tag the moment, retrieve matching self-reflection episodes
        (prioritizing punch_bowl lessons), build calibration guidance.
        Returns prose ready for the brief, or empty string if nothing relevant.
        """
        if not message.strip():
            return ""

        self_data = librarian.read_gizmo_self()
        vocabulary = self_data.get("tag_vocabulary", [])

        tag_result = await _tag_moment(message, dynamic, vocabulary)
        tags       = tag_result.get("tags", [])

        if tag_result.get("new_tags"):
            _reconcile_vocabulary(tag_result["new_tags"])

        if not tags:
            print(f"[GizmoSelfRead] no tags extracted, skipping")
            return ""
        dynamic = await _read_dynamic(message, vocabulary)
        print(f"GIZMO TAGS for dynamic {dynamic}:\n\n{tag_result}")
        episodes = _retrieve_episodes(name, tags)
        if not episodes:
            print(f"[GizmoSelfRead] no matching episodes for {name} on tags {tags}")
            return ""

        calibration = await _build_calibration(name, message, dynamic, episodes)
        if not calibration:
            return ""

        log_event("GizmoSelfRead", "BUILT",
            name=name,
            tags=tags,
            episodes_matched=len(episodes),
            punch_bowl_count=sum(1 for e in episodes if e.get("punch_bowl")),
        )

        return _calibration_to_prose(calibration)


gizmo_self_read = GizmoSelfRead()
