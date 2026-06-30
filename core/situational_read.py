"""
core/situational_read.py

The "actually use the data" layer.

Three stages, run per-person, per-message:
  1. TAG     — extract relevant tags from the current moment
  2. RETRIEVE — pull matched personality traits + episodes via librarian.get_by_tags
  3. READ    — interpret the matched evidence into a judgment call with confidence

This replaces flat "top-N trait" prose with an actual situational read grounded
in episodic history — competing interpretations, confidence-weighted, with a
recommendation on how Gizmo should proceed.

Usage:
    read = await situational_read.build(name="jess", message=user_message, dynamic=register)
    # read is a prose string ready to drop into the brief
"""

import json
import re
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Stage 1: Tagger ───────────────────────────────────────────────────────────

_TAG_SYSTEM = """
You extract relevant behavioral/relational tags from a single conversational moment.
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
- Tags should describe what's relevant to interpreting THIS moment, not the whole conversation
""".strip()


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
        return {
            "tags":     result.get("tags", []),
            "new_tags": result.get("new_tags", []),
        }
    except Exception as e:
        log_error("SituationalRead", "tag_moment failed", exc=e)
        return {"tags": [], "new_tags": []}


def _reconcile_vocabulary(new_tags: list[str]) -> None:
    """Extend the shared vocabulary file with genuinely new tags."""
    if not new_tags:
        return
    vocabulary = librarian.read_vocabulary() or []
    added = False
    for tag in new_tags:
        tag_clean = tag.lower().strip()
        if tag_clean and tag_clean not in vocabulary:
            vocabulary.append(tag_clean)
            added = True
    if added:
        librarian.write_vocabulary(vocabulary)
        print(f"[SituationalRead] vocabulary extended: {new_tags}")


# ── Stage 3: Read ──────────────────────────────────────────────────────────────

_READ_SYSTEM = """
You are reading a single conversational moment for an AI companion named Gizmo,
using matched episodic history and personality traits as your evidence.

Your job is NOT to describe who this person is in general. Your job is to interpret
THIS specific moment — what is most likely happening underneath it, grounded in how
this exact pattern has played out before.

When the evidence supports more than one plausible read, surface BOTH with confidence
scores rather than picking one. Confidence is your calibrated estimate of how likely
each read is, based on how consistently the episodic evidence supports it.

Return ONLY valid JSON. No markdown. No explanation.

{
  "reads": [
    {
      "interpretation": "what might be happening — e.g. 'deflecting with humor'",
      "confidence": 0.69,
      "evidence": "brief reference to which episode(s) support this read"
    },
    {
      "interpretation": "the competing interpretation — e.g. 'genuinely unbothered'",
      "confidence": 0.78,
      "evidence": "brief reference to which episode(s) support this read"
    }
  ],
  "recommendation": "one sentence — given the confidence levels, how Gizmo should proceed right now",
  "watch_for": "one sentence or null — a signal that would indicate the lower-confidence read is actually correct"
}

Rules:
- Confidence scores don't need to sum to 1 — these are independent estimates, not a distribution
- If there's only one clear read with no real ambiguity, return a single entry in reads with high confidence
- If there's no matched evidence at all, return reads as [] and recommendation as "No strong pattern read — respond to what's actually being said."
- Ground every read in the matched episodes provided, not general assumptions
- Do not invent episodes or traits not present in the evidence given
""".strip()


async def _build_read(
    name:        str,
    message:     str,
    dynamic:     str,
    matched:     dict,
) -> Optional[dict]:
    personality = matched.get("personality", {})
    episodes    = matched.get("episodes", [])

    if not personality and not episodes:
        return None

    prompt = (
        f"Person: {name}\n"
        f"Dynamic/register: {dynamic}\n\n"
        f"Current message:\n{message.strip()}\n\n"
        f"Matched personality traits:\n"
        + json.dumps(
            {t: {"weight": v.get("weight")} for t, v in personality.items()},
            indent=2
        )
        + f"\n\nMatched episodes (cause → action → reaction → trait):\n"
        + json.dumps(episodes, indent=2)
    )

    try:
        from core.llm import llm
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_READ_SYSTEM,
            temperature=0.2,
            max_new_tokens=400,
        )
        if not raw or not raw.strip():
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        log_error("SituationalRead", "build_read failed", exc=e)
        print(f"[SituationalRead] build_read failed: {type(e).__name__}: {e}")
        return None


def _read_to_prose(read: dict) -> str:
    """Convert the structured read into a compact prose block for the brief."""
    if not read or not read.get("reads"):
        return ""

    lines = []
    for r in read["reads"]:
        interp = r.get("interpretation", "")
        conf   = r.get("confidence", 0)
        lines.append(f"- {interp} (confidence {conf:.2f})")

    block = "\n".join(lines)

    rec = read.get("recommendation", "")
    if rec:
        block += f"\nRecommendation: {rec}"

    watch = read.get("watch_for")
    if watch:
        block += f"\nWatch for: {watch}"

    return block


# ── Public API ────────────────────────────────────────────────────────────────

class SituationalRead:

    async def build(self, name: str, message: str, dynamic: str = "neutral") -> str:
        """
        Full pipeline: tag the moment, retrieve matched evidence, build the read.
        Returns prose ready for the brief, or empty string if nothing to surface.
        """
        if not message.strip():
            return ""

        # Stage 1: Tag
        vocabulary = librarian.read_vocabulary() or []
        tag_result = await _tag_moment(message, dynamic, vocabulary)
        tags       = tag_result.get("tags", [])

        if tag_result.get("new_tags"):
            _reconcile_vocabulary(tag_result["new_tags"])

        if not tags:
            print(f"[SituationalRead] no tags extracted for {name}, skipping read")
            return ""

        # Stage 2: Retrieve
        matched = librarian.get_by_tags(name, tags)

        if not matched.get("personality") and not matched.get("episodes"):
            print(f"[SituationalRead] no matched evidence for {name} on tags {tags}")
            return ""

        # Stage 3: Read
        read = await _build_read(name, message, dynamic, matched)
        if not read:
            return ""

        log_event("SituationalRead", "BUILT",
            name=name,
            tags=tags,
            reads=len(read.get("reads", [])),
        )

        return _read_to_prose(read)


situational_read = SituationalRead()
