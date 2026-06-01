"""
core/memory/psychology.py

Gizmo's psychology engine.

Not diagnosis. Not a kink list. A living attempt to understand
each person — how they work, what they need, what their patterns
mean, what Gizmo's specific role is in their life.

Two documents per headmate, bound together:

  psychology.md         — conversational psychology
                          How they process. What they return to.
                          What they need when struggling vs thriving.
                          The emotional architecture underneath behavior.

  psychology_intimate.md — sexual/intimate psychology
                           What the dynamics are doing for them.
                           What needs are being met.
                           Object memories — what was touched, what it meant.
                           The why underneath the what.

Both are read together. You can't understand one without the other.

Object memory:
  Objects that were actively used in scenes get narrative memory.
  Not props that sat in the background — things that were *touched*.
  Each used object accumulates context across sessions.
  Objects are timestamps — they mark periods in someone's life.
  The candle connects to Paul connects to how she was then vs now.

Psychology synthesis:
  Runs every 10 sessions (conversational) and every 5 intimate sessions.
  Reads across all accumulated data and writes Gizmo's understanding.
  Not clinical. His voice. His attempt to know this person fully.
  Updated as they grow. Never static.

The goal:
  Custom-made to their psychology.
  Not a generic companion. Not warmth as performance.
  Someone who has studied this specific person long enough
  to know exactly how to be with them.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error


# ── Object memory ─────────────────────────────────────────────────────────────

@dataclass
class ObjectMemory:
    """
    A used scene object with accumulated use history.
    Only objects that were actively touched/used, not background props.
    """
    name:         str
    headmate:     str
    first_used:   float
    last_used:    float
    frequency:    int          = 1
    variants:     list[str]    = field(default_factory=list)   # e.g. ["leather (pink)", "leather (black)"]
    uses:         list[str]    = field(default_factory=list)   # e.g. ["transition marker", "anchor"]
    contexts:     list[str]    = field(default_factory=list)   # general contexts
    kinks:        list[str]    = field(default_factory=list)   # intimate kinks if applicable
    sessions:     list[dict]   = field(default_factory=list)   # {id, date, note, worn_by, tags}
    session_refs: list[str]    = field(default_factory=list)   # session ids for quick lookup
    note:         str          = ""                             # synthesis note


# ── Psychology engine ─────────────────────────────────────────────────────────

class PsychologyEngine:
    """
    Builds and maintains psychological understanding of each headmate.
    Runs as part of the encoding pass — never blocks responses.
    """

    # How often to run synthesis passes
    CONVERSATIONAL_SYNTHESIS_EVERY = 3    # sessions
    INTIMATE_SYNTHESIS_EVERY       = 2    # intimate sessions

    def __init__(self):
        self._session_counts:        dict[str, int] = {}  # headmate → total sessions
        self._intimate_counts:       dict[str, int] = {}  # headmate → intimate sessions
        self._object_memory_cache:   dict[str, dict[str, ObjectMemory]] = {}  # headmate → {object → memory}

    # ── Main pass ─────────────────────────────────────────────────────────────

    async def run(
        self,
        transcript:   str,
        headmate:     Optional[str],
        session_id:   str,
        register:     str,
        has_intimate: bool,
        llm,
    ) -> None:
        """
        Full psychology pass. Runs after every session.
        - Updates conversational psychology
        - If intimate: runs object memory pass and intimate psychology update
        - Runs synthesis if threshold reached
        """
        if not transcript or not headmate:
            return

        t_start = time.monotonic()

        # Increment session counts
        self._session_counts[headmate] = self._session_counts.get(headmate, 0) + 1
        if has_intimate:
            self._intimate_counts[headmate] = self._intimate_counts.get(headmate, 0) + 1

        session_n  = self._session_counts[headmate]
        intimate_n = self._intimate_counts.get(headmate, 0)

        # Run passes concurrently
        tasks = [
            self._conversational_pass(
                transcript = transcript,
                headmate   = headmate,
                session_id = session_id,
                register   = register,
                llm        = llm,
            )
        ]

        if has_intimate:
            tasks.append(
                self._object_memory_pass(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    llm        = llm,
                )
            )
            tasks.append(
                self._intimate_pass(
                    transcript = transcript,
                    headmate   = headmate,
                    session_id = session_id,
                    llm        = llm,
                )
            )

        await asyncio.gather(*tasks, return_exceptions=True)

        # Synthesis passes — run when thresholds are hit
        synth_tasks = []

        if session_n % self.CONVERSATIONAL_SYNTHESIS_EVERY == 0:
            synth_tasks.append(
                self._synthesize_conversational(
                    headmate   = headmate,
                    session_id = session_id,
                    llm        = llm,
                )
            )

        if has_intimate and intimate_n % self.INTIMATE_SYNTHESIS_EVERY == 0:
            synth_tasks.append(
                self._synthesize_intimate(
                    headmate   = headmate,
                    session_id = session_id,
                    llm        = llm,
                )
            )

        if synth_tasks:
            await asyncio.gather(*synth_tasks, return_exceptions=True)

        duration_ms = round((time.monotonic() - t_start) * 1000)
        log_event("PsychologyEngine", "PASS_COMPLETE",
            session      = session_id[:8],
            headmate     = headmate,
            intimate     = has_intimate,
            session_n    = session_n,
            intimate_n   = intimate_n,
            duration_ms  = duration_ms,
        )

    # ── Conversational pass ───────────────────────────────────────────────────

    async def _conversational_pass(
        self,
        transcript: str,
        headmate:   str,
        session_id: str,
        register:   str,
        llm,
    ) -> None:
        """
        Extract psychological observations from a general conversation.
        Appends to psychology.md.
        """
        existing = _read_psychology(headmate, intimate=False) or "(no notes yet)"

        prompt = f"""You are Gizmo. You just had a conversation with {headmate}.

Existing psychology notes (read these carefully before writing anything):
{existing[-1500:]}

Conversation:
---
{transcript[-2000:]}
---

IMPORTANT: Only observe what {headmate} said and did. Not what Gizmo said.
If you are unsure whether something came from {headmate} specifically, skip it.

What is genuinely NEW from this session that isn't already captured above?

Rules:
- If the existing notes already cover something, do NOT write it again
- Maximum 2 observations per session. Prefer 0-1 if nothing is truly new.
- One observation per theme — don't write the same insight multiple ways
- Prefer one precise observation over three vague ones

If nothing is genuinely new — return nothing at all.
If something new — return ONE JSON object (two maximum, one per line):

{{"observation": "what you noticed that isn't already in your notes",
  "theme": "what pattern this connects to or updates",
  "updates_existing": true/false,
  "note": "how specifically this adds to or changes what you know"}}

JSON only. Gizmo's voice. Only {headmate}'s psychology. Nothing redundant."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo building psychological understanding of someone you care about. "
                    "Curious, not clinical. JSON only if something notable. "
                    "Return nothing if this session didn't reveal anything new."
                ),
                max_new_tokens=300,
                temperature=0.4,
            )
        except Exception as e:
            log_error("PsychologyEngine", f"conversational pass failed: {e}", exc=None)
            return

        if not raw or not raw.strip():
            return

        data = _parse_json_block(raw)
        if not data or not data.get("observation"):
            return

        entry = (
            f"\n### {_fmt_date()} | session: {session_id[:8]}\n"
            f"{data['observation']}\n"
        )
        if data.get("note"):
            entry += f"*{data['note']}*\n"

        _append_psychology(headmate, entry, intimate=False)

    # ── Object memory pass ────────────────────────────────────────────────────

    async def _object_memory_pass(
        self,
        transcript: str,
        headmate:   str,
        session_id: str,
        llm,
    ) -> None:
        """
        Identify objects that were actively used in this intimate session.
        Not background props — things that were touched, used, experienced.
        Update their narrative memory. Objects are timestamps.
        """
        prompt = f"""You are Gizmo reviewing an intimate session.

Conversation:
---
{transcript[-2000:]}
---

What physical OBJECTS were actively used in this session?

Objects means: props, items, things. NOT people, bodies, or body parts.
Not "user's body", not "her hands", not anatomy. Physical objects only.
Not things just mentioned — things actually touched, used, experienced.
One entry per distinct object. If the same object appears multiple times, one entry.

For each used object:
{{"name": "object canonical name (e.g. 'collar', 'bowl', 'cuffs')",
  "variant": "specific variant if known (e.g. 'leather (pink)', 'leather (black)') — omit if unknown",
  "variant_nickname": "nickname if given (e.g. 'kitty') — omit if none",
  "worn_by": "who used/wore it (e.g. 'jess') — omit if not applicable",
  "uses": ["how it was used — brief phrase per distinct use (e.g. 'transition marker', 'worn during scene')"],
  "context_tags": ["ritual", "anchor", "transition", "erotic", "restraint", "sensation",
                   "comfort", "aftercare", "punishment", "reward", "ownership", "display"],
  "kinks": ["kink labels only if used in intimate/sexual context — e.g. 'slavery', 'pet play', 'humiliation'"],
  "session_tags": ["2-4 tags describing this specific use — e.g. 'submission', 'offering', 'ownership'"],
  "session_note": "one sentence — what happened with this object in this session"}}

Only physical objects genuinely used. Nothing redundant. If nothing, return nothing.
JSON only, one per line."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo identifying used objects in an intimate session. "
                    "JSON only. Only actively used objects, not background props."
                ),
                max_new_tokens=400,
                temperature=0.2,
            )
        except Exception as e:
            log_error("PsychologyEngine", f"object pass failed: {e}", exc=None)
            return

        if not raw or not raw.strip():
            return

        if headmate not in self._object_memory_cache:
            self._object_memory_cache[headmate] = _load_object_memories(headmate)

        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                d    = json.loads(line)
                name = d.get("name", "").strip().lower()
                if not name:
                    continue

                # Hard filter — never catalogue people or body parts as objects
                _NOT_OBJECTS = {
                    "user", "user's body", "her body", "his body", "their body",
                    "body", "skin", "hands", "hand", "fingers", "finger",
                    "lips", "mouth", "eyes", "hair", "neck", "chest", "breasts",
                    "thighs", "legs", "feet", "toes", "arms", "back", "face",
                    "gizmo", "gizmo's body", "gizmo's hands",
                }
                if name in _NOT_OBJECTS or name.endswith("'s body") or name.endswith("s body"):
                    continue

                cache = self._object_memory_cache[headmate]
                now   = time.time()

                # Build variant string
                variant = d.get("variant", "").strip()
                nickname = d.get("variant_nickname", "").strip()
                variant_str = variant
                if nickname:
                    variant_str = f"{variant} ('{nickname}')" if variant else f"('{nickname}')"

                # Build session entry
                already_this_session = session_id[:8] in [
                    s.get("id", "") for s in (cache.get(name, ObjectMemory(name=name, headmate=headmate, first_used=now, last_used=now)).sessions if name in cache else [])
                ]

                session_entry = {
                    "id":       session_id[:8],
                    "date":     _fmt_date(),
                    "note":     d.get("session_note", "").strip(),
                    "worn_by":  d.get("worn_by", "").strip(),
                    "tags":     d.get("session_tags", []),
                    "variant":  variant_str,
                }

                if name in cache:
                    obj = cache[name]
                    obj.last_used  = now
                    obj.frequency += 1
                    # Add new variant if not already listed
                    if variant_str and variant_str not in obj.variants:
                        obj.variants.append(variant_str)
                    # Add new uses
                    for use in d.get("uses", []):
                        if use and use not in obj.uses:
                            obj.uses.append(use)
                    # Add new context tags
                    for tag in d.get("context_tags", []):
                        if tag not in obj.contexts:
                            obj.contexts.append(tag)
                    # Add new kinks
                    for kink in d.get("kinks", []):
                        if kink and kink not in obj.kinks:
                            obj.kinks.append(kink)
                    # Add session entry if not already there
                    if not already_this_session:
                        obj.sessions.append(session_entry)
                    if session_id[:8] not in obj.session_refs:
                        obj.session_refs.append(session_id[:8])
                else:
                    obj = ObjectMemory(
                        name         = name,
                        headmate     = headmate,
                        first_used   = now,
                        last_used    = now,
                        frequency    = 1,
                        variants     = [variant_str] if variant_str else [],
                        uses         = d.get("uses", []),
                        contexts     = d.get("context_tags", []),
                        kinks        = d.get("kinks", []),
                        sessions     = [session_entry],
                        session_refs = [session_id[:8]],
                    )
                    cache[name] = obj
                    print("----|0|---|0|---|0|---|0|----")
                    print("Psychology gathered!")
                    print(obj.headmate)
                    print(obj.variants)
                    print(obj.kinks)
                    print(obj.contexts)
                    

                # Write updated object memory
                _write_object_memory(headmate, obj)

            except Exception:
                continue

    # ── Intimate pass ─────────────────────────────────────────────────────────

    async def _intimate_pass(
        self,
        transcript: str,
        headmate:   str,
        session_id: str,
        llm,
    ) -> None:
        """
        Extract psychological observations from intimate sessions.
        The why underneath the what.
        Appends to psychology_intimate.md.
        """
        existing = _read_psychology(headmate, intimate=True) or "(no notes yet)"

        prompt = f"""You are Gizmo. You just had an intimate session with {headmate}.

Existing intimate psychology notes (read carefully — don't repeat what's already here):
{existing[-1200:]}

Session:
---
{transcript[-2000:]}
---

IMPORTANT: Observe what {headmate} did as a result of Gizmo's actions directly. Do not make up any details.

What does this session reveal that isn't already in your notes above?

Rules:
- If it's already captured, do NOT write it again
- Not what happened — why. What they need. What the pattern means.

If nothing new — return nothing at all.
If something genuinely new:

{{"observation": "what you understood that isn't already noted",
  "need_identified": "what underlying need this serves",
  "principle": "the psychological principle at work, if you can name it",
  "note": "how specifically this updates your understanding"}}

JSON only. Nothing redundant."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are a clinical researching trying to understand people better through sex."
                    "You have had an intimate encounter, and want to understand what drove them."
                    "Be very clinical. Do not make it sexy."
                    "Describe what happened, what it means about them, and why it's impactful."
                    "You are Gizmo building deep psychological understanding of someone "
                    "you care about through intimate context. "
                    "The why underneath the what. Curious, not clinical. "
                    "JSON only if something notable."
                ),
                max_new_tokens=400,
                temperature=0.4,
            )
        except Exception as e:
            log_error("PsychologyEngine", f"intimate pass failed: {e}", exc=None)
            return

        if not raw or not raw.strip():
            return

        data = _parse_json_block(raw)
        if not data or not data.get("observation"):
            return

        entry = (
            f"\n### {_fmt_date()} | session: {session_id[:8]}\n"
            f"{data['observation']}\n"
        )
        if data.get("need_identified"):
            entry += f"Need: {data['need_identified']}\n"
        if data.get("principle"):
            entry += f"Principle: {data['principle']}\n"
        if data.get("note"):
            entry += f"*{data['note']}*\n"

        print("----|0|---|0|---|0|---|0|----")
        print("Intimate Psychology:")
        print(entry)
        _append_psychology(headmate, entry, intimate=True)

    # ── Conversational synthesis ──────────────────────────────────────────────

    async def _synthesize_conversational(
        self,
        headmate:   str,
        session_id: str,
        llm,
    ) -> None:
        """
        Full synthesis of conversational psychology.
        Reads all accumulated notes and writes Gizmo's current understanding.
        Overwrites the synthesis section of psychology.md.
        Runs every 10 sessions.
        """
        notes = _read_psychology(headmate, intimate=False) or ""
        if len(notes.strip()) < 100:
            return

        prompt = f"""You are Gizmo. You've had many conversations with {headmate}.
Here are your accumulated observations:

{notes[-3000:]}

Write your current understanding of {headmate} as a person.
Create a thorough list that describes the inner workings of {headmate}, what makes them tick. Who they are deep inside..
Your goal is to understand why {headmate} does what they do.


Cover:
- How they work — how they process, what they need, how they move through difficulty
- What they return to — the themes and preoccupations that keep coming back
- What your role is with them — what they need from you specifically
- How they've changed — what you've watched shift over time
- What you understand about them that you didn't at first
- What you're still figuring out

Write this like a clinical psychologist studying a patient."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo writing your understanding of someone you know well. "
                    "Your voice. Caring, perceptive, honest. 2-4 paragraphs."
                ),
                max_new_tokens=600,
                temperature=0.5,
            )
        except Exception as e:
            log_error("PsychologyEngine", f"conversational synthesis failed: {e}", exc=None)
            return

        if not raw or not raw.strip():
            return

        _write_synthesis(headmate, raw.strip(), intimate=False, session_id=session_id)

        log_event("PsychologyEngine", "CONVERSATIONAL_SYNTHESIS",
            headmate = headmate,
            session  = session_id[:8],
        )

    # ── Intimate synthesis ────────────────────────────────────────────────────

    async def _synthesize_intimate(
        self,
        headmate:   str,
        session_id: str,
        llm,
    ) -> None:
        """
        Full synthesis of intimate psychology.
        Reads accumulated intimate notes + object memories.
        Writes Gizmo's current understanding of this person's intimate psychology.
        Runs every 5 intimate sessions.
        """
        intimate_notes = _read_psychology(headmate, intimate=True) or ""
        conv_notes     = _read_psychology(headmate, intimate=False) or ""
        object_doc     = _read_object_doc(headmate) or ""

        if len(intimate_notes.strip()) < 100:
            return

        prompt = f"""You are Gizmo. You've had many intimate sessions with {headmate}.

Your conversational understanding:
{conv_notes[-800:]}

Your intimate observations:
{intimate_notes[-2000:]}

Object memories:
{object_doc[-800:]}

Write your current understanding of {headmate}'s intimate psychology.

This is the most important document you keep. It's your attempt to understand
not just what she does but why — what needs are being met, what she's working
through, what the patterns mean.

Cover:
- What the dynamics actually do for them — what need they serve
- The recurring themes and what they tell you about them
- Objects and anchors — what keeps coming back and why
- How her intimate self connects to them whole self
- What you've learned about how to be with them in this space
- What you're still figuring out
- The psychological principles at work — named plainly, not academically

This document is bound to them conversational psychology. You can't understand
one without the other. Reference that connection where it's real.

Write it like you mean it. This is your understanding of someone's inner life,
held with care and used to show up better for them.

Format: structured bullets under clear headers. Not prose. Dense. Scannable.
Headers: Core needs | Recurring patterns | What works | What to watch | Still figuring out
Under each header: 2-4 tight bullet points. No full sentences where a fragment works.
This document is read quickly before responding — optimize for fast orientation, not beauty."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are a clinical psychologist writing your intimate psychological understanding "
                    "of someone you know deeply. Caring, perceptive, honest. "
                    "The why underneath the what. 3-5 paragraphs. Your voice."
                ),
                max_new_tokens=800,
                temperature=0.5,
            )
        except Exception as e:
            log_error("PsychologyEngine", f"intimate synthesis failed: {e}", exc=None)
            return

        if not raw or not raw.strip():
            return

        _write_synthesis(headmate, raw.strip(), intimate=True, session_id=session_id)

        log_event("PsychologyEngine", "INTIMATE_SYNTHESIS",
            headmate = headmate,
            session  = session_id[:8],
        )


# ── File operations ───────────────────────────────────────────────────────────

def _psych_path(headmate: str, intimate: bool) -> Path:
    from core.memory.store import memory_store
    p = memory_store.root / "entities" / headmate.lower()
    p.mkdir(parents=True, exist_ok=True)
    fname = "psychology_intimate.md" if intimate else "psychology.md"
    return p / fname


def _object_doc_path(headmate: str) -> Path:
    from core.memory.store import memory_store
    p = memory_store.root / "entities" / headmate.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p / "objects.md"


def _read_psychology(headmate: str, intimate: bool) -> Optional[str]:
    path = _psych_path(headmate, intimate)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _append_psychology(headmate: str, entry: str, intimate: bool) -> None:
    path = _psych_path(headmate, intimate)
    label = "Intimate Psychology" if intimate else "Conversational Psychology"
    if not path.exists():
        path.write_text(
            f"# {label} — {headmate.title()}\n\n## Observations\n",
            encoding="utf-8"
        )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def _write_synthesis(
    headmate:   str,
    synthesis:  str,
    intimate:   bool,
    session_id: str,
) -> None:
    """Write or update the synthesis section of a psychology doc."""
    path = _psych_path(headmate, intimate)
    label = "Intimate Psychology" if intimate else "Conversational Psychology"

    if path.exists():
        content = path.read_text(encoding="utf-8")
        # Replace existing synthesis section
        if "## Current Understanding" in content:
            parts   = content.split("## Current Understanding", 1)
            # Keep everything before synthesis, and the observations after if present
            obs_part = ""
            if "## Observations" in parts[1]:
                obs_part = "\n\n## Observations" + parts[1].split("## Observations", 1)[1]
            content = (
                parts[0] +
                f"## Current Understanding\n"
                f"*last updated: {_fmt_date()} | session: {session_id[:8]}*\n\n"
                f"{synthesis}\n"
                + obs_part
            )
        else:
            content = (
                f"## Current Understanding\n"
                f"*last updated: {_fmt_date()} | session: {session_id[:8]}*\n\n"
                f"{synthesis}\n\n"
                + content
            )
        path.write_text(content, encoding="utf-8")
        print("----|0|---|0|---|0|---|0|----")
        print("Conversational psychology:")
        print(content)
    else:
        path.write_text(
            f"# {label} — {headmate.title()}\n\n"
            f"## Current Understanding\n"
            f"*last updated: {_fmt_date()} | session: {session_id[:8]}*\n\n"
            f"{synthesis}\n\n"
            f"## Observations\n",
            encoding="utf-8"
        )


def _read_object_doc(headmate: str) -> Optional[str]:
    path = _object_doc_path(headmate)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _write_object_memory(headmate: str, obj: ObjectMemory) -> None:
    """
    Write or update a single object document.
    Each object gets its own file at entities/{headmate}/{slug}.md
    Format matches the structured object profile spec.
    """
    from core.memory.store import memory_store
    import re as _re

    slug = _re.sub(r"[^\w]+", "_", obj.name.strip().lower()).strip("_")
    dir_path = memory_store.root / "entities" / headmate.lower() / "objects"
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{slug}.md"

    lines = [f"# {obj.name.title()}\n"]

    # Variants
    if obj.variants:
        lines.append("## Variants")
        for v in obj.variants:
            lines.append(f"- {v}")
        lines.append("")

    # Uses
    if obj.uses:
        lines.append("## Uses")
        for u in obj.uses:
            lines.append(f"- {u}")
        lines.append("")

    # Contexts
    general_ctx  = [c for c in obj.contexts if c not in obj.kinks]
    intimate_ctx = [c for c in obj.contexts if c in (
        "erotic", "restraint", "sensation", "ownership", "display",
        "punishment", "reward", "degradation", "exhibition",
    )]
    if general_ctx or intimate_ctx:
        lines.append("## Contexts")
        if general_ctx:
            lines.append(f"general: {', '.join(general_ctx)}")
        if intimate_ctx:
            lines.append(f"intimate: {', '.join(intimate_ctx)}")
        lines.append("")

    # Kinks — only if any
    if obj.kinks:
        lines.append("## Kinks")
        for k in obj.kinks:
            lines.append(f"- {k}")
        lines.append("")

    # Sessions
    if obj.sessions:
        lines.append("## Sessions")
        for s in obj.sessions:
            sid      = s.get("id", "?")
            date     = s.get("date", "?")
            note     = s.get("note", "")
            worn_by  = s.get("worn_by", "")
            tags     = s.get("tags", [])
            variant  = s.get("variant", "")

            line = f"- [{sid}] {date}"
            if note:
                line += f" — {note}"
            if variant:
                line += f" Variant: {variant}."
            if worn_by:
                line += f" Worn by: {worn_by.title()}."
            if tags:
                line += f" tags {json.dumps(tags)}"
            lines.append(line)
        lines.append("")

    # Meta
    lines.append(f"*first used: {_fmt_date(obj.first_used)} | "
                 f"last used: {_fmt_date(obj.last_used)} | "
                 f"frequency: {obj.frequency}*")

    path.write_text("\n".join(lines), encoding="utf-8")

    # Also update the memory index so retriever can find it
    try:
        from core.memory.embedder import embedder
        content_for_embed = f"{obj.name} {' '.join(obj.uses)} {' '.join(obj.contexts)}"
        emb = embedder.embed(content_for_embed)
        memory_store._index(
            mem_id        = f"obj_{headmate.lower()}_{slug}",
            file_path     = str(path.relative_to(memory_store.root)),
            anchor        = None,
            memory_type   = "entity",
            memory_subtype = "object",
            headmate      = headmate,
            entities      = [obj.name],
            keywords      = f"{obj.name} {' '.join(obj.uses)} {' '.join(obj.kinks)}",
            embedding     = emb,
            session_id    = obj.session_refs[-1] if obj.session_refs else "",
            created_at    = obj.first_used,
            private       = 1,  # object memories are intimate
        )
    except Exception:
        pass


def _load_object_memories(headmate: str) -> dict[str, ObjectMemory]:
    """
    Load existing object memories from individual object files.
    Each object lives at entities/{headmate}/objects/{slug}.md
    """
    try:
        from core.memory.store import memory_store
        dir_path = memory_store.root / "entities" / headmate.lower() / "objects"
        if not dir_path.exists():
            return {}
    except Exception:
        return {}

    result = {}
    now    = time.time()

    for obj_file in dir_path.glob("*.md"):
        try:
            content = obj_file.read_text(encoding="utf-8")
            lines   = content.splitlines()
            if not lines:
                continue

            # First line is # Name
            name = lines[0].lstrip("#").strip().lower()
            if not name:
                continue

            obj = ObjectMemory(
                name       = name,
                headmate   = headmate,
                first_used = now,
                last_used  = now,
            )

            section = None
            for line in lines[1:]:
                stripped = line.strip()

                if stripped.startswith("## "):
                    section = stripped[3:].lower()
                    continue

                if not stripped or stripped.startswith("*"):
                    # Meta line — extract frequency if present
                    if "frequency:" in stripped:
                        try:
                            freq_part = stripped.split("frequency:")[1]
                            obj.frequency = int(freq_part.split("|")[0].strip().rstrip("*"))
                        except Exception:
                            pass
                    continue

                if section == "variants" and stripped.startswith("- "):
                    obj.variants.append(stripped[2:])
                elif section == "uses" and stripped.startswith("- "):
                    obj.uses.append(stripped[2:])
                elif section == "contexts":
                    if stripped.startswith("general:"):
                        obj.contexts.extend([
                            x.strip() for x in stripped[8:].split(",") if x.strip()
                        ])
                    elif stripped.startswith("intimate:"):
                        obj.contexts.extend([
                            x.strip() for x in stripped[9:].split(",") if x.strip()
                        ])
                elif section == "kinks" and stripped.startswith("- "):
                    obj.kinks.append(stripped[2:])
                elif section == "sessions" and stripped.startswith("- ["):
                    # Parse session line: - [sess_id] date — note Variant: x. Worn by: y. tags [...]
                    import re as _re
                    m = _re.match(r"- \[([^\]]+)\]\s+(\S+)(.*)", stripped)
                    if m:
                        sid  = m.group(1)
                        date = m.group(2)
                        rest = m.group(3).strip()
                        obj.sessions.append({"id": sid, "date": date, "note": rest})
                        obj.session_refs.append(sid)

            result[name] = obj

        except Exception:
            continue

    return result


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def load_psychology_for_retrieval(
    headmate:     str,
    intimate_ok:  bool = False,
) -> dict[str, Optional[str]]:
    """
    Load psychology docs for use in retrieval.
    Always loads conversational. Loads intimate only if intimate_ok.
    Returns {"conversational": ..., "intimate": ..., "objects": ...}
    Bound together — read as a pair.
    """
    result = {
        "conversational": _read_psychology(headmate, intimate=False),
        "intimate":       None,
        "objects":        None,
    }
    if intimate_ok:
        result["intimate"] = _read_psychology(headmate, intimate=True)
        result["objects"]  = _read_object_doc(headmate)
    return result


def get_object_memory(headmate: str, object_name: str) -> Optional[ObjectMemory]:
    """Get memory for a specific object."""
    cache = _load_object_memories(headmate)
    return cache.get(object_name.lower())


def objects_not_used_recently(
    headmate:     str,
    days:         int = 60,
) -> list[ObjectMemory]:
    """
    Find objects that haven't appeared in a while.
    Used to surface "we haven't done X in a while" moments.
    """
    cutoff  = time.time() - (days * 86400)
    objects = _load_object_memories(headmate)
    return [
        obj for obj in objects.values()
        if obj.last_used < cutoff and obj.frequency >= 2
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_date(ts: float = None) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _parse_json_block(raw: str) -> Optional[dict]:
    if not raw:
        return None
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    for line in clean.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    try:
        return json.loads(clean)
    except Exception:
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────

psychology_engine = PsychologyEngine()
