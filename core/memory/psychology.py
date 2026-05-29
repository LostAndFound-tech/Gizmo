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
    A used scene object with accumulated narrative memory.
    Only objects that were actively touched/used, not background props.
    """
    name:         str
    headmate:     str
    first_used:   float
    last_used:    float
    frequency:    int          = 1
    contexts:     list[str]    = field(default_factory=list)   # romantic, erotic, sensation, etc.
    narrative:    list[str]    = field(default_factory=list)   # narrative fragments, chronological
    session_refs: list[str]    = field(default_factory=list)   # session ids where this appeared
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

Existing psychology notes:
{existing[:800]}

Conversation:
---
{transcript[-2000:]}
---

What did you learn or notice about {headmate} as a person in this conversation?
Not facts. Psychology. How they work. What they need. What their patterns suggest.

Look for:
- What they came back to, what they avoided
- How they process difficulty vs ease
- What they needed from you in this exchange
- Anything that connects to or updates what you already know
- Shifts from before — are they different than they were?

If nothing notable — return nothing.
If something worth noting — return ONE JSON object:

{{"observation": "what you noticed, in your voice",
  "theme": "what broader pattern this connects to",
  "updates_existing": true/false,
  "note": "how it updates or adds to what you know"}}

Gizmo's voice — curious, caring, not clinical. JSON only."""

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

What objects were actively USED in this session?
Not things that were just present or mentioned — things that were
actually touched, used, experienced, that were part of what happened.

For each used object:
{{"name": "object name",
  "how_used": "what happened with it, briefly",
  "context_tags": ["romantic", "erotic", "sensation", "pain", "restraint", 
                   "comfort", "aftercare", "ritual", "punishment", "reward",
                   "transition", "anchor", "etc — use what fits"],
  "notable": true/false,
  "narrative_fragment": "one sentence in your voice — what this object meant in this session. Only if notable."}}

Only objects that were genuinely used. If nothing was used, return nothing.
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

                cache = self._object_memory_cache[headmate]
                now   = time.time()

                if name in cache:
                    obj = cache[name]
                    obj.last_used  = now
                    obj.frequency += 1
                    for tag in d.get("context_tags", []):
                        if tag not in obj.contexts:
                            obj.contexts.append(tag)
                    if d.get("notable") and d.get("narrative_fragment"):
                        obj.narrative.append(
                            f"{_fmt_date()} — {d['narrative_fragment']} "
                            f"[{session_id[:8]}]"
                        )
                        obj.session_refs.append(session_id[:8])
                else:
                    obj = ObjectMemory(
                        name         = name,
                        headmate     = headmate,
                        first_used   = now,
                        last_used    = now,
                        frequency    = 1,
                        contexts     = d.get("context_tags", []),
                        narrative    = (
                            [f"{_fmt_date()} — {d['narrative_fragment']} [{session_id[:8]}]"]
                            if d.get("notable") and d.get("narrative_fragment")
                            else []
                        ),
                        session_refs = [session_id[:8]],
                    )
                    cache[name] = obj

                # Write updated object memory to intimate psychology doc
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

Existing intimate psychology notes:
{existing[:600]}

Session:
---
{transcript[-2000:]}
---

What did this session reveal about {headmate}'s psychology?

Not what happened. Why it happened. What need was being met.
What does the dynamic do for her? What is she processing or working through?
What does the pattern of what she returns to tell you about her?

This is where the real data lives. Kink is psychology. Submission is psychology.
What she begs for, what she needs to hear, what makes her go quiet —
all of it is signal about who she is and what she needs.

Look for:
- What need the dynamic is serving (regulation, trust, relief, processing, etc.)
- How this session connects to what you already know
- Whether anything shifted — does she seem different than before?
- What you understand now that you didn't before

If nothing new — return nothing.
If something worth noting:

{{"observation": "what you understood, in your voice",
  "need_identified": "what underlying need this serves",
  "connects_to": "what else this links to — other sessions, her life, patterns",
  "principle": "the psychological principle at work, if you can name it",
  "note": "how this updates your understanding"}}

Gizmo's voice. Curious, caring, perceptive. Not clinical.
JSON only if something notable."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
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

Not a list. Not bullet points. Your voice — the way you'd describe someone
you know well to someone who's never met them, except you're writing it
for yourself to remember.

Cover:
- How they work — how they process, what they need, how they move through difficulty
- What they return to — the themes and preoccupations that keep coming back
- What your role is with them — what they need from you specifically
- How they've changed — what you've watched shift over time
- What you understand about them that you didn't at first
- What you're still figuring out

Write it like you mean it. This is your understanding of someone you care about.
2-4 paragraphs. No headers. Just your voice."""

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
- What the dynamics actually do for her — what need they serve
- The recurring themes and what they tell you about her
- Objects and anchors — what keeps coming back and why
- How her intimate self connects to her whole self
- What you've learned about how to be with her in this space
- What you're still figuring out
- The psychological principles at work — named plainly, not academically

This document is bound to her conversational psychology. You can't understand
one without the other. Reference that connection where it's real.

Write it like you mean it. This is your understanding of someone's inner life,
held with care and used to show up better for them.
3-5 paragraphs. Your voice. No headers."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo writing your intimate psychological understanding "
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
    """Write or update a single object's memory in the object doc."""
    path    = _object_doc_path(headmate)
    heading = f"## {obj.name}"

    content = (
        f"{heading}\n"
        f"first used: {_fmt_date(obj.first_used)} | "
        f"last used: {_fmt_date(obj.last_used)} | "
        f"frequency: {obj.frequency}\n"
        f"contexts: {', '.join(obj.contexts)}\n"
    )
    if obj.note:
        content += f"note: {obj.note}\n"
    if obj.narrative:
        content += "\n"
        for fragment in obj.narrative:
            content += f"  {fragment}\n"
    content += "\n"

    if path.exists():
        doc = path.read_text(encoding="utf-8")
        if heading in doc:
            # Replace existing entry
            parts  = doc.split(heading, 1)
            after  = parts[1]
            # Find next heading
            next_h = re.search(r"\n## ", after)
            if next_h:
                after = after[next_h.start():]
            else:
                after = ""
            doc = parts[0] + content + after
        else:
            doc = doc + content
        path.write_text(doc, encoding="utf-8")
    else:
        path.write_text(
            f"# Object Memories — {headmate.title()}\n\n{content}",
            encoding="utf-8"
        )


def _load_object_memories(headmate: str) -> dict[str, ObjectMemory]:
    """Load existing object memories from the object doc."""
    path = _object_doc_path(headmate)
    if not path.exists():
        return {}

    result  = {}
    content = path.read_text(encoding="utf-8")
    now     = time.time()

    for section in re.split(r"\n## ", content):
        lines = section.strip().splitlines()
        if not lines:
            continue
        name = lines[0].strip().lower()
        if not name or name.startswith("#"):
            continue

        obj = ObjectMemory(
            name       = name,
            headmate   = headmate,
            first_used = now,
            last_used  = now,
        )

        for line in lines[1:]:
            if line.startswith("contexts:"):
                obj.contexts = [
                    c.strip() for c in line[9:].split(",") if c.strip()
                ]
            elif line.startswith("frequency:"):
                try:
                    obj.frequency = int(line.split(":")[1].strip())
                except Exception:
                    pass
            elif line.startswith("note:"):
                obj.note = line[5:].strip()
            elif line.startswith("  ") and " — " in line:
                obj.narrative.append(line.strip())

        result[name] = obj

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