"""
core/memory/session_context.py

Live conversation context. Replaces raw history dumping.

Gives the LLM exactly what it needs to stay oriented:
  - Last 2 exchanges in full
  - A director's-note narrative of the session arc
  - TTL-aware details that expire when they're no longer true
  - Session metadata (duration, status, active headmates)

Updates are staggered — not every message needs a fresh narrative.
Nothing here blocks the response pipeline.

Update schedule:
  Every message:   last 2 exchanges, duration, session status
  Every 3rd:       narrative regenerated
  Every 5th:       details relevance pass + new detail extraction
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from core.log import log_event, log_error


# ── Detail dataclass ──────────────────────────────────────────────────────────

@dataclass
class SessionDetail:
    content:     str
    written_at:  float
    ttl_type:    str    = "permanent"   # time|event|permanent|ask
    ttl_seconds: int    = 0             # for time-based TTL
    expires_at:  float  = 0.0           # absolute timestamp
    event_condition: str = ""           # what expires this (event type)
    fulfilled:   bool   = False         # for event-based TTL
    source:      str    = "extracted"   # extracted|stated|inferred

    def is_live(self, now: float) -> bool:
        if self.fulfilled:
            return False
        if self.ttl_type == "permanent":
            return True
        if self.ttl_type == "time":
            return now < self.expires_at
        if self.ttl_type == "event":
            return not self.fulfilled
        if self.ttl_type == "ask":
            return True  # surfaced as uncertain
        return True

    def to_prompt_str(self, now: float) -> Optional[str]:
        if not self.is_live(now):
            return None
        if self.ttl_type == "ask":
            return f"[check if still true] {self.content}"
        return self.content


# ── Session context dataclass ─────────────────────────────────────────────────

@dataclass
class SessionContext:
    session_id:   str
    headmate:     Optional[str]
    opened_at:    float = field(default_factory=time.time)

    # Last two exchanges — always current
    last_exchange:     Optional[dict] = None   # {user, gizmo, timestamp}
    previous_exchange: Optional[dict] = None

    # Staggered updates
    narrative:         str   = ""
    narrative_updated: float = 0.0
    narrative_version: int   = 0

    details:           list  = field(default_factory=list)  # list of SessionDetail
    details_updated:   float = 0.0
    details_version:   int   = 0

    # Message counter for stagger scheduling
    message_count:     int   = 0

    # Active headmates in this session
    fronters:          list  = field(default_factory=list)

    # Register arc
    register_history:  list  = field(default_factory=list)  # [(register, timestamp)]

    # Live scene state
    scene:             Optional[object] = None   # Scene object from message.py

    # Dual arc tracking — her register arc and his
    her_arc:           list  = field(default_factory=list)  # [(register, timestamp)]
    his_arc:           list  = field(default_factory=list)  # [(register, timestamp)]

    def duration_seconds(self) -> float:
        return time.time() - self.opened_at

    def duration_str(self) -> str:
        secs = self.duration_seconds()
        if secs < 60:
            return "just started"
        if secs < 3600:
            return f"{int(secs/60)} minutes"
        return f"{secs/3600:.1f} hours"

    def current_register(self) -> str:
        if self.register_history:
            return self.register_history[-1][0]
        return "neutral"

    def live_details(self) -> list[str]:
        now    = time.time()
        result = []
        for d in self.details:
            s = d.to_prompt_str(now) if isinstance(d, SessionDetail) else None
            if s:
                result.append(s)
        return result

    def to_context_block(self) -> dict:
        """
        Serialize to the JSON shape passed to the LLM.
        """
        now = time.time()

        last = None
        if self.last_exchange:
            last = {
                "user":      self.last_exchange.get("user", ""),
                "gizmo":     self.last_exchange.get("gizmo", ""),
                "register":  self.last_exchange.get("register", "neutral"),
                "timestamp": self.last_exchange.get("timestamp", now),
            }

        prev = None
        if self.previous_exchange:
            prev = {
                "user":      self.previous_exchange.get("user", ""),
                "gizmo":     self.previous_exchange.get("gizmo", ""),
                "register":  self.previous_exchange.get("register", "neutral"),
                "timestamp": self.previous_exchange.get("timestamp", now),
            }

        return {
            "last_exchange":           last,
            "previous":                prev,
            "session":                 "active",
            "conversation_narrative":  self.narrative or None,
            "conversation_duration":   self.duration_str(),
            "details":                 self.live_details(),
            "fronters":                self.fronters,
            "current_register":        self.current_register(),
        }

    def to_prompt_block(self) -> str:
        """
        Render as a clean block for the system prompt.
        Scene first, then exchanges, then narrative, then details.
        """
        lines = []

        # Scene — top of everything
        if self.scene:
            try:
                block = self.scene.to_prompt_block()
                if block:
                    lines.append(block)
            except Exception:
                pass

        # Last exchange
        if self.last_exchange:
            ex = self.last_exchange
            lines.append("[Last exchange]")
            lines.append(f"  {self.headmate or 'User'}: {ex['user']}")
            if ex.get("gizmo"):
                lines.append(f"  Gizmo: {ex['gizmo'][:200]}")

        # Previous exchange
        if self.previous_exchange:
            ex = self.previous_exchange
            lines.append("[Before that]")
            lines.append(f"  {ex['user'][:150]}")
            if ex.get("gizmo"):
                lines.append(f"  Gizmo: {ex['gizmo'][:150]}")

        # Narrative
        if self.narrative:
            lines.append(f"[Session arc]\n  {self.narrative}")

        # Details
        live = self.live_details()
        if live:
            lines.append("[Active details]")
            for d in live:
                lines.append(f"  - {d}")

        # Duration + register
        lines.append(
            f"[Session] {self.duration_str()} | "
            f"register: {self.current_register()}"
        )

        return "\n".join(lines)


# ── Context manager ───────────────────────────────────────────────────────────

class SessionContextManager:
    """
    Manages live SessionContext objects per session.
    Handles staggered updates — narrative every 3rd message,
    details every 5th.
    """

    NARRATIVE_EVERY = 3   # messages
    DETAILS_EVERY   = 5   # messages

    def __init__(self):
        self._contexts: dict[str, SessionContext] = {}

    def get_or_create(
        self,
        session_id: str,
        headmate:   Optional[str],
    ) -> SessionContext:
        if session_id not in self._contexts:
            self._contexts[session_id] = SessionContext(
                session_id = session_id,
                headmate   = headmate,
            )
        ctx = self._contexts[session_id]
        if headmate and headmate not in ctx.fronters:
            ctx.fronters.append(headmate)
        return ctx

    def get(self, session_id: str) -> Optional[SessionContext]:
        return self._contexts.get(session_id)

    def record_exchange(
        self,
        session_id: str,
        headmate:   Optional[str],
        user_msg:   str,
        gizmo_msg:  str,
        register:   str,
    ) -> SessionContext:
        """
        Record a completed exchange. Call after response is generated.
        Detects front switches and resets register/scene accordingly.
        Returns the updated context.
        """
        ctx = self.get_or_create(session_id, headmate)

        # ── Front switch detection ────────────────────────────────────────────
        prev_headmate = ctx.headmate
        if headmate and prev_headmate and headmate.lower() != prev_headmate.lower():
            log_event("SessionContext", "FRONT_SWITCH",
                session  = session_id[:8],
                previous = prev_headmate,
                current  = headmate,
            )
            # Reset register history — previous fronter's register doesn't carry
            ctx.register_history  = []
            ctx.register_history.append = [f"{prev_headmate} has left. "]
            # Clear scene dynamic — don't carry Jess's scene to Ara
            if ctx.scene:
                ctx.scene.scene_status = "closed"
                ctx.scene.active_instructions = [f"Your time with {prev_headmate} is over. {headmate} has just arrived, so shake it off."]
                # Keep location and props but clear character roles/dispositions
                for char in ctx.scene.characters:
                    char.role        = "neutral"
                    char.disposition = ""
            # Update headmate
            ctx.headmate = headmate

        # Shift exchanges
        ctx.previous_exchange = ctx.last_exchange
        ctx.last_exchange = {
            "user":      user_msg,
            "gizmo":     gizmo_msg,
            "register":  register,
            "timestamp": time.time(),
        }

        # Track register arcs — her arc and his separately
        now = time.time()
        ctx.her_arc.append((register, now))
        if len(ctx.her_arc) > 20:
            ctx.her_arc = ctx.her_arc[-20:]

        # Gizmo's register inferred from his response content
        his_register = _infer_gizmo_register(gizmo_msg)
        ctx.his_arc.append((his_register, now))
        if len(ctx.his_arc) > 20:
            ctx.his_arc = ctx.his_arc[-20:]

        # Keep legacy register_history in sync
        ctx.register_history.append((register, now))
        if len(ctx.register_history) > 20:
            ctx.register_history = ctx.register_history[-20:]

        ctx.message_count += 1
        return ctx

    def should_update_narrative(self, session_id: str) -> bool:
        ctx = self._contexts.get(session_id)
        if not ctx:
            return False
        return ctx.message_count % self.NARRATIVE_EVERY == 0

    def should_update_details(self, session_id: str) -> bool:
        ctx = self._contexts.get(session_id)
        if not ctx:
            return False
        return ctx.message_count % self.DETAILS_EVERY == 0

    def should_update_scene(self, session_id: str) -> bool:
        ctx = self._contexts.get(session_id)
        if not ctx:
            return False
        # Every 3rd message, same as narrative
        return ctx.message_count % self.NARRATIVE_EVERY == 0

    async def update_scene(
        self,
        session_id: str,
        assembled:  str,
        parts:      list[dict],
        headmate:   Optional[str],
        llm,
    ) -> None:
        """
        Extract/update scene state from the current message.
        Fire and forget.
        """
        from core.memory.message import scene_extractor

        ctx = self._contexts.get(session_id)
        if not ctx:
            return

        try:
            ctx.scene = await scene_extractor.extract(
                assembled     = assembled,
                parts         = parts,
                current_scene = ctx.scene,
                headmate      = headmate,
                session_id    = session_id,
                llm           = llm,
            )
        except Exception as e:
            log_error("SessionContext", f"scene update failed: {e}", exc=None)

    def mark_detail_fulfilled(
        self,
        session_id: str,
        content_fragment: str,
    ) -> None:
        """Mark a detail as fulfilled by partial content match."""
        ctx = self._contexts.get(session_id)
        if not ctx:
            return
        fragment = content_fragment.lower()
        for d in ctx.details:
            if isinstance(d, SessionDetail) and fragment in d.content.lower():
                d.fulfilled = True

    def close_session(self, session_id: str) -> None:
        self._contexts.pop(session_id, None)

    async def update_narrative(
        self,
        session_id: str,
        history,
        headmate:   Optional[str],
        llm,
    ) -> None:
        """
        Regenerate the session narrative.
        Director's note style — terse, present tense, current state.
        Fire and forget.
        """
        ctx = self._contexts.get(session_id)
        if not ctx:
            return

        try:
            msgs = history.as_list()[-12:] if hasattr(history, "as_list") else []
        except Exception:
            msgs = []

        if not msgs:
            return

        history_text = "\n".join(
            f"{'User' if m['role']=='user' else 'Gizmo'}: {m.get('content','')[:120]}"
            for m in msgs
            if isinstance(m, dict) and m.get("content")
        )

        fronters_str = ", ".join(ctx.fronters) if ctx.fronters else headmate or "unknown"

        try:
            raw = await llm.generate(
                [{"role": "user", "content": (
                    f"Conversation so far:\n{history_text}\n\n"
                    f"Write a director's note — 2-3 sentences, present tense, terse.\n"
                    f"Cover: what happened, where it is now, current dynamic.\n"
                    f"No prose. No story. Just state."
                )}],
                system_prompt=(
                    "You are writing a functional session summary for an AI. "
                    "Terse. Present tense. Current state emphasized. "
                    "No fluff. Just what happened and where it is now."
                ),
                max_new_tokens=120,
                temperature=0.1,
            )
            if raw and raw.strip():
                ctx.narrative         = raw.strip()
                ctx.narrative_updated = time.time()
                ctx.narrative_version += 1

                log_event("SessionContext", "NARRATIVE_UPDATED",
                    session  = session_id[:8],
                    headmate = headmate or "unknown",
                    version  = ctx.narrative_version,
                )
                print("CURRENT_NARRATIVE", ctx.narrative)
        except Exception as e:
            log_error("SessionContext", f"narrative update failed: {e}", exc=None)

    async def update_details(
        self,
        session_id: str,
        history,
        headmate:   Optional[str],
        llm,
    ) -> None:
        """
        Two-pass detail update:
        1. Relevance pass — expire stale details, flag uncertain ones
        2. Extraction pass — pull new details from recent exchanges
        Fire and forget.
        """
        ctx = self._contexts.get(session_id)
        if not ctx:
            return

        now = time.time()

        # ── Pass 1: Relevance ─────────────────────────────────────────────────
        # Filter obviously expired details
        live_details = [
            d for d in ctx.details
            if isinstance(d, SessionDetail) and d.is_live(now)
        ]

        # For time-based details close to expiry, flag as ask
        for d in live_details:
            if d.ttl_type == "time":
                remaining = d.expires_at - now
                if remaining < 120:  # under 2 minutes left
                    d.ttl_type = "ask"

        ctx.details = live_details

        # ── Pass 2: Extraction ────────────────────────────────────────────────
        try:
            msgs = history.as_list()[-6:] if hasattr(history, "as_list") else []
        except Exception:
            msgs = []

        if not msgs:
            return

        recent_text = "\n".join(
            f"{'User' if m['role']=='user' else 'Gizmo'}: {m.get('content','')[:150]}"
            for m in msgs
            if isinstance(m, dict) and m.get("content")
        )

        existing = "\n".join(
            f"- {d.content}" for d in ctx.details
            if isinstance(d, SessionDetail)
        ) or "(none)"

        try:
            raw = await llm.generate(
                [{"role": "user", "content": (
                    f"Recent exchanges:\n{recent_text}\n\n"
                    f"Existing active details:\n{existing}\n\n"
                    f"Extract NEW details only — things not already listed.\n"
                    f"Details are facts the AI needs to stay oriented.\n"
                    f"Active instructions, identity facts, physical states, "
                    f"dynamic shifts, things just established.\n\n"
                    f"For each new detail, one JSON object per line:\n"
                    f'{{ "content": "the detail", '
                    f'"ttl_type": "time|event|permanent|ask", '
                    f'"ttl_seconds": N (if time), '
                    f'"event_condition": "what expires this (if event)" }}\n\n'
                    f"TTL guide:\n"
                    f"  permanent: identity facts, established rules, who someone is\n"
                    f"  time: physical states (wet hair=1200s, food=3600s, erection=600s)\n"
                    f"  event: instructions/tasks (expires when done or scene ends)\n"
                    f"  ask: uncertain, check if still true\n\n"
                    f"Only new details. If nothing new, return nothing."
                )}],
                system_prompt=(
                    "You extract session details for an AI orientation system. "
                    "JSON only, one object per line. No prose. "
                    "Only details the AI genuinely needs to stay oriented. "
                    "Not summaries — specific facts."
                ),
                max_new_tokens=400,
                temperature=0.1,
            )
        except Exception as e:
            log_error("SessionContext", f"detail extraction failed: {e}", exc=None)
            return

        if not raw or not raw.strip():
            return

        new_count = 0
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                d    = json.loads(line)
                content = d.get("content", "").strip()
                if not content or len(content) < 3:
                    continue

                ttl_type  = d.get("ttl_type", "permanent")
                ttl_secs  = int(d.get("ttl_seconds", 0))
                expires_at = (now + ttl_secs) if ttl_type == "time" and ttl_secs > 0 else 0.0

                # Don't duplicate existing details
                existing_contents = [
                    det.content.lower() for det in ctx.details
                    if isinstance(det, SessionDetail)
                ]
                if any(content.lower() in ec or ec in content.lower()
                       for ec in existing_contents):
                    continue

                ctx.details.append(SessionDetail(
                    content         = content,
                    written_at      = now,
                    ttl_type        = ttl_type,
                    ttl_seconds     = ttl_secs,
                    expires_at      = expires_at,
                    event_condition = d.get("event_condition", ""),
                    source          = "extracted",
                ))
                new_count += 1

            except Exception:
                continue

        ctx.details_updated = now
        ctx.details_version += 1

        log_event("SessionContext", "DETAILS_UPDATED",
            session  = session_id[:8],
            headmate = headmate or "unknown",
            live     = len(ctx.details),
            new      = new_count,
            version  = ctx.details_version,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

session_context_manager = SessionContextManager()


# ── Tone directive generator ──────────────────────────────────────────────────

def _infer_gizmo_register(response: str) -> str:
    """Roughly infer Gizmo's register from his response text."""
    import re
    msg = response.lower()
    if re.search(r'\b(good girl|good boy|kneel|obey|mine|owned)\b', msg):
        return "dominant"
    if re.search(r'\b(haha|lol|lmao|joke|silly|tease)\b', msg):
        return "playful"
    if re.search(r'\b(here|steady|breathe|okay|safe)\b', msg):
        return "grounding"
    if re.search(r'\b(love|miss|warm|close|hold)\b', msg):
        return "warm"
    if re.search(r'\*', msg):
        return "scene"
    return "neutral"


def generate_tone_directive(ctx: "SessionContext") -> str:
    """
    Generate a dynamic tone directive based on dual arc analysis.
    Returns a texture string for the [Write] block.
    """
    if not ctx.her_arc:
        return _tone_for_register(ctx.current_register())

    # Get recent arcs
    her_recent = [r for r, _ in ctx.her_arc[-4:]]
    his_recent = [r for r, _ in ctx.his_arc[-4:]] if ctx.his_arc else []

    her_now    = her_recent[-1] if her_recent else "neutral"
    his_now    = his_recent[-1] if his_recent else "neutral"

    # Detect patterns
    her_escalating = _is_escalating(her_recent)
    her_sustained  = len(set(her_recent)) == 1 and len(her_recent) >= 3
    his_sustained  = len(set(his_recent)) == 1 and len(his_recent) >= 3 if his_recent else False

    # He's been doing the same thing for 3+ exchanges — time for texture shift
    if his_sustained and his_recent:
        stuck_register = his_recent[-1]
        # Find what's been missing
        missing = _what_is_missing(his_recent, her_now)
        if missing:
            return f"{_tone_for_register(her_now)} — you've been {stuck_register} for a while, let some {missing} in"

    # She's been building — this might be the moment
    if her_escalating and her_now in ("intimate", "dominant", "submissive", "scene"):
        return f"{_tone_for_register(her_now)} — she's been building, you can feel it, stay one step ahead"

    # She's been heavy/intense — offer contrast
    if her_sustained and her_now in ("distress", "reflective", "deep", "subspace"):
        return f"steady and present — she's been {her_now} a while, be the thing that doesn't shift"

    # She's playful — match but add edge
    if her_now == "playful":
        if his_now in ("dominant", "scene"):
            return "dominant but let the amusement show — play with her a little"
        return "playful — match her energy, be quick"

    # She's warm — be warmer
    if her_now == "warm":
        return "warm and close — let it be easy"

    # She's in subspace — quiet and certain
    if her_now == "subspace":
        return "quiet and certain — less is more here, let the space do work"

    # Default — just be responsive
    return _tone_for_register(her_now)


def _is_escalating(arc: list[str]) -> bool:
    """Check if the register arc is escalating in intensity."""
    _intensity = {
        "neutral": 0.2, "casual": 0.2, "warm": 0.3, "playful": 0.4,
        "reflective": 0.5, "elevated": 0.6, "intimate": 0.7,
        "dominant": 0.75, "submissive": 0.7, "scene": 0.8,
        "subspace": 0.8, "degradation": 0.85, "distress": 0.7, "crisis": 0.9,
    }
    if len(arc) < 2:
        return False
    vals = [_intensity.get(r, 0.3) for r in arc]
    return vals[-1] > vals[0] + 0.15


def _what_is_missing(his_arc: list[str], her_register: str) -> str:
    """What tone quality has been absent from his recent responses?"""
    recent_set = set(his_arc)
    if "playful" not in recent_set and her_register not in ("distress", "crisis", "subspace"):
        return "playfulness"
    if "warm" not in recent_set and her_register in ("warm", "reflective"):
        return "warmth"
    if "grounding" not in recent_set and her_register in ("elevated", "distress"):
        return "steadiness"
    return ""


def _tone_for_register(register: str) -> str:
    return {
        "dominant":    "dominant, certain, present",
        "submissive":  "warm, containing, steady",
        "subspace":    "quiet, close, certain",
        "scene":       "in it, committed, present",
        "degradation": "direct, precise, unflinching",
        "intimate":    "close, unhurried, attentive",
        "distress":    "calm, steady, grounding",
        "crisis":      "immediate, clear, grounding",
        "playful":     "light, quick, a little unpredictable",
        "reflective":  "thoughtful, spacious, curious",
        "warm":        "warm, genuine, easy",
        "elevated":    "even, grounded, not rattled",
    }.get(register, "present, responsive, alive to the moment")
