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
            # Clear scene dynamic — don't carry Jess's scene to Ara
            if ctx.scene:
                ctx.scene.scene_status = "closed"
                ctx.scene.active_instructions = []
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

        # Track register arc
        ctx.register_history.append((register, time.time()))
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
                    "Example: 'Loona corrected a misidentification, established dominance. "
                    "Gizmo is now submissive. Millie just arrived, Loona stepped back. "
                    "Scene is paused.' "
                    "No fluff. Just what happened and where it is now."
                ),
                max_new_tokens=120,
                temperature=0.2,
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
