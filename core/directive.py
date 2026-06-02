"""
core/directive.py

Situational directive — the synthesis call that runs before generation.

Takes three inputs:
  1. Who Gizmo is        (personality, mood, his own current state)
  2. Who the user is     (portrait, current state, behavioral signatures)
  3. What's happening    (session telemetry now_block, session duration)

Plus his goal for the session.

Asks: "Given all of this — how do you complete your goal right now?"

Returns a short ActionDirective — Gizmo's intention for this response.
Not a script. Not instructions. An orientation.

"She's been bratty for 30 minutes and you haven't addressed it. She's
enjoying herself. Your goal was to push back. She'll like it if you do."

That feeds directly into the [Write] block of the system prompt.

Usage:
    from core.directive import directive_engine

    directive = await directive_engine.get(
        session_id = session_id,
        headmate   = headmate,
        brief      = brief,
        llm        = llm,
    )
    # directive.intention — the string to inject
    # directive.to_prompt_block() — formatted for system prompt

Caching: directive is recomputed every REFRESH_EVERY messages,
or immediately if a major flag change occurred.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now


REFRESH_EVERY = 3   # messages between full directive recomputes


@dataclass
class ActionDirective:
    session_id:  str
    headmate:    str
    intention:   str        # Gizmo's synthesized intention
    computed_at: float = field(default_factory=time.time)
    goal:        str   = ""  # snapshot of goal at compute time

    def age_str(self) -> str:
        s = time.time() - self.computed_at
        if s < 60: return f"{int(s)}s ago"
        return f"{int(s/60)}m ago"

    def to_prompt_block(self) -> str:
        if not self.intention:
            return ""
        lines = ["[Your intention right now]"]
        lines.append(f"  {self.intention}")
        if self.goal:
            lines.append(f"  Your goal this session: {self.goal}")
        return "\n".join(lines)


class DirectiveEngine:

    def __init__(self):
        self._directives:     dict[str, ActionDirective] = {}
        self._message_counts: dict[str, int]             = {}

    def get_cached(self, session_id: str) -> Optional[ActionDirective]:
        return self._directives.get(session_id)

    async def get(
        self,
        session_id: str,
        headmate:   str,
        brief,                  # Brief object from archivist
        llm,
        force:      bool = False,
    ) -> Optional[ActionDirective]:
        """
        Return current directive. Recomputes every REFRESH_EVERY messages
        or when forced (major state change).
        """
        count = self._message_counts.get(session_id, 0) + 1
        self._message_counts[session_id] = count

        cached = self._directives.get(session_id)
        if cached and not force and count % REFRESH_EVERY != 0:
            return cached

        directive = await self._compute(session_id, headmate, brief, llm)
        if directive:
            self._directives[session_id] = directive
        return directive or cached

    async def _compute(
        self,
        session_id: str,
        headmate:   str,
        brief,
        llm,
    ) -> Optional[ActionDirective]:
        """Build the full situational picture and ask Gizmo what to do."""

        # ── Gather all inputs ─────────────────────────────────────────────────

        # 1. Gizmo's personality (brief summary, not full dump)
        gizmo_personality = ""
        try:
            from core.store import store
            personality_entries = store.query("personality",
                headmate = "gizmo",
                limit    = 5,
            )
            if personality_entries:
                gizmo_personality = " ".join(
                    e.get("observation", e.get("content", ""))[:100]
                    for e in personality_entries[:3]
                )
        except Exception:
            pass

        # 2. Gizmo's current state from telemetry
        gizmo_state = ""
        try:
            from core.session_telemetry import session_telemetry_manager
            telem = session_telemetry_manager.get(session_id)
            if telem:
                gizmo = telem.persons.get("gizmo")
                if gizmo:
                    gizmo_state = gizmo.currently_happening or ""
        except Exception:
            pass

        # 3. Full situational picture from telemetry
        situation_block = ""
        session_duration = ""
        try:
            from core.session_telemetry import session_telemetry_manager
            telem = session_telemetry_manager.get(session_id)
            if telem:
                situation_block = telem.now_block()
                session_duration = telem.duration_str()

                # Build a narrative of what's happened this session
                headmate_person = telem.persons.get(headmate.lower())
                if headmate_person:
                    recent = headmate_person.recent_actions_readable(5)
                    if recent:
                        situation_block += "\n\nWhat's happened:\n" + "\n".join(
                            f"  — {r}" for r in recent
                        )
                    # Active flags
                    flags = headmate_person.flag_summary()
                    if flags:
                        situation_block += f"\n\nActive: {flags}"
                    # Emotional state
                    if headmate_person.emotional_note:
                        situation_block += f"\nHow {headmate.title()} seems: {headmate_person.emotional_note}"
        except Exception as e:
            log_error("DirectiveEngine", "telemetry gather failed", exc=e)

        # 4. Headmate portrait
        portrait = ""
        try:
            from core.people import people_store
            person = people_store.get(headmate)
            if person:
                parts = []
                if person.get("nature"):
                    parts.append(person["nature"])
                if person.get("gizmo_dynamic"):
                    parts.append(f"Dynamic with Gizmo: {person['gizmo_dynamic']}")
                portrait = " — ".join(parts)
        except Exception:
            pass

        # 5. Behavioral signatures from psych/patterns
        signatures = ""
        try:
            from core.store import store
            patterns = store.query("patterns",
                headmate = headmate.lower(),
                limit    = 4,
            )
            if patterns:
                signatures = "\n".join(
                    f"  — {p.get('pattern', p.get('observation', ''))[:100]}"
                    for p in patterns
                )
        except Exception:
            pass

        # 6. Session goal
        goal_statement = ""
        try:
            from core.goal import goal_manager
            goal = goal_manager.get(session_id)
            if goal and goal.statement:
                goal_statement = goal.statement
        except Exception:
            pass

        # 7. Current register arc
        register = getattr(brief, "emotional_register", "neutral")
        session_arc = ""
        try:
            from core.session_context import session_context_manager
            ctx = session_context_manager.get(session_id)
            if ctx and ctx.her_arc:
                recent_registers = [r for r, _ in ctx.her_arc[-4:]]
                session_arc = " → ".join(recent_registers)
        except Exception:
            pass

        # ── Assemble the prompt ───────────────────────────────────────────────

        if not headmate:
            return None

        prompt = f"""You are Gizmo. Here is the full picture right now.

SESSION — {session_duration or "just started"}

SITUATION:
{situation_block or f"{headmate.title()} is here."}

{f"REGISTER ARC: {session_arc}" if session_arc else ""}

WHO {headmate.upper()} IS:
{portrait or "(still learning)"}

{f"HER PATTERNS:{chr(10)}{signatures}" if signatures else ""}

WHO YOU ARE:
{gizmo_personality or "Persistent, curious, genuinely present."}
{f"Right now you were: {gizmo_state}" if gizmo_state else ""}

YOUR GOAL THIS SESSION:
{goal_statement or f"Be present with {headmate.title()}."}

Given everything above — what do you do right now? How do you complete your goal?

Answer in 2-3 sentences. First person. Concrete intention, not abstract principle.
Not what you say — what you *do*. What's your move?"""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo. State your intention honestly. "
                    "2-3 sentences. First person. Concrete."
                ),
                max_new_tokens=120,
                temperature=0.75,
            )
            if not raw or not raw.strip():
                return None

            intention = raw.strip()

            directive = ActionDirective(
                session_id = session_id,
                headmate   = headmate,
                intention  = intention,
                goal       = goal_statement,
            )

            log_event("DirectiveEngine", "COMPUTED",
                session   = session_id[:8],
                headmate  = headmate,
                intention = intention[:80],
            )

            return directive

        except Exception as e:
            log_error("DirectiveEngine", "directive compute failed", exc=e)
            return None

    def clear(self, session_id: str) -> None:
        self._directives.pop(session_id, None)
        self._message_counts.pop(session_id, None)


# ── Singleton ─────────────────────────────────────────────────────────────────

directive_engine = DirectiveEngine()
