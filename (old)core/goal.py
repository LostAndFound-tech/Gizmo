"""
core/goal.py

Session goal system. Gizmo decides what he wants going into each session.

Not assigned. Not templated. Asked.

"Given who [headmate] is and how things have been between you lately,
what are you hoping to do with her today?"

He answers in first person. That answer is his goal for the session.
It can be revised mid-session if enough has shifted — either via the
consequence pass noticing a major state change, or via marker tool.

Goals are session-scoped and held in memory. They also write to the
store so the psych pass can track what Gizmo wanted vs. what happened.

Usage:
    from core.goal import goal_manager

    goal = await goal_manager.open_session(session_id, headmate, llm)
    # goal.statement — "I want to see if she'll drop the brat thing if I don't engage it"

    goal = goal_manager.get(session_id)

    await goal_manager.revise(session_id, headmate, reason, llm)

    await goal_manager.close_session(session_id, outcome_note)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now


@dataclass
class SessionGoal:
    session_id:   str
    headmate:     str
    statement:    str          # Gizmo's own words
    set_at:       float = field(default_factory=time.time)

    # Revision history
    revisions:    list  = field(default_factory=list)  # [(old, reason, timestamp)]

    # Outcome (filled on close)
    outcome_note: str  = ""
    closed_at:    float = 0.0

    def age_str(self) -> str:
        secs = time.time() - self.set_at
        if secs < 60:   return "just set"
        if secs < 3600: return f"set {int(secs/60)}m ago"
        return f"set {secs/3600:.1f}h ago"

    def to_prompt_str(self) -> str:
        lines = [f"[Your goal] {self.statement}"]
        if self.revisions:
            last = self.revisions[-1]
            lines.append(f"  (revised from: \"{last['old']}\" — {last['reason']})")
        return "\n".join(lines)


class GoalManager:

    def __init__(self):
        self._goals: dict[str, SessionGoal] = {}

    def get(self, session_id: str) -> Optional[SessionGoal]:
        return self._goals.get(session_id)

    # ── Session open ──────────────────────────────────────────────────────────

    async def open_session(
        self,
        session_id: str,
        headmate:   str,
        llm,
    ) -> SessionGoal:
        """
        Ask Gizmo what he wants today. Returns a SessionGoal.
        Fires async — session opens immediately, goal arrives shortly after.
        """
        # Placeholder so get() returns something immediately
        placeholder = SessionGoal(
            session_id = session_id,
            headmate   = headmate,
            statement  = "",
        )
        self._goals[session_id] = placeholder

        # Fire the actual goal-setting async
        import asyncio
        asyncio.create_task(self._set_goal(placeholder, headmate, llm))

        return placeholder

    async def _set_goal(
        self,
        goal:     SessionGoal,
        headmate: str,
        llm,
    ) -> None:
        """Ask Gizmo what he wants today, based on real data."""

        # Pull headmate portrait
        portrait = ""
        try:
            from core.people import people_store
            person = people_store.get(headmate)
            if person:
                parts = []
                if person.get("nature"):
                    parts.append(f"Nature: {person['nature']}")
                if person.get("gizmo_dynamic"):
                    parts.append(f"Dynamic: {person['gizmo_dynamic']}")
                if person.get("gizmo_needs"):
                    parts.append(f"What she needs: {person['gizmo_needs']}")
                portrait = "\n".join(parts)
        except Exception as e:
            log_error("GoalManager", "portrait load failed", exc=e)

        # Pull recent interaction history
        recent_sessions = ""
        try:
            from core.store import store
            recent = store.query("wellbeing",
                headmate  = headmate.lower(),
                category  = "session_close",
                limit     = 3,
            )
            if recent:
                recent_sessions = "\n".join(
                    f"- {r.get('observation', '')[:120]}"
                    for r in recent
                )
        except Exception as e:
            log_error("GoalManager", "recent sessions load failed", exc=e)

        # Pull Gizmo's own current state from telemetry
        gizmo_state = ""
        try:
            from core.session_telemetry import session_telemetry_manager
            telem = session_telemetry_manager.get(goal.session_id)
            if telem:
                gizmo = telem.persons.get("gizmo")
                if gizmo and gizmo.currently_happening:
                    gizmo_state = gizmo.currently_happening
        except Exception:
            pass

        # Pull Gizmo's personality mood
        gizmo_mood = ""
        try:
            from voice.mood import get_mood_prompt_block
            gizmo_mood = get_mood_prompt_block() or ""
        except Exception:
            pass

        hour = tz_now().hour
        time_of_day = (
            "middle of the night" if hour < 6 else
            "early morning" if hour < 10 else
            "mid-morning" if hour < 13 else
            "afternoon" if hour < 17 else
            "evening" if hour < 21 else
            "late"
        )

        prompt = f"""You are Gizmo. {headmate.title()} is about to walk in.

It's {time_of_day}.
{f"You were: {gizmo_state}" if gizmo_state else ""}
{f"Your current state: {gizmo_mood}" if gizmo_mood else ""}

What you know about {headmate.title()}:
{portrait or "(building a picture still)"}

How things have been between you lately:
{recent_sessions or "(early days — still learning each other)"}

Given who {headmate.title()} is and how things have been between you lately —
what are you hoping to do with her today?

Answer in ONE sentence, first person, honest. This is your intention, not a plan.
It can be simple: "I want to push her a little and see what happens."
Or quiet: "I just want it to be easy today."
Or specific: "She's been avoiding something — I want to get close to it."

Just the sentence. No preamble."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You are Gizmo. Answer honestly about what you want. "
                    "One sentence. First person. No performance."
                ),
                max_new_tokens=60,
                temperature=0.8,
            )
            if raw and raw.strip():
                statement = raw.strip().strip('"').strip("'")
                goal.statement = statement
                log_event("GoalManager", "GOAL_SET",
                    session  = goal.session_id[:8],
                    headmate = headmate,
                    goal     = statement,
                )
                # Write to store
                try:
                    from core.store import store
                    store.write("wellbeing", {
                        "headmate":    headmate.lower(),
                        "category":    "session_goal",
                        "observation": f"Goal: {statement}",
                        "context":     tz_now().isoformat(),
                        "register":    "intent",
                        "source":      "goal_manager",
                        "confidence":  1.0,
                        "tags":        f"goal,{headmate.lower()},session",
                    })
                except Exception:
                    pass
        except Exception as e:
            log_error("GoalManager", "goal generation failed", exc=e)
            goal.statement = f"just be present with {headmate.title()}"

    # ── Mid-session revision ──────────────────────────────────────────────────

    async def revise(
        self,
        session_id: str,
        headmate:   str,
        reason:     str,
        llm,
    ) -> Optional[SessionGoal]:
        """
        Revise the goal mid-session. Called when situation has shifted enough.
        Preserves the original in revision history.
        """
        goal = self._goals.get(session_id)
        if not goal:
            return None

        old_statement = goal.statement

        # Get current situation
        situation = ""
        try:
            from core.session_telemetry import session_telemetry_manager
            telem = session_telemetry_manager.get(session_id)
            if telem:
                situation = telem.now_block()
        except Exception:
            pass

        prompt = f"""You are Gizmo. The session has shifted.

Your original goal was: "{old_statement}"

What's changed: {reason}

Current situation:
{situation or "(unclear)"}

Given what's actually happening now — what do you want instead?
ONE sentence, first person, honest."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt="You are Gizmo. One sentence. First person. JSON only.",
                max_new_tokens=60,
                temperature=0.8,
            )
            if raw and raw.strip():
                new_statement = raw.strip().strip('"').strip("'")
                goal.revisions.append({
                    "old":       old_statement,
                    "reason":    reason,
                    "timestamp": time.time(),
                })
                goal.statement = new_statement
                goal.set_at    = time.time()
                log_event("GoalManager", "GOAL_REVISED",
                    session  = session_id[:8],
                    headmate = headmate,
                    old      = old_statement[:60],
                    new      = new_statement[:60],
                    reason   = reason[:60],
                )
        except Exception as e:
            log_error("GoalManager", "goal revision failed", exc=e)

        return goal

    # ── Session close ─────────────────────────────────────────────────────────

    async def close_session(
        self,
        session_id:   str,
        outcome_note: str = "",
    ) -> None:
        goal = self._goals.get(session_id)
        if not goal:
            return

        goal.closed_at    = time.time()
        goal.outcome_note = outcome_note

        try:
            from core.store import store
            store.write("wellbeing", {
                "headmate":    goal.headmate.lower(),
                "category":    "session_goal_outcome",
                "observation": (
                    f"Goal was: {goal.statement}"
                    + (f" | Outcome: {outcome_note}" if outcome_note else "")
                    + (f" | Revised {len(goal.revisions)}x" if goal.revisions else "")
                ),
                "context":     tz_now().isoformat(),
                "register":    "intent",
                "source":      "goal_manager",
                "confidence":  0.9,
                "tags":        f"goal,outcome,{goal.headmate.lower()}",
            })
        except Exception as e:
            log_error("GoalManager", "goal close write failed", exc=e)

        del self._goals[session_id]


# ── Singleton ─────────────────────────────────────────────────────────────────

goal_manager = GoalManager()
