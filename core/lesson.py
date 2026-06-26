"""
core/lesson.py

Teaching system for Gizmo.

Triggered by: "I wanna teach you..." in any register.
Stays open until natural close ("that's it", "got it", "perfect", session end).

During a lesson:
  - A teaching pass runs parallel to BehaviorCatcher
  - Gizmo maintains a working understanding of what's being taught
  - He demonstrates it in responses — the responder is told a lesson is active
  - Corrections update the working version
  - When it lands, the lesson saves to the right target and synthesis fires

Write targets:
  - Global opinion/preference → behaviors/gizmo.json :: Taught
  - Headmate-specific         → behaviors/gizmo.json :: PerHeadmate.{name}

Conflict detection:
  - Incoming lesson checked against existing Taught entries
  - Conflicts surfaced to Gizmo so he can flag them in response
"""

import json
import re
from typing import Optional

from core.log import log_event, log_error
import core.librarian as librarian


# ── Trigger detection ─────────────────────────────────────────────────────────

_TRIGGER_PHRASES = [
    "i wanna teach you",
    "i want to teach you",
    "let me teach you",
    "i'm gonna teach you",
    "i'm going to teach you",
]

_CLOSE_PHRASES = [
    "that's it",
    "that's all",
    "got it",
    "perfect",
    "good",
    "yes, like that",
    "yeah, like that",
    "exactly",
    "okay that's enough",
    "stop there",
    "save that",
]


def is_lesson_trigger(message: str) -> bool:
    low = message.lower().strip()
    return any(phrase in low for phrase in _TRIGGER_PHRASES)


def is_lesson_close(message: str) -> bool:
    low = message.lower().strip()
    return any(phrase in low for phrase in _CLOSE_PHRASES)


# ── Lesson session state ──────────────────────────────────────────────────────

class LessonSession:
    def __init__(self, headmate: str, raw_trigger: str):
        self.headmate       = headmate       # who is teaching
        self.raw_trigger    = raw_trigger    # original "I wanna teach you..." message
        self.domain         = None           # extracted on first pass
        self.scope          = None           # "global" or "headmate"
        self.working        = {}             # current working version of the lesson
        self.exchanges      = 0             # how many back-and-forths since open
        self.conflicts      = []            # conflicts with existing Taught entries
        self.ready_to_save  = False

    def is_active(self) -> bool:
        return not self.ready_to_save


# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """
You extract the intent and scope of a teaching statement directed at an AI companion.
Return ONLY valid JSON. No markdown. No explanation.

{
  "domain": "one of: dynamic, functional, opinion, personality, relational",
  "scope": "one of: global, headmate",
  "key": "short identifier for what is being taught — e.g. 'dominance style', 'yellow is ugly', 'task management'",
  "value": "what is being taught, in plain prose — the actual lesson content",
  "notes": "anything relevant about how this should be applied"
}

domain definitions:
- dynamic: power exchange, dominance/submission, scene behavior
- functional: how to help with tasks, organization, assistance
- opinion: preferences, likes, dislikes, aesthetic judgments
- personality: general demeanor, how he shows up, communication style
- relational: how he is with this specific person

scope:
- headmate: this lesson applies only to the relationship with the person teaching
- global: this lesson applies across all interactions

When in doubt about scope: dynamic and relational lessons are almost always headmate.
Opinion and personality lessons are usually global unless the person specifies.
""".strip()


async def _extract_lesson(message: str, headmate: str) -> Optional[dict]:
    try:
        from core.llm import llm
        prompt = f"Teacher: {headmate}\n\nStatement: {message}"
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_EXTRACT_SYSTEM,
            temperature=0.0,
            max_new_tokens=300,
        )
        if not raw or not raw.strip():
            return None
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        log_error("Lesson", "extract failed", exc=e)
        return None


# ── Working version update ────────────────────────────────────────────────────

_UPDATE_SYSTEM = """
You maintain the current working version of a lesson being taught to an AI companion.
You receive the existing working version and a new message from the teacher.
Return ONLY valid JSON. No markdown. No explanation.

{
  "value": "updated lesson content incorporating the new instruction or correction",
  "notes": "updated notes if anything changed",
  "confidence": "low | medium | high — how complete and clear the lesson feels now"
}

Rules:
- Corrections override the previous version — don't average them
- Additions extend the previous version
- If the teacher says something like "yes", "exactly", "like that" — confidence goes up
- If the teacher corrects something — incorporate the correction, confidence stays medium until confirmed
- Keep it concise — this is a working definition, not an essay
""".strip()


async def _update_working(session: LessonSession, new_message: str) -> dict:
    try:
        from core.llm import llm
        prompt = (
            f"Current working version:\n{json.dumps(session.working, indent=2)}\n\n"
            f"New message from teacher ({session.headmate}):\n{new_message}"
        )
        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_UPDATE_SYSTEM,
            temperature=0.0,
            max_new_tokens=300,
        )
        if not raw or not raw.strip():
            return session.working
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        log_error("Lesson", "update_working failed", exc=e)
        return session.working


# ── Conflict detection ────────────────────────────────────────────────────────

def _check_conflicts(session: LessonSession) -> list[str]:
    """Check incoming lesson against existing Taught entries for conflicts."""
    data   = librarian._read_file("behaviors/gizmo.json") or {}
    taught = data.get("Taught", {})
    key    = session.working.get("key") or session.domain

    conflicts = []
    if key and key in taught:
        existing = taught[key]
        existing_val = existing.get("value", "")
        new_val      = session.working.get("value", "")
        if existing_val and existing_val != new_val:
            conflicts.append(
                f"I already have something about '{key}': {existing_val!r}. "
                f"This lesson would change it to: {new_val!r}."
            )
    return conflicts


# ── Save ──────────────────────────────────────────────────────────────────────

def _save_lesson(session: LessonSession) -> None:
    data = librarian._read_file("behaviors/gizmo.json") or {}

    key   = session.working.get("key") or session.domain or "unknown"
    value = session.working.get("value", "")
    notes = session.working.get("notes", "")

    entry = {
        "value":    value,
        "notes":    notes,
        "scope":    session.scope,
        "headmate": session.headmate if session.scope == "headmate" else None,
        "domain":   session.domain,
    }

    if session.scope == "headmate":
        per = data.setdefault("PerHeadmate", {})
        hm  = per.setdefault(session.headmate.lower(), {})
        hm[key] = entry
    else:
        taught = data.setdefault("Taught", {})
        taught[key] = entry

    librarian._write_json("behaviors/gizmo.json", data)
    print(f"[Lesson] saved: {key} → {session.scope} ({session.headmate})")
    log_event("Lesson", "SAVED", key=key, scope=session.scope, headmate=session.headmate)


# ── Public API ────────────────────────────────────────────────────────────────

class LessonManager:

    def __init__(self):
        self._session: Optional[LessonSession] = None

    @property
    def active(self) -> bool:
        return self._session is not None and self._session.is_active()

    @property
    def session(self) -> Optional[LessonSession]:
        return self._session

    async def open(self, message: str, headmate: str) -> dict:
        """
        Open a new lesson session. Extracts domain and scope from the trigger message.
        Returns a dict with what was understood — used to inform Gizmo's first response.
        """
        extracted = await _extract_lesson(message, headmate)
        if not extracted:
            return {"error": "could not extract lesson intent"}

        self._session = LessonSession(headmate=headmate, raw_trigger=message)
        self._session.domain  = extracted.get("domain")
        self._session.scope   = extracted.get("scope", "global")
        self._session.working = extracted

        conflicts = _check_conflicts(self._session)
        self._session.conflicts = conflicts

        log_event("Lesson", "OPENED",
            headmate=headmate,
            domain=self._session.domain,
            scope=self._session.scope,
            key=extracted.get("key"),
        )

        return {
            "domain":    self._session.domain,
            "scope":     self._session.scope,
            "key":       extracted.get("key"),
            "value":     extracted.get("value"),
            "conflicts": conflicts,
        }

    async def update(self, message: str) -> dict:
        """
        Process a message during an active lesson.
        Updates working version. Checks if lesson is ready to close.
        Returns current state for the responder brief.
        """
        if not self._session:
            return {}

        self._session.exchanges += 1
        updated = await _update_working(self._session, message)
        self._session.working.update(updated)

        confidence = updated.get("confidence", "low")
        if confidence == "high" or is_lesson_close(message):
            self._session.ready_to_save = True

        return {
            "domain":     self._session.domain,
            "scope":      self._session.scope,
            "key":        self._session.working.get("key"),
            "value":      self._session.working.get("value"),
            "confidence": confidence,
            "conflicts":  self._session.conflicts,
            "closing":    self._session.ready_to_save,
        }

    async def close(self) -> None:
        """Save the lesson and trigger synthesis."""
        if not self._session:
            return

        _save_lesson(self._session)
        self._session = None

        # Re-synthesize immediately — teaching writes should propagate to next response
        try:
            from core.gizmo_synthesis import synthesize
            await synthesize()
        except Exception as e:
            log_error("Lesson", "synthesis after save failed", exc=e)

    def abandon(self) -> None:
        """Close without saving — e.g. session ended before lesson landed."""
        if self._session:
            log_event("Lesson", "ABANDONED",
                headmate=self._session.headmate,
                exchanges=self._session.exchanges,
            )
        self._session = None

    def brief(self) -> Optional[dict]:
        """Return current lesson state for injection into the responder brief."""
        if not self._session:
            return None
        return {
            "active":    True,
            "domain":    self._session.domain,
            "scope":     self._session.scope,
            "key":       self._session.working.get("key"),
            "value":     self._session.working.get("value"),
            "conflicts": self._session.conflicts,
            "exchanges": self._session.exchanges,
            "closing":   self._session.ready_to_save,
        }


lesson_manager = LessonManager()
