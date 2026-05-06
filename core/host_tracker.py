"""
core/host_tracker.py
Server-side session state for host and fronter tracking.

Owns the question: who is speaking right now, and who else is present?

This replaces the client-side fronter input field. The client no longer
tells the server who is fronting — the server figures it out from conversation.

Host identification triggers:
  - "hey it's Jess" / "this is Oren" / "Jess here" → set current_host
  - Known headmate name appears as subject of first-person statement → candidate
  - Switch tool fires → update host

Doubt triggers (prompts Ego to check in):
  - New session + known headmate files exist → greet neutrally, let them ID
  - Tone/register shift from established baseline → flag for Ego
  - Explicit "someone else is here now" → clear host, ask

Fronter awareness:
  - "hanging with Oren and Kaylee" → add to active_fronters
  - "Oren just left" → remove from active_fronters
  - All active fronters are hot context for Mind (RAG queries their collections)

Usage:
    from core.host_tracker import host_tracker

    # Get current context for a session (replaces client-sent context)
    context = host_tracker.get_context(session_id)

    # Update after identification detected
    host_tracker.set_host(session_id, "jess")

    # Add fronters mentioned as present
    host_tracker.add_fronters(session_id, ["oren", "kaylee"])

    # Called by Archivist after each message to apply any detected changes
    host_tracker.apply_brief_updates(session_id, brief)
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from core.log import log, log_event, log_error

# ── Identification patterns ───────────────────────────────────────────────────
# Match self-identification statements. Capture group 1 = the name.

_IDENT_PATTERNS = [
    # "hey it's Jess" / "it's me, Oren" / "it's Kaylee!"
    re.compile(r"\bit'?s\s+(?:me[,\s]+)?([A-Z][a-z]{1,20})\b", re.IGNORECASE),
    # "this is Jess" / "this is Oren here"
    re.compile(r"\bthis\s+is\s+([A-Z][a-z]{1,20})\b", re.IGNORECASE),
    # "Jess here" / "Oren here!"
    re.compile(r"\b([A-Z][a-z]{1,20})\s+here\b", re.IGNORECASE),
    # "I'm Jess" / "I'm Kaylee btw"
    re.compile(r"\bI'?m\s+([A-Z][a-z]{1,20})\b", re.IGNORECASE),
    # "hey, Jess speaking"
    re.compile(r"\b([A-Z][a-z]{1,20})\s+speaking\b", re.IGNORECASE),
    # "btw it's Oren" / "btw, Jess"
    re.compile(r"\bbtw[,\s]+(?:it'?s\s+)?([A-Z][a-z]{1,20})\b", re.IGNORECASE),
]

# Fronter presence patterns — "hanging with X and Y", "X is here too"
_PRESENCE_PATTERNS = [
    # "hanging with Oren and Kaylee" / "chilling with Millie"
    re.compile(
        r"\b(?:hanging|chilling|vibing|here|sitting|watching)\s+with\s+"
        r"([A-Z][a-z]{1,20}(?:\s*(?:,|and)\s*[A-Z][a-z]{1,20})*)",
        re.IGNORECASE,
    ),
    # "Oren is here" / "Kaylee's around"
    re.compile(
        r"\b([A-Z][a-z]{1,20})\s+(?:is|'s)\s+(?:here|around|with\s+me|fronting|co-?fronting)\b",
        re.IGNORECASE,
    ),
    # "me and Oren" / "Oren and I"
    re.compile(
        r"\b(?:me\s+and\s+|with\s+me\s+and\s+)([A-Z][a-z]{1,20})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][a-z]{1,20})\s+and\s+(?:I|me)\b",
        re.IGNORECASE,
    ),
]

# Departure patterns — "Oren just left" / "Kaylee switched out"
_DEPARTURE_PATTERNS = [
    re.compile(
        r"\b([A-Z][a-z]{1,20})\s+(?:left|switched\s+out|went\s+back|stepped\s+out|is\s+gone)\b",
        re.IGNORECASE,
    ),
]

# Words that look like names in "I'm [word]" constructions but aren't
_NOT_NAMES = {
    "I", "Me", "My", "We", "Us", "You", "It", "He", "She", "They",
    "The", "A", "An", "And", "But", "Or", "So", "Ok", "Okay",
    "Hey", "Hi", "Hello", "Bye", "Yes", "No", "Not", "Now",
    "Gizmo", "Just", "Still", "Here", "There", "Back", "Out",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "Today", "Tomorrow", "Yesterday",
    # Common adjectives/states that follow "I'm"
    "Curious", "Excited", "Tired", "Happy", "Sad", "Good", "Fine",
    "Great", "Okay", "Bad", "Bored", "Anxious", "Stressed", "Ready",
    "Sure", "Sorry", "Glad", "Worried", "Nervous", "Scared", "Lost",
    "Here", "Back", "Home", "Done", "Free", "Busy", "Late", "Early",
    "Hungry", "Tired", "Sick", "Well", "Better", "Worse", "New",
    "Old", "Young", "Big", "Small", "Right", "Wrong", "Different",
    "Thinking", "Working", "Trying", "Going", "Coming", "Leaving",
    "Serious", "Kidding", "Joking", "Playing", "Watching", "Listening",
    "Confused", "Surprised", "Shocked", "Impressed", "Interested",
    "Distressed", "Overwhelmed", "Frustrated", "Annoyed", "Angry",
    "Positive", "Neutral", "Negative", "Elevated", "Subdued",
    # Family titles and terms of endearment that aren't names
    "Baby", "Babe", "Gramma", "Grandma", "Grandpa", "Nana", "Papa",
    "Mama", "Momma", "Daddy", "Auntie", "Uncle", "Sis", "Bro",
    "Honey", "Sugar", "Sweetie", "Darling", "Love", "Hun",
}


def _clean_name(name: str) -> Optional[str]:
    """Normalize a detected name. Returns None if it looks like a non-name."""
    name = name.strip().title()
    if name in _NOT_NAMES:
        return None
    if len(name) < 2 or len(name) > 20:
        return None
    return name


def _extract_names_from_match(match_group: str) -> list[str]:
    """Extract individual names from a match that may contain 'and' or commas."""
    # Split on "and" and commas
    parts = re.split(r"\s+and\s+|,\s*", match_group, flags=re.IGNORECASE)
    names = []
    for part in parts:
        name = _clean_name(part.strip())
        if name:
            names.append(name)
    return names


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """
    Per-session fronter state. Lives server-side, updated from conversation.
    """
    session_id:      str
    current_host:    Optional[str] = None
    active_fronters: list[str]     = field(default_factory=list)
    timezone:        str           = "UTC"
    created_at:      float         = field(default_factory=time.time)
    updated_at:      float         = field(default_factory=time.time)
    host_confidence: float         = 0.0   # 0 = unknown, 1 = explicit identification
    last_host:       Optional[str] = None  # for change detection

    def set_host(self, name: str, confidence: float = 1.0) -> bool:
        """
        Set current host. Returns True if this is a change.
        """
        name = name.lower()
        changed = name != self.current_host
        if changed:
            self.last_host = self.current_host
        self.current_host = name
        self.host_confidence = confidence
        self.updated_at = time.time()

        # Host is always a fronter
        if name not in [f.lower() for f in self.active_fronters]:
            self.active_fronters.insert(0, name)

        return changed

    def add_fronters(self, names: list[str]) -> list[str]:
        """
        Add names to active fronters. Returns list of newly added names.
        """
        added = []
        current_lower = [f.lower() for f in self.active_fronters]
        for name in names:
            nl = name.lower()
            if nl not in current_lower:
                self.active_fronters.append(name)
                current_lower.append(nl)
                added.append(name)
        self.updated_at = time.time()
        return added

    def remove_fronter(self, name: str) -> bool:
        """Remove a fronter. Returns True if removed."""
        name_lower = name.lower()
        before = len(self.active_fronters)
        self.active_fronters = [
            f for f in self.active_fronters
            if f.lower() != name_lower
        ]
        # If current host left, clear host
        if self.current_host and self.current_host.lower() == name_lower:
            self.current_host = None
            self.host_confidence = 0.0
        self.updated_at = time.time()
        return len(self.active_fronters) < before

    def to_context(self) -> dict:
        """Export as context dict consumed by agent/archivist."""
        return {
            "current_host": self.current_host or "",
            "fronters":     list(self.active_fronters),
            "timezone":     self.timezone,
        }


# ── Host Tracker ──────────────────────────────────────────────────────────────

class HostTracker:
    """
    Singleton. Manages per-session fronter state.
    Called by server.py to get context, and by Archivist after each message.
    """

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        log("HostTracker", "initialised")

    def _get(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def get_context(self, session_id: str) -> dict:
        """
        Get current context dict for a session.
        This replaces the client-sent context entirely.
        """
        return self._get(session_id).to_context()

    def set_timezone(self, session_id: str, tz: str) -> None:
        """Set timezone for a session (sent once by client on connect)."""
        state = self._get(session_id)
        state.timezone = tz

    def set_host(self, session_id: str, name: str, confidence: float = 1.0) -> bool:
        """Explicitly set the current host. Returns True if changed."""
        state = self._get(session_id)
        changed = state.set_host(name, confidence)
        if changed:
            log_event("HostTracker", "HOST_SET",
                session=session_id[:8],
                host=name,
                confidence=confidence,
                previous=state.last_host or "none",
            )
        return changed

    def add_fronters(self, session_id: str, names: list[str]) -> list[str]:
        """Add fronters to the active set. Returns newly added names."""
        state = self._get(session_id)
        added = state.add_fronters(names)
        if added:
            log_event("HostTracker", "FRONTERS_ADDED",
                session=session_id[:8],
                added=added,
                all_fronters=state.active_fronters,
            )
        return added

    def remove_fronter(self, session_id: str, name: str) -> None:
        """Remove a fronter from the active set."""
        state = self._get(session_id)
        removed = state.remove_fronter(name)
        if removed:
            log_event("HostTracker", "FRONTER_REMOVED",
                session=session_id[:8],
                name=name,
                remaining=state.active_fronters,
            )

    def process_message(self, session_id: str, message: str, known_headmates: set) -> dict:
        """
        Scan a message for host identification and fronter presence signals.
        Updates state and returns a dict of what changed:
          {
            "host_identified": str | None,
            "fronters_added":  list[str],
            "fronters_removed": list[str],
          }

        Called by Archivist.receive() so the brief reflects current state.
        """
        state = self._get(session_id)
        changes = {
            "host_identified":  None,
            "fronters_added":   [],
            "fronters_removed": [],
        }

        # ── Self-identification ───────────────────────────────────────────────
        for pattern in _IDENT_PATTERNS:
            match = pattern.search(message)
            if match:
                name = _clean_name(match.group(1))
                if not name:
                    continue

                # Only accept if:
                # 1. It's a known headmate, OR
                # 2. No headmate files exist yet (cold start — trust explicit statements)
                #    AND the original message had it capitalized (proper noun signal)
                name_is_known = name.lower() in known_headmates
                cold_start = not known_headmates
                was_capitalized = bool(re.search(
                    rf"\b{re.escape(match.group(1))}\b", message
                ) and match.group(1)[0].isupper())

                if name_is_known or (cold_start and was_capitalized):
                    changed = state.set_host(name, confidence=1.0)
                    if changed:
                        changes["host_identified"] = name
                        log_event("HostTracker", "HOST_IDENTIFIED",
                            session=session_id[:8],
                            name=name,
                            known=name_is_known,
                            cold_start=cold_start,
                        )
                    break  # one identification per message

        # ── Fronter presence ──────────────────────────────────────────────────
        for pattern in _PRESENCE_PATTERNS:
            for match in pattern.finditer(message):
                names = _extract_names_from_match(match.group(1))
                for name in names:
                    # Only add as fronter if known headmate or cold start
                    if name.lower() in known_headmates or not known_headmates:
                        added = state.add_fronters([name])
                        changes["fronters_added"].extend(added)

        # ── Fronter departures ────────────────────────────────────────────────
        for pattern in _DEPARTURE_PATTERNS:
            for match in pattern.finditer(message):
                name = _clean_name(match.group(1))
                if name:
                    removed = state.remove_fronter(name)
                    if removed:
                        changes["fronters_removed"].append(name)
                        log_event("HostTracker", "FRONTER_DEPARTED",
                            session=session_id[:8],
                            name=name,
                        )

        if any(changes.values()):
            log_event("HostTracker", "MESSAGE_PROCESSED",
                session=session_id[:8],
                host=state.current_host or "unknown",
                fronters=state.active_fronters,
                changes={k: v for k, v in changes.items() if v},
            )

        return changes

    def get_state(self, session_id: str) -> SessionState:
        return self._get(session_id)

    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# ── Singleton ─────────────────────────────────────────────────────────────────
host_tracker = HostTracker()