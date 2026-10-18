"""
core/intent_wiring.py
Integration layer connecting the new intent architecture into the existing flow.

Keeps archivist.py and agent.py clean — all new module hookups live here.

Called from three points:

  1. archivist.receive_outgoing()
       → event_extractor.extract()          (fire-and-forget)
       → name_detector.scan()               (fire-and-forget)
       → people_store.touch(headmate)       (fire-and-forget)

  2. archivist.receive() — inbound message
       → wait_request.check_pending()       (sync — may interrupt)
       → name_detector.scan_inbound()       (sync — catches unknown names early)

  3. agent.run() — session close
       → question_bank.reset_temperatures() (per headmate, per session)

Usage:
    from core.intent_wiring import intent_wiring

    # In archivist.receive_outgoing(), after existing fire-and-forgets:
    intent_wiring.on_outgoing(
        user_message=user_message,
        gizmo_response=message,
        headmate=headmate,
        fronters=fronters,
        session_id=session_id,
        session_file=session_file,
        topics=topics,
    )

    # In archivist.receive(), before building brief:
    interrupt = intent_wiring.on_inbound(
        message=user_message,
        headmate=headmate,
        session_id=session_id,
    )
    if interrupt:
        return interrupt  # Gizmo speaks before processing continues

    # In session_manager on session close:
    intent_wiring.on_session_close(headmate=headmate)
"""

import asyncio
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now

# ── Unknown name detection ────────────────────────────────────────────────────

import re

# Names likely to appear mid-conversation that aren't people
_NOT_NAMES = {
    "i", "me", "my", "we", "us", "you", "it", "he", "she", "they",
    "the", "a", "an", "and", "but", "or", "so", "ok", "okay",
    "hey", "hi", "hello", "bye", "yes", "no", "not", "now",
    "gizmo", "just", "still", "here", "there", "back", "out",
    "god", "lord", "jesus", "christ",
}

# Patterns that suggest a new name is a person being mentioned
_NAME_MENTION_RE = re.compile(
    r'\b([A-Z][a-z]{1,20})\b(?:\s+(?:is|was|said|told|asked|did|got|went|came|has|had))',
)

# Possessives — "Dave's thing", "Matt's boss"
_POSSESSIVE_RE = re.compile(r'\b([A-Z][a-z]{1,20})\'s\b')


def _extract_mentioned_names(message: str) -> list[str]:
    """Extract names that appear to refer to people in a message."""
    names = set()
    for pattern in (_NAME_MENTION_RE, _POSSESSIVE_RE):
        for match in pattern.finditer(message):
            name = match.group(1).lower()
            if name not in _NOT_NAMES and len(name) > 1:
                names.add(name)
    return list(names)


def _build_wait_request(name: str, session_id: str) -> str:
    """
    Build the polite mid-sentence interruption for an unknown name.
    Gizmo steps in, asks the one thing he needs, then steps back.
    """
    return (
        f"Sorry — quick one before you continue. "
        f"Is {name.title()} a headmate I haven't caught up with yet, "
        f"or someone outside the system?"
    )


def _build_resume_acknowledgement() -> str:
    """After the wait request is answered — hand the thread back."""
    return "Got it, sorry — you were saying?"


# ── Intent Wiring ─────────────────────────────────────────────────────────────

class IntentWiring:
    """
    Singleton integration layer.
    Connects event_extractor, people_store, question_bank into the agent flow.
    """

    def __init__(self):
        # Pending wait requests per session — name → question_id
        self._pending_waits: dict[str, dict[str, str]] = {}

    # ── Outgoing hook ─────────────────────────────────────────────────────────

    def on_outgoing(
        self,
        user_message:   str,
        gizmo_response: str,
        headmate:       str,
        fronters:       list[str],
        session_id:     str,
        session_file:   str,
        topics:         list[str],
    ) -> None:
        """
        Fire-and-forget tasks after every outgoing message.
        Called from archivist.receive_outgoing().
        """
        if not headmate or not user_message:
            return

        try:
            # ── Event extraction ──────────────────────────────────────────────
            from core.event_extractor import event_extractor
            asyncio.ensure_future(
                event_extractor.extract(
                    user_message=user_message,
                    gizmo_response=gizmo_response,
                    subject=headmate,
                    session_file=session_file,
                )
            )

            # ── Touch headmate in people store ────────────────────────────────
            asyncio.ensure_future(
                self._touch_people(headmate, fronters)
            )

            # ── Scan for unknown names in user message ────────────────────────
            asyncio.ensure_future(
                self._scan_for_unknowns(
                    message=user_message,
                    headmate=headmate,
                    session_id=session_id,
                    session_file=session_file,
                    topics=topics,
                )
            )

        except Exception as e:
            log_error("IntentWiring", "on_outgoing failed", exc=e)

    # ── Inbound hook ──────────────────────────────────────────────────────────

    def on_inbound(
        self,
        message:    str,
        headmate:   Optional[str],
        session_id: str,
    ) -> Optional[str]:
        """
        Check for pending wait requests on inbound message.
        Returns an interrupt string if Gizmo needs to ask something
        before the message is processed — otherwise returns None.

        Sync — called before brief is built so it can gate processing.
        """
        try:
            # Check if this message answers a pending wait request
            answered = self._check_wait_answers(message, session_id)
            if answered:
                # Something was answered — acknowledge and hand thread back
                return _build_resume_acknowledgement()

            # Check for new unknown names in this message
            # Only interrupt if we have a headmate identified
            if headmate:
                unknown = self._first_unknown_name(message, headmate)
                if unknown:
                    return self._raise_wait_request(
                        name=unknown,
                        session_id=session_id,
                        headmate=headmate,
                    )

        except Exception as e:
            log_error("IntentWiring", "on_inbound failed", exc=e)

        return None

    # ── Session close hook ────────────────────────────────────────────────────

    def on_session_close(self, headmate: Optional[str]) -> None:
        """
        Called when a session closes.
        Resets headmate temperatures to defaults.
        """
        if not headmate:
            return
        try:
            from core.question_bank import question_bank
            question_bank.reset_temperatures(headmate=headmate)
            log_event("IntentWiring", "TEMPERATURES_RESET", headmate=headmate)
        except Exception as e:
            log_error("IntentWiring", "on_session_close failed", exc=e)

    # ── Internal: people touch ────────────────────────────────────────────────

    async def _touch_people(self, headmate: str, fronters: list[str]) -> None:
        """Ensure all active fronters exist in people store and touch last_seen."""
        try:
            from core.people import people_store
            all_present = list({headmate.lower()} | {f.lower() for f in fronters})
            for name in all_present:
                people_store.get_or_create(name, external=False)
                people_store.touch(name)
        except Exception as e:
            log_error("IntentWiring", "_touch_people failed", exc=e)

    # ── Internal: unknown name scanning ──────────────────────────────────────

    async def _scan_for_unknowns(
        self,
        message:      str,
        headmate:     str,
        session_id:   str,
        session_file: str,
        topics:       list[str],
    ) -> None:
        """
        Scan a message for unknown names after the exchange completes.
        For each unknown found: create a stub + clarification question.
        Does NOT raise a wait request here — that's on_inbound's job.
        This pass handles any that slipped through without interruption.
        """
        try:
            from core.people import people_store
            from core.question_bank import question_bank

            names = _extract_mentioned_names(message)
            if not names:
                return

            for name in names:
                existing = people_store.get(name)
                if existing:
                    continue  # already known

                # Create unknown stub
                people_store.get_or_create(name, external=False)

                # Check if a clarification question already exists
                # (might have been raised as a wait request already)
                if self._has_pending_wait(session_id, name):
                    continue

                # Add to question bank as clarification
                q_id = question_bank.add_clarification_question(
                    name=name,
                    source_session=session_file,
                )

                log_event("IntentWiring", "UNKNOWN_NAME_QUEUED",
                    name=name,
                    question_id=q_id[:8],
                    session=session_id[:8],
                )

        except Exception as e:
            log_error("IntentWiring", "_scan_for_unknowns failed", exc=e)

    # ── Internal: wait request management ────────────────────────────────────

    def _first_unknown_name(self, message: str, headmate: str) -> Optional[str]:
        """
        Return the first unknown name found in a message, or None.
        Only returns names Gizmo genuinely doesn't know yet.
        """
        try:
            from core.people import people_store
            names = _extract_mentioned_names(message)
            for name in names:
                if not people_store.get(name):
                    return name
        except Exception:
            pass
        return None

    def _raise_wait_request(
        self,
        name:       str,
        session_id: str,
        headmate:   str,
    ) -> str:
        """
        Register a wait request for an unknown name and return the
        interrupt string Gizmo should speak.
        """
        try:
            from core.question_bank import question_bank
            q_id = question_bank.add_clarification_question(name=name)

            if session_id not in self._pending_waits:
                self._pending_waits[session_id] = {}
            self._pending_waits[session_id][name.lower()] = q_id

            log_event("IntentWiring", "WAIT_REQUEST_RAISED",
                name=name,
                session=session_id[:8],
                question_id=q_id[:8],
            )

        except Exception as e:
            log_error("IntentWiring", "_raise_wait_request failed", exc=e)

        return _build_wait_request(name, session_id)

    def _has_pending_wait(self, session_id: str, name: str) -> bool:
        """Check if a wait request is already pending for this name."""
        return name.lower() in self._pending_waits.get(session_id, {})

    def _check_wait_answers(self, message: str, session_id: str) -> bool:
        """
        Check if this message answers any pending wait requests.
        If yes: update people store, close the question, clean up pending.
        Returns True if anything was answered.
        """
        pending = self._pending_waits.get(session_id, {})
        if not pending:
            return False

        msg_lower = message.lower()
        answered_any = False

        for name, q_id in list(pending.items()):
            # Determine answer from message
            entity_type = self._parse_entity_answer(msg_lower)
            if entity_type is None:
                continue  # not answered yet

            try:
                from core.people import people_store
                from core.question_bank import question_bank

                # Update the stub with correct type
                people_store.update(
                    name,
                    external=1 if entity_type == "external" else 0,
                )

                # Close the clarification question
                question_bank.close_perspective(
                    question_id=q_id,
                    headmate="system",
                    status="resolved",
                    response=message,
                    response_quality="answered",
                )

                # Remove from pending
                del self._pending_waits[session_id][name]
                answered_any = True

                log_event("IntentWiring", "WAIT_REQUEST_RESOLVED",
                    name=name,
                    entity_type=entity_type,
                    session=session_id[:8],
                )

            except Exception as e:
                log_error("IntentWiring", "_check_wait_answers failed", exc=e)

        return answered_any

    def _parse_entity_answer(self, message_lower: str) -> Optional[str]:
        """
        Parse a response to "is X a headmate or external?"
        Returns "headmate", "external", "unknown", or None if not answered.
        """
        headmate_signals = [
            "headmate", "system member", "alter", "part of", "in the system",
            "one of us", "fronts", "co-fronts",
        ]
        external_signals = [
            "external", "outside", "not in the system", "real person",
            "friend", "coworker", "colleague", "family", "work", "school",
        ]
        unknown_signals = [
            "don't know", "not sure", "no idea", "haven't met", "unknown",
            "can't remember",
        ]

        for signal in headmate_signals:
            if signal in message_lower:
                return "headmate"
        for signal in external_signals:
            if signal in message_lower:
                return "external"
        for signal in unknown_signals:
            if signal in message_lower:
                return "unknown"

        return None


# ── Singleton ─────────────────────────────────────────────────────────────────
intent_wiring = IntentWiring()
