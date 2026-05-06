"""
core/archivist.py
The Archivist — first receiver of all input, regardless of source.

Responsibilities (v1):
  - Receive a message and classify it without LLM
  - Build a lightweight conversation field for the session
  - Package a structured brief (plain dict — no language)
  - Track topic temperature: hot / warm / cool
  - Save raw message to history
  - Log everything in structured, reflection-ready format

What this is NOT yet (grows into later):
  - Full tensor temperature management
  - Thought bubble accumulation across multi-part messages
  - Cross-session memory queries
  - Baseline reader integration
  - Empath / Id briefs

The brief returned is a plain dict consumed by the rest of the pipeline.
No LLM is called here. Ever. Classification is heuristic + keyword.

Usage:
    from core.archivist import Archivist

    archivist = Archivist()
    brief = archivist.receive(
        message="I'm getting hungry",
        session_id="abc123",
        history=history,
        context={"current_host": "alastor", "fronters": ["alastor"]},
    )
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from core.log import log, log_event, log_error

# ── Topic keyword map ─────────────────────────────────────────────────────────
# Lightweight heuristic classification. No LLM.
# Each entry: topic_tag → list of trigger patterns (regex or plain substrings)
# Ordered by specificity — first match wins per message.
# Grows over time as Gizmo encounters new domains.

_TOPIC_PATTERNS: list[tuple[str, list[str]]] = [
    # Emotional / wellbeing
    ("distress",        [r"\b(help|scared|panic|crisis|can't cope|overwhelmed|hurting)\b"]),
    ("sadness",         [r"\b(sad|cry|crying|depressed|grief|lost|miss|missing)\b"]),
    ("anger",           [r"\b(angry|furious|pissed|frustrated|annoyed|rage|mad)\b"]),
    ("anxiety",         [r"\b(anxious|anxiety|nervous|worried|worry|stress|stressed)\b"]),
    ("happiness",       [r"\b(happy|excited|great|amazing|love it|so good|delighted)\b"]),

    # Physical needs
    ("food",            [r"\b(hungry|hunger|eat|eating|food|meal|snack|cook|cooking|dinner|lunch|breakfast|recipe)\b"]),
    ("sleep",           [r"\b(tired|sleepy|sleep|nap|exhausted|rest|bed|bedtime|insomnia)\b"]),
    ("health",          [r"\b(sick|ill|pain|hurt|doctor|medicine|medication|symptom|headache|fever)\b"]),

    # Creative / making
    ("sewing",          [r"\b(sew|sewing|seam|stitch|fabric|thread|needle|hem|pattern|garment|dress|skirt)\b"]),
    ("fashion",         [r"\b(fashion|style|outfit|clothing|clothes|wear|wearing|color|colour|aesthetic|look)\b"]),
    ("art",             [r"\b(draw|drawing|paint|painting|sketch|art|illustrat|design|creative)\b"]),
    ("music",           [r"\b(music|song|sing|singing|listen|album|track|playlist|band|guitar|piano)\b"]),
    ("writing",         [r"\b(write|writing|story|poem|novel|chapter|draft|edit|prose|fiction)\b"]),

    # Technology
    ("coding",          [r"\b(code|coding|program|function|bug|debug|error|python|javascript|server|api|deploy)\b"]),
    ("hardware",        [r"\b(raspberry|pi|gpio|sensor|microphone|circuit|solder|hardware|device|wiring)\b"]),
    ("gizmo_dev",       [r"\b(gizmo|archivist|agent|pipeline|tensor|embedding|rag|chroma|llm|model)\b"]),
    ("technology",      [r"\b(tech|computer|phone|app|software|internet|wifi|network|device|screen)\b"]),

    # Social / relational
    ("relationship",    [r"\b(friend|family|partner|relationship|love|trust|fight|argument|together|apart)\b"]),
    ("work",            [r"\b(work|job|boss|coworker|colleague|office|meeting|project|deadline|salary)\b"]),
    ("social",          [r"\b(social|party|event|gathering|people|crowd|group|conversation|talk)\b"]),

    # Practical / logistics
    ("reminder",        [r"\b(remind|reminder|don't forget|remember to|at \d+|tomorrow|later today)\b"]),
    ("planning",        [r"\b(plan|planning|schedule|organise|organize|todo|list|goal|agenda)\b"]),
    ("question",        [r"\b(what|how|why|where|when|who|can you|could you|do you know|tell me)\b"]),

    # Reflective / philosophical
    ("reflection",      [r"\b(think|thinking|wonder|wondering|feel like|feels like|realise|realize|notice|noticed)\b"]),
    ("identity",        [r"\b(who am i|who are we|alter|headmate|front|fronting|system|plural|switch)\b"]),

    # Catch-all
    ("general",         []),  # always matches as fallback
]

# Compile patterns once
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (topic, [re.compile(p, re.IGNORECASE) for p in patterns])
    for topic, patterns in _TOPIC_PATTERNS
]


# ── Emotional register classifier ────────────────────────────────────────────
# Very lightweight — maps to one of five registers.
# The baseline reader will replace this with per-headmate deviation scoring later.

_REGISTER_PATTERNS = {
    "distress":  re.compile(
        r"\b(help|scared|panic|can't|cannot|overwhelm|hurt|crisis|please|desperate)\b",
        re.IGNORECASE
    ),
    "elevated":  re.compile(
        r"\b(angry|furious|pissed|hate|fucking|fuck|shit|annoyed|frustrated|rage)\b",
        re.IGNORECASE
    ),
    "positive":  re.compile(
        r"\b(great|amazing|love|happy|excited|wonderful|perfect|yes|awesome|good)\b",
        re.IGNORECASE
    ),
    "subdued":   re.compile(
        r"^.{0,40}$",   # very short messages often signal withdrawal or low energy
    ),
}

_DEFAULT_REGISTER = "neutral"


# ── Conversation field ────────────────────────────────────────────────────────

@dataclass
class ConversationField:
    """
    Per-session living field. Tracks topic temperature and participant history.
    Hot = active now. Warm = earlier this session. Cool = present but fading.

    v1: simple weight map with decay.
    Future: full tensor with cross-session links.
    """
    session_id: str
    topic_weights: dict[str, float] = field(default_factory=dict)
    participants: set = field(default_factory=set)
    message_count: int = 0
    last_topics: list[str] = field(default_factory=list)
    last_register: str = "neutral"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Temperature thresholds
    HOT_THRESHOLD:  float = 0.6
    WARM_THRESHOLD: float = 0.3
    DECAY_RATE:     float = 0.15   # per message, applied to all non-active topics

    def update(self, topics: list[str], participant: Optional[str] = None) -> None:
        """Integrate new topics — boost active, decay inactive."""
        # Decay all existing weights
        for t in list(self.topic_weights):
            self.topic_weights[t] = max(0.0, self.topic_weights[t] - self.DECAY_RATE)
            if self.topic_weights[t] == 0.0:
                del self.topic_weights[t]

        # Boost active topics
        for topic in topics:
            current = self.topic_weights.get(topic, 0.0)
            self.topic_weights[topic] = min(1.0, current + 0.4)

        if participant:
            self.participants.add(participant.lower())

        self.last_topics = topics
        self.message_count += 1
        self.updated_at = time.time()

    def temperature(self, topic: str) -> str:
        """Return hot / warm / cool / absent for a topic."""
        w = self.topic_weights.get(topic, 0.0)
        if w >= self.HOT_THRESHOLD:
            return "hot"
        elif w >= self.WARM_THRESHOLD:
            return "warm"
        elif w > 0.0:
            return "cool"
        return "absent"

    def hot_topics(self) -> list[str]:
        return [t for t, w in self.topic_weights.items() if w >= self.HOT_THRESHOLD]

    def warm_topics(self) -> list[str]:
        return [t for t, w in self.topic_weights.items()
                if self.WARM_THRESHOLD <= w < self.HOT_THRESHOLD]

    def snapshot(self) -> dict:
        """Plain dict snapshot for the brief."""
        return {
            "hot":          self.hot_topics(),
            "warm":         self.warm_topics(),
            "participants": list(self.participants),
            "message_count": self.message_count,
            "last_topics":  self.last_topics,
            "last_register": self.last_register,
        }


# ── Brief dataclass ───────────────────────────────────────────────────────────

@dataclass
class Brief:
    """
    Structured brief produced by the Archivist for the rest of the pipeline.
    Plain data — no language, no LLM, no strings intended for human reading.

    Consumed by: Mind (topic scope), Ego (context + register), Empath (participants).
    """
    # Message basics
    message:          str
    session_id:       str
    timestamp:        float
    headmate:         Optional[str]
    fronters:         list[str]

    # Classification
    topics:           list[str]
    emotional_register: str        # neutral / positive / elevated / distress / subdued
    is_directed_at_gizmo: bool     # explicit address ("hey gizmo", "gizmo,")
    is_question:      bool
    is_correction:    bool         # "that's wrong", "no, actually", etc.
    word_count:       int
    char_count:       int

    # Conversation field snapshot
    field_snapshot:   dict         # hot / warm / cool topic lists + participants

    # Source — who initiated this (user message, or which agent flagged it)
    source:           str          # "user" | "archivist" | "ego" | "mind" | etc.

    def to_dict(self) -> dict:
        return {
            "message":            self.message,
            "session_id":         self.session_id,
            "timestamp":          self.timestamp,
            "headmate":           self.headmate,
            "fronters":           self.fronters,
            "topics":             self.topics,
            "emotional_register": self.emotional_register,
            "is_directed":        self.is_directed_at_gizmo,
            "is_question":        self.is_question,
            "is_correction":      self.is_correction,
            "word_count":         self.word_count,
            "char_count":         self.char_count,
            "field":              self.field_snapshot,
            "source":             self.source,
        }


# ── Archivist ─────────────────────────────────────────────────────────────────

# Gizmo address patterns
_GIZMO_RE = re.compile(r"\b(gizmo|hey gizmo|giz)\b", re.IGNORECASE)

# Correction signal patterns
_CORRECTION_RE = re.compile(
    r"\b(that'?s wrong|no,? actually|not quite|incorrect|you'?re wrong|"
    r"that'?s not right|wait no|actually,?|i meant|i said|polar bears)\b",
    re.IGNORECASE,
)

# Question detection — ends with ? or starts with question word
_QUESTION_RE = re.compile(
    r"(\?$|^\s*(what|how|why|where|when|who|can you|could you|do you|is there|"
    r"are there|will you|would you|have you|did you))",
    re.IGNORECASE | re.MULTILINE,
)


class Archivist:
    """
    Singleton-style — one per process. Sessions are keyed by session_id.
    """

    def __init__(self):
        self._fields: dict[str, ConversationField] = {}
        log("Archivist", "initialised")

    def _get_field(self, session_id: str) -> ConversationField:
        if session_id not in self._fields:
            self._fields[session_id] = ConversationField(session_id=session_id)
            log_event("Archivist", "FIELD_CREATED", session=session_id[:8])
        return self._fields[session_id]

    # ── Classification ────────────────────────────────────────────────────────

    def _classify_topics(self, text: str) -> list[str]:
        """
        Extract 1-4 topic tags from text using compiled regex patterns.
        No LLM. Fast. Falls back to 'general' if nothing matches.
        """
        found = []
        for topic, patterns in _COMPILED:
            if topic == "general":
                continue
            for pattern in patterns:
                if pattern.search(text):
                    found.append(topic)
                    break
            if len(found) >= 4:
                break

        return found if found else ["general"]

    def _classify_register(self, text: str) -> str:
        """
        Classify emotional register. Returns one of:
        distress / elevated / positive / subdued / neutral
        Priority: distress > elevated > positive > subdued > neutral
        """
        for register, pattern in _REGISTER_PATTERNS.items():
            if pattern.search(text):
                return register
        return _DEFAULT_REGISTER

    # ── Main entry point ──────────────────────────────────────────────────────

    def receive(
        self,
        message: str,
        session_id: str,
        history,                      # ConversationHistory instance
        context: Optional[dict] = None,
        source: str = "user",
    ) -> Brief:
        """
        Receive a message, classify it, update the conversation field,
        save to history, and return a structured brief.

        This is the only public method most callers need.
        """
        t_start = time.monotonic()
        now = time.time()

        # Extract context
        ctx = context or {}
        headmate = ctx.get("current_host") or None
        fronters = list(ctx.get("fronters") or [])
        if headmate and headmate not in fronters:
            fronters.insert(0, headmate)

        # Classify
        topics    = self._classify_topics(message)
        register  = self._classify_register(message)
        directed  = bool(_GIZMO_RE.search(message))
        question  = bool(_QUESTION_RE.search(message.strip()))
        correction = bool(_CORRECTION_RE.search(message))

        # Update conversation field
        field = self._get_field(session_id)
        field.update(topics=topics, participant=headmate)
        field.last_register = register
        snapshot = field.snapshot()

        # Save to history (raw — no modification)
        try:
            history.add("user", message, context=ctx)
        except Exception as e:
            log_error("Archivist", "failed to save message to history", exc=e)

        # Build brief
        brief = Brief(
            message=message,
            session_id=session_id,
            timestamp=now,
            headmate=headmate,
            fronters=fronters,
            topics=topics,
            emotional_register=register,
            is_directed_at_gizmo=directed,
            is_question=question,
            is_correction=correction,
            word_count=len(message.split()),
            char_count=len(message),
            field_snapshot=snapshot,
            source=source,
        )

        # Log — structured and reflection-ready
        duration_ms = round((time.monotonic() - t_start) * 1000, 1)
        log_event(
            "Archivist", "RECEIVE",
            source=source,
            headmate=headmate or "unknown",
            topics=topics,
            register=register,
            directed=directed,
            question=question,
            correction=correction,
            words=brief.word_count,
            hot=snapshot["hot"],
            warm=snapshot["warm"],
            duration_ms=duration_ms,
        )

        return brief

    def receive_outgoing(
        self,
        message: str,
        session_id: str,
        history,
        context: Optional[dict] = None,
        source: str = "body",
    ) -> None:
        """
        Receive an outgoing message (what Gizmo just said).
        Updates the conversation field and logs — closes the loop.
        Does not build a full brief — outgoing messages are field updates only.
        """
        ctx = context or {}
        topics = self._classify_topics(message)

        field = self._get_field(session_id)
        field.update(topics=topics)

        try:
            history.add("assistant", message, context=ctx)
        except Exception as e:
            log_error("Archivist", "failed to save outgoing message to history", exc=e)

        log_event(
            "Archivist", "OUTGOING",
            source=source,
            topics=topics,
            words=len(message.split()),
            hot=field.hot_topics(),
        )

    def get_field(self, session_id: str) -> Optional[ConversationField]:
        """Return the conversation field for a session, if it exists."""
        return self._fields.get(session_id)

    def field_snapshot(self, session_id: str) -> dict:
        """Return a plain dict snapshot of the conversation field."""
        field = self._fields.get(session_id)
        if field is None:
            return {"hot": [], "warm": [], "participants": [], "message_count": 0,
                    "last_topics": [], "last_register": "neutral"}
        return field.snapshot()

    def active_sessions(self) -> list[str]:
        return list(self._fields.keys())


# ── Singleton ─────────────────────────────────────────────────────────────────
archivist = Archivist()