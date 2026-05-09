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
  - Detect host/fronter changes between turns

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

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from core.log import log, log_event, log_error

# ── Topic keyword map ─────────────────────────────────────────────────────────
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
    ("general",         []),
]

_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (topic, [re.compile(p, re.IGNORECASE) for p in patterns])
    for topic, patterns in _TOPIC_PATTERNS
]

# ── Emotional register classifier ─────────────────────────────────────────────
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
        r"^.{0,40}$",
    ),
}

_DEFAULT_REGISTER = "neutral"


# ── Conversation field ────────────────────────────────────────────────────────

@dataclass
class ConversationField:
    """
    Per-session living field. Tracks topic temperature and participant history.
    Hot = active now. Warm = earlier this session. Cool = present but fading.
    """
    session_id: str
    topic_weights: dict[str, float] = field(default_factory=dict)
    participants: set = field(default_factory=set)
    message_count: int = 0
    last_topics: list[str] = field(default_factory=list)
    last_register: str = "neutral"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    HOT_THRESHOLD:  float = 0.6
    WARM_THRESHOLD: float = 0.3
    DECAY_RATE:     float = 0.15

    def update(self, topics: list[str], participant: Optional[str] = None) -> None:
        for t in list(self.topic_weights):
            self.topic_weights[t] = max(0.0, self.topic_weights[t] - self.DECAY_RATE)
            if self.topic_weights[t] == 0.0:
                del self.topic_weights[t]

        for topic in topics:
            current = self.topic_weights.get(topic, 0.0)
            self.topic_weights[topic] = min(1.0, current + 0.4)

        if participant:
            self.participants.add(participant.lower())

        self.last_topics = topics
        self.message_count += 1
        self.updated_at = time.time()

    def temperature(self, topic: str) -> str:
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
        return {
            "hot":           self.hot_topics(),
            "warm":          self.warm_topics(),
            "participants":  list(self.participants),
            "message_count": self.message_count,
            "last_topics":   self.last_topics,
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
    message:            str
    session_id:         str
    timestamp:          float
    headmate:           Optional[str]
    fronters:           list[str]

    # Classification
    topics:             list[str]
    emotional_register: str
    is_directed_at_gizmo: bool
    is_question:        bool
    is_correction:      bool
    word_count:         int
    char_count:         int

    # Conversation field snapshot
    field_snapshot:     dict

    # Source
    source:             str

    # Host/fronter change detection — populated by Archivist, consumed by Ego
    host_changed:       bool = False
    previous_host:      Optional[str] = None
    fronters_joined:    list[str] = field(default_factory=list)
    fronters_left:      list[str] = field(default_factory=list)

    # Emotional state — populated by Archivist via emotion_tracker
    valence:            float = 0.0
    intensity:          float = 0.2
    chaos:              float = 0.0
    emotion_arc_block:  str   = ""   # pre-formatted for Ego's system prompt

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
            "host_changed":       self.host_changed,
            "previous_host":      self.previous_host,
            "fronters_joined":    self.fronters_joined,
            "fronters_left":      self.fronters_left,
        }


# ── Archivist ─────────────────────────────────────────────────────────────────

_GIZMO_RE = re.compile(r"\b(gizmo|hey gizmo|giz)\b", re.IGNORECASE)
_CORRECTION_RE = re.compile(
    r"\b(that'?s wrong|no,? actually|not quite|incorrect|you'?re wrong|"
    r"that'?s not right|wait no|actually,?|i meant|i said|polar bears)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"(\?$|^\s*(what|how|why|where|when|who|can you|could you|do you|is there|"
    r"are there|will you|would you|have you|did you))",
    re.IGNORECASE | re.MULTILINE,
)


class Archivist:
    """
    Singleton-style — one per process. Sessions are keyed by session_id.
    Also owns host/fronter change detection — last context per session lives here.
    """

    def __init__(self):
        self._fields: dict[str, ConversationField] = {}
        self._last_context: dict[str, dict] = {}  # session_id → last known context
        log("Archivist", "initialised")

    def _get_field(self, session_id: str) -> ConversationField:
        if session_id not in self._fields:
            self._fields[session_id] = ConversationField(session_id=session_id)
            log_event("Archivist", "FIELD_CREATED", session=session_id[:8])
        return self._fields[session_id]

    # ── Classification ────────────────────────────────────────────────────────

    def _classify_topics(self, text: str) -> list[str]:
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
        for register, pattern in _REGISTER_PATTERNS.items():
            if pattern.search(text):
                return register
        return _DEFAULT_REGISTER

    # ── Host change detection ─────────────────────────────────────────────────

    def _detect_changes(self, session_id: str, context: dict) -> dict:
        """
        Compare current context to last known context for this session.
        Returns a changes dict consumed by Brief and Ego.
        """
        last = self._last_context.get(session_id, {})

        current_host = context.get("current_host", "")
        last_host    = last.get("current_host", "")

        current_fronters = set(context.get("fronters") or [])
        last_fronters    = set(last.get("fronters") or [])

        host_changed    = bool(current_host and current_host != last_host and last_host)
        previous_host   = last_host if host_changed else None
        fronters_joined = list(current_fronters - last_fronters)
        fronters_left   = list(last_fronters - current_fronters)

        # Update stored context
        self._last_context[session_id] = dict(context)

        if host_changed:
            log_event("Archivist", "HOST_CHANGED",
                session=session_id[:8],
                previous=last_host,
                current=current_host,
            )
        if fronters_joined:
            log_event("Archivist", "FRONTERS_JOINED",
                session=session_id[:8],
                joined=fronters_joined,
            )
        if fronters_left:
            log_event("Archivist", "FRONTERS_LEFT",
                session=session_id[:8],
                left=fronters_left,
            )

        return {
            "host_changed":    host_changed,
            "previous_host":   previous_host,
            "fronters_joined": fronters_joined,
            "fronters_left":   fronters_left,
        }

    # ── Main entry point ──────────────────────────────────────────────────────

    def receive(
        self,
        message: str,
        session_id: str,
        history,
        context: Optional[dict] = None,
        source: str = "user",
    ) -> Brief:
        """
        Receive a message, classify it, detect host changes, update the
        conversation field, save to history, and return a structured brief.

        Host identification and fronter presence are detected here via
        host_tracker — context passed in is the server-side state, which
        gets updated in-place if the message contains identification signals.
        """
        t_start = time.monotonic()
        now = time.time()

        ctx = context or {}

        # ── Host tracker: scan message for identification signals ──────────────
        # Do this BEFORE reading headmate/fronters from ctx, so if someone
        # identifies themselves in this very message, the brief reflects it.
        try:
            from core.host_tracker import host_tracker
            from core.ego import _get_known_headmates
            known_headmates = _get_known_headmates()
            tracker_changes = host_tracker.process_message(
                session_id, message, known_headmates
            )
            # Pull updated context after tracker has processed the message
            updated_ctx = host_tracker.get_context(session_id)
            # Preserve timezone from original ctx if tracker doesn't have it
            if not updated_ctx.get("timezone") and ctx.get("timezone"):
                updated_ctx["timezone"] = ctx["timezone"]
            ctx = updated_ctx
        except Exception as e:
            log_error("Archivist", "host_tracker processing failed", exc=e)

        headmate = ctx.get("current_host") or None
        fronters = list(ctx.get("fronters") or [])
        if headmate and headmate not in [f.lower() for f in fronters]:
            fronters.insert(0, headmate)

        # Detect host/fronter changes BEFORE updating stored context
        changes = self._detect_changes(session_id, ctx)

        # Classify
        topics     = self._classify_topics(message)
        register   = self._classify_register(message)
        directed   = bool(_GIZMO_RE.search(message))
        question   = bool(_QUESTION_RE.search(message.strip()))
        correction = bool(_CORRECTION_RE.search(message))

        # ── Emotion tracking ──────────────────────────────────────────────────
        emotion_point = None
        try:
            from core.emotion_tracker import emotion_tracker
            primary_topic = topics[0] if topics else "general"
            emotion_point = emotion_tracker.record(
                session_id=session_id,
                message=message,
                headmate=headmate,
                register=register,
                topic=primary_topic,
                timestamp=now,
            )
        except Exception as e:
            log_error("Archivist", "emotion_tracker.record failed", exc=e)

        # Update conversation field — all active fronters warm their collections
        conv_field = self._get_field(session_id)
        conv_field.update(topics=topics, participant=headmate)
        for f in fronters:
            conv_field.participants.add(f.lower())
        conv_field.last_register = register
        snapshot = conv_field.snapshot()

        # Save to history
        try:
            history.add("user", message, context=ctx)
        except Exception as e:
            log_error("Archivist", "failed to save message to history", exc=e)

        # Build brief
        # Pull emotion arc block for Ego
        arc_block = ""
        ep_valence = 0.0
        ep_intensity = 0.2
        ep_chaos = 0.0
        try:
            from core.emotion_tracker import emotion_tracker
            arc_block = emotion_tracker.ego_block(session_id)
            if emotion_point:
                ep_valence   = emotion_point.valence
                ep_intensity = emotion_point.intensity
                ep_chaos     = emotion_point.chaos
        except Exception:
            pass

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
            host_changed=changes["host_changed"],
            previous_host=changes["previous_host"],
            fronters_joined=changes["fronters_joined"],
            fronters_left=changes["fronters_left"],
            valence=ep_valence,
            intensity=ep_intensity,
            chaos=ep_chaos,
            emotion_arc_block=arc_block,
        )

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
            host_changed=changes["host_changed"],
            duration_ms=duration_ms,
        )

        try:
            from core.session_manager import session_manager, is_close_signal
            if is_close_signal(message):
                session_manager.signal_close(session_id)
                log_event("Archivist", "CLOSE_SIGNAL_DETECTED",
                    session=session_id[:8],
                    preview=message[:40],
                )
        except Exception as e:
            log_error("Archivist", "close signal detection failed", exc=e)

        try:
            from core.session_manager import session_manager, is_close_signal
            if is_close_signal(message):
                session_manager.signal_close(session_id)
                log_event("Archivist", "CLOSE_SIGNAL_DETECTED",
                    session=session_id[:8],
                    preview=message[:40],
                )
        except Exception as e:
            log_error("Archivist", "close signal detection failed", exc=e)
            
        return brief

    def receive_outgoing(
        self,
        message: str,
        session_id: str,
        history,
        context: Optional[dict] = None,
        source: str = "body",
        user_message: str = "",  # the message that triggered this response
    ) -> None:
        """
        Receive an outgoing message (what Gizmo just said).
        Updates the conversation field, logs, and fires background observation.
        """
        ctx = context or {}
        topics = self._classify_topics(message)

        conv_field = self._get_field(session_id)
        conv_field.update(topics=topics)

        try:
            history.add("assistant", message, context=ctx)
        except Exception as e:
            log_error("Archivist", "failed to save outgoing message to history", exc=e)
        try:
            from core.conversation_archive import append_exchange
            from core.session_manager import session_manager
            state = session_manager._sessions.get(session_id)
            opened_at = state.opened_at if state else time.time()
            fronters = list(ctx.get("fronters") or [])
            host = fronters[0] if fronters else ""
            append_exchange(
                session_id=session_id,
                opened_at=opened_at,
                user_message=user_message,
                gizmo_response=message,
                host=host,
                timestamp=time.time(),
            )
            try:
                from core.session_manager import session_manager
                session_manager.touch(
                    session_id=session_id,
                    hosts=fronters,
                    topics=topics,
                )
            except Exception as e:
                log_error("Archivist", "session_manager.touch failed", exc=e)
        except Exception as e:
            log_error("Archivist", "append_exchange failed", exc=e)

        log_event(
            "Archivist", "OUTGOING",
            source=source,
            topics=topics,
            words=len(message.split()),
            hot=conv_field.hot_topics(),
        )

        # ── Background observation — fire and forget ──────────────────────────
        # Extract facts from this exchange and write to entity files.
        # Runs async so it never blocks the response pipeline.
        fronters = list(ctx.get("fronters") or [])
        if fronters and user_message:
            try:
                from core.observer import observe
                from core.llm import llm
                asyncio.ensure_future(
                    observe(
                        user_message=user_message,
                        gizmo_response=message,
                        fronters=fronters,
                        session_id=session_id,
                        llm=llm,
                    )
                )
            except Exception as e:
                log_error("Archivist", "failed to schedule observation", exc=e)
            
        try:
            from core.session_manager import session_manager
            session_manager.touch(
                session_id=session_id,
                hosts=fronters,
                topics=topics,
            )
        except Exception as e:
            log_error("Archivist", "session_manager.touch failed", exc=e)

    def get_field(self, session_id: str) -> Optional[ConversationField]:
        return self._fields.get(session_id)

    def field_snapshot(self, session_id: str) -> dict:
        conv_field = self._fields.get(session_id)
        if conv_field is None:
            return {"hot": [], "warm": [], "participants": [], "message_count": 0,
                    "last_topics": [], "last_register": "neutral"}
        return conv_field.snapshot()

    def active_sessions(self) -> list[str]:
        return list(self._fields.keys())


# ── Singleton ─────────────────────────────────────────────────────────────────
archivist = Archivist()