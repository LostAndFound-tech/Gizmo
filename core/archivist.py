"""
core/archivist.py
The Archivist — first receiver of all input, regardless of source.

Responsibilities:
  - Receive a message and classify it without LLM
  - Build a lightweight conversation field for the session
  - Package a structured brief (plain dict — no language)
  - Track topic temperature: hot / warm / cool
  - Save raw message to history
  - Log everything in structured, reflection-ready format
  - Detect host/fronter changes between turns

On every outgoing message, fires these async (fire-and-forget):
  1. Observer          — extract explicit facts, write to memory + JSON
  2. Inference         — every 2 user messages, ask LLM what we can learn
                         passes headmate persona so inference is contextual
  3. append_exchange   — write exchange to transcript file incrementally
  4. session_manager   — touch session, detect close signals
  5. message_store     — persist structured exchange for recall
  6. tagger            — async LLM tagging pass (topics, mood, cause, effect)

No LLM is called here directly. Ever. Classification is heuristic + keyword.
LLM calls are delegated to Observer, InferenceEngine, and Tagger.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

from core.log import log, log_event, log_error

# ── Topic keyword map ─────────────────────────────────────────────────────────
_TOPIC_PATTERNS: list[tuple[str, list[str]]] = [
    ("distress",        [r"\b(help|scared|panic|crisis|can't cope|overwhelmed|hurting)\b"]),
    ("sadness",         [r"\b(sad|cry|crying|depressed|grief|lost|miss|missing)\b"]),
    ("anger",           [r"\b(angry|furious|pissed|frustrated|annoyed|rage|mad)\b"]),
    ("anxiety",         [r"\b(anxious|anxiety|nervous|worried|worry|stress|stressed)\b"]),
    ("happiness",       [r"\b(happy|excited|great|amazing|love it|so good|delighted)\b"]),
    ("food",            [r"\b(hungry|hunger|eat|eating|food|meal|snack|cook|cooking|dinner|lunch|breakfast|recipe)\b"]),
    ("sleep",           [r"\b(tired|sleepy|sleep|nap|exhausted|rest|bed|bedtime|insomnia)\b"]),
    ("health",          [r"\b(sick|ill|pain|hurt|doctor|medicine|medication|symptom|headache|fever)\b"]),
    ("sewing",          [r"\b(sew|sewing|seam|stitch|fabric|thread|needle|hem|pattern|garment|dress|skirt)\b"]),
    ("fashion",         [r"\b(fashion|style|outfit|clothing|clothes|wear|wearing|color|colour|aesthetic|look)\b"]),
    ("art",             [r"\b(draw|drawing|paint|painting|sketch|art|illustrat|design|creative)\b"]),
    ("music",           [r"\b(music|song|sing|singing|listen|album|track|playlist|band|guitar|piano)\b"]),
    ("writing",         [r"\b(write|writing|story|poem|novel|chapter|draft|edit|prose|fiction)\b"]),
    ("coding",          [r"\b(code|coding|program|function|bug|debug|error|python|javascript|server|api|deploy)\b"]),
    ("hardware",        [r"\b(raspberry|pi|gpio|sensor|microphone|circuit|solder|hardware|device|wiring)\b"]),
    ("gizmo_dev",       [r"\b(gizmo|archivist|agent|pipeline|tensor|embedding|rag|chroma|llm|model)\b"]),
    ("technology",      [r"\b(tech|computer|phone|app|software|internet|wifi|network|device|screen)\b"]),
    ("relationship",    [r"\b(friend|family|partner|relationship|love|trust|fight|argument|together|apart)\b"]),
    ("work",            [r"\b(work|job|boss|coworker|colleague|office|meeting|project|deadline|salary)\b"]),
    ("social",          [r"\b(social|party|event|gathering|people|crowd|group|conversation|talk)\b"]),
    ("reminder",        [r"\b(remind|reminder|don't forget|remember to|at \d+|tomorrow|later today)\b"]),
    ("planning",        [r"\b(plan|planning|schedule|organise|organize|todo|list|goal|agenda)\b"]),
    ("question",        [r"\b(what|how|why|where|when|who|can you|could you|do you know|tell me)\b"]),
    ("reflection",      [r"\b(think|thinking|wonder|wondering|feel like|feels like|realise|realize|notice|noticed)\b"]),
    ("identity",        [r"\b(who am i|who are we|alter|headmate|front|fronting|system|plural|switch)\b"]),
    ("general",         []),
]

_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (topic, [re.compile(p, re.IGNORECASE) for p in patterns])
    for topic, patterns in _TOPIC_PATTERNS
]

# ── Emotional register classifier ─────────────────────────────────────────────
_REGISTER_PATTERNS = {
    "distress": re.compile(
        r"\b(help|scared|panic|can't|cannot|overwhelm|hurt|crisis|please|desperate)\b",
        re.IGNORECASE
    ),
    "elevated": re.compile(
        r"\b(angry|furious|pissed|hate|fucking|fuck|shit|annoyed|frustrated|rage)\b",
        re.IGNORECASE
    ),
    "positive": re.compile(
        r"\b(great|amazing|love|happy|excited|wonderful|perfect|yes|awesome|good)\b",
        re.IGNORECASE
    ),
    "subdued": re.compile(r"^.{0,40}$"),
}
_DEFAULT_REGISTER = "neutral"


# ── Conversation field ────────────────────────────────────────────────────────

@dataclass
class ConversationField:
    session_id:    str
    topic_weights: dict  = field(default_factory=dict)
    participants:  set   = field(default_factory=set)
    message_count: int   = 0
    last_topics:   list  = field(default_factory=list)
    last_register: str   = "neutral"
    created_at:    float = field(default_factory=time.time)
    updated_at:    float = field(default_factory=time.time)

    HOT_THRESHOLD:  float = 0.6
    WARM_THRESHOLD: float = 0.3
    DECAY_RATE:     float = 0.15

    def update(self, topics: list, participant: Optional[str] = None) -> None:
        for t in list(self.topic_weights):
            self.topic_weights[t] = max(0.0, self.topic_weights[t] - self.DECAY_RATE)
            if self.topic_weights[t] == 0.0:
                del self.topic_weights[t]
        for topic in topics:
            current = self.topic_weights.get(topic, 0.0)
            self.topic_weights[topic] = min(1.0, current + 0.4)
        if participant:
            self.participants.add(participant.lower())
        self.last_topics  = topics
        self.message_count += 1
        self.updated_at   = time.time()

    def hot_topics(self) -> list:
        return [t for t, w in self.topic_weights.items() if w >= self.HOT_THRESHOLD]

    def warm_topics(self) -> list:
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
    message:              str
    session_id:           str
    timestamp:            float
    headmate:             Optional[str]
    fronters:             list

    topics:               list
    emotional_register:   str
    is_directed_at_gizmo: bool
    is_question:          bool
    is_correction:        bool
    word_count:           int
    char_count:           int

    field_snapshot:       dict
    source:               str

    host_changed:         bool          = False
    previous_host:        Optional[str] = None
    fronters_joined:      list          = field(default_factory=list)
    fronters_left:        list          = field(default_factory=list)

    valence:              float = 0.0
    intensity:            float = 0.2
    chaos:                float = 0.0
    emotion_arc_block:    str   = ""
    stage_directions:     list  = field(default_factory=list)
    lore:                 list  = field(default_factory=list)

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
            "stage_directions":   self.stage_directions,
            "lore":               self.lore,
        }


# ── Inference helpers ─────────────────────────────────────────────────────────

def _load_headmate_persona(headmate: str) -> str:
    if not headmate:
        return ""
    try:
        personality_dir = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
        path = personality_dir / "headmates" / f"{headmate.lower()}.json"
        if not path.exists():
            return ""
        data  = json.loads(path.read_text(encoding="utf-8"))
        prefs = data.get("interaction_prefs", {})
        persona  = prefs.get("persona", "")
        explicit = prefs.get("explicit", [])
        parts = []
        if persona:
            parts.append(persona)
        if explicit:
            parts.extend(explicit)
        return "\n".join(parts) if parts else ""
    except Exception:
        return ""


async def _run_inference(
    recent_messages: list,
    current_host: str,
    fronters: list,
    session_id: str,
    resolved_subject: str,
    llm,
) -> None:
    if not recent_messages:
        return

    persona = _load_headmate_persona(resolved_subject or current_host)

    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content'][:300]}"
        for m in recent_messages
        if m.get("role") in ("user", "assistant")
    )

    persona_context = (
        f"\nWhat I know about how to engage with {resolved_subject}:\n{persona}\n"
        if persona else ""
    )

    prompt = [{
        "role": "user",
        "content": (
            f"Based on this conversation, what can we learn about {resolved_subject}?\n"
            f"{persona_context}\n"
            f"Conversation:\n{transcript}\n\n"
            f"Rules:\n"
            f"- Infer beyond explicit statements — word choice, tone, concerns, humor, values\n"
            f"- Be specific. 'Finds comfort in dark humor' beats 'has a sense of humor'\n"
            f"- Only infer what is genuinely supported\n"
            f"- Skip obvious filler. Skip anything already clear from the persona context\n"
            f"- Respond with ONLY a JSON array of strings, no markdown:\n"
            f'["inference one", "inference two", ...]'
        )
    }]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You extract implicit inferences about a person from conversation. "
                "JSON array of strings only. No markdown. No preamble. "
                "Each string is a single, specific, grounded inference."
            ),
            max_new_tokens=400,
            temperature=0.3,
        )

        raw = raw.strip().strip("```json").strip("```").strip()
        inferences = json.loads(raw)

        if not isinstance(inferences, list):
            return

        from tools.memory_tool import _get_collection, MEMORY_COLLECTION
        from datetime import datetime
        import uuid

        col = _get_collection(MEMORY_COLLECTION)
        now = datetime.now().isoformat(timespec="seconds")
        count = 0

        for text in inferences:
            text = text.strip()
            if not text:
                continue
            col.add(
                documents=[text],
                metadatas=[{
                    "subject":    (resolved_subject or current_host or "").lower(),
                    "type":       "observation",
                    "written_at": now,
                    "session_id": session_id,
                    "source":     "inference",
                    "tags":       f"observation,inference,{(resolved_subject or '').lower()}",
                }],
                ids=[f"inf_{uuid.uuid4().hex[:12]}"],
            )
            count += 1

        log_event("Archivist", "INFERENCE_COMPLETE",
            session=session_id[:8],
            subject=resolved_subject or current_host,
            count=count,
        )

    except Exception as e:
        log_error("Archivist", "inference failed", exc=e)


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

    def __init__(self):
        self._fields:            dict[str, ConversationField] = {}
        self._last_context:      dict[str, dict]  = {}
        self._user_msg_counts:   dict[str, int]   = {}
        self._inference_subject: dict[str, str]   = {}
        log("Archivist", "initialised")

    def _get_field(self, session_id: str) -> ConversationField:
        if session_id not in self._fields:
            self._fields[session_id] = ConversationField(session_id=session_id)
            log_event("Archivist", "FIELD_CREATED", session=session_id[:8])
        return self._fields[session_id]

    def _classify_topics(self, text: str) -> list:
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

    def _detect_changes(self, session_id: str, context: dict) -> dict:
        last = self._last_context.get(session_id, {})

        current_host     = context.get("current_host", "")
        last_host        = last.get("current_host", "")
        current_fronters = set(context.get("fronters") or [])
        last_fronters    = set(last.get("fronters") or [])

        host_changed    = bool(current_host and current_host != last_host and last_host)
        previous_host   = last_host if host_changed else None
        fronters_joined = list(current_fronters - last_fronters)
        fronters_left   = list(last_fronters - current_fronters)

        self._last_context[session_id] = dict(context)

        if host_changed:
            log_event("Archivist", "HOST_CHANGED",
                session=session_id[:8], previous=last_host, current=current_host)
        if fronters_joined:
            log_event("Archivist", "FRONTERS_JOINED",
                session=session_id[:8], joined=fronters_joined)
        if fronters_left:
            log_event("Archivist", "FRONTERS_LEFT",
                session=session_id[:8], left=fronters_left)

        return {
            "host_changed":    host_changed,
            "previous_host":   previous_host,
            "fronters_joined": fronters_joined,
            "fronters_left":   fronters_left,
        }

    def _resolve_inference_subject(
        self,
        session_id: str,
        message: str,
        current_host: str,
        known_headmates: set,
    ) -> str:
        msg_lower = message.lower()
        speaker   = (current_host or "").lower()

        current_subject = self._inference_subject.get(session_id, current_host or "")

        if speaker and speaker in msg_lower:
            current_subject = current_host or ""

        for name in known_headmates:
            if name != speaker and name in msg_lower:
                current_subject = name
                break

        self._inference_subject[session_id] = current_subject
        return current_subject

    # ── Main entry point ──────────────────────────────────────────────────────

    def receive(
        self,
        message: str,
        session_id: str,
        history,
        context: Optional[dict] = None,
        source: str = "user",
    ) -> Brief:
        t_start = time.monotonic()
        now     = time.time()
        ctx     = context or {}

        # Host tracker — scan for identification signals
        try:
            from core.host_tracker import host_tracker
            from core.ego import _get_known_headmates
            known_headmates = _get_known_headmates()
            host_tracker.process_message(session_id, message, known_headmates)
            updated_ctx = host_tracker.get_context(session_id)
            if not updated_ctx.get("timezone") and ctx.get("timezone"):
                updated_ctx["timezone"] = ctx["timezone"]
            ctx = updated_ctx
        except Exception as e:
            log_error("Archivist", "host_tracker processing failed", exc=e)

        headmate = ctx.get("current_host") or None
        fronters = list(ctx.get("fronters") or [])
        if headmate and headmate not in [f.lower() for f in fronters]:
            fronters.insert(0, headmate)

        changes = self._detect_changes(session_id, ctx)

        topics     = self._classify_topics(message)
        register   = self._classify_register(message)
        directed   = bool(_GIZMO_RE.search(message))
        question   = bool(_QUESTION_RE.search(message.strip()))
        correction = bool(_CORRECTION_RE.search(message))

        # Emotion tracking
        emotion_point = None
        try:
            from core.emotion_tracker import emotion_tracker
            emotion_point = emotion_tracker.record(
                session_id=session_id,
                message=message,
                headmate=headmate,
                register=register,
                topic=topics[0] if topics else "general",
                timestamp=now,
            )
        except Exception as e:
            log_error("Archivist", "emotion_tracker.record failed", exc=e)

        conv_field = self._get_field(session_id)
        conv_field.update(topics=topics, participant=headmate)
        for f in fronters:
            conv_field.participants.add(f.lower())
        conv_field.last_register = register
        snapshot = conv_field.snapshot()

        try:
            history.add("user", message, context=ctx)
        except Exception as e:
            log_error("Archivist", "failed to save message to history", exc=e)

        arc_block    = ""
        ep_valence   = 0.0
        ep_intensity = 0.2
        ep_chaos     = 0.0
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
        log_event("Archivist", "RECEIVE",
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

        # Close signal detection
        try:
            from core.session_manager import session_manager, is_close_signal
            if is_close_signal(message):
                session_manager.signal_close(session_id)
                log_event("Archivist", "CLOSE_SIGNAL_DETECTED",
                    session=session_id[:8], preview=message[:40])
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
        user_message: str = "",
    ) -> None:
        ctx      = context or {}
        topics   = self._classify_topics(message)
        fronters = list(ctx.get("fronters") or [])
        headmate = ctx.get("current_host") or (fronters[0] if fronters else "")

        conv_field = self._get_field(session_id)
        conv_field.update(topics=topics)

        # Save to history
        try:
            history.add("assistant", message, context=ctx)
        except Exception as e:
            log_error("Archivist", "failed to save outgoing message to history", exc=e)

        # Append exchange to transcript incrementally
        try:
            from core.conversation_archive import append_exchange
            from core.session_manager import session_manager as _sm
            state     = _sm._sessions.get(session_id)
            opened_at = state.opened_at if state else time.time()
            append_exchange(
                session_id=session_id,
                opened_at=opened_at,
                user_message=user_message,
                gizmo_response=message,
                host=headmate,
                timestamp=time.time(),
            )
        except Exception as e:
            log_error("Archivist", "append_exchange failed", exc=e)

        # Touch session manager
        try:
            from core.session_manager import session_manager
            session_manager.touch(
                session_id=session_id,
                hosts=fronters,
                topics=topics,
            )
        except Exception as e:
            log_error("Archivist", "session_manager.touch failed", exc=e)

        # Message store — persist structured exchange
        msg_id = None
        try:
            from core.message_store import insert_exchange
            msg_id = insert_exchange(
                session_id         = session_id,
                timestamp          = time.time(),
                host               = headmate,
                fronters           = fronters,
                user_message       = user_message,
                gizmo_response     = message,
                topics             = topics,
                emotional_register = "neutral",
                mood               = "neutral",
                tags               = topics,
                notable            = False,
                stage_directions   = [],
                lore               = [],
            )
        except Exception as e:
            log_error("Archivist", "message_store insert failed", exc=e)

        # Tagger — async LLM tagging pass
        if msg_id:
            try:
                from core.tagger import tag_exchange
                from core.llm import llm as _llm
                asyncio.ensure_future(
                    tag_exchange(
                        msg_id         = msg_id,
                        session_id     = session_id,
                        user_message   = user_message,
                        gizmo_response = message,
                        host           = headmate,
                        fronters       = fronters,
                        prior_topics   = list(conv_field.last_topics),
                        prior_mood     = "",
                        llm            = _llm,
                    )
                )
            except Exception as e:
                log_error("Archivist", "tagger failed to schedule", exc=e)

        log_event("Archivist", "OUTGOING",
            source=source,
            topics=topics,
            words=len(message.split()),
            hot=conv_field.hot_topics(),
        )

        if not fronters or not user_message:
            return

        try:
            from core.llm import llm

            # ── 1. Observer — explicit fact extraction ────────────────────────
            from core.observer import observe
            asyncio.ensure_future(
                observe(
                    user_message=user_message,
                    gizmo_response=message,
                    fronters=fronters,
                    session_id=session_id,
                    llm=llm,
                )
            )

            # ── 2. Inference — every 2 user messages ─────────────────────────
            self._user_msg_counts[session_id] = (
                self._user_msg_counts.get(session_id, 0) + 1
            )

            if self._user_msg_counts[session_id] % 2 == 0:
                try:
                    from core.ego import _get_known_headmates
                    known = _get_known_headmates()
                except Exception:
                    known = set()

                resolved_subject = self._resolve_inference_subject(
                    session_id=session_id,
                    message=user_message,
                    current_host=headmate,
                    known_headmates=known,
                )

                try:
                    recent = history.as_list()[-8:]
                except Exception:
                    recent = []

                asyncio.ensure_future(
                    _run_inference(
                        recent_messages=recent,
                        current_host=headmate,
                        fronters=fronters,
                        session_id=session_id,
                        resolved_subject=resolved_subject,
                        llm=llm,
                    )
                )

                log_event("Archivist", "INFERENCE_FIRED",
                    session=session_id[:8],
                    subject=resolved_subject or headmate,
                    msg_count=self._user_msg_counts[session_id],
                )

        except Exception as e:
            log_error("Archivist", "background tasks failed to schedule", exc=e)

    def get_field(self, session_id: str) -> Optional[ConversationField]:
        return self._fields.get(session_id)

    def field_snapshot(self, session_id: str) -> dict:
        conv_field = self._fields.get(session_id)
        if conv_field is None:
            return {
                "hot": [], "warm": [], "participants": [],
                "message_count": 0, "last_topics": [], "last_register": "neutral"
            }
        return conv_field.snapshot()

    def active_sessions(self) -> list:
        return list(self._fields.keys())


# ── Singleton ─────────────────────────────────────────────────────────────────
archivist = Archivist()