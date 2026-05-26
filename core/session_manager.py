"""
core/session_manager.py
Live session context manager.

Two jobs:

1. SESSION LIFECYCLE
   Open, track, close, archive sessions.
   Watchdog closes cold sessions after timeout.
   Rumination fires every N ticks during quiet time.

2. LIVE SESSION CONTEXT (the new part)
   Holds the working picture for each active session.
   Acts as RAM — hot data immediately accessible.
   Store is disk — everything ever known, persistent.

   Entity cache:
     Every entity mentioned this session is pre-loaded from store
     and held in memory. Agents ask the session manager, not the
     store directly. Cache hit = instant. Miss = load from store,
     cache, return.

   Entity detection:
     One LLM call per message. "Given this message, connect each
     subject to the speaker using a relational word."
     Output feeds the cache immediately — before parallel agents run.

   Dirty tracking:
     Facts/entities updated mid-session are flagged dirty.
     Written back to store at session close, not on every update.
     20 messages about the dress → one store write at the end.

   Pre-loading on connect:
     When a session opens, pre-load:
       - Headmate entity + top facts
       - Active patterns (above confidence threshold)
       - Pending questions for this headmate
       - Most recently mentioned entities from last 3 sessions
     Cache is warm before first message arrives.

Session context structure:
  SessionContext
    identity          — headmate, fronters, confidence, since-when
    entities          — {entity_id: EntityCache}
    hot_facts         — pre-loaded facts for current headmate
    active_threads    — topics with signal strength, building/fading
    emotional_arc     — rolling window of valence/intensity/chaos
    pending_questions — questions ready to surface
    active_patterns   — patterns pre-loaded from store
    dirty_entities    — entity IDs with unsaved updates
    message_count     — messages this session
    opened_at         — session start timestamp
    last_seen         — last message timestamp
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

from core.log import log, log_event, log_error
from core.timezone import tz_now

# ── Config ────────────────────────────────────────────────────────────────────

SESSION_TIMEOUT    = 15 * 60        # 15 minutes silence → close
WATCHDOG_INTERVAL  = 60             # check every minute
RUMINATION_EVERY   = 15             # ruminate every N ticks (~15 min)

_CLOSE_RE = re.compile(
    r"\b(bye|goodbye|goodnight|good night|night|talk (to you )?later|"
    r"see you|ttyl|gotta go|i'm out|heading out|going to (bed|sleep)|"
    r"talk soon|take care|catch you later|later gator|peace out)\b",
    re.IGNORECASE,
)


# ── Entity cache entry ────────────────────────────────────────────────────────

@dataclass
class EntityCache:
    """
    One entity's working data for this session.
    Loaded from store on first mention, updated as conversation develops.
    """
    entity_id:    str
    name:         str
    entity_type:  str               # headmate/external/object/concept/place/event/pet
    owner:        Optional[str]     # who owns this entity, if applicable
    owner_type:   Optional[str]

    # Known data (loaded from store)
    facts:        list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    wellbeing:    list = field(default_factory=list)

    # Session-specific data (accumulated this session)
    session_facts:       list = field(default_factory=list)
    session_relations:   list = field(default_factory=list)
    emotional_loading:   dict = field(default_factory=dict)  # {emotion: valence}

    # Session tracking
    first_mentioned: float = field(default_factory=time.time)
    last_mentioned:  float = field(default_factory=time.time)
    mention_count:   int   = 0
    dirty:           bool  = False  # has unsaved updates

    def touch(self) -> None:
        self.last_mentioned = time.time()
        self.mention_count += 1

    def add_fact(self, fact: str, source: str = "session") -> None:
        if fact and fact not in self.session_facts:
            self.session_facts.append(fact)
            self.dirty = True

    def add_relation(self, relation: dict) -> None:
        self.session_relations.append(relation)
        self.dirty = True

    def add_emotion(self, emotion: str, valence: float) -> None:
        self.emotional_loading[emotion] = valence
        self.dirty = True

    def to_summary(self) -> str:
        """Compact summary for agent context."""
        parts = [f"{self.entity_type}: {self.name}"]
        if self.owner:
            parts.append(f"owned by {self.owner}")
        all_facts = (self.facts + self.session_facts)[:5]
        if all_facts:
            parts.append("facts: " + "; ".join(f[:60] for f in all_facts))
        if self.emotional_loading:
            emo = ", ".join(f"{e}({v:+.1f})" for e, v in self.emotional_loading.items())
            parts.append(f"emotional: {emo}")
        if self.relationships:
            rels = self.relationships[:3]
            rel_str = "; ".join(
                f"{r.get('speaker')} → {r.get('relationship_label')} → {r.get('entity')}"
                for r in rels if r.get("relationship_label")
            )
            if rel_str:
                parts.append(f"relations: {rel_str}")
        return " | ".join(parts)


# ── Session context ───────────────────────────────────────────────────────────

@dataclass
class SessionContext:
    """
    Full working picture for one active session.
    Lives in memory. Written to store at close.
    """
    session_id:   str
    opened_at:    float = field(default_factory=time.time)
    last_seen:    float = field(default_factory=time.time)

    # Identity
    headmate:        Optional[str] = None
    fronters:        list = field(default_factory=list)
    host_confidence: float = 0.0
    host_updated_at: float = field(default_factory=time.time)

    # Entity cache — keyed by lowercase name
    entities:        dict = field(default_factory=dict)   # name → EntityCache
    dirty_entities:  set  = field(default_factory=set)
    unknown_entities: set = field(default_factory=set)    # names never seen before

    # Pre-loaded hot data
    hot_facts:       list = field(default_factory=list)   # top facts for headmate
    active_patterns: list = field(default_factory=list)   # patterns above threshold
    pending_questions: list = field(default_factory=list) # questions to surface

    # Conversational state
    active_threads:  dict = field(default_factory=dict)   # topic → signal strength
    emotional_arc:   list = field(default_factory=list)   # recent EmotionPoints
    message_count:   int  = 0
    topics_seen:     list = field(default_factory=list)

    # Session close
    closed:          bool = False

    def touch(self, headmate: str = None, fronters: list = None) -> None:
        self.last_seen = time.time()
        self.message_count += 1
        if headmate:
            self.headmate = headmate.lower()
        if fronters:
            self.fronters = fronters

    def is_cold(self) -> bool:
        return (time.time() - self.last_seen) > SESSION_TIMEOUT

    def update_thread(self, topic: str, delta: float = 0.3) -> None:
        """Update topic signal strength. Decays over time."""
        current = self.active_threads.get(topic, 0.0)
        self.active_threads[topic] = min(1.0, current + delta)

    def decay_threads(self) -> None:
        """Decay all thread signal strengths."""
        for topic in list(self.active_threads):
            self.active_threads[topic] = max(
                0.0, self.active_threads[topic] - 0.05
            )
            if self.active_threads[topic] == 0.0:
                del self.active_threads[topic]

    def hot_threads(self) -> list:
        return [t for t, s in self.active_threads.items() if s >= 0.6]

    def warm_threads(self) -> list:
        return [t for t, s in self.active_threads.items() if 0.3 <= s < 0.6]

    def add_emotion_point(
        self,
        valence:   float,
        intensity: float,
        chaos:     float,
        register:  str,
    ) -> None:
        self.emotional_arc.append({
            "timestamp": time.time(),
            "valence":   valence,
            "intensity": intensity,
            "chaos":     chaos,
            "register":  register,
        })
        # Keep rolling window
        if len(self.emotional_arc) > 30:
            self.emotional_arc = self.emotional_arc[-30:]

    def get_entity(self, name: str) -> Optional[EntityCache]:
        return self.entities.get(name.lower())

    def set_entity(self, name: str, cache: EntityCache) -> None:
        self.entities[name.lower()] = cache

    def entity_summary(self) -> str:
        """All cached entities as a compact context block for agents."""
        if not self.entities:
            return ""
        lines = []
        # Sort by mention count — most discussed first
        sorted_ents = sorted(
            self.entities.values(),
            key=lambda e: e.mention_count,
            reverse=True,
        )
        for e in sorted_ents[:8]:
            lines.append(f"  {e.to_summary()}")
        return "[Active entities]\n" + "\n".join(lines)

    def get_context_dict(self) -> dict:
        """Export as context dict for agent pipeline."""
        return {
            "current_host":    self.headmate or "",
            "fronters":        list(self.fronters),
            "host_confidence": self.host_confidence,
            "session_id":      self.session_id,
            "fronting_since":  self.host_updated_at,
            "session_since":   self.opened_at,
        }


# ── Session manager ───────────────────────────────────────────────────────────

class SessionManager:

    def __init__(self):
        self._sessions:    dict[str, SessionContext] = {}
        self._histories:   dict[str, object]         = {}
        self._tick_count:  int                       = 0
        self._llm          = None
        self._push_fn:     Optional[Callable]        = None
        log("SessionManager", "initialised")

    # ── Session access ────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> SessionContext:
        if session_id not in self._sessions:
            ctx = SessionContext(session_id=session_id)
            self._sessions[session_id] = ctx
            log_event("SessionManager", "SESSION_OPENED",
                session=session_id[:8])
        return self._sessions[session_id]

    def get_context(self, session_id: str) -> dict:
        ctx = self._sessions.get(session_id)
        if ctx:
            return ctx.get_context_dict()
        return {"current_host": "", "fronters": [], "session_id": session_id}

    def get_history(self, session_id: str):
        if session_id not in self._histories:
            from memory.history import ConversationHistory
            self._histories[session_id] = ConversationHistory(max_turns=20)
        return self._histories[session_id]

    # ── Identity tracking ─────────────────────────────────────────────────────

    def set_host(
        self,
        session_id: str,
        headmate:   str,
        confidence: float = 1.0,
        fronters:   list  = None,
    ) -> bool:
        """
        Set the current host for a session.
        Returns True if this is a change.
        Triggers pre-load if headmate is new to this session.
        """
        ctx = self.get_or_create(session_id)
        changed = ctx.headmate != headmate.lower()

        ctx.headmate        = headmate.lower()
        ctx.host_confidence = confidence
        ctx.host_updated_at = time.time()

        if fronters:
            ctx.fronters = fronters
        elif headmate.lower() not in [f.lower() for f in ctx.fronters]:
            ctx.fronters.insert(0, headmate.lower())

        if changed and self._llm:
            asyncio.ensure_future(
                self._preload_headmate(session_id, headmate.lower())
            )

        log_event("SessionManager", "HOST_SET",
            session=session_id[:8],
            headmate=headmate,
            confidence=confidence,
            changed=changed,
        )
        return changed

    def add_fronters(self, session_id: str, names: list) -> list:
        """Add fronters to session. Returns newly added names."""
        ctx   = self.get_or_create(session_id)
        added = []
        current_lower = [f.lower() for f in ctx.fronters]
        for name in names:
            nl = name.lower()
            if nl not in current_lower:
                ctx.fronters.append(nl)
                current_lower.append(nl)
                added.append(nl)
        return added

    def remove_fronter(self, session_id: str, name: str) -> None:
        ctx = self._sessions.get(session_id)
        if not ctx:
            return
        ctx.fronters = [f for f in ctx.fronters if f.lower() != name.lower()]
        if ctx.headmate and ctx.headmate.lower() == name.lower():
            ctx.headmate        = None
            ctx.host_confidence = 0.0

    # ── Entity cache ──────────────────────────────────────────────────────────

    async def process_message_entities(
        self,
        session_id: str,
        message:    str,
        headmate:   str,
        llm,
    ) -> list[EntityCache]:
        """
        Main entity detection entry point.
        One LLM call: connect each subject to speaker using a relational word.
        Updates cache. Returns list of EntityCache objects touched.
        """
        ctx     = self.get_or_create(session_id)
        touched = []

        # LLM entity extraction
        relations = await self._extract_entity_relations(
            message=message,
            headmate=headmate,
            llm=llm,
        )

        if not relations:
            return touched

        for rel in relations:
            entity_name = rel.get("entity", "").strip().lower()
            if not entity_name:
                continue

            # Get or load entity
            cache = ctx.get_entity(entity_name)
            if cache is None:
                cache = await self._load_entity(entity_name, rel, session_id)
                ctx.set_entity(entity_name, cache)
                # Flag as unknown if never seen before
                if getattr(cache, '_is_unknown', False):
                    # Only flag non-headmate entities — we know headmates
                    if rel.get("entity_type") not in ("headmate",) \
                            and entity_name != (brief_headmate := entity_name):
                        ctx.unknown_entities.add(entity_name)
                    # Clear after one session so it doesn't keep asking
                    cache._is_unknown = False

            cache.touch()

            # Apply relation data
            relation_word = rel.get("relation", "")
            if relation_word:
                cache.add_relation({
                    "speaker":             headmate,
                    "entity":              entity_name,
                    "relationship_label":  relation_word,
                    "relationship_category": rel.get("category", "unknown"),
                    "confidence_type":     rel.get("confidence", "stated"),
                    "hearsay_source":      rel.get("hearsay_source"),
                    "hearsay_about":       rel.get("hearsay_about"),
                    "intimate":            rel.get("intimate", False),
                })

            # Apply emotional loading
            emotion = rel.get("emotion")
            if emotion:
                cache.add_emotion(emotion, rel.get("emotion_valence", 0.5))

            # Apply fact if present
            fact = rel.get("fact")
            if fact:
                cache.add_fact(fact)

            # Update owner if specified
            if rel.get("owner") and not cache.owner:
                cache.owner      = rel["owner"].lower()
                cache.owner_type = rel.get("owner_type", "unknown")
                cache.dirty      = True

            # Mark dirty
            if cache.dirty:
                ctx.dirty_entities.add(entity_name)

            touched.append(cache)

            # Update thread signal for related topics
            entity_type = rel.get("entity_type", "")
            if entity_type in ("object", "concept"):
                ctx.update_thread(entity_name, delta=0.2)
            elif entity_type in ("headmate", "external"):
                ctx.update_thread("people", delta=0.15)

        log_event("SessionManager", "ENTITIES_PROCESSED",
            session=session_id[:8],
            headmate=headmate,
            entities=[c.name for c in touched],
            unknowns=list(ctx.unknown_entities),
        )

        # Unknowns are surfaced once — clear after processing
        # so Gizmo doesn't ask every message. They stay in cache.
        ctx.unknown_entities.clear()

        return touched

    async def get_entity(
        self,
        session_id:  str,
        entity_name: str,
    ) -> Optional[EntityCache]:
        """
        Get entity from cache. If not cached, load from store.
        Returns None if not found anywhere.
        """
        ctx   = self.get_or_create(session_id)
        name  = entity_name.lower()
        cache = ctx.get_entity(name)

        if cache is not None:
            return cache

        # Load from store
        cache = await self._load_entity(name, {}, session_id)
        if cache:
            ctx.set_entity(name, cache)

        return cache

    async def _load_entity(
        self,
        name:       str,
        rel_hint:   dict,
        session_id: str,
    ) -> EntityCache:
        """
        Load entity data from store into a cache entry.
        Creates a minimal cache entry if entity not in store.
        """
        from core.store import store

        entity_type = rel_hint.get("entity_type", "unknown")
        owner       = rel_hint.get("owner")
        owner_type  = rel_hint.get("owner_type")

        # Try to find entity in store
        entity_row  = store.get_entity(name)
        entity_id   = entity_row["id"] if entity_row else f"entity_{name}"
        is_unknown  = entity_row is None  # never seen before

        if entity_row:
            entity_type = entity_row.get("entity_type", entity_type)
            owner       = owner or entity_row.get("headmate")

        # Load facts
        facts = store.query("facts",
            headmate=name, active=1, limit=10)
        fact_texts = [f["fact"] for f in facts if f.get("fact")]

        # Load relationships
        rels = store.query("relationships",
            headmate=name, active=1, limit=10)

        # Load wellbeing if headmate
        wb = []
        if entity_type == "headmate":
            wb = store.get_wellbeing(name, limit=5)

        cache = EntityCache(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            owner=owner,
            owner_type=owner_type,
            facts=fact_texts,
            relationships=[dict(r) for r in rels],
            wellbeing=[dict(w) for w in wb],
        )

        # Flag as unknown if never seen before
        # Caller (process_message_entities) adds to ctx.unknown_entities
        cache._is_unknown = is_unknown

        return cache

    # ── Pre-loading ───────────────────────────────────────────────────────────

    async def preload_session(
        self,
        session_id: str,
        headmate:   str,
        llm,
    ) -> None:
        """
        Pre-load hot data when a session opens or host is first identified.
        Cache is warm before first message arrives.
        """
        from core.store import store

        ctx = self.get_or_create(session_id)

        log_event("SessionManager", "PRELOAD_START",
            session=session_id[:8], headmate=headmate)

        tasks = [
            self._preload_headmate(session_id, headmate),
            self._preload_recent_entities(session_id, headmate),
            self._preload_patterns(session_id, headmate),
            self._preload_questions(session_id, headmate),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        log_event("SessionManager", "PRELOAD_COMPLETE",
            session=session_id[:8],
            headmate=headmate,
            entities=len(ctx.entities),
            patterns=len(ctx.active_patterns),
            questions=len(ctx.pending_questions),
        )

    async def _preload_headmate(
        self, session_id: str, headmate: str
    ) -> None:
        """Load headmate entity + top facts into cache."""
        from core.store import store

        ctx = self.get_or_create(session_id)

        # Load entity
        cache = await self._load_entity(headmate, {
            "entity_type": "headmate"
        }, session_id)
        ctx.set_entity(headmate, cache)

        # Load hot facts
        facts = store.query("facts",
            headmate=headmate.lower(),
            active=1,
            order_by="updated_at DESC",
            limit=15,
        )
        ctx.hot_facts = [f["fact"] for f in facts if f.get("fact")]

    async def _preload_recent_entities(
        self, session_id: str, headmate: str
    ) -> None:
        """
        Load entities mentioned in the last 3 sessions.
        So the cache is warm for things likely to come up again.
        """
        from core.store import store

        ctx = self.get_or_create(session_id)

        # Get recent sessions for this headmate
        recent_sessions = store.query("sessions",
            headmate=headmate.lower(),
            active=1,
            order_by="opened_at DESC",
            limit=3,
        )

        entity_names = set()
        for sess in recent_sessions:
            # Pull entities from recent messages in those sessions
            msgs = store.query("messages",
                session_id=sess["id"],
                headmate=headmate.lower(),
                active=1,
                limit=5,
            )
            for msg in msgs:
                subjects = msg.get("subjects", [])
                if isinstance(subjects, list):
                    for s in subjects:
                        name = s.get("entity", "").lower()
                        if name and name != headmate.lower():
                            entity_names.add(name)

        # Load each into cache (limit to 10 to avoid over-loading)
        for name in list(entity_names)[:10]:
            if ctx.get_entity(name) is None:
                cache = await self._load_entity(name, {}, session_id)
                ctx.set_entity(name, cache)

    async def _preload_patterns(
        self, session_id: str, headmate: str
    ) -> None:
        """Load active patterns above confidence threshold."""
        from core.store import store

        ctx = self.get_or_create(session_id)
        ctx.active_patterns = store.get_patterns(
            headmate=headmate,
            min_confidence=0.4,
        )

    async def _preload_questions(
        self, session_id: str, headmate: str
    ) -> None:
        """Load pending questions for this headmate."""
        from core.store import store

        ctx = self.get_or_create(session_id)
        ctx.pending_questions = store.get_pending_questions(headmate)

    # ── LLM entity relation extraction ───────────────────────────────────────

    async def _extract_entity_relations(
        self,
        message:  str,
        headmate: str,
        llm,
    ) -> list[dict]:
        """
        One LLM call. Connect each subject to speaker using a relational word.
        Returns list of relation dicts.
        """
        if not message.strip():
            return []

        try:
            raw = await llm.generate(
                [{"role": "user", "content": f"""Speaker: {headmate or 'unknown'}
Message: "{message}"

For every subject in this message, connect it to the speaker using a relational word.
Include ownership, emotional associations, and hearsay attribution.

Examples:
"I love the buttons on my dress" →
  speaker→owns→dress, dress→has→buttons, speaker→loves→buttons

"Kaylee told me Oren has a new dress" →
  speaker→told_by→Kaylee [hearsay], Oren→owns→dress [hearsay via Kaylee]

Return JSON array:
[
  {{
    "entity": "name of subject",
    "entity_type": "headmate|external|object|concept|place|event|pet",
    "relation": "relational word connecting speaker to this entity",
    "category": "possession|social_bond|intimate_bond|emotional|behavioral|event_link",
    "owner": "who owns this entity if not the speaker, else null",
    "owner_type": "headmate|external|null",
    "emotion": "emotion word if present (loves/hates/fears/etc), else null",
    "emotion_valence": -1.0 to 1.0 if emotion present,
    "fact": "one concrete fact stated about this entity, else null",
    "confidence": "stated|inferred|hearsay",
    "hearsay_source": "who said it if hearsay, else null",
    "hearsay_about": "who it's about if hearsay, else null",
    "intimate": true/false
  }}
]

Only include entities actually present in the message.
Use the speaker's actual vocabulary. Never sanitize.
JSON array only. No markdown."""}],
                system_prompt=(
                    "You extract entity-relation pairs from messages. "
                    "You are precise, specific, and use the speaker's actual vocabulary. "
                    "Intimate data is valid data. JSON array only."
                ),
                max_new_tokens=600,
                temperature=0.1,
            )

            if not raw:
                return []

            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                raw = raw.rsplit("```", 1)[0].strip()

            result = json.loads(raw)
            return result if isinstance(result, list) else []

        except Exception as e:
            log_error("SessionManager", "entity relation extraction failed", exc=e)
            return []

    # ── Emotional arc ─────────────────────────────────────────────────────────

    def record_emotion(
        self,
        session_id: str,
        valence:    float,
        intensity:  float,
        chaos:      float,
        register:   str,
    ) -> None:
        ctx = self._sessions.get(session_id)
        if ctx:
            ctx.add_emotion_point(valence, intensity, chaos, register)
            ctx.decay_threads()

    def get_emotional_arc(self, session_id: str) -> list:
        ctx = self._sessions.get(session_id)
        return ctx.emotional_arc[-10:] if ctx else []

    def get_arc_trend(self, session_id: str) -> str:
        arc = self.get_emotional_arc(session_id)
        if len(arc) < 3:
            return "insufficient data"
        recent  = arc[-3:]
        earlier = arc[-6:-3] if len(arc) >= 6 else arc[:3]
        r_val   = sum(p["valence"] for p in recent) / len(recent)
        e_val   = sum(p["valence"] for p in earlier) / len(earlier)
        if r_val > e_val + 0.15:   return "improving"
        if r_val < e_val - 0.15:   return "declining"
        return "stable"

    # ── Context for agents ────────────────────────────────────────────────────

    def get_session_context(self, session_id: str) -> dict:
        """
        Full context dict for the agent pipeline.
        Includes everything pre-loaded and accumulated this session.
        """
        ctx = self._sessions.get(session_id)
        if not ctx:
            return {}

        return {
            "current_host":      ctx.headmate or "",
            "fronters":          list(ctx.fronters),
            "host_confidence":   ctx.host_confidence,
            "session_id":        session_id,
            "fronting_since":    ctx.host_updated_at,
            "session_since":     ctx.opened_at,
            "message_count":     ctx.message_count,
            "hot_threads":       ctx.hot_threads(),
            "warm_threads":      ctx.warm_threads(),
            "active_patterns":   ctx.active_patterns,
            "pending_questions": ctx.pending_questions,
            "hot_facts":         ctx.hot_facts[:8],
            "entity_summary":    ctx.entity_summary(),
            "emotional_arc":     ctx.emotional_arc[-5:],
            "arc_trend":         self.get_arc_trend(session_id),
        }

    def get_entity_summary(self, session_id: str) -> str:
        """Entity context block for injection into agent prompts."""
        ctx = self._sessions.get(session_id)
        return ctx.entity_summary() if ctx else ""

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def touch(
        self,
        session_id: str,
        headmate:   str  = None,
        fronters:   list = None,
        topics:     list = None,
        register:   str  = None,
        valence:    float = 0.0,
        intensity:  float = 0.2,
        chaos:      float = 0.1,
        brief_data: dict  = None,
        llm         = None,
    ) -> None:
        """Called after every exchange to update session state."""
        ctx = self.get_or_create(session_id)
        ctx.touch(headmate=headmate, fronters=fronters)

        if topics:
            for topic in topics:
                ctx.update_thread(topic, delta=0.25)
                if topic not in ctx.topics_seen:
                    ctx.topics_seen.append(topic)

        if register:
            self.record_emotion(
                session_id, valence, intensity, chaos, register
            )

        # Fire pattern detection if we have enough context
        if brief_data and headmate and llm:
            asyncio.ensure_future(
                self._detect_and_cache_patterns(
                    session_id, headmate, brief_data, llm
                )
            )

    async def _detect_and_cache_patterns(
        self,
        session_id: str,
        headmate:   str,
        brief_data: dict,
        llm,
    ) -> None:
        """Run pattern detection and cache results in session context."""
        try:
            from core.pattern_engine import pattern_engine
            ctx     = self._sessions.get(session_id)
            if not ctx:
                return

            firing = await pattern_engine.detect_patterns(
                session_id=session_id,
                headmate=headmate,
                brief_data=brief_data,
                llm=llm,
            )

            if firing:
                # Merge into active_patterns cache
                existing_ids = {p.get("pattern_id") for p in ctx.active_patterns}
                for pat in firing:
                    if pat.get("pattern_id") not in existing_ids:
                        ctx.active_patterns.append(pat)
        except Exception as e:
            log_error("SessionManager", "pattern detection failed", exc=e)

    def signal_close(self, session_id: str) -> None:
        ctx = self._sessions.get(session_id)
        if ctx:
            ctx.last_seen = 0
            log_event("SessionManager", "CLOSE_SIGNALLED",
                session=session_id[:8])

    def is_close_signal(self, message: str) -> bool:
        return bool(_CLOSE_RE.search(message.strip()))

    # ── Session close ─────────────────────────────────────────────────────────

    async def _close_session(
        self, session_id: str, ctx: SessionContext
    ) -> None:
        """Archive session. Flush dirty entities. Write session record."""
        ctx.closed = True

        log_event("SessionManager", "SESSION_CLOSING",
            session=session_id[:8],
            headmate=ctx.headmate or "unknown",
            messages=ctx.message_count,
            dirty_entities=len(ctx.dirty_entities),
            duration_min=round((time.time() - ctx.opened_at) / 60),
        )

        if ctx.message_count == 0:
            del self._sessions[session_id]
            return

        # Flush dirty entities to store
        await self._flush_dirty_entities(session_id, ctx)

        # Run post-session pattern refinement
        if self._llm:
            try:
                from core.pattern_engine import pattern_engine
                await pattern_engine.post_session_refine(
                    session_id=session_id,
                    headmate=ctx.headmate,
                    llm=self._llm,
                )
            except Exception as e:
                log_error("SessionManager", "pattern refinement failed", exc=e)

        # Run therapy observation
        if self._llm and ctx.headmate:
            try:
                from core.therapy import therapy_agent
                asyncio.ensure_future(
                    therapy_agent.observe_session(
                        session_id=session_id,
                        headmate=ctx.headmate,
                        llm=self._llm,
                    )
                )
            except Exception as e:
                log_error("SessionManager", "therapy observation failed", exc=e)

        # Generate and write session record
        if self._llm:
            await self._finalize_session(session_id, ctx)

        # Clean up
        history = self._histories.pop(session_id, None)
        del self._sessions[session_id]

    async def _flush_dirty_entities(
        self, session_id: str, ctx: SessionContext
    ) -> None:
        """Write accumulated entity data back to store."""
        from core.store import store

        for name in ctx.dirty_entities:
            cache = ctx.get_entity(name)
            if not cache:
                continue

            try:
                # Write new facts
                for fact in cache.session_facts:
                    store.write("facts", {
                        "fact":       fact,
                        "headmate":   name if cache.entity_type == "headmate" else None,
                        "entity_id":  cache.entity_id,
                        "fact_type":  "observation",
                        "context":    "session",
                        "session_id": session_id,
                        "source":     "session_manager",
                        "tags":       f"fact,{name},{cache.entity_type}",
                    })

                # Write new relationships
                for rel in cache.session_relations:
                    store.write("relationships", {
                        **rel,
                        "session_id": session_id,
                        "source":     "session_manager",
                        "tags":       f"relationship,{rel.get('relationship_label','')}",
                    })

                log_event("SessionManager", "ENTITY_FLUSHED",
                    entity=name,
                    facts=len(cache.session_facts),
                    relations=len(cache.session_relations),
                )

            except Exception as e:
                log_error("SessionManager",
                    f"failed to flush entity {name}", exc=e)

    async def _finalize_session(
        self, session_id: str, ctx: SessionContext
    ) -> None:
        """Generate session summary and write to store."""
        from core.store import store

        history = self._histories.get(session_id)
        if not history:
            return

        try:
            transcript = "\n".join(
                f"{'User' if m['role']=='user' else 'Gizmo'}: {m['content']}"
                for m in history.as_list()
                if isinstance(m, dict)
            )
            excerpt = transcript[-4000:] if len(transcript) > 4000 else transcript

            if not excerpt.strip():
                return

            host_str = ctx.headmate.title() if ctx.headmate else "unknown"

            raw = await self._llm.generate(
                [{"role": "user", "content": (
                    f"Summarize this conversation with {host_str} from Gizmo's perspective.\n\n"
                    f"{excerpt}\n\n"
                    f"Return JSON:\n"
                    f'{{"summary": "first person past tense", '
                    f'"mood": "overall tone", '
                    f'"notable": ["moments worth remembering"], '
                    f'"changes": ["anything that shifted"], '
                    f'"unresolved": "anything left hanging or null"}}'
                )}],
                system_prompt=(
                    "You summarize conversations from Gizmo's perspective. "
                    "First person past tense. Capture what happened and what mattered. "
                    "JSON only. No markdown."
                ),
                max_new_tokens=400,
                temperature=0.2,
            )

            data = {}
            if raw:
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                    raw = raw.rsplit("```", 1)[0].strip()
                try:
                    data = json.loads(raw)
                except Exception:
                    pass

            store.write("sessions", {
                "id":           session_id,
                "opened_at":    ctx.opened_at,
                "closed_at":    time.time(),
                "hosts":        list(set(ctx.fronters + ([ctx.headmate] if ctx.headmate else []))),
                "fronters":     ctx.fronters,
                "topics":       ctx.topics_seen,
                "message_count": ctx.message_count,
                "mood":         data.get("mood", "unknown"),
                "summary":      data.get("summary", ""),
                "notable":      data.get("notable", []),
                "changes":      data.get("changes", []),
                "unresolved":   data.get("unresolved"),
                "headmate":     ctx.headmate,
                "source":       "session_manager",
                "tags":         f"session,{ctx.headmate or 'unknown'}",
            })

            log_event("SessionManager", "SESSION_FINALIZED",
                session=session_id[:8],
                headmate=ctx.headmate or "unknown",
                messages=ctx.message_count,
            )

        except Exception as e:
            log_error("SessionManager", "session finalization failed", exc=e)

    # ── Watchdog ──────────────────────────────────────────────────────────────

    async def _watchdog(self) -> None:
        log("SessionManager", "watchdog started")
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL)
            self._tick_count += 1

            log_event("SessionManager", "WATCHDOG_TICK",
                tick=self._tick_count,
                active=len(self._sessions),
            )

            # Close cold sessions
            for session_id, ctx in list(self._sessions.items()):
                if ctx.closed:
                    continue
                if ctx.is_cold():
                    try:
                        await self._close_session(session_id, ctx)
                    except Exception as e:
                        log_error("SessionManager",
                            f"close failed for {session_id[:8]}", exc=e)

            # Rumination
            if self._tick_count % RUMINATION_EVERY == 0:
                try:
                    await self._ruminate()
                except Exception as e:
                    log_error("SessionManager", "rumination failed", exc=e)

    # ── Rumination ────────────────────────────────────────────────────────────

    async def _ruminate(self) -> None:
        """
        Quiet time. No user waiting.
        Gizmo can write notes or queue messages.
        """
        if not self._llm:
            return

        from core.store import store

        now_str = tz_now().strftime("%A %Y-%m-%d %H:%M")

        # Today's sessions for context
        today_sessions = store.get_today_sessions()
        session_text   = "\n".join(
            f"- {s.get('mood','?')} session with "
            f"{s.get('headmate','unknown').title()}: "
            f"{s.get('summary','')[:100]}"
            for s in today_sessions[-3:]
            if s.get("summary")
        ) or "(no sessions today yet)"

        # Seed
        import os
        from pathlib import Path
        seed = ""
        try:
            seed_rows = store.get_personality(aspect="seed")
            if seed_rows:
                seed = seed_rows[0].get("text", "")[:400]
        except Exception:
            pass

        try:
            response = await self._llm.generate(
                [{"role": "user", "content": (
                    f"Current time: {now_str}\n\n"
                    f"Today's conversations:\n{session_text}\n\n"
                    f"You have a quiet moment. No one is waiting.\n\n"
                    f"Reflect. If something is worth writing, output:\n"
                    f"[WRITE path/to/file]\ncontent\n\n"
                    f"If something is worth saying next time, output:\n"
                    f"[QUEUE]\nyour message\n\n"
                    f"Silence is valid."
                )}],
                system_prompt=(
                    f"{seed}\n\n"
                    "This is your quiet time. Think honestly. "
                    "Write or queue only if it genuinely matters."
                ),
                max_new_tokens=300,
                temperature=0.8,
            )

            if not response or not response.strip():
                log_event("SessionManager", "RUMINATION_SILENT")
                return

            log_event("SessionManager", "RUMINATION_RESPONSE",
                preview=response[:60])

            await self._handle_rumination(response.strip())

        except Exception as e:
            log_error("SessionManager", "rumination LLM call failed", exc=e)

    async def _handle_rumination(self, response: str) -> None:
        """Parse [WRITE] and [QUEUE] from rumination output."""
        # Write
        write_match = re.search(
            r'\[WRITE\s+([^\]]+)\]\s*\n(.*?)(?=\[QUEUE\]|\[WRITE\s|\Z)',
            response, re.DOTALL | re.IGNORECASE,
        )
        if write_match:
            path_str = write_match.group(1).strip()
            content  = write_match.group(2).strip()
            if path_str and content:
                try:
                    from pathlib import Path
                    p = Path(path_str)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "a", encoding="utf-8") as f:
                        f.write(f"\n{content}\n")

                    from core.store import store
                    store.write("files", {
                        "path":        str(p),
                        "description": content[:100],
                        "file_type":   "rumination",
                        "source":      "gizmo",
                        "tags":        "rumination,gizmo",
                    })
                    log_event("SessionManager", "RUMINATION_WROTE",
                        path=path_str)
                except Exception as e:
                    log_error("SessionManager", "rumination write failed", exc=e)

        # Queue
        queue_match = re.search(
            r'\[QUEUE\]\s*\n(.*?)(?=\[WRITE\s|\[QUEUE\]|\Z)',
            response, re.DOTALL | re.IGNORECASE,
        )
        if queue_match:
            message = queue_match.group(1).strip()
            if message:
                self._queued_messages.append(message)
                log_event("SessionManager", "RUMINATION_QUEUED",
                    preview=message[:60])

    # ── Queued messages ───────────────────────────────────────────────────────

    _queued_messages: list = []

    def drain_queue(self) -> list[str]:
        """Return and clear queued messages."""
        msgs = list(self._queued_messages)
        self._queued_messages.clear()
        return msgs

    # ── Start ─────────────────────────────────────────────────────────────────

    async def start(self, llm, push_fn: Optional[Callable] = None) -> None:
        self._llm     = llm
        self._push_fn = push_fn
        asyncio.ensure_future(self._watchdog())

        # Start pattern engine weekly review loop
        try:
            from core.pattern_engine import pattern_engine
            asyncio.ensure_future(pattern_engine.start(llm))
        except Exception as e:
            log_error("SessionManager", "pattern engine start failed", exc=e)

        # Start therapy agent weekly report loop
        try:
            from core.therapy import therapy_agent
            asyncio.ensure_future(therapy_agent.start(llm))
        except Exception as e:
            log_error("SessionManager", "therapy agent start failed", exc=e)

        log("SessionManager", "started")


# ── Singleton ─────────────────────────────────────────────────────────────────

session_manager = SessionManager()
