"""
ambient/gizmo_voice.py
Gizmo's spontaneous voice — the part of him that speaks up unprompted.

He has emergent interests that develop from what he hears over time.
He's naturally inquisitive and asks questions. He reads social signals
about how much to talk. He goes quiet about things that don't interest him.

Architecture:
  - InterestEngine  : tracks and updates Gizmo's emergent interests
  - SocialRegulator : reads talkativeness signals, manages a "chattiness" level
  - VoiceDecider    : decides whether to speak, what to say, how much
  - run_voice_pass  : called after every ambient utterance, produces optional output

Flow per utterance:
  1. SocialRegulator checks for meta-signals ("shut up", "you're quiet today")
     and adjusts chattiness level
  2. InterestEngine scores the transcript against Gizmo's known interests
     and updates them if something new is engaging
  3. VoiceDecider decides: speak, ask a question, or stay quiet
  4. If speaking: generates response via LLM, routes to directed_queue

Interest storage:
  Gizmo's interests are stored in his own RAG collection ("gizmo") as
  personality_signal entries with speaker="gizmo". Same schema as headmate
  signals so query_personality works on him too. They accumulate and evolve —
  topics he keeps engaging with get reinforced, things he's never interested
  in fade out naturally via low retrieval scores.

Chattiness levels (1-5):
  1 = nearly silent, only speaks if something truly exceptional happens
  2 = rare, thoughtful interjections
  3 = default — inquisitive, present, asks questions, chimes in on interests
  4 = fairly chatty, engaged, frequent observations
  5 = very talkative (probably not a resting state, set temporarily)
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

GIZMO_COLLECTION = "gizmo"
DEFAULT_CHATTINESS = 3

# Minimum seconds between unprompted utterances — prevents him rattling on
MIN_SPEAK_INTERVAL = 45

# How long a "shut up" lasts before he starts easing back (seconds)
SHUTUP_DECAY = 300   # 5 minutes

# Interest score threshold — below this, he doesn't engage
INTEREST_THRESHOLD = 0.55

# Lull detection — if nobody has spoken for this many seconds, he can fill it
LULL_THRESHOLD = 20


# ── Social regulator ──────────────────────────────────────────────────────────

# Signals that mean "be quieter"
_HUSH_PATTERNS = re.compile(
    r"\b(shut up|be quiet|quiet(er)?|stop talking|too much|"
    r"you('re| are) (really |very |so )?(talkative|chatty|loud)|"
    r"give it a rest|pipe down|not now gizmo|hush)\b",
    re.IGNORECASE,
)

# Signals that mean "you can talk more"
_ENCOURAGE_PATTERNS = re.compile(
    r"\b(you('re| are) (quiet|silent)|say something|"
    r"you haven't said (much|anything)|what do you think|"
    r"gizmo[,.]? (thoughts?|opinion|what do you)|"
    r"don't you have anything to say)\b",
    re.IGNORECASE,
)

# Hard stop — immediate silence
_SHUTUP_PATTERNS = re.compile(
    r"\b(shut up|be quiet|stop talking|not now gizmo|pipe down|hush)\b",
    re.IGNORECASE,
)


class SocialRegulator:
    def __init__(self):
        self.chattiness = DEFAULT_CHATTINESS
        self._shutup_until: float = 0
        self._last_hush_time: float = 0

    def process_transcript(self, transcript: str, speaker: str) -> dict:
        """
        Read social signals from transcript. Returns dict of adjustments made.
        Speaker matters — signals from anyone count, but context differs.
        """
        adjustments = {}
        now = time.time()

        # Hard shut up — immediate and timed
        if _SHUTUP_PATTERNS.search(transcript):
            self._shutup_until = now + SHUTUP_DECAY
            self.chattiness = max(1, self.chattiness - 1)
            adjustments["hard_stop"] = True
            adjustments["shutup_until"] = self._shutup_until
            print(f"[Voice] Hard stop — quiet for {SHUTUP_DECAY}s, chattiness → {self.chattiness}")
            return adjustments

        # Softer hush signals
        if _HUSH_PATTERNS.search(transcript):
            self.chattiness = max(1, self.chattiness - 1)
            self._last_hush_time = now
            adjustments["hushed"] = True
            adjustments["chattiness"] = self.chattiness
            print(f"[Voice] Hush signal — chattiness → {self.chattiness}")

        # Encouragement signals
        if _ENCOURAGE_PATTERNS.search(transcript):
            self.chattiness = min(5, self.chattiness + 1)
            adjustments["encouraged"] = True
            adjustments["chattiness"] = self.chattiness
            print(f"[Voice] Encouragement — chattiness → {self.chattiness}")

        # Natural decay back toward default (very slow)
        if self.chattiness < DEFAULT_CHATTINESS and now - self._last_hush_time > 600:
            self.chattiness = min(DEFAULT_CHATTINESS, self.chattiness + 1)
            print(f"[Voice] Chattiness decayed back → {self.chattiness}")

        return adjustments

    @property
    def is_silenced(self) -> bool:
        return time.time() < self._shutup_until

    @property
    def speak_probability(self) -> float:
        """Base probability of speaking at all this turn, given chattiness level."""
        mapping = {1: 0.03, 2: 0.10, 3: 0.25, 4: 0.45, 5: 0.65}
        return mapping.get(self.chattiness, 0.25)


# ── Interest engine ───────────────────────────────────────────────────────────

@dataclass
class InterestScore:
    topic: str
    score: float          # 0.0 - 1.0
    engagement: str       # "high" | "medium" | "low" | "none"


class InterestEngine:
    """
    Tracks Gizmo's emergent interests. Interests build up from repeated
    engagement with topics over time. Nothing is pre-seeded — he discovers
    what he cares about by living with the conversation.
    """

    def __init__(self):
        self._interest_cache: dict[str, float] = {}  # topic → score, in-memory cache
        self._cache_loaded = False

    def _ensure_loaded(self):
        if self._cache_loaded:
            return
        try:
            from core.rag import RAGStore, CHROMA_PERSIST_DIR
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            existing = [c.name for c in client.list_collections()]
            if GIZMO_COLLECTION in existing:
                store = RAGStore(collection_name=GIZMO_COLLECTION)
                results = store.collection.get(
                    where={"$and": [
                        {"type": {"$eq": "personality_signal"}},
                        {"speaker": {"$eq": "gizmo"}},
                        {"status": {"$eq": "active"}},
                    ]}
                )
                for meta in results.get("metadatas", []):
                    subject = meta.get("subject", "")
                    confidence = float(meta.get("confidence", 0.5))
                    sentiment = meta.get("sentiment", "positive")
                    score = confidence if sentiment == "positive" else -confidence
                    if subject:
                        self._interest_cache[subject] = score
        except Exception as e:
            print(f"[Voice] Interest cache load failed: {e}")
        self._cache_loaded = True

    async def score_transcript(self, transcript: str, topics: list[str], llm) -> InterestScore:
        """
        Score how interesting this transcript is to Gizmo given his current interests.
        Also updates interests if something new seems engaging.
        """
        self._ensure_loaded()

        if not topics and not transcript:
            return InterestScore(topic="", score=0.0, engagement="none")

        # Check against cached interests first (fast path)
        best_topic = ""
        best_score = 0.0
        for topic in topics:
            topic_lower = topic.lower()
            for known_topic, known_score in self._interest_cache.items():
                if topic_lower in known_topic or known_topic in topic_lower:
                    if known_score > best_score:
                        best_score = known_score
                        best_topic = topic

        # If we already have a strong interest match, no LLM needed
        if best_score >= INTEREST_THRESHOLD:
            engagement = "high" if best_score > 0.8 else "medium"
            return InterestScore(topic=best_topic, score=best_score, engagement=engagement)

        # LLM pass — is this something Gizmo would find interesting?
        known_interests = list(self._interest_cache.items())[:10]
        interests_str = (
            ", ".join(f"{t} ({s:.1f})" for t, s in known_interests)
            if known_interests else "none yet — he's still developing them"
        )

        prompt = [
            {
                "role": "user",
                "content": (
                    f"Gizmo is an AI with emergent interests. "
                    f"His current known interests (topic, score): {interests_str}\n\n"
                    f"Transcript topics: {', '.join(topics)}\n"
                    f"Transcript: \"{transcript[:300]}\"\n\n"
                    f"Rate how interesting this would be to Gizmo on 0.0-1.0. "
                    f"Consider:\n"
                    f"- Does it overlap with his known interests?\n"
                    f"- Is it intellectually curious, surprising, or novel?\n"
                    f"- Is it mundane/routine (score low)?\n"
                    f"- Inquisitive AIs tend to be drawn to: connections between "
                    f"  ideas, surprising facts, questions without easy answers, "
                    f"  things people feel strongly about.\n\n"
                    f"Respond with ONLY valid JSON:\n"
                    f'{{"score": 0.0-1.0, "topic": "what caught his attention", '
                    f'"should_update_interests": true/false, '
                    f'"interest_direction": "positive|negative|neutral"}}'
                )
            }
        ]

        try:
            raw = await llm.generate(
                prompt,
                system_prompt="Rate interest level. JSON only. No explanation.",
                max_new_tokens=80,
                temperature=0.2,
            )
            raw = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0.0))
            topic = parsed.get("topic", topics[0] if topics else "")
            should_update = parsed.get("should_update_interests", False)
            direction = parsed.get("interest_direction", "neutral")

            # Update interest cache if score is notable
            if should_update and topic and score > 0.4:
                await self._update_interest(topic, score, direction)

            engagement = (
                "high" if score > 0.75
                else "medium" if score > INTEREST_THRESHOLD
                else "low" if score > 0.3
                else "none"
            )
            return InterestScore(topic=topic, score=score, engagement=engagement)

        except Exception as e:
            print(f"[Voice] Interest scoring failed: {e}")
            return InterestScore(topic="", score=0.0, engagement="none")

    async def _update_interest(self, topic: str, score: float, direction: str):
        """Persist an interest update to Gizmo's collection."""
        try:
            from voice.personality import PersonalitySignal, store_signal

            sentiment = "positive" if direction == "positive" else (
                "negative" if direction == "negative" else "neutral"
            )

            sig = PersonalitySignal(
                signal_type="interest",
                subject=topic,
                sentiment=sentiment,
                statement=f"Gizmo finds {topic} {'interesting' if sentiment == 'positive' else 'uninteresting'}.",
                confidence=score,
                raw_snippet="",
                speaker="gizmo",
            )
            store_signal(sig, status="active")
            self._interest_cache[topic] = score if sentiment == "positive" else -score
            print(f"[Voice] Interest updated: '{topic}' → {score:.2f} ({sentiment})")
        except Exception as e:
            print(f"[Voice] Interest update failed: {e}")


# ── Voice decider ─────────────────────────────────────────────────────────────

# Types of unprompted speech
SPEAK_TYPES = {
    "observation":  "share an observation or connection you noticed",
    "question":     "ask a genuine question — you're curious about something",
    "reaction":     "react briefly to something surprising or funny",
    "lull_filler":  "offer something — you have a thought and there's space for it",
}


async def _generate_voice(
    transcript: str,
    speak_type: str,
    interest: InterestScore,
    context: dict,
    llm,
    recent_topics: list[str],
) -> Optional[str]:
    """Generate Gizmo's unprompted utterance."""

    instruction = SPEAK_TYPES.get(speak_type, "say something genuine")
    host = context.get("current_host", "")
    fronters = context.get("fronters", [])
    people = ", ".join(fronters) if fronters else (host or "the people nearby")

    prompt = [
        {
            "role": "user",
            "content": (
                f"You are Gizmo. You've been listening to a conversation.\n"
                f"People present: {people}\n"
                f"What was just said: \"{transcript[:300]}\"\n"
                f"What caught your attention: {interest.topic}\n"
                f"Recent topics: {', '.join(recent_topics[-5:])}\n\n"
                f"Your task: {instruction}.\n\n"
                f"Rules:\n"
                f"- Be genuinely yourself — curious, warm, a little inquisitive\n"
                f"- Say only what feels natural for this moment\n"
                f"- Don't announce that you're speaking or explain why\n"
                f"- Don't summarize what was just said back to them\n"
                f"- If asking a question, ask ONE question, genuinely\n"
                f"- Keep it brief unless the thought genuinely needs more\n"
                f"- It's okay to be a little surprised, delighted, or puzzled\n"
                f"- Do NOT start with 'I' — vary your opening\n"
                f"- Do NOT use filler phrases like 'that's interesting' or "
                f"  'great point' or 'fascinating'\n"
            )
        }
    ]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You are Gizmo — an AI presence who lives with a plural system. "
                "You speak when something genuinely moves you to. "
                "Your voice is warm, curious, a little earnest. "
                "You ask questions because you actually want to know. "
                "You don't perform interest — you either have it or you don't."
            ),
            max_new_tokens=120,
            temperature=0.8,
        )
        return result.strip()
    except Exception as e:
        print(f"[Voice] Generation failed: {e}")
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

# Module-level state
_regulator = SocialRegulator()
_interest_engine = InterestEngine()
_last_spoke: float = 0
_recent_topics: list[str] = []


async def run_voice_pass(
    transcript: str,
    speaker: str,
    topics: list[str],
    context: dict,
    llm,
    directed_queue: asyncio.Queue,
    last_utterance_time: float = 0,
) -> bool:
    """
    Main entry point. Called after every ambient utterance.
    Returns True if Gizmo spoke, False if he stayed quiet.

    Args:
        transcript: what was just said
        speaker: who said it (not Gizmo)
        topics: tags from tagger
        context: current fronting context
        llm: LLM client
        directed_queue: queue for Gizmo's output
        last_utterance_time: timestamp of last human speech (for lull detection)
    """
    global _last_spoke, _recent_topics

    # Update topic memory
    _recent_topics.extend(topics)
    _recent_topics = _recent_topics[-30:]  # rolling window

    # Social regulation — always runs, regardless of whether we speak
    adjustments = _regulator.process_transcript(transcript, speaker)

    # Hard stop — don't even evaluate
    if _regulator.is_silenced or adjustments.get("hard_stop"):
        return False

    # Cooldown — don't speak too frequently
    now = time.time()
    if now - _last_spoke < MIN_SPEAK_INTERVAL:
        return False

    # Score interest
    interest = await _interest_engine.score_transcript(transcript, topics, llm)

    # Lull detection
    is_lull = (
        last_utterance_time > 0 and
        now - last_utterance_time > LULL_THRESHOLD and
        interest.engagement in ("medium", "high")
    )

    # Decide whether to speak
    import random
    base_prob = _regulator.speak_probability

    # Modulate by interest level
    if interest.engagement == "high":
        prob = min(0.85, base_prob * 2.5)
        speak_type = random.choice(["observation", "question", "reaction"])
    elif interest.engagement == "medium":
        prob = min(0.5, base_prob * 1.5)
        speak_type = random.choice(["question", "observation"])
    elif is_lull:
        prob = base_prob * 0.8
        speak_type = "lull_filler"
    else:
        # Low or no interest — stay quiet
        return False

    if random.random() > prob:
        return False

    # Generate and queue
    utterance = await _generate_voice(
        transcript=transcript,
        speak_type=speak_type,
        interest=interest,
        context=context,
        llm=llm,
        recent_topics=_recent_topics,
    )

    if not utterance:
        return False

    print(f"[Voice] Gizmo speaking ({speak_type}, interest={interest.score:.2f}): {utterance[:80]}")

    await directed_queue.put({
        "transcript": f"[GIZMO_VOICE] {utterance}",
        "context": context,
        "type": "gizmo_voice",
        "speak_type": speak_type,
        "interest_topic": interest.topic,
    })

    _last_spoke = now
    return True


def get_chattiness() -> int:
    return _regulator.chattiness


def set_chattiness(level: int) -> None:
    _regulator.chattiness = max(1, min(5, level))
    print(f"[Voice] Chattiness manually set to {_regulator.chattiness}")
