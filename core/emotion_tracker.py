"""
core/emotion_tracker.py
Per-session emotional arc tracking.

Tracks valence, intensity, and chaos per message across the whole session.
Not per-headmate — the arc is one conversation, everyone in it.
But each data point is tagged with who was fronting, so per-headmate
analysis is possible in retrospect.

Three axes:
  valence:   -1.0 to +1.0  (negative ←→ positive)
  intensity:  0.0 to  1.0  (quiet ←→ overwhelming)
  chaos:      0.0 to  1.0  (coherent ←→ fragmented)

Heuristic scoring — fast, zero LLM calls, runs on every message.
LLM retrospective scoring happens in observer.py after session archives.

The pressure gauge accumulates from content weight + emotional signal
+ time density. When it crosses threshold, reflector.py fires.

Usage:
    from core.emotion_tracker import emotion_tracker

    # After each message
    point = emotion_tracker.record(
        session_id, message, headmate, register, timestamp
    )

    # Get real-time summary for Ego
    summary = emotion_tracker.arc_summary(session_id)

    # Check pressure gauge
    if emotion_tracker.should_reflect(session_id):
        emotion_tracker.reset_pressure(session_id)
        # → fire reflector
"""

import re
import time
import math
from dataclasses import dataclass, field
from typing import Optional

from core.log import log, log_event, log_error

# ── Emotion point ─────────────────────────────────────────────────────────────

@dataclass
class EmotionPoint:
    timestamp:  float
    headmate:   Optional[str]
    register:   str           # from Archivist (neutral/positive/elevated/distress/subdued)
    valence:    float         # -1.0 to +1.0
    intensity:  float         # 0.0 to 1.0
    chaos:      float         # 0.0 to 1.0
    message:    str           # raw message for retrospective
    word_count: int
    topic:      str           # primary topic from Archivist


# ── Register → valence/intensity base mappings ────────────────────────────────

_REGISTER_BASES = {
    "neutral":  (0.0,  0.2),
    "positive": (0.6,  0.5),
    "elevated": (-0.2, 0.7),
    "distress": (-0.7, 0.8),
    "subdued":  (-0.1, 0.15),
}

# ── Chaos heuristics ──────────────────────────────────────────────────────────

_ELLIPSIS_RE    = re.compile(r'\.{2,}|…')
_FRAGMENT_RE    = re.compile(r'\b\w{1,3}\b')
_CAPS_RE        = re.compile(r'\b[A-Z]{2,}\b')
_REPEAT_RE      = re.compile(r'(\w+)\s+\1\b', re.IGNORECASE)
_TRAIL_PUNCT_RE = re.compile(r'[!?]{2,}')


def _heuristic_chaos(message: str, session_arc: list[EmotionPoint]) -> float:
    """
    Estimate chaos from message content and recent arc.
    Returns 0.0–1.0.
    """
    score = 0.0
    msg = message.strip()
    words = msg.split()
    word_count = len(words)

    # Very short message after longer ones — fragmentation signal
    if session_arc:
        recent = session_arc[-3:]
        avg_words = sum(p.word_count for p in recent) / len(recent)
        if word_count < avg_words * 0.4 and word_count < 8:
            score += 0.2

    # Ellipsis — trailing off, uncertainty
    ellipsis_count = len(_ELLIPSIS_RE.findall(msg))
    score += min(0.2, ellipsis_count * 0.07)

    # Excessive punctuation
    trail_count = len(_TRAIL_PUNCT_RE.findall(msg))
    score += min(0.15, trail_count * 0.08)

    # All-caps words — shouting, emphasis, intensity spikes
    caps_count = len(_CAPS_RE.findall(msg))
    score += min(0.15, caps_count * 0.05)

    # Word repetition — verbal loop, fixation
    repeat_count = len(_REPEAT_RE.findall(msg))
    score += min(0.15, repeat_count * 0.08)

    # Register oscillation — if last 3 points bounce registers a lot
    if len(session_arc) >= 3:
        recent_registers = [p.register for p in session_arc[-3:]]
        unique = len(set(recent_registers))
        if unique == 3:  # all different
            score += 0.15

    # High fragment ratio — lots of tiny words
    if word_count > 4:
        fragment_ratio = len(_FRAGMENT_RE.findall(msg)) / word_count
        if fragment_ratio > 0.6:
            score += 0.1

    return min(1.0, score)


def _heuristic_valence_intensity(
    message: str,
    register: str,
    session_arc: list[EmotionPoint],
) -> tuple[float, float]:
    """
    Refine valence and intensity from register base + message signals.
    """
    base_valence, base_intensity = _REGISTER_BASES.get(register, (0.0, 0.3))

    msg_lower = message.lower()
    words = message.split()
    word_count = len(words)

    # Valence modifiers
    valence = base_valence

    positive_words = {"love", "great", "awesome", "happy", "excited", "good",
                      "wonderful", "yay", "yes", "perfect", "amazing", "fun",
                      "glad", "grateful", "proud", "relieved", "better"}
    negative_words = {"hate", "awful", "terrible", "sad", "angry", "bad",
                      "hurt", "scared", "anxious", "frustrated", "tired",
                      "exhausted", "overwhelmed", "stressed", "worried", "lost"}

    pos_hits = sum(1 for w in words if w.lower().strip(".,!?") in positive_words)
    neg_hits = sum(1 for w in words if w.lower().strip(".,!?") in negative_words)

    valence += pos_hits * 0.08
    valence -= neg_hits * 0.08
    valence = max(-1.0, min(1.0, valence))

    # Intensity modifiers
    intensity = base_intensity

    # Message length — longer messages often carry more weight
    if word_count > 50:
        intensity += 0.1
    elif word_count < 5:
        intensity -= 0.05

    # Intensifiers
    intensifiers = {"really", "very", "so", "extremely", "absolutely",
                    "completely", "totally", "utterly", "fucking", "literally"}
    intensifier_hits = sum(1 for w in words if w.lower() in intensifiers)
    intensity += min(0.2, intensifier_hits * 0.06)

    # Softeners
    softeners = {"maybe", "perhaps", "kind of", "sort of", "a little", "slightly"}
    softener_hits = sum(1 for s in softeners if s in msg_lower)
    intensity -= softener_hits * 0.05

    intensity = max(0.0, min(1.0, intensity))

    return valence, intensity


# ── Pressure gauge ────────────────────────────────────────────────────────────

@dataclass
class PressureGauge:
    """
    Accumulates pressure from content weight + emotional signal + time density.
    When threshold is crossed, reflection should fire.
    """
    value:           float = 0.0
    threshold:       float = 1.0
    last_reset:      float = field(default_factory=time.time)
    last_message_ts: float = field(default_factory=time.time)
    messages_since_reset: int = 0

    def add(
        self,
        word_count: int,
        intensity: float,
        chaos: float,
        timestamp: float,
    ) -> None:
        """Accumulate pressure from a new message."""
        # Time density — how fast are messages coming?
        elapsed = timestamp - self.last_message_ts
        # Fast messages (< 30s) = higher density weight
        density_weight = max(0.5, min(2.0, 30.0 / max(elapsed, 1)))

        # Content weight — meaningful messages count more
        content_weight = math.log1p(word_count) / math.log1p(50)

        # Emotional signal — high intensity/chaos raises pressure faster
        emotional_signal = (intensity * 0.6 + chaos * 0.4)

        pressure_delta = content_weight * density_weight * (0.4 + 0.6 * emotional_signal)
        self.value += pressure_delta
        self.last_message_ts = timestamp
        self.messages_since_reset += 1

    def should_fire(self) -> bool:
        return self.value >= self.threshold

    def reset(self) -> None:
        self.value = 0.0
        self.last_reset = time.time()
        self.messages_since_reset = 0


# ── Session arc ───────────────────────────────────────────────────────────────

@dataclass
class SessionArc:
    session_id:   str
    points:       list[EmotionPoint] = field(default_factory=list)
    pressure:     PressureGauge      = field(default_factory=PressureGauge)
    created_at:   float              = field(default_factory=time.time)

    def add_point(self, point: EmotionPoint) -> None:
        self.points.append(point)
        self.pressure.add(
            word_count=point.word_count,
            intensity=point.intensity,
            chaos=point.chaos,
            timestamp=point.timestamp,
        )

    def recent(self, n: int = 5) -> list[EmotionPoint]:
        return self.points[-n:]

    def for_headmate(self, name: str) -> list[EmotionPoint]:
        return [p for p in self.points if p.headmate and p.headmate.lower() == name.lower()]

    def trend(self, n: int = 4) -> dict:
        """Compute trend over last N points."""
        recent = self.recent(n)
        if not recent:
            return {"valence": 0.0, "intensity": 0.0, "chaos": 0.0, "direction": "stable"}

        first = recent[0]
        last  = recent[-1]

        valence_delta   = last.valence   - first.valence
        intensity_delta = last.intensity - first.intensity
        chaos_delta     = last.chaos     - first.chaos

        if intensity_delta > 0.15:
            direction = "escalating"
        elif intensity_delta < -0.15:
            direction = "de-escalating"
        elif chaos_delta > 0.2:
            direction = "fragmenting"
        elif chaos_delta < -0.2:
            direction = "stabilizing"
        else:
            direction = "stable"

        return {
            "valence":   round(valence_delta, 3),
            "intensity": round(intensity_delta, 3),
            "chaos":     round(chaos_delta, 3),
            "direction": direction,
        }

    def current_state(self) -> dict:
        """Most recent emotion point as a dict."""
        if not self.points:
            return {"valence": 0.0, "intensity": 0.2, "chaos": 0.0}
        p = self.points[-1]
        return {
            "valence":   round(p.valence, 3),
            "intensity": round(p.intensity, 3),
            "chaos":     round(p.chaos, 3),
            "headmate":  p.headmate,
            "register":  p.register,
        }

    def headmate_summary(self) -> dict:
        """Per-headmate current state and arc."""
        result = {}
        seen = set()
        for p in reversed(self.points):
            hm = (p.headmate or "unknown").lower()
            if hm not in seen:
                seen.add(hm)
                hm_points = self.for_headmate(hm)
                result[hm] = {
                    "current": {
                        "valence":   round(p.valence, 3),
                        "intensity": round(p.intensity, 3),
                        "chaos":     round(p.chaos, 3),
                        "register":  p.register,
                    },
                    "messages": len(hm_points),
                    "arc": [
                        {
                            "v": round(pt.valence, 2),
                            "i": round(pt.intensity, 2),
                            "c": round(pt.chaos, 2),
                        }
                        for pt in hm_points[-6:]
                    ],
                }
        return result

    def ego_block(self) -> str:
        """
        Format arc data for Ego's system prompt.
        Concise — just what Gizmo needs to read the room.
        """
        if not self.points:
            return ""

        lines = ["[Emotional arc]"]

        # Per-headmate summary
        hm_summary = self.headmate_summary()
        for hm, data in hm_summary.items():
            cur = data["current"]
            arc = data["arc"]
            name = hm.title()

            state_parts = []
            v = cur["valence"]
            i = cur["intensity"]
            c = cur["chaos"]

            if v > 0.4:
                state_parts.append("positive")
            elif v < -0.4:
                state_parts.append("negative")

            if i > 0.7:
                state_parts.append("high intensity")
            elif i < 0.2:
                state_parts.append("low energy")

            if c > 0.5:
                state_parts.append("fragmented")
            elif c > 0.3:
                state_parts.append("some scatter")

            state_str = ", ".join(state_parts) if state_parts else "neutral"

            # Trend
            if len(arc) >= 2:
                i_delta = arc[-1]["i"] - arc[0]["i"]
                c_delta = arc[-1]["c"] - arc[0]["c"]
                if i_delta > 0.2:
                    state_str += " (escalating)"
                elif c_delta > 0.2:
                    state_str += " (fragmenting)"
                elif i_delta < -0.2:
                    state_str += " (calming)"

            lines.append(f"  {name}: {state_str} | v={v:+.2f} i={i:.2f} c={c:.2f} ({data['messages']} messages)")

        # Session trend
        trend = self.trend()
        if trend["direction"] != "stable":
            lines.append(f"  Session trend: {trend['direction']}")

        return "\n".join(lines)


# ── EmotionTracker singleton ──────────────────────────────────────────────────

class EmotionTracker:
    """Singleton. Manages per-session arcs."""

    def __init__(self):
        self._sessions: dict[str, SessionArc] = {}
        log("EmotionTracker", "initialised")

    def _get(self, session_id: str) -> SessionArc:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionArc(session_id=session_id)
        return self._sessions[session_id]

    def record(
        self,
        session_id: str,
        message: str,
        headmate: Optional[str],
        register: str,
        topic: str = "general",
        timestamp: Optional[float] = None,
    ) -> EmotionPoint:
        """
        Record a message's emotional signature.
        Called by Archivist.receive() for every incoming message.
        Returns the EmotionPoint for immediate use.
        """
        ts = timestamp or time.time()
        arc = self._get(session_id)

        chaos    = _heuristic_chaos(message, arc.points)
        valence, intensity = _heuristic_valence_intensity(message, register, arc.points)

        point = EmotionPoint(
            timestamp=ts,
            headmate=headmate,
            register=register,
            valence=valence,
            intensity=intensity,
            chaos=chaos,
            message=message,
            word_count=len(message.split()),
            topic=topic,
        )

        arc.add_point(point)

        log_event("EmotionTracker", "POINT",
            session=session_id[:8],
            headmate=headmate or "unknown",
            register=register,
            valence=round(valence, 2),
            intensity=round(intensity, 2),
            chaos=round(chaos, 2),
        )

        return point

    def should_reflect(self, session_id: str) -> bool:
        """Check if the pressure gauge has crossed threshold."""
        return self._get(session_id).pressure.should_fire()

    def reset_pressure(self, session_id: str) -> None:
        """Reset pressure gauge after reflection fires."""
        self._get(session_id).pressure.reset()
        log_event("EmotionTracker", "PRESSURE_RESET", session=session_id[:8])

    def ego_block(self, session_id: str) -> str:
        """Get formatted arc block for Ego's system prompt."""
        return self._get(session_id).ego_block()

    def get_arc(self, session_id: str) -> SessionArc:
        return self._get(session_id)

    def current_state(self, session_id: str) -> dict:
        return self._get(session_id).current_state()

    def headmate_summary(self, session_id: str) -> dict:
        return self._get(session_id).headmate_summary()

    def recent_points(self, session_id: str, n: int = 8) -> list[EmotionPoint]:
        return self._get(session_id).recent(n)

    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# ── Singleton ─────────────────────────────────────────────────────────────────
emotion_tracker = EmotionTracker()
