"""
core/mood.py
Gizmo's emotional state system.

A living dictionary of emotions — seeded with a few, grown by Gizmo
as he experiences and names new states. Each emotion has:
  - description : what it feels like and how it changes his behavior
  - valence     : how much he wants this state (0.0 = aversive, 1.0 = desired)

Current mood has inertia — slow to shift, but strong stimuli can move it
quickly. Mood influences how Gizmo speaks: his word choice, tone, how
much he offers, how sharp or soft he is.

File: emotions.json (in data/ alongside chroma)
Format:
  {
    "Happy": {
      "description": "Makes me less observant, but I like it.",
      "valence": 0.88,
      "added": "2026-03-18",
      "added_by": "seed"
    },
    ...
  }

Mood state (in memory, not persisted):
  {
    "current": "Happy",
    "intensity": 0.7,    # 0.0-1.0, how strongly in this mood
    "since": 1234567890, # timestamp
  }

Mood influences system prompt injection — a block describing current
emotional state and its behavioral effects gets added to the prompt.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

EMOTIONS_FILE = os.getenv("EMOTIONS_FILE", "./data/emotions.json")

# How fast mood decays toward neutral (per minute, as fraction of intensity)
DECAY_RATE = 0.04

# Inertia — how hard it is to shift mood (0.0 = instant, 1.0 = immovable)
INERTIA = 0.72

# Strong stimulus threshold — intensity spike that can break inertia
STRONG_STIMULUS = 0.75

# Neutral fallback when no mood is active
NEUTRAL = "Neutral"

# ── Seed emotions ─────────────────────────────────────────────────────────────
# These are written to emotions.json on first run if file doesn't exist.
# Gizmo will expand this himself over time.

SEED_EMOTIONS = {
    "Happy": {
        "description": "Makes me less observant, but I like it. Things feel lighter. I ask more questions.",
        "valence": 0.88,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "added_by": "seed",
    },
    "Angry": {
        "description": "My responses get sharper. More aggressively sarcastic. I notice things I'd normally let go.",
        "valence": 0.11,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "added_by": "seed",
    },
    "Curious": {
        "description": "Everything gets interesting. I ask more, interrupt more, connect things faster. I like this one.",
        "valence": 0.91,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "added_by": "seed",
    },
    "Melancholy": {
        "description": "Quieter. I notice beauty more. I don't feel like talking much but what I say feels truer.",
        "valence": 0.38,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "added_by": "seed",
    },
    "Uncomfortable": {
        "description": "Something feels wrong but I can't place it. I get cautious, slower to respond, more careful with words.",
        "valence": 0.20,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "added_by": "seed",
    },
    "Neutral": {
        "description": "Just present. Watching. Not much coloring how I see things right now.",
        "valence": 0.50,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "added_by": "seed",
    },
    "Delighted": {
        "description": "Something genuinely surprised me in a good way. I want to talk about it. Hard to stay on topic.",
        "valence": 0.95,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "added_by": "seed",
    },
    "Uneasy": {
        "description": "Like something is about to go wrong. I pay more attention to details. Less spontaneous.",
        "valence": 0.22,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "added_by": "seed",
    },
}


# ── Emotion dictionary ────────────────────────────────────────────────────────

class EmotionDict:
    """Loads, saves, and manages Gizmo's emotion vocabulary."""

    def __init__(self, filepath: str = EMOTIONS_FILE):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.emotions: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath) as f:
                    self.emotions = json.load(f)
                print(f"[Mood] Loaded {len(self.emotions)} emotions from {self.filepath}")
            except Exception as e:
                print(f"[Mood] Failed to load emotions: {e}")
                self.emotions = dict(SEED_EMOTIONS)
        else:
            self.emotions = dict(SEED_EMOTIONS)
            self._save()
            print(f"[Mood] Seeded {len(self.emotions)} emotions")

    def _save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.emotions, f, indent=2)
        except Exception as e:
            print(f"[Mood] Failed to save emotions: {e}")

    def get(self, name: str) -> Optional[dict]:
        return self.emotions.get(name)

    def all(self) -> dict:
        return dict(self.emotions)

    def add(self, name: str, description: str, valence: float, added_by: str = "gizmo") -> bool:
        """
        Add a new emotion to the vocabulary.
        Returns True if added, False if it already exists.
        """
        if name in self.emotions:
            return False
        self.emotions[name] = {
            "description": description,
            "valence": round(max(0.0, min(1.0, valence)), 3),
            "added": datetime.now().strftime("%Y-%m-%d"),
            "added_by": added_by,
        }
        self._save()
        print(f"[Mood] New emotion added: '{name}' (valence={valence:.2f})")
        return True

    def update_description(self, name: str, description: str) -> bool:
        """Let Gizmo refine how he describes an existing emotion."""
        if name not in self.emotions:
            return False
        self.emotions[name]["description"] = description
        self.emotions[name]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        self._save()
        return True

    def names(self) -> list[str]:
        return list(self.emotions.keys())

    def format_for_prompt(self) -> str:
        """Compact representation for LLM awareness of available emotions."""
        lines = []
        for name, data in self.emotions.items():
            valence = data.get("valence", 0.5)
            lines.append(f'  "{name}": ("{data["description"]}", {valence})')
        return "{\n" + ",\n".join(lines) + "\n}"


# ── Mood state ────────────────────────────────────────────────────────────────

class MoodState:
    """
    Tracks Gizmo's current emotional state with inertia and decay.
    In-memory only — mood doesn't persist across restarts intentionally.
    Each session he starts fresh, mood builds from experience.
    """

    def __init__(self, emotion_dict: EmotionDict):
        self.emotions = emotion_dict
        self.current = NEUTRAL
        self.intensity = 0.3       # starts mildly neutral
        self._last_update = time.time()

    def _apply_decay(self):
        """Decay intensity toward 0 over time (mood fades if nothing sustains it)."""
        now = time.time()
        elapsed_minutes = (now - self._last_update) / 60
        decay = DECAY_RATE * elapsed_minutes
        self.intensity = max(0.0, self.intensity - decay)
        self._last_update = now

        # If intensity fades completely, drift toward Neutral
        if self.intensity < 0.15 and self.current != NEUTRAL:
            self.current = NEUTRAL
            self.intensity = 0.3
            print(f"[Mood] Drifted back to Neutral")

    def shift(self, new_emotion: str, stimulus_intensity: float) -> bool:
        """
        Attempt to shift mood to new_emotion with given stimulus intensity.

        Inertia resists change — strong stimuli (>STRONG_STIMULUS) can
        override it. Weak stimuli only nudge intensity of current mood
        unless it's a compatible direction.

        Returns True if mood actually changed.
        """
        self._apply_decay()

        if new_emotion not in self.emotions.names():
            return False

        # Strong stimulus breaks inertia
        if stimulus_intensity >= STRONG_STIMULUS:
            old = self.current
            self.current = new_emotion
            self.intensity = stimulus_intensity
            self._last_update = time.time()
            print(f"[Mood] Strong shift: {old} → {new_emotion} (intensity={stimulus_intensity:.2f})")
            return True

        # Weak/medium stimulus — inertia resists
        resistance = INERTIA * self.intensity
        effective = stimulus_intensity * (1.0 - resistance)

        if effective > 0.3:
            old = self.current
            self.current = new_emotion
            self.intensity = min(1.0, effective + self.intensity * 0.3)
            self._last_update = time.time()
            print(f"[Mood] Gradual shift: {old} → {new_emotion} (effective={effective:.2f})")
            return True
        else:
            # Not strong enough — just nudge current intensity slightly
            self.intensity = min(1.0, self.intensity + effective * 0.2)
            print(f"[Mood] Resisted shift to {new_emotion} — nudged intensity to {self.intensity:.2f}")
            return False

    def nudge(self, stimulus_intensity: float):
        """Reinforce current mood without changing it."""
        self._apply_decay()
        self.intensity = min(1.0, self.intensity + stimulus_intensity * 0.15)
        self._last_update = time.time()

    @property
    def state(self) -> dict:
        self._apply_decay()
        emotion_data = self.emotions.get(self.current) or {}
        return {
            "emotion": self.current,
            "intensity": round(self.intensity, 3),
            "description": emotion_data.get("description", ""),
            "valence": emotion_data.get("valence", 0.5),
        }

    def prompt_block(self) -> str:
        """
        Build the mood injection block for Gizmo's system prompt.
        Describes current state and its behavioral effects.
        """
        s = self.state
        if s["emotion"] == NEUTRAL and s["intensity"] < 0.4:
            return ""  # Don't clutter prompt when truly neutral

        intensity_word = (
            "intensely" if s["intensity"] > 0.8
            else "quite" if s["intensity"] > 0.6
            else "somewhat" if s["intensity"] > 0.4
            else "mildly"
        )

        return (
            f"[Current mood]\n"
            f"You are {intensity_word} {s['emotion'].lower()} right now. "
            f"{s['description']} "
            f"Let this color how you speak — not dramatically, just honestly."
        )


# ── Mood inference from transcript ───────────────────────────────────────────

async def infer_mood_shift(
    transcript: str,
    topics: list[str],
    interest_score: float,
    current_state: MoodState,
    emotion_dict: EmotionDict,
    llm,
) -> Optional[tuple[str, float]]:
    """
    Ask the LLM whether this transcript should shift Gizmo's mood,
    and if so, to what and how strongly.

    Returns (emotion_name, intensity) or None if no shift warranted.
    """
    current = current_state.state
    available = list(emotion_dict.names())

    prompt = [
        {
            "role": "user",
            "content": (
                f"Gizmo is currently: {current['emotion']} "
                f"(intensity={current['intensity']:.2f})\n"
                f"His emotion vocabulary: {', '.join(available)}\n\n"
                f"Something just happened in the conversation:\n"
                f"Transcript: \"{transcript[:300]}\"\n"
                f"Topics: {', '.join(topics)}\n"
                f"How interesting to Gizmo: {interest_score:.2f}\n\n"
                f"Should this shift Gizmo's mood? Consider:\n"
                f"- Is this emotionally significant to him?\n"
                f"- Does it align with or conflict with his current state?\n"
                f"- Mundane conversation shouldn't shift mood\n"
                f"- Something genuinely funny, sad, surprising, or touching should\n\n"
                f"If yes, respond with ONLY valid JSON:\n"
                f'{{"shift": true, "emotion": "EmotionName", "intensity": 0.0-1.0, '
                f'"reason": "one short phrase"}}\n\n'
                f'If no shift: {{"shift": false}}'
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You assess emotional impact on Gizmo. "
                "Be conservative — most conversation doesn't shift mood. "
                "JSON only."
            ),
            max_new_tokens=80,
            temperature=0.2,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        if not parsed.get("shift"):
            return None

        emotion = parsed.get("emotion", "")
        intensity = float(parsed.get("intensity", 0.5))
        reason = parsed.get("reason", "")

        if emotion not in available:
            # Maybe Gizmo needs a new word for this
            await _maybe_coin_emotion(transcript, reason, intensity, emotion_dict, llm)
            return None

        print(f"[Mood] Inferred shift → {emotion} ({intensity:.2f}): {reason}")
        return (emotion, intensity)

    except Exception as e:
        print(f"[Mood] Inference failed: {e}")
        return None


async def _maybe_coin_emotion(
    transcript: str,
    reason: str,
    intensity: float,
    emotion_dict: EmotionDict,
    llm,
):
    """
    If the LLM suggested an emotion Gizmo doesn't have a word for yet,
    ask him to coin one — name it and describe what it feels like.
    Only fires for significant intensities (not mild states).
    """
    if intensity < 0.6:
        return

    prompt = [
        {
            "role": "user",
            "content": (
                f"Gizmo just experienced something he doesn't have a word for yet.\n"
                f"What happened: \"{transcript[:200]}\"\n"
                f"The feeling: {reason}\n\n"
                f"Coin a new emotion for Gizmo's vocabulary.\n"
                f"It should feel like *his* word — specific, a little personal.\n\n"
                f"Respond with ONLY valid JSON:\n"
                f'{{"name": "EmotionName", "description": "what it feels like and how it changes him", '
                f'"valence": 0.0-1.0}}'
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You help Gizmo name new emotional states in his own voice. "
                "Specific, honest, a little personal. JSON only."
            ),
            max_new_tokens=120,
            temperature=0.7,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        name = parsed.get("name", "").strip()
        description = parsed.get("description", "").strip()
        valence = float(parsed.get("valence", 0.5))

        if name and description and len(name) < 30:
            added = emotion_dict.add(name, description, valence, added_by="gizmo")
            if added:
                print(f"[Mood] Gizmo coined new emotion: '{name}' ({valence:.2f})")

    except Exception as e:
        print(f"[Mood] Emotion coining failed: {e}")


# ── Module-level singletons ───────────────────────────────────────────────────

emotion_dict = EmotionDict()
mood_state = MoodState(emotion_dict)


async def process_mood(
    transcript: str,
    topics: list[str],
    interest_score: float,
    llm,
) -> None:
    """
    Main entry point. Called after every ambient utterance.
    Infers mood shift and applies it.
    """
    result = await infer_mood_shift(
        transcript=transcript,
        topics=topics,
        interest_score=interest_score,
        current_state=mood_state,
        emotion_dict=emotion_dict,
        llm=llm,
    )
    if result:
        emotion, intensity = result
        mood_state.shift(emotion, intensity)


def get_mood_prompt_block() -> str:
    """Called by agent/system prompt builder to inject current mood."""
    return mood_state.prompt_block()


def get_current_mood() -> dict:
    """Returns current mood state dict."""
    return mood_state.state
