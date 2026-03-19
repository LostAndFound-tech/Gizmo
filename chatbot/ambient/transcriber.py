"""
ambient/transcriber.py
Whisper-based speech-to-text for ambient audio chunks.

Filters out meaningless audio before it ever hits the LLM or RAG store:
  1. Word count check (too short = discard)
  2. Filler-word check (only "uh", "okay", "yeah" etc. = discard)
  3. Confidence check (Whisper no-speech probability)

Requirements:
    pip install openai-whisper

For GPU acceleration:
    pip install torch torchvision torchaudio
    (will auto-use CUDA if available, falls back to CPU)
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Optional

# Whisper model size. Tradeoffs:
#   tiny   — fastest, least accurate (~1GB RAM)
#   base   — good balance (~1GB RAM)             ← recommended for ambient
#   small  — more accurate (~2GB RAM)
#   medium — high accuracy (~5GB RAM)
WHISPER_MODEL = "base"

# Meaningfulness thresholds
MIN_WORD_COUNT = 8          # fewer words than this → discard
NO_SPEECH_THRESHOLD = 0.6   # Whisper's no_speech_prob above this → discard

# Filler phrases — if transcript is ONLY these words, discard
FILLER_PATTERNS = re.compile(
    r"^[\s,.]*(uh+|um+|mm+|hmm+|hm+|okay|ok|yeah|yep|yes|no|nope|"
    r"right|sure|alright|anyway|so|like|well|just|oh|ah|ow|ow+|"
    r"thanks|thank you|bye|goodbye|hello|hi|hey)[\s,.]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Whisper sometimes hallucinates these when there's background noise
HALLUCINATION_PATTERNS = re.compile(
    r"(thank you for watching|please subscribe|subtitles by|"
    r"translation by|www\.|\.com)",
    re.IGNORECASE,
)


@dataclass
class TranscriptResult:
    text: str
    language: str
    no_speech_prob: float
    meaningful: bool
    discard_reason: Optional[str] = None


_model_cache = {}


def _load_model(model_name: str = WHISPER_MODEL):
    """Load Whisper model, cached after first load."""
    if model_name not in _model_cache:
        print(f"[Transcriber] Loading Whisper model '{model_name}'...")
        try:
            import whisper
            _model_cache[model_name] = whisper.load_model(model_name)
            print(f"[Transcriber] Whisper '{model_name}' ready.")
        except ImportError:
            raise ImportError(
                "openai-whisper is required: pip install openai-whisper"
            )
    return _model_cache[model_name]


def _is_meaningful(text: str, no_speech_prob: float) -> tuple[bool, Optional[str]]:
    """
    Returns (is_meaningful, discard_reason).
    discard_reason is None if meaningful.
    """
    text = text.strip()

    if not text:
        return False, "empty"

    if no_speech_prob > NO_SPEECH_THRESHOLD:
        return False, f"low confidence (no_speech_prob={no_speech_prob:.2f})"

    if HALLUCINATION_PATTERNS.search(text):
        return False, "whisper hallucination pattern"

    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False, f"too short ({len(words)} words)"

    # Check if it's ONLY filler words (strip punctuation first)
    stripped = re.sub(r"[^\w\s]", "", text).strip()
    if FILLER_PATTERNS.match(stripped):
        return False, "filler only"

    return True, None


async def transcribe(
    audio_bytes: bytes,
    model_name: str = WHISPER_MODEL,
) -> TranscriptResult:
    """
    Transcribe raw PCM bytes (16kHz mono 16-bit) using Whisper.
    Returns TranscriptResult with meaningfulness assessment.
    Runs Whisper in executor to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    def _run():
        import numpy as np
        import whisper

        model = _load_model(model_name)

        # Convert raw PCM bytes to float32 numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Pad or trim to 30s (Whisper's expected input length)
        audio_np = whisper.pad_or_trim(audio_np)

        # Run transcription with no-speech detection
        result = model.transcribe(
            audio_np,
            language=None,          # auto-detect language
            verbose=False,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            condition_on_previous_text=False,  # better for ambient snippets
        )

        text = result.get("text", "").strip()
        language = result.get("language", "en")

        # Whisper returns per-segment no_speech_prob; take the max
        segments = result.get("segments", [])
        no_speech_prob = max(
            (s.get("no_speech_prob", 0.0) for s in segments),
            default=0.0,
        )

        meaningful, reason = _is_meaningful(text, no_speech_prob)

        return TranscriptResult(
            text=text,
            language=language,
            no_speech_prob=no_speech_prob,
            meaningful=meaningful,
            discard_reason=reason,
        )

    result = await loop.run_in_executor(None, _run)

    if result.meaningful:
        print(f"[Transcriber] ✓ '{result.text[:80]}...' " if len(result.text) > 80 else f"[Transcriber] ✓ '{result.text}'")
    else:
        print(f"[Transcriber] ✗ Discarded: {result.discard_reason}")

    return result
