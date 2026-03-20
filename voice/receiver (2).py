"""
voice/receiver.py
GPU-side WebSocket server. Receives raw PCM from the Pi streamer,
runs real-time transcription (Faster-Whisper) and speaker ID (pyannote)
in parallel, then feeds results into the existing ambient pipeline.

Architecture:
    Pi streamer → WebSocket → receiver buffer
                                    ├── VAD → Faster-Whisper → transcript
                                    └── pyannote embedding → speaker ID
                                    ↓
                              SpeakerResult(transcript, speaker, confidence)
                                    ↓
                              existing: tagger → rag → directed_queue

Requirements (GPU box):
    pip install faster-whisper pyannote.audio websockets torch webrtcvad numpy

Environment:
    GPU_PORT=8766
    HF_TOKEN=...             (for pyannote model)
    VOICE_PROFILES_DIR=...   (default: ./data/voice_profiles)

Usage:
    python -m voice.receiver   # starts the WebSocket server
"""

import asyncio
import collections
import os
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

GPU_PORT = int(os.getenv("GPU_PORT", "8766"))
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2

# VAD settings
VAD_AGGRESSIVENESS = 2
SILENCE_THRESHOLD_FRAMES = 25  # ~0.75s
MIN_SPEECH_FRAMES = 10         # ~0.3s

# Speaker ID settings
ID_MIN_AUDIO_SECONDS = 1.5   # minimum audio length for reliable speaker embedding


@dataclass
class SpeakerResult:
    transcript: str
    speaker_name: Optional[str]
    speaker_confidence: float
    speaker_status: str          # "confident" | "possible" | "unknown"
    audio_bytes: bytes           # raw PCM of the utterance
    embedding: Optional[np.ndarray]
    timestamp: float


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_whisper():
    if not hasattr(_load_whisper, "_model"):
        try:
            from faster_whisper import WhisperModel
            print("[Receiver] Loading Faster-Whisper model...")
            _load_whisper._model = WhisperModel(
                "base",
                device="cuda",
                compute_type="float16",
            )
            print("[Receiver] Faster-Whisper ready.")
        except ImportError:
            raise ImportError("faster-whisper required: pip install faster-whisper")
    return _load_whisper._model


def _load_vad():
    if not hasattr(_load_vad, "_vad"):
        try:
            import webrtcvad
            _load_vad._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        except ImportError:
            raise ImportError("webrtcvad required: pip install webrtcvad")
    return _load_vad._vad


# ── Transcription ─────────────────────────────────────────────────────────────

async def transcribe_utterance(audio_bytes: bytes) -> Optional[str]:
    """
    Run Faster-Whisper on a complete utterance (VAD-bounded PCM bytes).
    Returns transcript text or None if nothing meaningful.
    """
    loop = asyncio.get_event_loop()

    def _run():
        model = _load_whisper()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = model.transcribe(
            audio_np,
            language=None,
            beam_size=5,
            vad_filter=True,        # Faster-Whisper has built-in VAD too
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )

        text = " ".join(s.text for s in segments).strip()

        # Quick meaningfulness checks
        words = text.split()
        if len(words) < 4:
            return None
        if info.language_probability < 0.5:
            return None

        return text

    try:
        return await loop.run_in_executor(None, _run)
    except Exception as e:
        print(f"[Receiver] Transcription error: {e}")
        return None


# ── Speaker ID ────────────────────────────────────────────────────────────────

async def identify_speaker(audio_bytes: bytes) -> tuple[Optional[np.ndarray], Optional[str], float, str]:
    """
    Compute speaker embedding and identify against profile store.
    Returns (embedding, name, confidence, status).
    """
    from voice.enrollment import compute_embedding, profile_store

    # Need enough audio for a reliable embedding
    duration = len(audio_bytes) / (SAMPLE_RATE * 2)
    if duration < ID_MIN_AUDIO_SECONDS:
        return None, None, 0.0, "too_short"

    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(None, compute_embedding, audio_bytes)

    if embedding is None:
        return None, None, 0.0, "embedding_failed"

    name, confidence, status = profile_store.identify(embedding)
    return embedding, name, confidence, status


# ── VAD + chunking ────────────────────────────────────────────────────────────

class VADBuffer:
    """
    Accumulates streaming PCM frames and yields complete utterances
    when silence is detected. Same logic as listener.py but live.
    """

    def __init__(self):
        self.vad = _load_vad()
        self.ring = collections.deque(maxlen=15)
        self.speech_frames = []
        self.silence_count = 0
        self.in_speech = False
        self.max_frames = int(60 * 1000 / FRAME_DURATION_MS)  # 60s force-flush

    def push(self, frame: bytes) -> Optional[bytes]:
        """
        Push a frame. Returns a complete utterance (bytes) if one just ended,
        else None.
        """
        if len(frame) != FRAME_SIZE:
            return None

        try:
            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            return None

        if is_speech:
            if not self.in_speech:
                self.speech_frames.extend(self.ring)
                self.in_speech = True
                self.silence_count = 0
            self.speech_frames.append(frame)
            self.silence_count = 0
        else:
            self.ring.append(frame)
            if self.in_speech:
                self.silence_count += 1
                self.speech_frames.append(frame)

                if self.silence_count >= SILENCE_THRESHOLD_FRAMES:
                    if len(self.speech_frames) >= MIN_SPEECH_FRAMES:
                        utterance = b"".join(self.speech_frames)
                        self._reset()
                        return utterance
                    self._reset()

        # Force-flush long utterances
        if self.in_speech and len(self.speech_frames) >= self.max_frames:
            utterance = b"".join(self.speech_frames)
            self._reset()
            return utterance

        return None

    def _reset(self):
        self.speech_frames = []
        self.silence_count = 0
        self.in_speech = False


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def _handle_client(websocket, result_queue: asyncio.Queue):
    """
    Handle one Pi client connection.
    Parses framed PCM, runs VAD, dispatches utterances for processing.
    """
    client_addr = websocket.remote_address
    print(f"[Receiver] Pi connected from {client_addr}")

    vad_buf = VADBuffer()
    buffer = b""  # reassembly buffer for partial WebSocket messages

    HEADER_SIZE = 6  # 4B timestamp + 2B frame_size

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue

            buffer += message

            # Parse all complete frames from buffer
            while len(buffer) >= HEADER_SIZE:
                ts, frame_size = struct.unpack(">IH", buffer[:HEADER_SIZE])
                total = HEADER_SIZE + frame_size

                if len(buffer) < total:
                    break  # wait for more data

                frame = buffer[HEADER_SIZE:total]
                buffer = buffer[total:]

                utterance = vad_buf.push(frame)
                if utterance:
                    # Dispatch for parallel processing — don't block the receive loop
                    asyncio.ensure_future(_process_utterance(utterance, result_queue))

    except Exception as e:
        print(f"[Receiver] Client {client_addr} error: {e}")
    finally:
        print(f"[Receiver] Pi disconnected: {client_addr}")


async def _process_utterance(audio_bytes: bytes, result_queue: asyncio.Queue):
    """
    Run transcription and speaker ID concurrently for one utterance.
    Puts a SpeakerResult into the queue when both complete.
    """
    transcript_task = asyncio.ensure_future(transcribe_utterance(audio_bytes))
    speaker_task = asyncio.ensure_future(identify_speaker(audio_bytes))

    transcript, (embedding, name, confidence, status) = await asyncio.gather(
        transcript_task, speaker_task
    )

    if not transcript:
        return  # discard meaningless audio

    result = SpeakerResult(
        transcript=transcript,
        speaker_name=name,
        speaker_confidence=confidence,
        speaker_status=status,
        audio_bytes=audio_bytes,
        embedding=embedding,
        timestamp=time.time(),
    )

    speaker_str = f"{name} ({confidence:.2f})" if name else f"unknown ({status})"
    print(f"[Receiver] '{transcript[:60]}' → speaker: {speaker_str}")

    await result_queue.put(result)


# ── Result consumer (feeds existing pipeline) ─────────────────────────────────

async def consume_results(
    result_queue: asyncio.Queue,
    llm,
    context_fn=None,
    directed_queue: asyncio.Queue = None,
):
    """
    Reads SpeakerResults from the queue and feeds them into the existing
    tagger → rag → directed_queue pipeline.

    Voice ID is used ONLY for:
        - Tagging ambient transcripts with who's speaking
        - Adding newly detected voices to fronters (not switching host)
        - Auto-learning from confidently identified speech

    current_host is NEVER changed here — that's always a manual switch_host call.
    Conversation is constant and multi-voice; voice ID ≠ fronting.

    Unknown speakers: quietly noted in metadata, no verbal prompt needed
    unless a voice appears that has no profile at all (genuinely new person).
    """
    from ambient.tagger import tag
    from ambient.reminders import store_reminder
    from voice.enrollment import profile_store
    from voice.auto_learn import AutoLearner

    learner = AutoLearner(profile_store)

    print("[Receiver] Result consumer started.")

    while True:
        result: SpeakerResult = await result_queue.get()

        # Get current context (read-only — we never write current_host from here)
        context = context_fn() if context_fn else {}

        # ── Speaker ID: tag and optionally add to fronters ────────────────────
        if result.speaker_status == "confident" and result.speaker_name:
            speaker_name = result.speaker_name

            # Add to fronters if not already present — but DO NOT touch current_host
            fronters = [f.lower() for f in context.get("fronters", [])]
            if speaker_name not in fronters:
                print(f"[Receiver] Adding '{speaker_name}' to fronters (voice detected, confidence: {result.speaker_confidence:.2f})")
                context = _add_to_fronters(context, speaker_name)

            # Passive auto-learn — reinforce confident identifications
            if result.embedding is not None:
                learner.learn_passive(result, speaker_name)

        elif result.speaker_status == "unknown" and not profile_store.profiles:
            # No profiles at all yet — system isn't set up, stay quiet
            pass

        # "possible" and low-confidence unknowns: just log, don't interrupt

        # ── Feed into tagger + personality + conflict detection (parallel) ─────
        try:
            personality_speaker = (
                result.speaker_name
                if result.speaker_status == "confident" and result.speaker_name
                else context.get("current_host", "")
            )

            # All three run concurrently — none blocks the others
            tag_task = asyncio.ensure_future(
                tag(result.transcript, llm=llm, context=context)
            )
            personality_task = asyncio.ensure_future(
                _run_personality(
                    transcript=result.transcript,
                    speaker=personality_speaker,
                    llm=llm,
                    directed_queue=directed_queue,
                    context=context,
                )
            )
            conflict_task = asyncio.ensure_future(
                _run_conflict_detection(
                    transcript=result.transcript,
                    speaker=personality_speaker,
                    llm=llm,
                    directed_queue=directed_queue,
                    context=context,
                )
            )
            voice_task = asyncio.ensure_future(
                _run_voice_pass(
                    transcript=result.transcript,
                    speaker=personality_speaker,
                    topics=[],
                    context=context,
                    llm=llm,
                    directed_queue=directed_queue,
                    timestamp=result.timestamp,
                )
            )
            mood_task = asyncio.ensure_future(
                _run_mood(
                    transcript=result.transcript,
                    topics=[],  # filled from tag_result below
                    interest_score=0.0,
                    llm=llm,
                )
            )

            tag_result, _, _, _, _ = await asyncio.gather(
                tag_task, personality_task, conflict_task, voice_task, mood_task
            )

            # Handle reminder
            if tag_result.reminder:
                r = tag_result.reminder
                fronter = context.get("current_host", result.speaker_name or "")
                store_reminder(
                    due_iso=r["due_iso"],
                    due_date=r["due_date"],
                    due_hour=r["due_hour"],
                    due_minute=r["due_minute"],
                    message=r["message"],
                    set_by=fronter,
                    raw_transcript=result.transcript,
                )

            # Handle directed at Gizmo
            if tag_result.directed_at_gizmo and directed_queue:
                await directed_queue.put({
                    "transcript": result.transcript,
                    "context": context,
                })

            # Ingest into ambient log
            await _ingest_ambient(tag_result, result, context)

        except Exception as e:
            print(f"[Receiver] Pipeline error: {e}")


async def _run_personality(
    transcript: str,
    speaker: str,
    llm,
    directed_queue: Optional[asyncio.Queue],
    context: dict,
) -> None:
    """
    Wrapper that runs personality extraction and swallows errors
    so a personality failure never kills the main pipeline.
    """
    if not speaker:
        return
    try:
        from voice.personality import process_transcript
        await process_transcript(
            transcript=transcript,
            speaker=speaker,
            llm=llm,
            directed_queue=directed_queue,
            context=context,
        )
    except Exception as e:
        print(f"[Receiver] Personality extraction error (non-fatal): {e}")


async def _run_mood(
    transcript: str,
    topics: list,
    interest_score: float,
    llm,
) -> None:
    """Wrapper for mood processing — non-fatal."""
    try:
        from voice.mood import process_mood
        await process_mood(
            transcript=transcript,
            topics=topics,
            interest_score=interest_score,
            llm=llm,
        )
    except Exception as e:
        print(f"[Receiver] Mood processing error (non-fatal): {e}")


async def _run_voice_pass(
    transcript: str,
    speaker: str,
    topics: list,
    context: dict,
    llm,
    directed_queue: Optional[asyncio.Queue],
    timestamp: float = 0,
) -> None:
    """Wrapper for Gizmo's spontaneous voice — non-fatal."""
    if not directed_queue:
        return
    try:
        from voice.gizmo_voice import run_voice_pass
        await run_voice_pass(
            transcript=transcript,
            speaker=speaker,
            topics=topics,
            context=context,
            llm=llm,
            directed_queue=directed_queue,
            last_utterance_time=timestamp,
        )
    except Exception as e:
        print(f"[Receiver] Voice pass error (non-fatal): {e}")


async def _run_conflict_detection(
    transcript: str,
    speaker: str,
    llm,
    directed_queue: Optional[asyncio.Queue],
    context: dict,
) -> None:
    """
    Wrapper for conflict detection — swallows errors so it never
    kills the main pipeline.
    """
    if not speaker:
        return
    try:
        from voice.conflict_detector import run_conflict_detection
        await run_conflict_detection(
            transcript=transcript,
            speaker=speaker,
            llm=llm,
            directed_queue=directed_queue,
            context=context,
        )
    except Exception as e:
        print(f"[Receiver] Conflict detection error (non-fatal): {e}")
    """
    Return updated context with name added to fronters.
    Never modifies current_host — that's always a manual switch_host call.
    """
    updated = dict(context)
    fronters = list({f.lower() for f in updated.get("fronters", [])} | {name.lower()})
    updated["fronters"] = fronters
    return updated


def _looks_like_name(text: str) -> bool:
    """Heuristic: does this short utterance look like a name response?"""
    words = text.strip().split()
    if len(words) > 6:
        return False
    triggers = ["i'm", "i am", "it's", "it is", "this is", "it's me"]
    text_lower = text.lower()
    return any(t in text_lower for t in triggers) or len(words) <= 2


def _extract_name(text: str) -> Optional[str]:
    """Extract a name from responses like 'I'm Alice' or just 'Alice'."""
    import re
    text = text.strip()
    # "I'm X" / "It's X" / "This is X"
    match = re.search(r"(?:i'm|i am|it's|it is|this is)\s+(\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Just a single word
    words = text.split()
    if len(words) == 1 and words[0].isalpha():
        return words[0].lower()
    return None


def _update_host(context: dict, name: str) -> dict:
    """Return updated context dict with new host."""
    updated = dict(context)
    updated["current_host"] = name
    fronters = list(set(updated.get("fronters", []) + [name]))
    updated["fronters"] = fronters
    return updated


async def _ingest_ambient(tag_result, result: SpeakerResult, context: dict):
    """
    Ingest a tagged utterance into the ambient_log RAG collection.
    Speaker name (from voice ID) is stored in metadata so RAG queries
    like 'what did Alice say about X' can filter by speaker.
    """
    from core.rag import RAGStore
    import time

    now = datetime.now()

    # Use voice-identified speaker for attribution, fall back to current_host
    speaker = result.speaker_name or context.get("current_host", "")
    host = context.get("current_host", "")

    metadata = {
        "source": "ambient_mic_streaming",
        "type": "ambient_transcript",
        "date": now.strftime("%Y-%m-%d"),
        "hour": now.strftime("%H"),
        "time": now.strftime("%H:%M"),
        "timestamp": str(int(time.time())),
        "topics": ", ".join(tag_result.topics),
        "raw_transcript": result.transcript[:500],
        # Voice ID attribution — who was actually speaking this utterance
        "speaker": speaker,
        "speaker_confidence": str(round(result.speaker_confidence, 3)),
        "speaker_status": result.speaker_status,
        # Current host context (who's fronting, may differ from speaker)
        "fronter": host,
    }

    try:
        store = RAGStore(collection_name="ambient_log")
        store.ingest_texts([tag_result.summary], metadatas=[metadata])

        # Also ingest into the speaker's personal collection if confident
        if result.speaker_status == "confident" and result.speaker_name:
            personal_store = RAGStore(collection_name=result.speaker_name.lower())
            personal_store.ingest_texts(
                [tag_result.summary],
                metadatas=[{**metadata, "collection": result.speaker_name.lower()}],
            )
    except Exception as e:
        print(f"[Receiver] Ingest error: {e}")


# ── Server entrypoint ─────────────────────────────────────────────────────────

async def start_receiver(
    llm,
    context_fn=None,
    directed_queue: asyncio.Queue = None,
    port: int = GPU_PORT,
):
    """
    Start the WebSocket receiver server.
    Call this on the GPU box instead of start_ambient() from pipeline.py.
    """
    try:
        import websockets
    except ImportError:
        raise ImportError("websockets required: pip install websockets")

    result_queue = asyncio.Queue()

    # Start result consumer (feeds tagger/rag/directed_queue)
    asyncio.ensure_future(consume_results(
        result_queue,
        llm=llm,
        context_fn=context_fn,
        directed_queue=directed_queue,
    ))

    handler = lambda ws, path: _handle_client(ws, result_queue)

    print(f"[Receiver] Starting WebSocket server on port {port}...")
    async with websockets.serve(handler, "0.0.0.0", port, max_size=None):
        print(f"[Receiver] Listening for Pi connections on :{port}")
        await asyncio.Future()  # run forever
