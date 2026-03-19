"""
ambient/listener.py
Continuous microphone capture with voice activity detection (VAD).

Uses webrtcvad to detect speech frames. Accumulates speech into chunks,
then yields complete utterances when silence is detected.

Requirements:
    pip install webrtcvad pyaudio

Usage:
    async for audio_chunk in listen():
        # audio_chunk is raw PCM bytes (16kHz, mono, 16-bit)
        ...
"""

import asyncio
import collections
from typing import AsyncGenerator

# VAD aggressiveness: 0 (least aggressive) to 3 (most aggressive)
# 2 is a good balance — filters noise but catches normal speech
VAD_AGGRESSIVENESS = 2

# Audio settings — Whisper wants 16kHz mono 16-bit PCM
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30          # webrtcvad supports 10, 20, or 30ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2  # bytes (16-bit)

# Silence detection
SILENCE_THRESHOLD_FRAMES = 30   # ~0.9s of silence ends an utterance
MIN_SPEECH_FRAMES = 10          # ~0.3s minimum — ignore very short sounds
MAX_UTTERANCE_SECONDS = 60      # force-flush after 60s regardless


def _get_vad():
    try:
        import webrtcvad
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        return vad
    except ImportError:
        raise ImportError(
            "webrtcvad is required for ambient listening: pip install webrtcvad"
        )


def _get_pyaudio():
    try:
        import pyaudio
        return pyaudio
    except ImportError:
        raise ImportError(
            "pyaudio is required for ambient listening: pip install pyaudio"
        )


async def listen(
    device_index: int = None,
) -> AsyncGenerator[bytes, None]:
    """
    Async generator. Yields complete speech utterances as raw PCM bytes.
    Runs until cancelled.

    Each yielded chunk is a complete utterance — speech bounded by silence.
    """
    vad = _get_vad()
    pyaudio = _get_pyaudio()

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=FRAME_SIZE // 2,  # pyaudio counts samples, not bytes
    )

    print("[Listener] Microphone open. Listening...")

    # Ring buffer of recent frames for context (captures lead-in before VAD triggers)
    ring = collections.deque(maxlen=15)  # ~0.45s of pre-speech context

    speech_frames = []
    silence_count = 0
    in_speech = False
    max_frames = int(MAX_UTTERANCE_SECONDS * 1000 / FRAME_DURATION_MS)

    try:
        loop = asyncio.get_event_loop()

        while True:
            # Read is blocking — run in executor to not block event loop
            frame = await loop.run_in_executor(
                None,
                lambda: stream.read(FRAME_SIZE // 2, exception_on_overflow=False)
            )

            # webrtcvad expects exactly FRAME_SIZE bytes
            if len(frame) != FRAME_SIZE:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                if not in_speech:
                    # Speech just started — include ring buffer as pre-roll
                    speech_frames.extend(ring)
                    in_speech = True
                    silence_count = 0
                speech_frames.append(frame)
                silence_count = 0
            else:
                ring.append(frame)
                if in_speech:
                    silence_count += 1
                    speech_frames.append(frame)  # include trailing silence

                    # End of utterance
                    if silence_count >= SILENCE_THRESHOLD_FRAMES:
                        if len(speech_frames) >= MIN_SPEECH_FRAMES:
                            yield b"".join(speech_frames)
                        speech_frames = []
                        silence_count = 0
                        in_speech = False

            # Force-flush very long utterances
            if in_speech and len(speech_frames) >= max_frames:
                print("[Listener] Force-flushing long utterance")
                yield b"".join(speech_frames)
                speech_frames = []
                silence_count = 0
                in_speech = False

    except asyncio.CancelledError:
        print("[Listener] Cancelled.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("[Listener] Microphone closed.")
