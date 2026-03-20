"""
voice/tts.py
ElevenLabs text-to-speech pipeline with streaming playback.

Two sides:
  - GPU box: receives text, calls ElevenLabs streaming API, sends audio to Pi
  - Pi: receives audio stream, plays through earbuds immediately

Uses ElevenLabs streaming endpoint so playback starts before full
generation completes — critical for low-latency feel when Gizmo
is whispering something time-sensitive in your ear.

Architecture:
  GPU box:
    text → ElevenLabs streaming API → chunked audio → WebSocket → Pi

  Pi:
    WebSocket audio chunks → pyaudio playback buffer → earbuds

Requirements:
  GPU box: pip install elevenlabs websockets
  Pi:      pip install websockets pyaudio

Environment:
  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE_ID=...   (set after voice design)
  GPU_HOST=...              (Pi needs this to connect back)
  TTS_PORT=8767             (separate port from audio input stream)

Usage:
  GPU box: from voice.tts import TTSServer; await TTSServer().start()
  Pi:      python -m voice.tts_client
"""

import asyncio
import os
import struct
from typing import AsyncGenerator

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
TTS_PORT = int(os.getenv("TTS_PORT", "8767"))
GPU_HOST = os.getenv("GPU_HOST", "gpu-box.local")

# ElevenLabs model — flash is fastest, good quality
# multilingual-v2 if you need language flexibility
ELEVENLABS_MODEL = "eleven_flash_v2_5"

# Voice settings — tuned for "warm and close"
VOICE_SETTINGS = {
    "stability": 0.45,           # lower = more expressive, less robotic
    "similarity_boost": 0.85,    # high = stays true to the voice character
    "style": 0.20,               # subtle style variation
    "use_speaker_boost": True,   # enhances presence, makes it feel closer
}

# Audio format ElevenLabs will return
AUDIO_FORMAT = "pcm_22050"   # raw PCM 22050Hz, no encoding overhead
SAMPLE_RATE = 22050
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


# ── GPU box: TTS server ───────────────────────────────────────────────────────

class TTSServer:
    """
    Runs on the GPU box. Accepts text over WebSocket from the agent,
    streams it through ElevenLabs, forwards audio chunks to Pi clients.
    """

    def __init__(self):
        self._pi_connections: set = set()
        self._queue: asyncio.Queue = asyncio.Queue()

    async def start(self, port: int = TTS_PORT):
        """Start the TTS server. Call once at startup on GPU box."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets required: pip install websockets")

        # Start audio forwarding loop
        asyncio.ensure_future(self._process_queue())

        handler = lambda ws, path: self._handle_connection(ws, path)
        print(f"[TTS] Server starting on port {port}...")
        async with websockets.serve(handler, "0.0.0.0", port, max_size=None):
            print(f"[TTS] Ready — waiting for Pi and agent connections")
            await asyncio.Future()

    async def _handle_connection(self, websocket, path):
        """Route connections by path: /listen for Pi, /speak for agent."""
        if path == "/listen":
            # Pi connecting to receive audio
            self._pi_connections.add(websocket)
            print(f"[TTS] Pi connected for playback: {websocket.remote_address}")
            try:
                await websocket.wait_closed()
            finally:
                self._pi_connections.discard(websocket)
                print(f"[TTS] Pi disconnected")

        elif path == "/speak":
            # Agent sending text to speak
            print(f"[TTS] Agent connected for speech")
            try:
                async for message in websocket:
                    if isinstance(message, str):
                        await self._queue.put(message)
            except Exception as e:
                print(f"[TTS] Agent connection error: {e}")

    async def _process_queue(self):
        """Drain text queue, generate audio, broadcast to Pi clients."""
        while True:
            text = await self._queue.get()
            if not text.strip():
                continue
            print(f"[TTS] Generating: '{text[:60]}'")
            try:
                async for chunk in _stream_elevenlabs(text):
                    await self._broadcast(chunk)
                # Send end-of-utterance marker
                await self._broadcast(b"__END__")
            except Exception as e:
                print(f"[TTS] Generation error: {e}")

    async def _broadcast(self, audio_chunk: bytes):
        """Send audio chunk to all connected Pi clients."""
        dead = set()
        for ws in self._pi_connections:
            try:
                await ws.send(audio_chunk)
            except Exception:
                dead.add(ws)
        self._pi_connections -= dead


async def _stream_elevenlabs(text: str) -> AsyncGenerator[bytes, None]:
    """
    Stream PCM audio from ElevenLabs for the given text.
    Yields raw audio chunks as they arrive.
    """
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not set")
    if not ELEVENLABS_VOICE_ID:
        raise ValueError("ELEVENLABS_VOICE_ID not set")

    try:
        import httpx
    except ImportError:
        raise ImportError("httpx required: pip install httpx")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"

    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": VOICE_SETTINGS,
        "output_format": AUDIO_FORMAT,
    }

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk


# ── Convenience: speak from agent ─────────────────────────────────────────────

_tts_client_ws = None


async def speak(text: str, gpu_host: str = GPU_HOST, port: int = TTS_PORT):
    """
    Send text to the TTS server for playback.
    Called from the agent/queue drain loop on the GPU box.

    Since the TTS server is on the same machine as the agent,
    this connects to localhost.
    """
    global _tts_client_ws

    try:
        import websockets

        if _tts_client_ws is None or _tts_client_ws.closed:
            _tts_client_ws = await websockets.connect(
                f"ws://localhost:{port}/speak",
                ping_interval=10,
            )

        await _tts_client_ws.send(text)

    except Exception as e:
        print(f"[TTS] speak() failed: {e}")
        _tts_client_ws = None


# ── Pi-side client ─────────────────────────────────────────────────────────────

async def run_pi_client(gpu_host: str = GPU_HOST, port: int = TTS_PORT):
    """
    Runs on the Pi. Connects to TTS server, plays audio chunks through earbuds.
    Reconnects automatically if the connection drops.
    """
    try:
        import websockets
        import pyaudio
    except ImportError:
        raise ImportError("websockets and pyaudio required on Pi")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=1024,
    )

    uri = f"ws://{gpu_host}:{port}/listen"
    print(f"[TTS Client] Connecting to {uri}...")

    while True:
        try:
            async with websockets.connect(uri, max_size=None, ping_interval=10) as ws:
                print(f"[TTS Client] Connected — ready for audio")
                async for message in ws:
                    if isinstance(message, bytes):
                        if message == b"__END__":
                            # Utterance complete — small gap before next
                            await asyncio.sleep(0.05)
                        else:
                            # Write PCM directly to audio output
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None,
                                lambda: stream.write(message)
                            )

        except Exception as e:
            print(f"[TTS Client] Connection lost: {e} — reconnecting in 3s")
            await asyncio.sleep(3)


if __name__ == "__main__":
    # Run Pi client when executed directly
    import sys
    host = os.getenv("GPU_HOST", GPU_HOST)
    asyncio.run(run_pi_client(gpu_host=host))
