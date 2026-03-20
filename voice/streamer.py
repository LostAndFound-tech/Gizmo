"""
voice/streamer.py
Pi-side audio capture and streaming.

Replaces the batch listen() generator in ambient/listener.py.
Captures raw PCM from the microphone and streams it over WebSocket
to the GPU box receiver in real time.

No VAD or chunking here — that moves to the GPU box where we have
the compute to do it properly alongside speaker ID.

Requirements (Pi):
    pip install websockets pyaudio

Environment:
    GPU_HOST=192.168.x.x   (IP of GPU box)
    GPU_PORT=8766           (default)

Usage:
    python -m voice.streamer
    # or import and call stream_mic() in your Pi entrypoint
"""

import asyncio
import os
import sys
import struct
import time

# Audio settings — must match receiver expectations
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2  # bytes (16-bit)

GPU_HOST = os.getenv("GPU_HOST", "gpu-box.local")
GPU_PORT = int(os.getenv("GPU_PORT", "8766"))

# Reconnect settings
RECONNECT_DELAY = 3      # seconds between reconnect attempts
MAX_RECONNECT_TRIES = 0  # 0 = infinite


def _get_pyaudio():
    try:
        import pyaudio
        return pyaudio
    except ImportError:
        raise ImportError("pyaudio is required on the Pi: pip install pyaudio")


def _get_websockets():
    try:
        import websockets
        return websockets
    except ImportError:
        raise ImportError("websockets is required: pip install websockets")


async def _capture_and_stream(ws, device_index=None):
    """
    Open mic, read frames, send over WebSocket.
    Runs until the WebSocket closes or an exception occurs.

    Frame format sent over wire:
        4 bytes: timestamp (uint32, seconds since epoch)
        2 bytes: frame size in bytes (uint16)
        N bytes: raw PCM (16kHz mono 16-bit LE)
    """
    pyaudio = _get_pyaudio()
    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=FRAME_SIZE // 2,
    )

    print(f"[Streamer] Mic open → streaming to GPU box")
    loop = asyncio.get_event_loop()

    try:
        while True:
            frame = await loop.run_in_executor(
                None,
                lambda: stream.read(FRAME_SIZE // 2, exception_on_overflow=False),
            )

            if len(frame) != FRAME_SIZE:
                continue

            # Pack header: timestamp (4B) + frame_size (2B) + pcm
            ts = int(time.time())
            header = struct.pack(">IH", ts, len(frame))
            await ws.send(header + frame)

    except Exception as e:
        print(f"[Streamer] Capture error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("[Streamer] Mic closed.")


async def stream_mic(device_index=None):
    """
    Main streaming loop with auto-reconnect.
    Connects to GPU box WebSocket and streams PCM indefinitely.
    """
    websockets = _get_websockets()
    uri = f"ws://{GPU_HOST}:{GPU_PORT}/audio"
    attempts = 0

    while True:
        try:
            print(f"[Streamer] Connecting to {uri}...")
            async with websockets.connect(
                uri,
                ping_interval=10,
                ping_timeout=5,
                max_size=None,  # no message size limit
            ) as ws:
                attempts = 0
                print(f"[Streamer] Connected.")
                await _capture_and_stream(ws, device_index=device_index)

        except Exception as e:
            attempts += 1
            print(f"[Streamer] Connection lost: {e} (attempt {attempts})")
            if MAX_RECONNECT_TRIES and attempts >= MAX_RECONNECT_TRIES:
                print("[Streamer] Max reconnect attempts reached. Exiting.")
                break
            await asyncio.sleep(RECONNECT_DELAY)


def list_devices():
    """Print available audio input devices. Run this on the Pi to find device_index."""
    pyaudio = _get_pyaudio()
    pa = pyaudio.PyAudio()
    print("\nAvailable audio input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']} — {int(info['defaultSampleRate'])}Hz")
    pa.terminate()
    print()


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_devices()
    else:
        device = None
        for arg in sys.argv[1:]:
            if arg.startswith("--device="):
                device = int(arg.split("=")[1])
        asyncio.run(stream_mic(device_index=device))
