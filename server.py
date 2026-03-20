"""
server.py
Gizmo WebSocket server for Render deployment.

Clients connect, send JSON messages, receive streaming token responses.

Message format (client → server):
    {
        "message": "hey gizmo what do you think about...",
        "context": {                        # optional
            "current_host": "alice",
            "fronters": ["alice", "bob"]
        },
        "session_id": "abc123"             # optional, defaults to connection id
    }

Message format (server → client):
    {"type": "token",   "data": "..."}     # streaming token
    {"type": "done",    "data": ""}        # response complete
    {"type": "error",   "data": "..."}     # something went wrong
    {"type": "mood",    "data": {...}}      # current mood state (sent with done)

Run locally:
    python server.py

Deploy on Render:
    - Runtime: Python 3.11
    - Build command: pip install -r requirements.txt
    - Start command: python server.py
    - Persistent disk: mount at /data (needed for ChromaDB)
    - Environment variables: see .env.example
"""

import asyncio
import json
import os
import uuid
import websockets
from websockets.server import WebSocketServerProtocol

from core.agent import agent, build_system_prompt, TOOL_REGISTRY
from core.llm import llm
from memory.history import get_session, ConversationHistory

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8765"))


async def handle_client(websocket: WebSocketServerProtocol, path: str):
    """Handle one WebSocket client connection."""
    conn_id = str(uuid.uuid4())[:8]
    print(f"[Server] Client connected: {conn_id}")

    try:
        async for raw in websocket:
            await handle_message(websocket, raw, conn_id)

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[Server] Client {conn_id} disconnected unexpectedly: {e}")
    except Exception as e:
        print(f"[Server] Client {conn_id} error: {e}")
    finally:
        print(f"[Server] Client disconnected: {conn_id}")


async def handle_message(
    websocket: WebSocketServerProtocol,
    raw: str,
    conn_id: str,
):
    """Parse and handle one incoming message."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await send(websocket, "error", "Invalid JSON")
        return

    message = data.get("message", "").strip()
    if not message:
        await send(websocket, "error", "Empty message")
        return

    context = data.get("context") or {}
    session_id = data.get("session_id") or conn_id

    # Ensure context has at least empty fronters
    context.setdefault("fronters", [])

    history = get_session(session_id)

    print(f"[Server] [{session_id[:8]}] {context.get('current_host', '?')}: {message[:60]}")

    # Stream response tokens
    try:
        async for chunk in agent.run(
            user_message=message,
            history=history,
            session_id=session_id,
            use_rag=True,
            context=context,
        ):
            await send(websocket, "token", chunk)

    except Exception as e:
        print(f"[Server] Agent error: {e}")
        await send(websocket, "error", str(e))
        return

    # Send completion signal with current mood
    mood_data = {}
    try:
        from voice.mood import get_current_mood
        mood_data = get_current_mood()
    except Exception:
        pass

    await send(websocket, "done", "", extra={"mood": mood_data} if mood_data else {})


async def send(
    websocket: WebSocketServerProtocol,
    msg_type: str,
    data: str,
    extra: dict = None,
):
    """Send a typed message to the client."""
    payload = {"type": msg_type, "data": data}
    if extra:
        payload.update(extra)
    try:
        await websocket.send(json.dumps(payload))
    except Exception as e:
        print(f"[Server] Send error: {e}")


async def main():
    print(f"[Server] Gizmo starting on {HOST}:{PORT}")

    # Start background services
    await _start_background_services()

    async with websockets.serve(
        handle_client,
        HOST,
        PORT,
        ping_interval=20,
        ping_timeout=10,
        max_size=1_000_000,
    ):
        print(f"[Server] Ready. Listening on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


async def _start_background_services():
    """Start archiver and reminder checker."""
    loop = asyncio.get_event_loop()

    try:
        from memory.archiver import start_archiver
        start_archiver(llm, loop=loop)
        print("[Server] Archiver started")
    except Exception as e:
        print(f"[Server] Archiver failed to start: {e}")

    try:
        from ambient.reminders import start_reminder_checker
        from asyncio import Queue
        reminder_queue = Queue()
        start_reminder_checker(reminder_queue, loop=loop)
        # Drain reminder queue into a simple printer for now
        # (no TTS on Render — just log them)
        asyncio.ensure_future(_drain_reminder_queue(reminder_queue))
        print("[Server] Reminder checker started")
    except Exception as e:
        print(f"[Server] Reminder checker failed to start: {e}")


async def _drain_reminder_queue(queue: asyncio.Queue):
    """Log reminders — replace with TTS/push notification when ready."""
    while True:
        try:
            item = await queue.get()
            print(f"[Server] REMINDER: {item.get('transcript', '')}")
        except Exception as e:
            print(f"[Server] Reminder drain error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
