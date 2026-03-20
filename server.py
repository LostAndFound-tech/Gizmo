"""
server.py - Gizmo server using websockets library with built-in HTTP fallback.
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

import websockets
from websockets.server import serve

from core.agent import agent
from core.llm import llm
from memory.history import get_session

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8765"))
CHAT_HTML = Path(__file__).parent / "chat.html"


async def handler(websocket):
    """Handle both HTTP and WebSocket on the same port."""
    conn_id = str(uuid.uuid4())[:8]
    print(f"[Server] Client connected: {conn_id}")

    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"type": "error", "data": "Invalid JSON"}))
                continue

            message = data.get("message", "").strip()
            if not message:
                continue

            context = data.get("context") or {}
            session_id = data.get("session_id") or conn_id
            context.setdefault("fronters", [])
            history = get_session(session_id)

            print(f"[Server] [{session_id[:8]}] {context.get('current_host', '?')}: {message[:60]}")

            try:
                async for chunk in agent.run(
                    user_message=message,
                    history=history,
                    session_id=session_id,
                    use_rag=True,
                    context=context,
                ):
                    await websocket.send(json.dumps({"type": "token", "data": chunk}))

            except Exception as e:
                print(f"[Server] Agent error: {e}")
                await websocket.send(json.dumps({"type": "error", "data": str(e)}))
                continue

            mood_data = {}
            try:
                from voice.mood import get_current_mood
                mood_data = get_current_mood()
            except Exception:
                pass

            payload = {"type": "done", "data": ""}
            if mood_data:
                payload["mood"] = mood_data
            await websocket.send(json.dumps(payload))

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except Exception as e:
        print(f"[Server] Connection error: {e}")
    finally:
        print(f"[Server] Client disconnected: {conn_id}")


async def http_handler(path, request_headers):
    """Serve chat UI for HTTP requests, let WebSocket upgrades through."""
    if request_headers.get("Upgrade", "").lower() == "websocket":
        return None  # let websockets handle it
    if path in ("/", "/index.html"):
        html = CHAT_HTML.read_text()
        return (200, [("Content-Type", "text/html")], html.encode())
    if path == "/health":
        return (200, [("Content-Type", "text/plain")], b"OK")
    return (404, [], b"Not found")


async def start_background_services():
    loop = asyncio.get_event_loop()
    try:
        from memory.archiver import start_archiver
        start_archiver(llm, loop=loop)
        print("[Server] Archiver started")
    except Exception as e:
        print(f"[Server] Archiver failed: {e}")
    try:
        from ambient.reminders import start_reminder_checker
        reminder_queue = asyncio.Queue()
        start_reminder_checker(reminder_queue, loop=loop)
        asyncio.ensure_future(_drain_reminders(reminder_queue))
        print("[Server] Reminder checker started")
    except Exception as e:
        print(f"[Server] Reminder checker failed: {e}")


async def _drain_reminders(queue):
    while True:
        try:
            item = await queue.get()
            print(f"[Server] REMINDER: {item.get('transcript', '')}")
        except Exception as e:
            print(f"[Server] Reminder drain error: {e}")


async def main():
    await start_background_services()
    print(f"[Server] Gizmo starting on {HOST}:{PORT}")
    async with serve(
        handler,
        HOST,
        PORT,
        process_request=http_handler,
        ping_interval=20,
        ping_timeout=10,
        max_size=1_000_000,
    ):
        print(f"[Server] Ready at http://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
