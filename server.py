"""
server.py
Gizmo server for Render deployment.
Uses aiohttp to handle both HTTP health checks and WebSocket on the same port.

Message format (client → server):
    {
        "message": "hey gizmo...",
        "context": {"current_host": "alice", "fronters": ["alice"]},
        "session_id": "abc123"
    }

Message format (server → client):
    {"type": "token",   "data": "..."}
    {"type": "done",    "data": ""}
    {"type": "error",   "data": "..."}
    {"type": "mood",    "data": {...}}
"""

import asyncio
import json
import os
import uuid

from aiohttp import web, WSMsgType

from core.agent import agent
from core.llm import llm
from memory.history import get_session

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8765"))


async def health(request):
    return web.Response(text="OK")


async def chat_ui(request):
    here = os.path.dirname(os.path.abspath(__file__))
    return web.FileResponse(os.path.join(here, "chat.html"))


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    conn_id = str(uuid.uuid4())[:8]
    print(f"[Server] Client connected: {conn_id}")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await handle_message(ws, msg.data, conn_id)
            elif msg.type == WSMsgType.ERROR:
                print(f"[Server] WS error: {ws.exception()}")
    finally:
        print(f"[Server] Client disconnected: {conn_id}")

    return ws


async def handle_message(ws, raw: str, conn_id: str):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await send(ws, "error", "Invalid JSON")
        return

    message = data.get("message", "").strip()
    if not message:
        await send(ws, "error", "Empty message")
        return

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
            await send(ws, "token", chunk)

    except Exception as e:
        print(f"[Server] Agent error: {e}")
        await send(ws, "error", str(e))
        return

    mood_data = {}
    try:
        from voice.mood import get_current_mood
        mood_data = get_current_mood()
    except Exception:
        pass

    await send(ws, "done", "", extra={"mood": mood_data} if mood_data else {})


async def send(ws, msg_type: str, data: str, extra: dict = None):
    payload = {"type": msg_type, "data": data}
    if extra:
        payload.update(extra)
    try:
        await ws.send_str(json.dumps(payload))
    except Exception as e:
        print(f"[Server] Send error: {e}")


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


async def _drain_reminders(queue: asyncio.Queue):
    while True:
        try:
            item = await queue.get()
            print(f"[Server] REMINDER: {item.get('transcript', '')}")
        except Exception as e:
            print(f"[Server] Reminder drain error: {e}")


async def on_startup(app):
    await start_background_services()
    print(f"[Server] Gizmo ready on port {PORT}")


def create_app():
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/", chat_ui)
    app.router.add_get("/ws", websocket_handler)
    app.on_startup.append(on_startup)
    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host=HOST, port=PORT)
