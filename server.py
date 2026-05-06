"""
server.py — Gizmo server
Binds port first, then starts background services.

What changed from previous version:
  - greeter removed (Ego will handle return greetings)
  - voice.mood removed (gone)
  - memory.archiver removed (new background loop pending)
  - agent.run() no longer takes use_rag arg
  - agent.set_push_fn() registered on startup
  - all logging through core.log
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

import websockets
from websockets.server import serve

from core.agent import agent, set_push_fn
from core.llm import llm
from core.log import log, log_event, log_error
from memory.history import get_session

HOST     = "0.0.0.0"
PORT     = int(os.getenv("PORT", "8765"))
CHAT_HTML = Path(__file__).parent / "chat.html"

# All currently connected websockets — used for server-push
_connected: set = set()


async def _push_to_all(message: str) -> None:
    """Push a message to every connected client."""
    if not _connected:
        log("Server", f"push skipped — no clients connected: {message[:60]}")
        return
    payload = json.dumps({"type": "token", "data": message})
    done    = json.dumps({"type": "done",  "data": ""})
    for ws in list(_connected):
        try:
            await ws.send(payload)
            await ws.send(done)
        except Exception as e:
            log_error("Server", "push failed for client", exc=e)


async def handler(websocket):
    conn_id = str(uuid.uuid4())[:8]
    log_event("Server", "CLIENT_CONNECTED", conn=conn_id)
    _connected.add(websocket)

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

            context    = data.get("context") or {}
            session_id = data.get("session_id") or conn_id
            context.setdefault("fronters", [])

            # Apply client timezone if provided
            tz = context.get("timezone", "")
            if tz:
                from core.timezone import set_timezone
                set_timezone(tz)

            history = get_session(session_id)

            log_event("Server", "MESSAGE",
                session=session_id[:8],
                headmate=context.get("current_host", "?"),
                preview=message[:60],
            )

            try:
                async for chunk in agent.run(
                    user_message=message,
                    history=history,
                    session_id=session_id,
                    context=context,
                ):
                    await websocket.send(json.dumps({"type": "token", "data": chunk}))

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                log_error("Server", f"agent error: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "data": f"{e}\n\n{tb}",
                }))
                continue

            await websocket.send(json.dumps({"type": "done", "data": ""}))

    except websockets.exceptions.ConnectionClosedOK:
        pass
    except Exception as e:
        log_error("Server", f"connection error", exc=e)
    finally:
        _connected.discard(websocket)
        log_event("Server", "CLIENT_DISCONNECTED", conn=conn_id)


async def http_handler(path, request_headers):
    if path == "/ws":
        return None  # pass through to WebSocket handler
    if path in ("/", "/index.html"):
        html = CHAT_HTML.read_text()
        return (200, [("Content-Type", "text/html")], html.encode())
    if path == "/health":
        return (200, [("Content-Type", "text/plain")], b"OK")
    return None


async def _drain_reminders(queue):
    """
    Drain the reminder queue and push to all clients.
    Routes through agent.run() so delivery is natural.
    """
    while True:
        try:
            item       = await queue.get()
            transcript = item.get("transcript", "")
            context    = item.get("context") or {}
            session_id = item.get("session_id", "server")

            log_event("Server", "REMINDER_DELIVERING",
                session=session_id[:8],
                preview=transcript[:60],
            )

            if not _connected:
                log("Server", "no clients connected — reminder dropped")
                continue

            history       = get_session(session_id)
            response_text = ""
            async for chunk in agent.run(
                user_message=transcript,
                history=history,
                session_id=session_id,
                context=context,
                source="reminder",
            ):
                response_text += chunk

            await _push_to_all(response_text)

        except Exception as e:
            log_error("Server", "reminder drain error", exc=e)


async def start_background_services():
    loop = asyncio.get_event_loop()

    # Reminders
    try:
        from ambient.reminders import start_reminder_checker
        reminder_queue = asyncio.Queue()
        start_reminder_checker(reminder_queue, loop=loop)
        asyncio.ensure_future(_drain_reminders(reminder_queue))
        log("Server", "reminder checker started")
    except Exception as e:
        log_error("Server", "reminder checker failed to start", exc=e)


async def main():
    # Register push function with agent membrane
    # so any component can push unsolicited messages to clients
    set_push_fn(_push_to_all)

    # Bind port FIRST — Render needs this to know we're alive
    async with serve(
        handler,
        HOST,
        PORT,
        process_request=http_handler,
        ping_interval=20,
        ping_timeout=10,
        max_size=1_000_000,
    ):
        log_event("Server", "READY", host=HOST, port=PORT)
        await start_background_services()
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())