"""
server.py - Gizmo server
Binds port first, then starts background services.
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

# All currently connected websockets — used for server-push (reminders, timers)
_connected: set = set()


async def _push_to_all(message: str) -> None:
    """Push a message to every connected client."""
    if not _connected:
        print(f"[Server] Push skipped — no clients connected: {message[:60]}")
        return
    payload = json.dumps({"type": "token", "data": message})
    done = json.dumps({"type": "done", "data": ""})
    for ws in list(_connected):
        try:
            await ws.send(payload)
            await ws.send(done)
        except Exception as e:
            print(f"[Server] Push failed for client: {e}")


async def handler(websocket):
    conn_id = str(uuid.uuid4())[:8]
    print(f"[Server] Client connected: {conn_id}")
    _connected.add(websocket)

    # We need context to greet properly — wait for first message to get fronter/session
    # then decide if a greeting should fire. Track whether we've greeted this connection.
    _greeted = False

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

            # Apply client timezone if provided
            tz = context.get("timezone", "")
            if tz:
                from core.timezone import set_timezone
                set_timezone(tz)
            history = get_session(session_id)

            # ── Nuclear reset — intercept before agent, bypass LLM entirely ──
            if message.strip() == "sudo reset yourself motherfucker":
                print(f"[Server] Nuclear reset triggered by {context.get('current_host', 'unknown')}")
                try:
                    from tools.reset_tool import FactoryResetTool
                    reset = FactoryResetTool()
                    result = await reset.run(
                        passphrase="sudo reset yourself motherfucker",
                        session_id=session_id,
                    )
                    print(f"[Server] Reset result: {result.output}")
                except Exception as e:
                    print(f"[Server] Reset failed: {e}")

                # Tell client to reload — clears the UI completely
                await websocket.send(json.dumps({"type": "reload"}))

                # Push onboarding opening after client reconnects
                async def _push_onboarding():
                    await asyncio.sleep(2.0)
                    try:
                        from core.personality_growth import run_onboarding
                        opening = await run_onboarding(llm)
                        await _push_to_all(opening)
                    except Exception as e:
                        print(f"[Server] Onboarding push failed: {e}")

                asyncio.ensure_future(_push_onboarding())
                continue

            # ── Return greeting ───────────────────────────────────────────────
            # Fire once per connection if session was inactive long enough.
            # Greeting replaces the normal response for the first message.
            nonlocal_greeted = _greeted  # capture for closure
            if not _greeted:
                _greeted = True
                from core.greeter import should_greet, build_greeting
                if should_greet(history):
                    fronter = context.get("current_host", "")
                    print(f"[Server] Firing return greeting for {fronter or 'unknown'}")
                    try:
                        greeting = await build_greeting(
                            fronter=fronter,
                            session_id=session_id,
                            llm=llm,
                        )
                        await websocket.send(json.dumps({"type": "token", "data": greeting}))
                        await websocket.send(json.dumps({"type": "done", "data": ""}))
                        # Still process their first message normally after greeting
                    except Exception as e:
                        print(f"[Server] Greeting failed: {e}")

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
                import traceback
                tb = traceback.format_exc()
                print(f"[Server] Agent error: {e}\n{tb}")
                await websocket.send(json.dumps({"type": "error", "data": f"{e}\n\n{tb}"}))
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
        _connected.discard(websocket)
        print(f"[Server] Client disconnected: {conn_id}")


async def http_handler(path, request_headers):
    if path == "/ws":
        return None  # pass through to WebSocket handler
    if path in ("/", "/index.html"):
        html = CHAT_HTML.read_text()
        return (200, [("Content-Type", "text/html")], html.encode())
    if path == "/health":
        return (200, [("Content-Type", "text/plain")], b"OK")
    return None


async def start_background_services():
    # Initialize entity store DB on startup
    try:
        from core.entity_store import init_db
        init_db()
        print("[Server] Entity store initialized")
    except Exception as e:
        print(f"[Server] Entity store init failed: {e}")
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
    """
    Drain the directed queue (reminders, timer fires) and push to all clients.
    Runs the item through the agent so delivery is natural, not raw text.
    """
    while True:
        try:
            item = await queue.get()
            transcript = item.get("transcript", "")
            context = item.get("context") or {}
            session_id = item.get("session_id", "server")

            print(f"[Server] Delivering queued item: {transcript[:60]}")

            if not _connected:
                print("[Server] No clients connected — queued item dropped")
                continue

            history = get_session(session_id)
            response_text = ""
            async for chunk in agent.run(
                user_message=transcript,
                history=history,
                session_id=session_id,
                use_rag=False,
                context=context,
            ):
                response_text += chunk

            await _push_to_all(response_text)

        except Exception as e:
            print(f"[Server] Reminder drain error: {e}")


async def main():
    # Bind port FIRST so Render knows we're alive
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
        # Start background services after port is bound
        await start_background_services()
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())