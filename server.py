"""
server.py - Gizmo server
Binds port first, then starts background services.

IMPORTANT: ChromaDB health check runs at the very top before any imports
that might instantiate a RAGStore. This is necessary because rag.py creates
a module-level RAGStore() singleton which calls PersistentClient immediately
on import. If ChromaDB's schema is broken, we must fix it before that happens.
"""

# ── ChromaDB health check — MUST be before all other imports ─────────────────
import os
import shutil as _shutil
import sqlite3 as _sq

_chroma_path = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
_chroma_sqlite = os.path.join(_chroma_path, "chroma.sqlite3")
_needs_reinit = not os.path.exists(_chroma_sqlite)

if not _needs_reinit:
    try:
        _con = _sq.connect(_chroma_sqlite)
        _tables = [r[0] for r in _con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        _con.close()
        if "collections" not in _tables or "acquire_write" not in _tables:
            _needs_reinit = True
            print(f"[Server] ChromaDB schema incomplete — reinitializing")
    except Exception as _e:
        _needs_reinit = True
        print(f"[Server] ChromaDB validation failed ({_e}) — reinitializing")

if _needs_reinit:
    print(f"[Server] Reinitializing ChromaDB at {_chroma_path}")
    if os.path.exists(_chroma_path):
        _shutil.rmtree(_chroma_path)
    os.makedirs(_chroma_path, exist_ok=True)
    import chromadb as _cdb
    _cdb.PersistentClient(path=_chroma_path)
    # Ensure writable by all render processes
    _sqlite = os.path.join(_chroma_path, "chroma.sqlite3")
    try:
        os.chmod(_chroma_path, 0o775)
        if os.path.exists(_sqlite):
            os.chmod(_sqlite, 0o664)
    except Exception as _ce:
        print(f"[Server] ChromaDB chmod failed (non-fatal): {_ce}")
    print("[Server] ChromaDB reinitialized cleanly")
else:
    print("[Server] ChromaDB healthy")

# ── Now safe to import everything else ───────────────────────────────────────

import asyncio
import json
import uuid
from pathlib import Path

import websockets
from websockets.server import serve

from core.agent import agent
from core.llm import llm
from memory.history import get_session
from core.push import _connected, _push_to_all

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8765"))
CHAT_HTML = Path(__file__).parent / "chat.html"


async def handler(websocket):
    conn_id = str(uuid.uuid4())[:8]
    print(f"[Server] Client connected: {conn_id}")
    _connected.add(websocket)

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

                await websocket.send(json.dumps({"type": "reload"}))

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
        return None
    if path in ("/", "/index.html"):
        html = CHAT_HTML.read_text()
        return (200, [("Content-Type", "text/html")], html.encode())
    if path == "/health":
        return (200, [("Content-Type", "text/plain")], b"OK")
    return None


async def start_background_services():
    # Entity store init
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
        await start_background_services()
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
