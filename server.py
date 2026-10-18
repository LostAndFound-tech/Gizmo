"""
server.py
Gizmo's WebSocket server — bare bones.

Does: WebSocket chat, session persistence (JSON), short history, dedup,
      chunked streaming back to the client, /health, /sessions endpoints.
Does not: modes, safewords, scenes, intents, summaries, multi-speaker parsing.

Client protocol (unchanged):
  in : {"type": "message", "content": str, "session_id": str,
        "context": {"current_host": str}}
       {"type": "ping"}
       {"type": "switch_host", "headmate": str}
       {"type": "check_unsent", "headmate": str}
  out: {"type": "thinking"}
       {"type": "chunk", "content": str}
       {"type": "done", "session_id": str, "current_host": str}
       {"type": "error", "message": str}
       {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from aiohttp import web, WSMsgType

from core.log import log, log_event, log_error

DEDUP_WINDOW    = 5.0
THINKING_DELAY  = 0.3
CHUNK_SIZE      = 8
MAX_MESSAGE_LEN = 8000
HISTORY_LIMIT   = 12   # messages loaded from disk (6 pairs)

# ── Session persistence ───────────────────────────────────────────────────────

_DATA_DIR     = os.getenv("DATA_DIR", "./data")
_SESSIONS_DIR = os.path.join(_DATA_DIR, "sessions")


def _session_path(session_id: str) -> str:
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return os.path.join(_SESSIONS_DIR, f"{safe}.json")


def _load_session(session_id: str) -> Optional[dict]:
    try:
        with open(_session_path(session_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error("Sessions", f"load failed {session_id[:8]}", exc=e)
        return None


def _save_session(session_id: str, data: dict) -> None:
    try:
        with open(_session_path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log_error("Sessions", f"save failed {session_id[:8]}", exc=e)


def _new_session(session_id: str) -> dict:
    now = time.time()
    return {
        "session_id":  session_id,
        "opened_at":   now,
        "last_active": now,
        "hosts":       [],
        "messages":    [],
    }


def _list_sessions() -> list[dict]:
    results = []
    try:
        os.makedirs(_SESSIONS_DIR, exist_ok=True)
        for fname in os.listdir(_SESSIONS_DIR):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(_SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
                    d = json.load(f)
                results.append({
                    "id":          d.get("session_id", fname[:-5]),
                    "opened_at":   d.get("opened_at", 0),
                    "last_active": d.get("last_active", 0),
                    "hosts":       d.get("hosts", []),
                    "summary":     d.get("summary"),
                })
            except Exception:
                continue
        results.sort(key=lambda x: x["last_active"], reverse=True)
    except Exception as e:
        log_error("Sessions", "list failed", exc=e)
    return results


# ── In-memory state ───────────────────────────────────────────────────────────

_seen_messages: dict[str, float]   = {}
_live_sockets:  dict[str, object]  = {}   # client session_id -> ws
_server_to_client_sid: dict[str, str] = {}


def _is_duplicate(session_id: str, content: str) -> bool:
    key = hashlib.md5(f"{session_id}:{content}".encode()).hexdigest()
    now = time.time()
    if key in _seen_messages and now - _seen_messages[key] < DEDUP_WINDOW:
        return True
    _seen_messages[key] = now
    for k in [k for k, t in _seen_messages.items() if now - t > DEDUP_WINDOW * 2]:
        del _seen_messages[k]
    return False


def _history_from_saved(saved: Optional[dict]) -> list[dict]:
    if not saved:
        return []
    return [
        {"role": m["role"], "content": m["content"]}
        for m in saved.get("messages", [])[-HISTORY_LIMIT:]
        if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
    ]


# ── Server ────────────────────────────────────────────────────────────────────

class GizmoServer:

    def __init__(self):
        self._connections: dict[str, object] = {}
        log("GizmoServer", "initialised")

    async def start(self, host: str = "0.0.0.0", port: int = 10000) -> None:
        app = web.Application()

        async def handle_index(request):
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                return web.Response(text=html_path.read_text(encoding="utf-8"),
                                    content_type="text/html")
            return web.Response(text="<h1>Gizmo</h1>", content_type="text/html")

        async def handle_health(request):
            return web.json_response({"status": "ok"})

        async def handle_sessions(request):
            return web.json_response({"sessions": _list_sessions()})

        async def handle_session_detail(request):
            data = _load_session(request.match_info["session_id"])
            if not data:
                return web.json_response({"error": "not found"}, status=404)
            return web.json_response(data)

        async def handle_ws(request):
            ws = web.WebSocketResponse(heartbeat=15.0)
            await ws.prepare(request)
            server_sid = f"sess_{id(ws):016x}"
            self._connections[server_sid] = ws
            log_event("GizmoServer", "CONNECTION_OPENED", session=server_sid[:8])
            try:
                async for msg in ws:
                    if msg.type == WSMsgType.TEXT:
                        await self._handle_message(ws, server_sid, msg.data)
                    elif msg.type == WSMsgType.ERROR:
                        log_error("GizmoServer", f"ws error: {ws.exception()}", exc=None)
            finally:
                self._connections.pop(server_sid, None)
                client_sid = _server_to_client_sid.pop(server_sid, None)
                if client_sid and _live_sockets.get(client_sid) is ws:
                    _live_sockets.pop(client_sid, None)
                try:
                    from core.agent_simple import agent_simple
                    agent_simple.end_session(client_sid or server_sid)
                except Exception:
                    pass
                log_event("GizmoServer", "CONNECTION_CLOSED", session=server_sid[:8])
            return ws

        app.router.add_get("/",                      handle_index)
        app.router.add_get("/health",                handle_health)
        app.router.add_get("/sessions",              handle_sessions)
        app.router.add_get("/sessions/{session_id}", handle_session_detail)
        app.router.add_get("/ws",                    handle_ws)

        log_event("GizmoServer", "STARTING", host=host, port=port)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        log_event("GizmoServer", "LISTENING", host=host, port=port)
        await asyncio.Future()

    # ── Message routing ───────────────────────────────────────────────────────

    async def _handle_message(self, websocket, server_sid: str, raw_msg: str) -> None:
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            await self._send(websocket, {"type": "error", "message": "invalid JSON"})
            return

        msg_type = msg.get("type", "message")
        sid = msg.get("session_id") or server_sid

        _live_sockets[sid] = websocket
        _server_to_client_sid[server_sid] = sid

        if msg_type == "ping":
            await self._send(websocket, {"type": "pong"})
        elif msg_type == "message":
            await self._handle_chat_message(websocket, sid, msg)
        elif msg_type == "switch_host":
            hm = (msg.get("headmate") or "").strip().lower()
            if hm:
                log_event("GizmoServer", "HOST_SWITCH", session=sid[:8], headmate=hm)
        elif msg_type == "check_unsent":
            await self._handle_check_unsent(websocket, sid, msg)
        elif msg_type == "received":
            saved = _load_session(sid)
            if saved:
                saved.pop("pending_response", None)
                saved.pop("pending_response_ts", None)
                _save_session(sid, saved)
        else:
            log_event("GizmoServer", "MSG_TYPE_IGNORED", type=msg_type, session=sid[:8])

    # ── Chat ──────────────────────────────────────────────────────────────────

    async def _handle_chat_message(self, websocket, session_id: str, msg: dict) -> None:
        content  = msg.get("content", "")
        context  = msg.get("context") or {}
        headmate = (context.get("current_host") or msg.get("headmate") or "").strip().lower() or None

        if isinstance(content, list):
            # Tolerate the old multi-part format: flatten to plain text
            content = "\n".join(
                str(p.get("content", "")) for p in content if isinstance(p, dict)
            )
        content = (content or "").strip()

        if not content:
            return
        if len(content) > MAX_MESSAGE_LEN:
            await self._send(websocket, {"type": "error", "message": "message too long"})
            return
        if _is_duplicate(session_id, content):
            return

        context["current_host"] = headmate
        context["session_id"]   = session_id

        log_event("GizmoServer", "MESSAGE_RECEIVED",
                  session=session_id[:8], headmate=headmate or "unknown",
                  words=len(content.split()))

        # History comes from disk (before we add this message)
        saved   = _load_session(session_id) or _new_session(session_id)
        history = _history_from_saved(saved)

        # Persist the user message up front so a dropped connection can resend
        now = time.time()
        saved["last_active"] = now
        saved["pipeline_running"] = True
        if headmate and headmate not in saved["hosts"]:
            saved["hosts"].append(headmate)
        saved["messages"].append({
            "role": "user", "speaker": headmate or "unknown",
            "content": content, "ts": now,
        })
        _save_session(session_id, saved)

        await asyncio.sleep(THINKING_DELAY)
        await self._send(websocket, {"type": "thinking"})

        try:
            response = await self._run_agent(content, session_id, context, history)
        except Exception as e:
            log_error("GizmoServer", "agent failed", exc=e)
            saved = _load_session(session_id) or saved
            saved.pop("pipeline_running", None)
            _save_session(session_id, saved)
            await self._send(websocket, {"type": "error", "message": f"{type(e).__name__}: {e}"})
            return

        saved = _load_session(session_id) or saved
        saved.pop("pipeline_running", None)
        saved["last_active"] = time.time()
        saved["messages"].append({
            "role": "assistant", "speaker": "gizmo",
            "content": response, "ts": time.time(),
        })
        saved["pending_response"]    = response
        saved["pending_response_ts"] = time.time()
        _save_session(session_id, saved)

        # The socket may have reconnected mid-generation; prefer the live one
        live_ws = _live_sockets.get(session_id, websocket)
        await self._stream(live_ws, response)
        await self._send(live_ws, {"type": "done", "session_id": session_id,
                                   "current_host": headmate or ""})

        log_event("GizmoServer", "RESPONSE_SENT",
                  session=session_id[:8], words=len(response.split()))

    async def _handle_check_unsent(self, websocket, session_id: str, msg: dict) -> None:
        """On reconnect: deliver a response the client missed, or re-run an unanswered message."""
        saved = _load_session(session_id)
        if not saved:
            return

        headmate = (msg.get("headmate") or "").strip().lower()
        if not headmate and saved.get("hosts"):
            headmate = saved["hosts"][0].lower()

        pending = (saved.get("pending_response") or "").strip()
        if pending:
            await self._stream(websocket, pending)
            await self._send(websocket, {"type": "done", "session_id": session_id,
                                         "current_host": headmate})
            return

        messages = saved.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            return

        last_text = messages[-1].get("content", "")
        if not last_text:
            return

        log_event("GizmoServer", "RESENDING_UNSENT", session=session_id[:8])
        await self._send(websocket, {"type": "thinking"})

        history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1][-HISTORY_LIMIT:]
                   if m.get("role") in ("user", "assistant")]
        context = {"current_host": headmate, "session_id": session_id}

        try:
            response = await self._run_agent(last_text, session_id, context, history)
        except Exception as e:
            log_error("GizmoServer", "resend failed", exc=e)
            return
        if not response:
            return

        saved["messages"].append({
            "role": "assistant", "speaker": "gizmo",
            "content": response, "ts": time.time(),
        })
        saved["last_active"] = time.time()
        saved.pop("pipeline_running", None)
        _save_session(session_id, saved)

        await self._stream(websocket, response)
        await self._send(websocket, {"type": "done", "session_id": session_id,
                                     "current_host": headmate})

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run_agent(self, message: str, session_id: str, context: dict, history: list) -> str:
        from core.agent_simple import agent_simple
        parts = []
        async for chunk in agent_simple.respond(
            user_message=message,
            history=history,
            session_id=session_id,
            context=context,
        ):
            parts.append(chunk)
        return "".join(parts)

    async def _stream(self, websocket, text: str) -> None:
        for i in range(0, len(text), CHUNK_SIZE):
            await self._send(websocket, {"type": "chunk", "content": text[i:i + CHUNK_SIZE]})
            await asyncio.sleep(0)

    async def _send(self, websocket, data: dict) -> None:
        try:
            await websocket.send_str(json.dumps(data))
        except Exception:
            pass


server = GizmoServer()


async def main():
    await server.start(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "10000")),
    )


if __name__ == "__main__":
    asyncio.run(main())
