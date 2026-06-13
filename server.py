"""
server.py
Gizmo's WebSocket server. Bare bones testing branch.

Mobile reconnect fixes:
- _live_sockets updated immediately on ANY message type (not just "message")
- check_unsent registers new socket before checking, so in-flight pipeline
  delivers to the new socket naturally when it finishes
- pending_response write happens before pipeline starts (keyed on user message hash)
  so reconnect can detect "pipeline running" vs "response lost"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Optional

from pathlib import Path
from core.log import log, log_event, log_error
from core.timezone import tz_now
from dotenv import load_dotenv
load_dotenv()

DEDUP_WINDOW    = 5.0
THINKING_DELAY  = 0.3
CHUNK_SIZE      = 8
MAX_MESSAGE_LEN = 8000

# ── Session persistence ───────────────────────────────────────────────────────

_DATA_DIR = os.getenv("DATA_DIR", "./data")
_SESSIONS_DIR = os.path.join(_DATA_DIR, "sessions")


def _sessions_dir() -> str:
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    return _SESSIONS_DIR


def _session_path(session_id: str) -> str:
    return os.path.join(_sessions_dir(), f"{session_id}.json")


def _load_session(session_id: str) -> Optional[dict]:
    path = _session_path(session_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[Sessions] failed to load {session_id}: {e}")
        return None


def _save_session(session_id: str, data: dict) -> None:
    try:
        path = _session_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Sessions] failed to save {session_id}: {e}")


def _list_sessions() -> list[dict]:
    """Return session metadata sorted by last_active descending."""
    try:
        d = _sessions_dir()
        results = []
        for fname in os.listdir(d):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(d, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append({
                    "id":          data.get("session_id", fname[:-5]),
                    "opened_at":   data.get("opened_at", 0),
                    "last_active": data.get("last_active", 0),
                    "hosts":       data.get("hosts", []),
                    "tags":        data.get("tags", []),
                    "topics":      data.get("topics", []),
                    "mood":        data.get("mood"),
                    "summary":     data.get("summary"),
                    "notable":     data.get("notable", []),
                    "emotion_arc": data.get("emotion_arc", []),
                })
            except Exception:
                continue
        results.sort(key=lambda x: x["last_active"], reverse=True)
        return results
    except Exception as e:
        print(f"[Sessions] list failed: {e}")
        return []

_SPEECH_RE   = re.compile(r'^\[?([A-Za-z][A-Za-z0-9_\- ]{0,30})\]?\s*:\s*(.+)', re.DOTALL)
_ACTION_RE   = re.compile(r'^\*(.+)\*$', re.DOTALL)
_DIRECTED_RE = re.compile(r'\b(to|at|@)\s+([A-Za-z][A-Za-z0-9_\- ]{0,20})\b', re.IGNORECASE)

_seen_messages:   dict[str, float]       = {}
_session_history: dict[str, list[dict]] = {}

# Tracks which sessions are waiting on a scene resume answer
_pending_scene_resume: dict[str, str] = {}

# Live WebSocket registry — client_session_id -> current active websocket
# Updated on every incoming message (any type), not just "message" type.
# This ensures reconnect registers the new socket before check_unsent runs.
_live_sockets: dict[str, object] = {}

# Maps server socket session_id -> client session_id
_server_to_client_sid: dict[str, str] = {}

# Tracks in-flight pipelines — client_session_id -> asyncio.Task
# So reconnect can detect "still thinking" and wait rather than giving up
_pipeline_tasks: dict[str, asyncio.Task] = {}


def _is_duplicate(session_id: str, content: str) -> bool:
    key = hashlib.md5(f"{session_id}:{content}".encode()).hexdigest()
    now = time.time()
    if key in _seen_messages and now - _seen_messages[key] < DEDUP_WINDOW:
        return True
    _seen_messages[key] = now
    for k in [k for k, t in _seen_messages.items() if now - t > DEDUP_WINDOW * 2]:
        del _seen_messages[k]
    return False


def parse_exchange(raw: str, default_headmate: Optional[str] = None) -> list[dict]:
    parts = []
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    for line in lines:
        action_match = _ACTION_RE.match(line)
        if action_match:
            action_text = action_match.group(1).strip()
            actor = None
            words = action_text.split()
            if words and words[0][0].isupper() and len(words[0]) > 1:
                actor = words[0].rstrip("'s")
                action_text = " ".join(words[1:]) if len(words) > 1 else action_text
            directed = None
            d_match = _DIRECTED_RE.search(action_text)
            if d_match:
                directed = d_match.group(2).lower()
            parts.append({"headmate": actor.lower() if actor else default_headmate, "content": action_text, "content_type": "action", "directed_at": directed})
            continue
        speech_match = _SPEECH_RE.match(line)
        if speech_match:
            name    = speech_match.group(1).strip().lower()
            content = speech_match.group(2).strip()
            directed = None
            d_match  = _DIRECTED_RE.search(content)
            if d_match:
                directed = d_match.group(2).lower()
                if directed in ("gizmo", "you"):
                    directed = "gizmo"
            inline_action = re.search(r'\*([^*]+)\*', content)
            if inline_action:
                action_text = inline_action.group(1)
                speech_text = re.sub(r'\*[^*]+\*', '', content).strip()
                if speech_text:
                    parts.append({"headmate": name, "content": speech_text, "content_type": "speech", "directed_at": directed})
                parts.append({"headmate": name, "content": action_text, "content_type": "action", "directed_at": None})
            else:
                parts.append({"headmate": name, "content": content, "content_type": "speech", "directed_at": directed})
            continue
        if line and default_headmate:
            parts.append({"headmate": default_headmate.lower(), "content": line, "content_type": "speech", "directed_at": "gizmo"})
    if not parts and raw.strip() and default_headmate:
        parts.append({"headmate": default_headmate.lower(), "content": raw.strip(), "content_type": "speech", "directed_at": "gizmo"})
    return parts


def is_multi_part(raw: str) -> bool:
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    return sum(1 for l in lines if _SPEECH_RE.match(l) or _ACTION_RE.match(l)) >= 2


def assemble_scene_text(parts: list[dict]) -> str:
    lines = []
    for part in parts:
        name     = (part.get("headmate") or "unknown").title()
        content  = part.get("content", "")
        ctype    = part.get("content_type", "speech")
        directed = part.get("directed_at")
        if ctype == "action":
            line = f"*{name} {content}*"
        elif ctype == "presence":
            line = f"[{name} is present]"
        else:
            suffix = f" (to {directed})" if directed and directed != "gizmo" else ""
            line   = f"[{name}]{suffix}: {content}"
        lines.append(line)
    return "\n".join(lines)


def _time_of_day(hour: int) -> str:
    if 5  <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"


def _is_yes(text: str) -> bool:
    t = text.strip().lower()
    return any(w in t for w in ("yes", "yeah", "yep", "sure", "ok", "okay", "yea", "let's", "lets", "continue", "pick up", "resume"))


def _is_no(text: str) -> bool:
    t = text.strip().lower()
    return any(w in t for w in ("no", "nah", "nope", "not", "skip", "later", "don't", "dont", "pass", "forget"))


async def run_single_pipeline(message, session_id, headmate, context, history) -> str:
    from core.agent_simple import agent_simple as agent
    try:
        chunks = []
        async for chunk in agent.respond(
            user_message=message,
            history=history,
            session_id=session_id,
            context=context,
        ):
            print("CHUNK RECEIVED:", repr(chunk[:40]), flush=True)
            chunks.append(chunk)
        return "".join(chunks)
    except Exception as e:
        import traceback
        print(f"[SINGLE PIPELINE ERROR]\n{traceback.format_exc()}", flush=True)
        raise


class GizmoServer:

    def __init__(self):
        self._connections: dict[str, object] = {}
        log("GizmoServer", "initialised")

    async def start(self, host: str = "0.0.0.0", port: int = 10000) -> None:
        from aiohttp import web

        app = web.Application()

        async def handle_index(request):
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")
            return web.Response(text="<h1>Gizmo</h1>", content_type="text/html")

        async def handle_health(request):
            return web.Response(text='{"status":"ok"}', content_type="application/json")

        async def handle_ws(request):
            ws = web.WebSocketResponse(heartbeat=15.0)
            await ws.prepare(request)
            server_sid = f"sess_{id(ws):016x}"
            self._connections[server_sid] = ws
            log_event("GizmoServer", "CONNECTION_OPENED", session=server_sid[:8])
            try:
                async for msg in ws:
                    from aiohttp import WSMsgType
                    if msg.type == WSMsgType.TEXT:
                        await self._handle_message(ws, server_sid, msg.data)
                    elif msg.type == WSMsgType.ERROR:
                        log_error("GizmoServer", f"ws error: {ws.exception()}", exc=None)
            finally:
                self._connections.pop(server_sid, None)
                client_sid = _server_to_client_sid.pop(server_sid, None)
                _session_history.pop(server_sid, None)
                _pending_scene_resume.pop(server_sid, None)
                # Don't remove from _live_sockets — next reconnect overwrites it
                try:
                    from core.agent_simple import agent_simple
                    agent_simple.end_session(server_sid)
                except Exception:
                    pass
                log_event("GizmoServer", "CONNECTION_CLOSED", session=server_sid[:8])
            return ws

        async def handle_sessions(request):
            sessions = _list_sessions()
            return web.Response(
                text=json.dumps({"sessions": sessions}),
                content_type="application/json",
            )

        async def handle_session_detail(request):
            sid  = request.match_info["session_id"]
            data = _load_session(sid)
            if not data:
                return web.Response(
                    status=404,
                    text=json.dumps({"error": "not found"}),
                    content_type="application/json",
                )
            return web.Response(
                text=json.dumps(data),
                content_type="application/json",
            )

        app.router.add_get("/",                       handle_index)
        app.router.add_get("/health",                 handle_health)
        app.router.add_get("/sessions",               handle_sessions)
        app.router.add_get("/sessions/{session_id}",  handle_session_detail)
        app.router.add_get("/ws",                     handle_ws)

        log_event("GizmoServer", "STARTING", host=host, port=port)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        log_event("GizmoServer", "LISTENING", host=host, port=port)
        await asyncio.Future()

    async def _handle_message(self, websocket, server_sid: str, raw_msg: str) -> None:
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            await self._send(websocket, {"type": "error", "message": "invalid JSON"})
            return

        msg_type = msg.get("type", "message")
        # Client session ID — stable across reconnects
        sid = msg.get("session_id", server_sid)

        # ── Register live socket immediately, regardless of message type ──────
        # This is the fix: reconnect sends check_unsent first, which means
        # the socket must be registered before we check for pending responses
        # or in-flight pipelines, otherwise they deliver to the dead socket.
        if sid:
            _live_sockets[sid] = websocket
            _server_to_client_sid[server_sid] = sid

        if msg_type == "ping":
            await self._send(websocket, {"type": "pong"})
            return

        if msg_type == "message":
            await self._handle_chat_message(websocket, sid, msg)
            return

        if msg_type == "check_scene":
            await self._handle_check_scene(websocket, sid, msg)
            return

        if msg_type == "scene_resume_answer":
            await self._handle_scene_resume_answer(websocket, sid, msg)
            return

        if msg_type == "check_unsent":
            await self._handle_check_unsent(websocket, sid, msg)
            return

        log_event("GizmoServer", "MSG_TYPE_IGNORED", type=msg_type, session=sid[:8])

    async def _handle_check_scene(self, websocket, session_id: str, msg: dict) -> None:
        from core.scene_tracker import scene_tracker

        headmate = msg.get("headmate", "").strip().lower()
        if not headmate:
            await self._send(websocket, {"type": "scene_check", "has_scene": False})
            return

        try:
            reconnect_msg = await scene_tracker.check_reconnect(headmate)
            if reconnect_msg:
                _pending_scene_resume[session_id] = headmate
                await self._send(websocket, {
                    "type":              "scene_check",
                    "has_scene":         True,
                    "reconnect_message": reconnect_msg,
                })
                log_event("GizmoServer", "SCENE_CHECK_HIT",
                    session=session_id[:8], headmate=headmate)
            else:
                await self._send(websocket, {"type": "scene_check", "has_scene": False})

        except Exception as e:
            log_error("GizmoServer", "scene check failed", exc=e)
            await self._send(websocket, {"type": "scene_check", "has_scene": False})

    async def _handle_scene_resume_answer(self, websocket, session_id: str, msg: dict) -> None:
        from core.scene_tracker import scene_tracker

        headmate = _pending_scene_resume.pop(session_id, None)
        if not headmate:
            await self._handle_chat_message(websocket, session_id, msg)
            return

        content = msg.get("content", "")
        if isinstance(content, list):
            content = assemble_scene_text(content)

        if _is_yes(content):
            scene_tracker.confirm_resume(headmate)
            log_event("GizmoServer", "SCENE_RESUMED", session=session_id[:8], headmate=headmate)
        elif _is_no(content):
            scene_tracker.pause_scene(headmate)
            log_event("GizmoServer", "SCENE_PAUSED", session=session_id[:8], headmate=headmate)
        else:
            scene_tracker.confirm_resume(headmate)

        await self._handle_chat_message(websocket, session_id, msg)

    async def _handle_chat_message(self, websocket, session_id: str, msg: dict) -> None:
        content  = msg.get("content", "")
        context  = msg.get("context", {})
        headmate = context.get("current_host") or msg.get("headmate") or None

        if not content:
            return
        if isinstance(content, str) and len(content) > MAX_MESSAGE_LEN:
            await self._send(websocket, {"type": "error", "message": "message too long"})
            return

        if isinstance(content, list):
            parts    = content
            raw_text = assemble_scene_text(parts)
            multi    = len([p for p in parts if p.get("content_type") != "presence"]) > 1
        else:
            raw_text = content
            parts    = parse_exchange(raw_text, default_headmate=headmate)
            multi    = is_multi_part(raw_text)

        if _is_duplicate(session_id, raw_text):
            return

        speech_parts = [p for p in parts if p.get("content_type") == "speech"]
        if speech_parts:
            first_speaker = speech_parts[0].get("headmate")
            if first_speaker and first_speaker != headmate:
                context["current_host"] = first_speaker
                headmate = first_speaker

        all_speakers = list(dict.fromkeys(p["headmate"] for p in parts if p.get("headmate")))
        if all_speakers:
            context["fronters"] = list(set(context.get("fronters", []) + all_speakers))

        if not headmate and not multi and len(parts) == 1:
            c = parts[0].get("content", "").strip()
            name_match = (
                re.match(r"^([A-Za-z][A-Za-z0-9_\- ]{0,20})$", c) or
                re.search(r"(?:it'?s|i'?m|this is|call me|my name is)\s+([A-Za-z][A-Za-z0-9_\- ]{0,20})", c, re.IGNORECASE)
            )
            if name_match:
                detected = name_match.group(1).strip().lower()
                context["current_host"] = detected
                headmate = detected
                context["fronters"] = [detected]

        log_event("GizmoServer", "MESSAGE_RECEIVED",
            session=session_id[:8], headmate=headmate or "unknown",
            multi=multi, parts=len(parts), words=len(raw_text.split()))

        await asyncio.sleep(THINKING_DELAY)
        await self._send(websocket, {"type": "thinking"})

        if session_id not in _session_history:
            saved = _load_session(session_id)
            if saved:
                _session_history[session_id] = [
                    {"role": m["role"], "content": m["content"]}
                    for m in saved.get("messages", [])
                ]
        history = _session_history.get(session_id, [])

        # ── Mark pipeline as in-flight before starting ────────────────────────
        # Write a sentinel so check_unsent knows "still thinking" vs "response lost"
        saved_pre = _load_session(session_id) or {
            "session_id": session_id,
            "opened_at":  time.time(),
            "hosts":      [],
            "tags":       [],
            "messages":   [],
            "mood":       None,
            "summary":    None,
            "topics":     [],
            "notable":    [],
            "emotion_arc": [],
        }
        saved_pre["pipeline_running"] = True
        saved_pre["pipeline_started"] = time.time()
        # Append the user message now so reconnect can find it even mid-pipeline
        now = time.time()
        saved_pre["last_active"] = now
        if headmate and headmate not in saved_pre.get("hosts", []):
            saved_pre.setdefault("hosts", []).append(headmate)
        for h in context.get("fronters", []):
            if h and h not in saved_pre.get("hosts", []):
                saved_pre["hosts"].append(h)
        # Only append if not already the last user message (avoid duplicates on retry)
        messages = saved_pre.get("messages", [])
        if not messages or messages[-1].get("content") != raw_text or messages[-1].get("role") != "user":
            saved_pre.setdefault("messages", []).append({
                "role":    "user",
                "speaker": headmate or "unknown",
                "content": raw_text,
                "ts":      now,
            })
        _save_session(session_id, saved_pre)

        try:
            # Run pipeline as a tracked task so check_unsent can detect it
            loop = asyncio.get_event_loop()
            task = loop.create_task(run_single_pipeline(
                message=raw_text,
                session_id=session_id,
                headmate=headmate,
                context=context,
                history=history,
            ))
            _pipeline_tasks[session_id] = task
            response = await task
        except Exception as e:
            import traceback
            print(f"[PIPELINE ERROR]\n{traceback.format_exc()}", flush=True)
            # Clear pipeline sentinel on failure
            try:
                fail_saved = _load_session(session_id) or {}
                fail_saved.pop("pipeline_running", None)
                fail_saved.pop("pipeline_started", None)
                _save_session(session_id, fail_saved)
            except Exception:
                pass
            _pipeline_tasks.pop(session_id, None)
            await self._send(websocket, {"type": "error", "message": f"{type(e).__name__}: {e}"})
            return
        finally:
            _pipeline_tasks.pop(session_id, None)

        history.append({"role": "user",      "content": raw_text})
        history.append({"role": "assistant", "content": response})
        _session_history[session_id] = history

        # ── Persist full session ──────────────────────────────────────────────
        saved = _load_session(session_id) or saved_pre
        saved["last_active"] = time.time()
        saved.pop("pipeline_running", None)
        saved.pop("pipeline_started", None)
        # Append assistant response
        saved.setdefault("messages", []).append({
            "role":    "assistant",
            "speaker": "gizmo",
            "content": response,
            "ts":      time.time(),
        })
        # Write pending_response so reconnect can deliver if chunks don't make it
        saved["pending_response"]    = response
        saved["pending_response_ts"] = time.time()
        _save_session(session_id, saved)

        # ── Scene state update ────────────────────────────────────────────────
        if headmate:
            try:
                from core.scene_tracker import scene_tracker
                exchange_lines = raw_text.splitlines() + [f"Gizmo: {response}"]
                asyncio.create_task(scene_tracker.update(
                    chunk=exchange_lines,
                    chunk_id=session_id,
                    name=headmate,
                    session_id=session_id,
                ))
            except Exception as e:
                log_error("GizmoServer", "scene update failed", exc=e)

        # ── Deliver via current live socket ───────────────────────────────────
        # By the time we get here, a reconnect may have updated _live_sockets[session_id]
        # to the new socket. Use it.
        live_ws = _live_sockets.get(session_id, websocket)

        for i in range(0, len(response), CHUNK_SIZE):
            await self._send(live_ws, {"type": "chunk", "content": response[i:i+CHUNK_SIZE]})
            await asyncio.sleep(0)

        # Clear pending_response — delivered successfully
        try:
            delivered = _load_session(session_id) or {}
            delivered.pop("pending_response", None)
            delivered.pop("pending_response_ts", None)
            _save_session(session_id, delivered)
        except Exception:
            pass

        await self._send(live_ws, {"type": "done", "session_id": session_id, "current_host": headmate or ""})

        log_event("GizmoServer", "RESPONSE_SENT",
            session=session_id[:8], words=len(response.split()), multi=multi)

    async def _handle_check_unsent(self, websocket, session_id: str, msg: dict) -> None:
        """
        On reconnect, check for:
        1. Pipeline still running — wait for it, deliver when done
        2. Response generated but chunks never reached client (pending_response)
        3. User message with no response at all — re-run pipeline
        """
        headmate = msg.get("headmate", "").strip().lower()

        # ── Case 1: Pipeline still running ────────────────────────────────────
        # The socket was already registered in _handle_message before we got here.
        # The running pipeline will use _live_sockets[session_id] to deliver,
        # which now points to this new socket. Just wait and let it land.
        task = _pipeline_tasks.get(session_id)
        if task and not task.done():
            log_event("GizmoServer", "RECONNECT_PIPELINE_RUNNING",
                session=session_id[:8], headmate=headmate)
            await self._send(websocket, {"type": "thinking"})
            # Wait up to 90s for the pipeline to finish
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=90.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            # Pipeline delivers via _live_sockets — nothing else needed here
            return

        saved = _load_session(session_id)
        if not saved:
            return

        # ── Case 2: Response generated but not delivered ──────────────────────
        pending = saved.get("pending_response", "").strip()
        if pending:
            log_event("GizmoServer", "DELIVERING_PENDING",
                session=session_id[:8], headmate=headmate, words=len(pending.split()))
            for i in range(0, len(pending), CHUNK_SIZE):
                await self._send(websocket, {"type": "chunk", "content": pending[i:i+CHUNK_SIZE]})
                await asyncio.sleep(0)
            await self._send(websocket, {"type": "done", "session_id": session_id, "current_host": headmate})
            saved.pop("pending_response", None)
            saved.pop("pending_response_ts", None)
            _save_session(session_id, saved)
            return

        # ── Case 3: User message with no response ─────────────────────────────
        messages = saved.get("messages", [])
        if not messages:
            return

        last = messages[-1]
        if last.get("role") != "user":
            return

        raw_text = last.get("content", "")
        if not raw_text:
            return

        log_event("GizmoServer", "RESENDING_UNSENT",
            session=session_id[:8], headmate=headmate)

        await asyncio.sleep(THINKING_DELAY)
        await self._send(websocket, {"type": "thinking"})

        context = {"current_host": headmate, "fronters": [headmate]}
        history = _session_history.get(session_id, [])

        if not history:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in messages[:-1]
            ]

        try:
            response = await run_single_pipeline(
                message=raw_text,
                session_id=session_id,
                headmate=headmate,
                context=context,
                history=history,
            )
        except Exception as e:
            log_error("GizmoServer", "resend failed", exc=e)
            return

        if not response:
            return

        history.append({"role": "user",      "content": raw_text})
        history.append({"role": "assistant", "content": response})
        _session_history[session_id] = history

        saved["messages"].append({
            "role":    "assistant",
            "speaker": "gizmo",
            "content": response,
            "ts":      time.time(),
        })
        saved["last_active"] = time.time()
        saved.pop("pending_response",    None)
        saved.pop("pending_response_ts", None)
        saved.pop("pipeline_running",    None)
        saved.pop("pipeline_started",    None)
        _save_session(session_id, saved)

        for i in range(0, len(response), CHUNK_SIZE):
            await self._send(websocket, {"type": "chunk", "content": response[i:i+CHUNK_SIZE]})
            await asyncio.sleep(0)
        await self._send(websocket, {"type": "done", "session_id": session_id, "current_host": headmate})

        log_event("GizmoServer", "UNSENT_DELIVERED",
            session=session_id[:8], words=len(response.split()))

    async def _send(self, websocket, data: dict) -> None:
        try:
            await websocket.send_str(json.dumps(data))
        except Exception:
            pass


server = GizmoServer()


async def main():
    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10000"))
    await server.start(host=host, port=port)


if __name__ == "__main__":
    asyncio.run(main())
