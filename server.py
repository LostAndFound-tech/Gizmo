"""
server.py
Gizmo's WebSocket server. Bare bones testing branch.
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
# session_id -> headmate name
_pending_scene_resume: dict[str, str] = {}


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
    """Loose yes-detection for scene resume answers."""
    t = text.strip().lower()
    return any(w in t for w in ("yes", "yeah", "yep", "sure", "ok", "okay", "yea", "let's", "lets", "continue", "pick up", "resume"))


def _is_no(text: str) -> bool:
    """Loose no-detection for scene resume answers."""
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
            session_id = f"sess_{id(ws):016x}"
            self._connections[session_id] = ws
            log_event("GizmoServer", "CONNECTION_OPENED", session=session_id[:8])
            try:
                async for msg in ws:
                    from aiohttp import WSMsgType
                    if msg.type == WSMsgType.TEXT:
                        await self._handle_message(ws, session_id, msg.data)
                    elif msg.type == WSMsgType.ERROR:
                        log_error("GizmoServer", f"ws error: {ws.exception()}", exc=None)
            finally:
                self._connections.pop(session_id, None)
                _session_history.pop(session_id, None)
                _pending_scene_resume.pop(session_id, None)
                log_event("GizmoServer", "CONNECTION_CLOSED", session=session_id[:8])
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

    async def _handle_message(self, websocket, session_id: str, raw_msg: str) -> None:
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            await self._send(websocket, {"type": "error", "message": "invalid JSON"})
            return

        msg_type = msg.get("type", "message")
        sid      = msg.get("session_id", session_id)

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

        log_event("GizmoServer", "MSG_TYPE_IGNORED", type=msg_type, session=sid[:8])

    async def _handle_check_scene(self, websocket, session_id: str, msg: dict) -> None:
        """
        Client connects (or reconnects) and checks whether there's an active scene.
        If yes, generate Gizmo's re-entry offer and send it back.
        """
        from core.scene_tracker import scene_tracker

        headmate = msg.get("headmate", "").strip().lower()
        if not headmate:
            await self._send(websocket, {"type": "scene_check", "has_scene": False})
            return

        try:
            reconnect_msg = await scene_tracker.check_reconnect(headmate)
            if reconnect_msg:
                # Park the pending resume so the next message from this session
                # is interpreted as a yes/no answer
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
        """
        User answered Gizmo's re-entry offer. Route to confirm or pause.
        Then fall through to normal chat handling with scene context loaded.
        """
        from core.scene_tracker import scene_tracker

        headmate = _pending_scene_resume.pop(session_id, None)
        if not headmate:
            # No pending resume — treat as a normal message
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
            # Ambiguous — resume the scene, let Gizmo handle it naturally
            scene_tracker.confirm_resume(headmate)

        # Now handle as a normal chat message so Gizmo actually responds
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

        # Load from disk if not in memory (e.g. after server restart)
        if session_id not in _session_history:
            saved = _load_session(session_id)
            if saved:
                _session_history[session_id] = [
                    {"role": m["role"], "content": m["content"]}
                    for m in saved.get("messages", [])
                ]
        history = _session_history.get(session_id, [])

        try:
            response = await run_single_pipeline(
                message=raw_text,
                session_id=session_id,
                headmate=headmate,
                context=context,
                history=history,
            )
        except Exception as e:
            import traceback
            print(f"[PIPELINE ERROR]\n{traceback.format_exc()}", flush=True)
            await self._send(websocket, {"type": "error", "message": f"{type(e).__name__}: {e}"})
            return

        now = time.time()
        history.append({"role": "user",      "content": raw_text})
        history.append({"role": "assistant", "content": response})
        _session_history[session_id] = history

        # ── Persist session to disk ───────────────────────────────────────────
        saved = _load_session(session_id) or {
            "session_id": session_id,
            "opened_at":  now,
            "hosts":      [],
            "tags":       [],
            "messages":   [],
            "mood":       None,
            "summary":    None,
            "topics":     [],
            "notable":    [],
            "emotion_arc": [],
        }
        saved["last_active"] = now
        # Merge hosts
        if headmate and headmate not in saved["hosts"]:
            saved["hosts"].append(headmate)
        for h in context.get("fronters", []):
            if h and h not in saved["hosts"]:
                saved["hosts"].append(h)
        # Append messages with speaker and timestamp
        saved["messages"].append({
            "role":    "user",
            "speaker": headmate or "unknown",
            "content": raw_text,
            "ts":      now,
        })
        saved["messages"].append({
            "role":    "assistant",
            "speaker": "gizmo",
            "content": response,
            "ts":      now,
        })
        _save_session(session_id, saved)

        # ── Scene state update ────────────────────────────────────────────────
        # Run after response so the full exchange (user + Gizmo) informs the scene
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

        for i in range(0, len(response), CHUNK_SIZE):
            await self._send(websocket, {"type": "chunk", "content": response[i:i+CHUNK_SIZE]})
            await asyncio.sleep(0)

        await self._send(websocket, {"type": "done", "session_id": session_id, "current_host": headmate or ""})

        log_event("GizmoServer", "RESPONSE_SENT",
            session=session_id[:8], words=len(response.split()), multi=multi)

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
