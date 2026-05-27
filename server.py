"""
server.py — Gizmo WebSocket + HTTP server.
Uses aiohttp to serve both HTTP (health check, frontend) and WebSocket on one port.
Session manager starts via aiohttp on_startup AFTER the port is bound.
"""

from __future__ import annotations
import asyncio
import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Optional, Callable
from core.log import log, log_event, log_error

# ── Deduplication ─────────────────────────────────────────────────────────────
_seen: dict[str, float] = {}

def _is_duplicate(session_id: str, content: str) -> bool:
    key = hashlib.md5(f"{session_id}:{content}".encode()).hexdigest()
    now = time.time()
    if key in _seen and now - _seen[key] < 5.0:
        return True
    _seen[key] = now
    stale = [k for k, t in _seen.items() if now - t > 10]
    for k in stale:
        del _seen[k]
    return False

# ── Exchange parser ───────────────────────────────────────────────────────────
_SPEECH_RE = re.compile(r'^\[?([A-Za-z][A-Za-z0-9_\- ]{0,30})\]?\s*:\s*(.+)', re.DOTALL)
_ACTION_RE = re.compile(r'^\*(.+)\*$', re.DOTALL)

def parse_exchange(raw: str, default_headmate: Optional[str] = None) -> list[dict]:
    parts = []
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    for line in lines:
        action_match = _ACTION_RE.match(line)
        if action_match:
            parts.append({
                "headmate": default_headmate,
                "content": action_match.group(1).strip(),
                "content_type": "action",
                "directed_at": None,
            })
            continue
        speech_match = _SPEECH_RE.match(line)
        if speech_match:
            parts.append({
                "headmate": speech_match.group(1).strip().lower(),
                "content": speech_match.group(2).strip(),
                "content_type": "speech",
                "directed_at": None,
            })
            continue
        if line and default_headmate:
            parts.append({
                "headmate": default_headmate.lower(),
                "content": line,
                "content_type": "speech",
                "directed_at": "gizmo",
            })
    if not parts and raw.strip() and default_headmate:
        parts.append({
            "headmate": default_headmate.lower(),
            "content": raw.strip(),
            "content_type": "speech",
            "directed_at": "gizmo",
        })
    return parts

def is_multi_part(raw: str) -> bool:
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    return sum(1 for l in lines if _SPEECH_RE.match(l) or _ACTION_RE.match(l)) >= 2

# ── Server ────────────────────────────────────────────────────────────────────

class GizmoServer:

    def __init__(self):
        self._connections: dict = {}
        log("GizmoServer", "initialised")

    async def start(self, host: str = "0.0.0.0", port: int = 10000) -> None:
        from aiohttp import web

        # Create app FIRST
        app = web.Application()

        # Register startup handler SECOND
        async def on_startup(app):
            from core.llm import llm
            from core.session_manager import session_manager
            await session_manager.start(llm=llm)
            log_event("GizmoServer", "READY")

        app.on_startup.append(on_startup)

        # Register routes THIRD
        app.router.add_get("/",        self._handle_index)
        app.router.add_get("/health",  self._handle_health)
        app.router.add_get("/ws",      self._handle_ws)
        app.router.add_get("/sessions", self._handle_sessions)
        app.router.add_get("/sessions/{sid}", self._handle_session_detail)

        # Start server LAST
        log_event("GizmoServer", "STARTING", host=host, port=port)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        log_event("GizmoServer", "LISTENING", host=host, port=port)
        await asyncio.Future()

    # ── HTTP handlers ─────────────────────────────────────────────────────────

    async def _handle_index(self, request):
        from aiohttp import web
        html_path = Path(__file__).parent / "index.html"
        if html_path.exists():
            return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")
        return web.Response(text="<h1>Gizmo</h1>", content_type="text/html")

    async def _handle_health(self, request):
        from aiohttp import web
        return web.Response(text='{"status":"ok"}', content_type="application/json")

    async def _handle_sessions(self, request):
        from aiohttp import web
        from core.store import store
        try:
            sessions = store.query("sessions", active=1, order_by="opened_at DESC", limit=50)
            items = [{
                "id":                s["id"],
                "opened_at":         s.get("opened_at"),
                "mood":              s.get("mood"),
                "topics":            s.get("topics") or [],
                "hosts":             s.get("hosts") or [],
                "summary":           (s.get("summary") or "")[:100],
                "parent_session_id": s.get("parent_session_id"),
            } for s in sessions]
            return web.Response(text=json.dumps({"sessions": items}), content_type="application/json")
        except Exception as e:
            return web.Response(text=json.dumps({"sessions": [], "error": str(e)}), content_type="application/json")

    async def _handle_session_detail(self, request):
        from aiohttp import web
        from core.store import store
        sid = request.match_info.get("sid", "")
        try:
            session = store.get("sessions", sid)
            if not session:
                return web.Response(text=json.dumps({"error": "not found"}), content_type="application/json", status=404)
            arc = store.query("emotion_log", session_id=sid, active=1, order_by="created_at ASC", limit=100)
            session["emotion_arc"] = [{"valence": p.get("valence", 0), "intensity": p.get("intensity", 0), "register": p.get("register", "")} for p in arc]
            return web.Response(text=json.dumps(session), content_type="application/json")
        except Exception as e:
            return web.Response(text=json.dumps({"error": str(e)}), content_type="application/json", status=500)

    # ── WebSocket handler ─────────────────────────────────────────────────────

    async def _handle_ws(self, request):
        from aiohttp import web
        from aiohttp.web import WebSocketResponse
        from aiohttp import WSMsgType

        ws = WebSocketResponse()
        await ws.prepare(request)

        session_id = f"sess_{id(ws):016x}"
        self._connections[session_id] = ws
        log_event("GizmoServer", "CONNECTION_OPENED", session=session_id[:8])

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_message(ws, session_id, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    log_error("GizmoServer", f"ws error: {ws.exception()}", exc=None)
        finally:
            self._connections.pop(session_id, None)
            log_event("GizmoServer", "CONNECTION_CLOSED", session=session_id[:8])

        return ws

    # ── Message router ────────────────────────────────────────────────────────

    async def _handle_message(self, websocket, session_id: str, raw_msg: str) -> None:
        try:
            msg = json.loads(raw_msg)
        except Exception:
            await self._send(websocket, {"type": "error", "message": "invalid JSON"})
            return

        msg_type = msg.get("type", "message")
        sid      = msg.get("session_id", session_id)

        if msg_type == "ping":
            await self._send(websocket, {"type": "pong"})
            return
        if msg_type == "restore_session":
            await self._handle_restore_session(websocket, msg)
            return
        if msg_type == "switch_host":
            await self._handle_switch_host(sid, msg)
            return
        if msg_type == "add_fronter":
            await self._handle_add_fronter(sid, msg)
            return
        if msg_type == "remove_fronter":
            await self._handle_remove_fronter(sid, msg)
            return
        if msg_type == "message":
            await self._handle_chat_message(websocket, sid, msg)

    async def _handle_chat_message(self, websocket, session_id: str, msg: dict) -> None:
        from core.llm import llm
        from core.session_manager import session_manager
        from core.agent import agent

        content  = msg.get("content", "")
        context  = msg.get("context", {})
        history  = session_manager.get_history(session_id)
        sess_ctx = session_manager.get_or_create(session_id)

        live_ctx = session_manager.get_session_context(session_id)
        if live_ctx.get("current_host"):
            context.setdefault("current_host", live_ctx["current_host"])
        if live_ctx.get("fronters"):
            context.setdefault("fronters", live_ctx["fronters"])

        headmate = context.get("current_host") or ""

        if not content:
            return

        if isinstance(content, str) and len(content) > 8000:
            await self._send(websocket, {"type": "error", "message": "message too long"})
            return

        if isinstance(content, list):
            parts    = content
            raw_text = " ".join(p.get("content", "") for p in parts if p.get("content"))
            multi    = len([p for p in parts if p.get("content_type") != "presence"]) > 1
        else:
            raw_text = content
            parts    = parse_exchange(raw_text, default_headmate=headmate)
            multi    = is_multi_part(raw_text)

        if _is_duplicate(session_id, raw_text):
            return

        # Detect host from [Name]: prefix
        speech_parts = [p for p in parts if p.get("content_type") == "speech"]
        if speech_parts:
            first_speaker = speech_parts[0].get("headmate")
            if first_speaker and first_speaker != headmate:
                session_manager.set_host(session_id=session_id, headmate=first_speaker, confidence=0.95)
                context["current_host"] = first_speaker
                headmate = first_speaker

        # Identify host from plain answer to "who's there?"
        if not headmate and not multi and len(parts) == 1:
            c = parts[0].get("content", "").strip()
            nm = (
                re.match(r"^([A-Za-z][A-Za-z0-9_\- ]{0,20})$", c) or
                re.search(r"(?:it'?s|i'?m|this is|call me|my name is)\s+([A-Za-z][A-Za-z0-9_\- ]{0,20})", c, re.IGNORECASE)
            )
            if nm:
                detected = nm.group(1).strip().lower()
                session_manager.set_host(session_id=session_id, headmate=detected, confidence=0.9)
                context["current_host"] = detected
                headmate = detected

        all_speakers = list(dict.fromkeys(p["headmate"] for p in parts if p.get("headmate")))
        if all_speakers:
            session_manager.add_fronters(session_id, all_speakers)
            context["fronters"] = list(set(context.get("fronters", []) + all_speakers))

        fronters = context.get("fronters", [headmate] if headmate else [])

        log_event("GizmoServer", "MESSAGE_RECEIVED",
            session=session_id[:8], headmate=headmate,
            multi=multi, parts=len(parts), words=len(raw_text.split()))

        await asyncio.sleep(0.3)
        await self._send(websocket, {"type": "thinking"})

        try:
            single_msg = parts[0]["content"] if parts else raw_text
            chunks     = []
            async for chunk in agent.respond(
                user_message=single_msg,
                session_id=session_id,
                context=context,
                history=history,
            ):
                chunks.append(chunk)

            response_text = "".join(chunks)

            for i in range(0, len(response_text), 8):
                await self._send(websocket, {"type": "chunk", "content": response_text[i:i+8]})
                await asyncio.sleep(0)

            await self._send(websocket, {"type": "done", "session_id": session_id})

            session_manager.touch(
                session_id=session_id,
                headmate=headmate,
                fronters=fronters,
                topics=self._classify_topics(raw_text),
                register=context.get("register", "neutral"),
            )

        except Exception as e:
            import traceback
            print(f"[PIPELINE ERROR]\n{traceback.format_exc()}", flush=True)
            await self._send(websocket, {"type": "error", "message": f"{type(e).__name__}: {e}"})

    # ── Session restore / branch ──────────────────────────────────────────────

    async def _handle_restore_session(self, websocket, msg: dict) -> None:
        from core.store import store
        from core.session_manager import session_manager
        from core.llm import llm
        import random, string, time as _time

        sid = msg.get("session_id", "")
        if not sid:
            return

        session = store.get("sessions", sid)
        if not session:
            await self._send(websocket, {"type": "error", "message": "session not found"})
            return

        # Branch session ID
        suffix    = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        branch_id = f"{sid}_{suffix}"

        now          = _time.time()
        last_ts      = session.get("closed_at") or session.get("updated_at") or session.get("opened_at") or now
        elapsed_hrs  = (now - last_ts) / 3600

        if elapsed_hrs < 0.5:     time_note = "just a few minutes ago"
        elif elapsed_hrs < 2:     time_note = f"about {int(elapsed_hrs * 60)} minutes ago"
        elif elapsed_hrs < 24:    time_note = f"about {int(elapsed_hrs)} hours ago"
        elif elapsed_hrs < 48:    time_note = "yesterday"
        else:                     time_note = f"{int(elapsed_hrs / 24)} days ago"

        # Restore history into branch
        history   = session_manager.get_history(branch_id)
        messages  = store.query("messages",  session_id=sid, active=1, order_by="created_at ASC", limit=20)
        responses = store.query("responses", session_id=sid, active=1, order_by="created_at ASC", limit=20)

        all_msgs = sorted(
            [{"role": "user",      "content": m["content"], "ts": m.get("created_at", 0)} for m in messages  if m.get("content")] +
            [{"role": "assistant", "content": r["content"], "ts": r.get("created_at", 0)} for r in responses if r.get("content")],
            key=lambda x: x["ts"]
        )

        for m in all_msgs[-12:]:
            try:
                history.add(m["role"], m["content"])
            except Exception:
                pass

        hosts = session.get("hosts") or []
        if hosts:
            session_manager.set_host(session_id=branch_id, headmate=hosts[0], confidence=0.9, fronters=hosts)

        store.write("sessions", {
            "id":                branch_id,
            "opened_at":         now,
            "hosts":             hosts,
            "fronters":          hosts,
            "parent_session_id": sid,
            "branch_point":      now,
            "headmate":          hosts[0].lower() if hosts else None,
            "source":            "branch",
            "tags":              f"session,branch,{hosts[0].lower() if hosts else 'unknown'}",
        })

        await self._send(websocket, {"type": "session_restored", "session_id": branch_id, "hosts": hosts})

        log_event("GizmoServer", "SESSION_BRANCHED", parent=sid[:8], branch=branch_id, elapsed_hrs=round(elapsed_hrs, 1))

        # Generate opening
        host_name  = hosts[0].title() if hosts else "you"
        last_exchange = "\n".join(
            f"{'User' if m['role']=='user' else 'Gizmo'}: {m['content'][:80]}"
            for m in all_msgs[-3:]
        ) if all_msgs else ""

        try:
            await self._send(websocket, {"type": "thinking"})
            opening = await llm.generate(
                [{"role": "user", "content": (
                    f"You are reconnecting with {host_name}. It has been {time_note}.\n"
                    f"Last session mood: {session.get('mood','unknown')}\n"
                    f"Summary: {(session.get('summary') or '')[:150]}\n"
                    f"Unresolved: {session.get('unresolved') or 'nothing specific'}\n"
                    f"Last exchanges:\n{last_exchange or '(none)'}\n\n"
                    f"Write ONE short natural opening — 1-2 sentences. "
                    f"Acknowledge the time. Reference anything left hanging. "
                    f"Don't say 'welcome back'. Just be present."
                )}],
                system_prompt=f"You are Gizmo reconnecting with {host_name} after {time_note}. Natural, warm, present. 1-2 sentences only.",
                max_new_tokens=80,
                temperature=0.85,
            )
            if opening and opening.strip():
                await self._send(websocket, {"type": "chunk", "content": opening.strip()})
                await self._send(websocket, {"type": "done", "session_id": branch_id})
        except Exception as e:
            log_error("GizmoServer", "restore opening failed", exc=e)

    # ── Host/fronter control ──────────────────────────────────────────────────

    async def _handle_switch_host(self, session_id: str, msg: dict) -> None:
        from core.session_manager import session_manager
        headmate = msg.get("headmate", "")
        if headmate:
            session_manager.set_host(session_id=session_id, headmate=headmate,
                confidence=msg.get("confidence", 1.0), fronters=msg.get("fronters"))

    async def _handle_add_fronter(self, session_id: str, msg: dict) -> None:
        from core.session_manager import session_manager
        names = msg.get("fronters", []) or [msg.get("headmate", "")]
        session_manager.add_fronters(session_id, [n for n in names if n])

    async def _handle_remove_fronter(self, session_id: str, msg: dict) -> None:
        from core.session_manager import session_manager
        name = msg.get("headmate", "")
        if name:
            session_manager.remove_fronter(session_id, name)

    # ── Send ──────────────────────────────────────────────────────────────────

    async def _send(self, websocket, data: dict) -> None:
        try:
            await websocket.send_str(json.dumps(data))
        except Exception:
            pass

    def _classify_topics(self, message: str) -> list:
        import re
        topics = []
        for topic, pattern in [
            ("work",     r"\b(work|job|boss|office|meeting|deadline)\b"),
            ("health",   r"\b(sick|pain|tired|sleep|exhausted)\b"),
            ("emotion",  r"\b(feel|feeling|mood|sad|happy|angry)\b"),
            ("identity", r"\b(headmate|front|system|plural|switch)\b"),
        ]:
            if re.search(pattern, message, re.IGNORECASE):
                topics.append(topic)
        return topics or ["general"]


# ── Entry point ───────────────────────────────────────────────────────────────

server = GizmoServer()

async def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10000"))
    await server.start(host=host, port=port)

if __name__ == "__main__":
    asyncio.run(main())