"""
server.py
Gizmo's WebSocket server. aiohttp version — this one works.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Optional, Callable

from pathlib import Path
from core.log import log, log_event, log_error
from core.timezone import tz_now


DEDUP_WINDOW     = 5.0
THINKING_DELAY   = 0.3
CHUNK_SIZE       = 8
MAX_MESSAGE_LEN  = 8000

_SPEECH_RE   = re.compile(r'^\[?([A-Za-z][A-Za-z0-9_\- ]{0,30})\]?\s*:\s*(.+)', re.DOTALL)
_ACTION_RE   = re.compile(r'^\*(.+)\*$', re.DOTALL)
_DIRECTED_RE = re.compile(r'\b(to|at|@)\s+([A-Za-z][A-Za-z0-9_\- ]{0,20})\b', re.IGNORECASE)

_seen_messages: dict[str, float] = {}

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
        name    = (part.get("headmate") or "unknown").title()
        content = part.get("content", "")
        ctype   = part.get("content_type", "speech")
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

async def run_single_pipeline(message, session_id, headmate, context, history, llm) -> str:
    from core.agent import agent
    try:
        chunks = []
        async for chunk in agent.respond(
            user_message=message,
            session_id=session_id,
            context=context,
            history=history,
        ):
            chunks.append(chunk)
        return "".join(chunks)
    except Exception as e:
        import traceback
        print(f"[SINGLE PIPELINE ERROR]\n{traceback.format_exc()}", flush=True)
        raise

def _get_known_headmates() -> set:
    """
    Build the set of known headmate names from the memory store.
    Used by host_tracker.process_message() to validate identifications.
    Cached loosely — rebuilds on each call but fast (filesystem glob).
    """
    try:
        from core.memory.store import memory_store
        memories_dir = memory_store.root / "memories"
        if not memories_dir.exists():
            return set()
        return {p.name.lower() for p in memories_dir.iterdir() if p.is_dir()}
    except Exception:
        return set()

def _add_known_headmate(name: str) -> None:
    """
    Ensure a headmate has a memories directory.
    Anyone who appears in the speaker window is definitively a headmate.
    Creates the directory if it doesn't exist — that's all it takes.
    """
    try:
        from core.memory.store import memory_store
        p = memory_store.root / "memories" / name.lower()
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _extract_topics_from_parts(parts: list[dict]) -> list[str]:
    try:
        from core.agent import _classify_topics
        all_text = " ".join(p.get("content", "") for p in parts if p.get("content"))
        return _classify_topics(all_text) if all_text else ["general"]
    except Exception:
        return ["general"]

def _time_of_day(hour: int) -> str:
    if 5  <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"


class GizmoServer:

    def __init__(self):
        self._connections: dict[str, object] = {}
        log("GizmoServer", "initialised")

    async def start(self, host: str = "0.0.0.0", port: int = 10000) -> None:
        from aiohttp import web
        from pathlib import Path
        from core.llm import llm
        from core.session_manager import session_manager

        await session_manager.start(llm=llm)

        # ── Scheduler — psych batch processor + any future cron jobs ──────────
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from core.psych_processor import psych_processor

            scheduler = AsyncIOScheduler()
            psych_processor.schedule(scheduler)
            scheduler.start()
            log_event("GizmoServer", "SCHEDULER_STARTED")
        except Exception as e:
            log_error("GizmoServer", "scheduler failed to start", exc=e)

        app = web.Application()

        async def handle_index(request):
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                return web.Response(text=html_path.read_text(encoding="utf-8"), content_type="text/html")
            return web.Response(text="<h1>Gizmo</h1>", content_type="text/html")

        async def handle_health(request):
            return web.Response(text='{"status":"ok"}', content_type="application/json")

        async def handle_sessions(request):
            try:
                from core.store import store
                sessions = store.query("sessions", active=1, order_by="opened_at DESC", limit=50)
                items = [{"id": s["id"], "opened_at": s.get("opened_at"), "mood": s.get("mood"),
                          "topics": s.get("topics") or [], "hosts": s.get("hosts") or [],
                          "summary": (s.get("summary") or "")[:100],
                          "parent_session_id": s.get("parent_session_id")} for s in sessions]
                return web.Response(text=json.dumps({"sessions": items}), content_type="application/json")
            except Exception as e:
                return web.Response(text=json.dumps({"sessions": [], "error": str(e)}), content_type="application/json")

        async def handle_session_detail(request):
            sid = request.match_info.get("session_id", "")
            try:
                from core.store import store
                session = store.get("sessions", sid)
                if not session:
                    return web.Response(text=json.dumps({"error": "not found"}), content_type="application/json", status=404)
                arc = store.query("emotion_log", session_id=sid, active=1, order_by="created_at ASC", limit=100)
                session["emotion_arc"] = [{"valence": p.get("valence", 0), "intensity": p.get("intensity", 0), "register": p.get("register", "")} for p in arc]
                return web.Response(text=json.dumps(session), content_type="application/json")
            except Exception as e:
                return web.Response(text=json.dumps({"error": str(e)}), content_type="application/json", status=500)

        async def handle_ws(request):
            ws = web.WebSocketResponse()
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
                log_event("GizmoServer", "CONNECTION_CLOSED", session=session_id[:8])
            return ws

        app.router.add_get("/",                       handle_index)
        app.router.add_get("/health",                 handle_health)
        app.router.add_get("/ws",                     handle_ws)
        app.router.add_get("/sessions",               handle_sessions)
        app.router.add_get("/sessions/{session_id}",  handle_session_detail)

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
        from core.host_tracker import host_tracker

        content  = msg.get("content", "")
        context  = msg.get("context", {})
        history  = session_manager.get_history(session_id)

        live_ctx = session_manager.get_session_context(session_id)
        if live_ctx.get("current_host"):
            context.setdefault("current_host", live_ctx["current_host"])
        if live_ctx.get("fronters"):
            context.setdefault("fronters", live_ctx["fronters"])

        # ── Speaker window is ground truth ────────────────────────────────────
        # context.current_host comes directly from the client UI speaker window.
        # Trust it unconditionally. Register as known headmate immediately.
        headmate = context.get("current_host") or ""
        if headmate:
            _add_known_headmate(headmate)
            session_manager.set_host(session_id=session_id, headmate=headmate, confidence=1.0)

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

        # ── Host identification — natural speech patterns ─────────────────────
        # Catches "it's Jess", "I'm Ara", "Jess here" etc.
        # Fires before everything else so headmate is known for the full pipeline
        _prev_headmate = headmate
        try:
            known = _get_known_headmates()
            changes = host_tracker.process_message(session_id, raw_text, known)
            if changes.get("host_identified"):
                identified = changes["host_identified"].lower()
                # Add to known headmates — anyone seen in speaker window is a headmate
                _add_known_headmate(identified)
                context["current_host"] = identified
                headmate = identified
                log_event("GizmoServer", "HOST_IDENTIFIED",
                    session=session_id[:8], headmate=identified)
        except Exception as e:
            log_error("GizmoServer", "host identification failed", exc=e)

        # ── Host change — hard reset session context ──────────────────────────
        # If who we're talking to changed, tell Gizmo before he responds
        _host_changed = (
            headmate and _prev_headmate and
            headmate.lower() != _prev_headmate.lower()
        )

        # ── Speaker detection from structured [Name]: format ──────────────────
        speech_parts = [p for p in parts if p.get("content_type") == "speech"]
        if speech_parts:
            first_speaker = speech_parts[0].get("headmate")
            if first_speaker and first_speaker != headmate:
                session_manager.set_host(session_id=session_id, headmate=first_speaker, confidence=0.95)
                context["current_host"] = first_speaker
                headmate = first_speaker

        all_speakers = list(dict.fromkeys(p["headmate"] for p in parts if p.get("headmate")))
        if all_speakers:
            session_manager.add_fronters(session_id, all_speakers)
            context["fronters"] = list(set(context.get("fronters", []) + all_speakers))

        fronters = context.get("fronters", [headmate] if headmate else [])

        # ── Fallback — bare name answer ───────────────────────────────────────
        if not headmate and not multi and len(parts) == 1:
            c = parts[0].get("content", "").strip()
            import re as _re
            name_match = (
                _re.match(r"^([A-Za-z][A-Za-z0-9_\- ]{0,20})$", c) or
                _re.search(r"(?:it'?s|i'?m|this is|call me|my name is)\s+([A-Za-z][A-Za-z0-9_\- ]{0,20})", c, _re.IGNORECASE)
            )
            if name_match:
                detected = name_match.group(1).strip().lower()
                session_manager.set_host(session_id=session_id, headmate=detected, confidence=0.9)
                context["current_host"] = detected
                headmate = detected
                fronters = [detected]
                context["fronters"] = fronters

        # ── Inject host change into context if switch detected ───────────────
        if _host_changed:
            context["host_changed"] = True
            context["previous_host"] = _prev_headmate
            log_event("GizmoServer", "HOST_SWITCH_DETECTED",
                session=session_id[:8],
                from_host=_prev_headmate,
                to_host=headmate,
            )

        log_event("GizmoServer", "MESSAGE_RECEIVED",
            session=session_id[:8], headmate=headmate or "unknown", multi=multi,
            parts=len(parts), words=len(raw_text.split()))

        await asyncio.sleep(THINKING_DELAY)
        await self._send(websocket, {"type": "thinking"})

        try:
            response = await run_single_pipeline(
                message=raw_text,
                session_id=session_id,
                headmate=headmate, context=context,
                history=history, llm=llm,
            )
        except Exception as e:
            import traceback
            print(f"[PIPELINE ERROR]\n{traceback.format_exc()}", flush=True)
            await self._send(websocket, {"type": "error", "message": f"{type(e).__name__}: {e}"})
            return

        for i in range(0, len(response), CHUNK_SIZE):
            await self._send(websocket, {"type": "chunk", "content": response[i:i+CHUNK_SIZE]})
            await asyncio.sleep(0)

        await self._send(websocket, {"type": "done", "session_id": session_id, "current_host": headmate or ""})

        # ── Session touch — correct signature (hosts + topics only) ──────────
        session_manager.touch(
            session_id = session_id,
            hosts      = [headmate] if headmate else [],
            topics     = _extract_topics_from_parts(parts),
        )

        # ── Scene extraction — fire and forget after response ─────────────────
        try:
            from core.memory.session_context import session_context_manager
            ctx = session_context_manager.get(session_id)
            if ctx and session_context_manager.should_update_scene(session_id):
                asyncio.ensure_future(
                    session_context_manager.update_scene(
                        session_id = session_id,
                        assembled  = raw_text,
                        parts      = parts,
                        headmate   = headmate,
                        llm        = llm,
                    )
                )
        except Exception as e:
            log_error("GizmoServer", "scene update failed to schedule", exc=e)

        log_event("GizmoServer", "RESPONSE_SENT",
            session=session_id[:8], words=len(response.split()), multi=multi)

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

        suffix    = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        branch_id = f"{sid}_{suffix}"

        now         = _time.time()
        last_ts     = session.get("closed_at") or session.get("opened_at") or now
        elapsed_hrs = (now - last_ts) / 3600

        if elapsed_hrs < 0.5:  time_note = "just a few minutes ago"
        elif elapsed_hrs < 2:  time_note = f"about {int(elapsed_hrs*60)} minutes ago"
        elif elapsed_hrs < 24: time_note = f"about {int(elapsed_hrs)} hours ago"
        elif elapsed_hrs < 48: time_note = "yesterday"
        else:                  time_note = f"{int(elapsed_hrs/24)} days ago"

        history   = session_manager.get_history(branch_id)
        messages  = store.query("messages",  session_id=sid, active=1, order_by="created_at ASC", limit=20)
        responses = store.query("responses", session_id=sid, active=1, order_by="created_at ASC", limit=20)

        all_msgs = sorted(
            [{"role": "user",      "content": m["content"], "ts": m.get("created_at", 0)} for m in messages  if m.get("content")] +
            [{"role": "assistant", "content": r["content"], "ts": r.get("created_at", 0)} for r in responses if r.get("content")],
            key=lambda x: x["ts"]
        )
        for m in all_msgs[-12:]:
            try: history.add(m["role"], m["content"])
            except: pass

        hosts = session.get("hosts") or []
        if hosts:
            session_manager.set_host(session_id=branch_id, headmate=hosts[0], confidence=0.9, fronters=hosts)

        store.write("sessions", {
            "id": branch_id, "opened_at": now, "hosts": hosts, "fronters": hosts,
            "parent_session_id": sid, "branch_point": now,
            "headmate": hosts[0].lower() if hosts else None,
            "source": "branch", "tags": "session,branch",
        })

        await self._send(websocket, {"type": "session_restored", "session_id": branch_id, "hosts": hosts})
        log_event("GizmoServer", "SESSION_BRANCHED", parent=sid[:8], branch=branch_id[:8])

        host_name     = hosts[0].title() if hosts else "you"
        last_exchange = "\n".join(
            f"{'User' if m['role']=='user' else 'Gizmo'}: {m['content'][:80]}"
            for m in all_msgs[-3:]
        )

        try:
            await self._send(websocket, {"type": "thinking"})
            opening = await llm.generate(
                [{"role": "user", "content": (
                    f"Reconnecting with {host_name} after {time_note}.\n"
                    f"Last mood: {session.get('mood','unknown')}\n"
                    f"Unresolved: {session.get('unresolved') or 'nothing'}\n"
                    f"Last exchanges:\n{last_exchange or '(none)'}\n\n"
                    f"Write ONE natural opening, 1-2 sentences. Acknowledge the time. Be present."
                )}],
                system_prompt=f"You are Gizmo. Reconnecting with {host_name} after {time_note}. Warm, present, 1-2 sentences.",
                max_new_tokens=80, temperature=0.85,
            )
            if opening and opening.strip():
                await self._send(websocket, {"type": "chunk", "content": opening.strip()})
                await self._send(websocket, {"type": "done", "session_id": branch_id})
        except Exception as e:
            log_error("GizmoServer", "restore opening failed", exc=e)

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
