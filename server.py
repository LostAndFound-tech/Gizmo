"""
server.py — Gizmo WebSocket server for Render.
Uses websockets with process_request to handle health checks.
See: https://websockets.readthedocs.io/en/14.0/howto/render.html
"""

from __future__ import annotations
import asyncio
import hashlib
import http
import json
import os
import re
import signal
import time
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

PORT = int(os.getenv("PORT", "10000"))
HOST = os.getenv("HOST", "0.0.0.0")

# ── Health check + frontend via process_request ───────────────────────────────

def process_request(connection, request):
    """
    Handle HTTP requests before WebSocket upgrade.
    Render's health check hits GET /healthz — respond with 200.
    Also serve index.html at GET /.
    """
    path = request.path

    if path in ("/healthz", "/health"):
        return connection.respond(http.HTTPStatus.OK, "OK\n")

    if path == "/" or path == "/index.html":
        html_path = Path(__file__).parent / "index.html"
        body = html_path.read_bytes() if html_path.exists() else b"<h1>Gizmo</h1>"
        headers = {"Content-Type": "text/html", "Content-Length": str(len(body))}
        response = connection.respond(http.HTTPStatus.OK, body)
        return response

    if path == "/sessions":
        return _handle_sessions_http(connection)

    if path.startswith("/sessions/"):
        sid = path.split("/sessions/")[1].strip("/")
        return _handle_session_detail_http(connection, sid)

    # Let WebSocket upgrade proceed normally
    return None

def _handle_sessions_http(connection):
    from core.store import store
    try:
        rows  = store.query("sessions", active=1, order_by="opened_at DESC", limit=50)
        items = [{
            "id":                s["id"],
            "opened_at":         s.get("opened_at"),
            "mood":              s.get("mood"),
            "topics":            s.get("topics") or [],
            "hosts":             s.get("hosts") or [],
            "summary":           (s.get("summary") or "")[:100],
            "parent_session_id": s.get("parent_session_id"),
        } for s in rows]
        body = json.dumps({"sessions": items}).encode()
    except Exception as e:
        body = json.dumps({"sessions": [], "error": str(e)}).encode()
    return connection.respond(http.HTTPStatus.OK, body)

def _handle_session_detail_http(connection, sid: str):
    from core.store import store
    try:
        session = store.get("sessions", sid)
        if not session:
            return connection.respond(http.HTTPStatus.NOT_FOUND, b'{"error":"not found"}')
        arc = store.query("emotion_log", session_id=sid, active=1, order_by="created_at ASC", limit=100)
        session["emotion_arc"] = [{"valence": p.get("valence", 0), "intensity": p.get("intensity", 0)} for p in arc]
        body = json.dumps(session).encode()
        return connection.respond(http.HTTPStatus.OK, body)
    except Exception as e:
        return connection.respond(http.HTTPStatus.INTERNAL_SERVER_ERROR, json.dumps({"error": str(e)}).encode())

# ── Deduplication ─────────────────────────────────────────────────────────────
_seen: dict[str, float] = {}

def _is_duplicate(session_id: str, content: str) -> bool:
    key = hashlib.md5(f"{session_id}:{content}".encode()).hexdigest()
    now = time.time()
    if key in _seen and now - _seen[key] < 5.0:
        return True
    _seen[key] = now
    for k in [k for k, t in _seen.items() if now - t > 10]:
        del _seen[k]
    return False

# ── Exchange parser ───────────────────────────────────────────────────────────
_SPEECH_RE = re.compile(r'^\[?([A-Za-z][A-Za-z0-9_\- ]{0,30})\]?\s*:\s*(.+)', re.DOTALL)
_ACTION_RE = re.compile(r'^\*(.+)\*$', re.DOTALL)

def parse_exchange(raw: str, default_headmate: Optional[str] = None) -> list[dict]:
    parts = []
    for line in [l.strip() for l in raw.strip().split("\n") if l.strip()]:
        m = _ACTION_RE.match(line)
        if m:
            parts.append({"headmate": default_headmate, "content": m.group(1).strip(), "content_type": "action", "directed_at": None})
            continue
        m = _SPEECH_RE.match(line)
        if m:
            parts.append({"headmate": m.group(1).strip().lower(), "content": m.group(2).strip(), "content_type": "speech", "directed_at": None})
            continue
        if line and default_headmate:
            parts.append({"headmate": default_headmate.lower(), "content": line, "content_type": "speech", "directed_at": "gizmo"})
    if not parts and raw.strip() and default_headmate:
        parts.append({"headmate": default_headmate.lower(), "content": raw.strip(), "content_type": "speech", "directed_at": "gizmo"})
    return parts

def is_multi_part(raw: str) -> bool:
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    return len(lines) >= 2 and sum(1 for l in lines if _SPEECH_RE.match(l) or _ACTION_RE.match(l)) >= 2

# ── WebSocket handler ─────────────────────────────────────────────────────────
_connections: dict = {}

async def ws_handler(websocket):
    session_id = f"sess_{id(websocket):016x}"
    _connections[session_id] = websocket
    log_event("GizmoServer", "CONNECTION_OPENED", session=session_id[:8])

    try:
        async for raw_msg in websocket:
            await handle_message(websocket, session_id, raw_msg)
    except Exception as e:
        if "ConnectionClosed" not in type(e).__name__:
            log_error("GizmoServer", f"ws error: {e}", exc=None)
    finally:
        _connections.pop(session_id, None)
        log_event("GizmoServer", "CONNECTION_CLOSED", session=session_id[:8])

async def handle_message(websocket, session_id: str, raw_msg: str) -> None:
    try:
        msg = json.loads(raw_msg)
    except Exception:
        await send(websocket, {"type": "error", "message": "invalid JSON"})
        return

    msg_type = msg.get("type", "message")
    sid      = msg.get("session_id", session_id)

    if msg_type == "ping":
        await send(websocket, {"type": "pong"})
        return
    if msg_type == "switch_host":
        from core.session_manager import session_manager
        headmate = msg.get("headmate", "")
        if headmate:
            session_manager.set_host(session_id=sid, headmate=headmate, confidence=1.0)
        return
    if msg_type == "restore_session":
        await handle_restore(websocket, msg)
        return
    if msg_type == "message":
        await handle_chat(websocket, sid, msg)

async def handle_chat(websocket, session_id: str, msg: dict) -> None:
    from core.session_manager import session_manager
    from core.agent import agent

    content  = msg.get("content", "")
    context  = msg.get("context", {})
    history  = session_manager.get_history(session_id)

    live_ctx = session_manager.get_session_context(session_id)
    if live_ctx.get("current_host"):
        context.setdefault("current_host", live_ctx["current_host"])

    headmate = context.get("current_host") or ""

    if not content:
        return
    if isinstance(content, str) and len(content) > 8000:
        await send(websocket, {"type": "error", "message": "message too long"})
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

    # Host detection from [Name]: prefix
    speech_parts = [p for p in parts if p.get("content_type") == "speech"]
    if speech_parts:
        first = speech_parts[0].get("headmate")
        if first and first != headmate:
            session_manager.set_host(session_id=session_id, headmate=first, confidence=0.95)
            context["current_host"] = first
            headmate = first

    # Host detection from plain answer
    if not headmate and not multi and len(parts) == 1:
        c  = parts[0].get("content", "").strip()
        nm = (re.match(r"^([A-Za-z][A-Za-z0-9_\- ]{0,20})$", c) or
              re.search(r"(?:it'?s|i'?m|this is|call me|my name is)\s+([A-Za-z][A-Za-z0-9_\- ]{0,20})", c, re.IGNORECASE))
        if nm:
            detected = nm.group(1).strip().lower()
            session_manager.set_host(session_id=session_id, headmate=detected, confidence=0.9)
            context["current_host"] = detected
            headmate = detected

    all_speakers = list(dict.fromkeys(p["headmate"] for p in parts if p.get("headmate")))
    if all_speakers:
        session_manager.add_fronters(session_id, all_speakers)
        context["fronters"] = list(set(context.get("fronters", []) + all_speakers))

    log_event("GizmoServer", "MESSAGE_RECEIVED",
        session=session_id[:8], headmate=headmate,
        multi=multi, parts=len(parts), words=len(raw_text.split()))

    await asyncio.sleep(0.3)
    await send(websocket, {"type": "thinking"})

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
            await send(websocket, {"type": "chunk", "content": response_text[i:i+8]})
            await asyncio.sleep(0)

        await send(websocket, {"type": "done", "session_id": session_id})

        session_manager.touch(
            session_id=session_id,
            headmate=headmate,
            fronters=context.get("fronters", []),
            topics=_classify_topics(raw_text),
            register=context.get("register", "neutral"),
        )

    except Exception as e:
        import traceback
        print(f"[PIPELINE ERROR]\n{traceback.format_exc()}", flush=True)
        await send(websocket, {"type": "error", "message": f"{type(e).__name__}: {e}"})

async def handle_restore(websocket, msg: dict) -> None:
    from core.store import store
    from core.session_manager import session_manager
    from core.llm import llm
    import random, string, time as _time

    sid = msg.get("session_id", "")
    if not sid:
        return

    session = store.get("sessions", sid)
    if not session:
        await send(websocket, {"type": "error", "message": "session not found"})
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

    await send(websocket, {"type": "session_restored", "session_id": branch_id, "hosts": hosts})
    log_event("GizmoServer", "SESSION_BRANCHED", parent=sid[:8], branch=branch_id[:8])

    host_name     = hosts[0].title() if hosts else "you"
    last_exchange = "\n".join(
        f"{'User' if m['role']=='user' else 'Gizmo'}: {m['content'][:80]}"
        for m in all_msgs[-3:]
    )

    try:
        await send(websocket, {"type": "thinking"})
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
            await send(websocket, {"type": "chunk", "content": opening.strip()})
            await send(websocket, {"type": "done", "session_id": branch_id})
    except Exception as e:
        log_error("GizmoServer", "restore opening failed", exc=e)

async def send(websocket, data: dict) -> None:
    try:
        await websocket.send(json.dumps(data))
    except Exception:
        pass

def _classify_topics(message: str) -> list:
    topics = []
    for topic, pattern in [
        ("work",     r"\b(work|job|boss|office|deadline)\b"),
        ("health",   r"\b(sick|pain|tired|sleep)\b"),
        ("emotion",  r"\b(feel|feeling|mood|sad|happy|angry)\b"),
        ("identity", r"\b(headmate|front|system|plural)\b"),
    ]:
        if re.search(pattern, message, re.IGNORECASE):
            topics.append(topic)
    return topics or ["general"]

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    import websockets
    from core.llm import llm
    from core.session_manager import session_manager

    await session_manager.start(llm=llm)

    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)

    log_event("GizmoServer", "STARTING", host=HOST, port=PORT)

    async with websockets.serve(
        ws_handler,
        HOST,
        PORT,
        process_request=process_request,
    ):
        log_event("GizmoServer", "LISTENING", host=HOST, port=PORT)
        await stop

if __name__ == "__main__":
    asyncio.run(main())