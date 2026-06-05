"""
server.py
Gizmo's WebSocket server. Bare bones testing branch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Optional

from pathlib import Path
from core.log import log, log_event, log_error
from core.timezone import tz_now


DEDUP_WINDOW    = 5.0
THINKING_DELAY  = 0.3
CHUNK_SIZE      = 8
MAX_MESSAGE_LEN = 8000

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


async def run_single_pipeline(message, session_id, headmate, context, history) -> tuple[str, str]:
    from core.agent_simple import agent_simple as agent
    ctx = context if context is not None else {}
    try:
        chunks = []
        async for chunk in agent.run(
            user_message=message,
            history=history,
            session_id=session_id,
            context=ctx,
        ):
            chunks.append(chunk)
        prompt = ctx.get("_last_prompt", "")
        return prompt, "".join(chunks)
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

        app.router.add_get("/",       handle_index)
        app.router.add_get("/health", handle_health)
        app.router.add_get("/ws",     handle_ws)

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
        if msg_type == "regenerate":
            await self._handle_regenerate(websocket, sid, msg)
            return

        log_event("GizmoServer", "MSG_TYPE_IGNORED", type=msg_type, session=sid[:8])

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

        try:
            prompt, response = await run_single_pipeline(
                message=raw_text,
                session_id=session_id,
                headmate=headmate,
                context=context,
                history=[],
            )
        except Exception as e:
            import traceback
            print(f"[PIPELINE ERROR]\n{traceback.format_exc()}", flush=True)
            await self._send(websocket, {"type": "error", "message": f"{type(e).__name__}: {e}"})
            return

        await self._send(websocket, {
            "type":     "prompt_sections",
            "sections": {"extractor_prompt": prompt},
        })

        for i in range(0, len(response), CHUNK_SIZE):
            await self._send(websocket, {"type": "chunk", "content": response[i:i+CHUNK_SIZE]})
            await asyncio.sleep(0)

        await self._send(websocket, {"type": "done", "session_id": session_id, "current_host": headmate or ""})

        log_event("GizmoServer", "RESPONSE_SENT",
            session=session_id[:8], words=len(response.split()), multi=multi)

    async def _handle_regenerate(self, websocket, session_id: str, msg: dict) -> None:
        from core.llm import llm

        sections = msg.get("sections", {})
        prompt   = sections.get("extractor_prompt", "").strip()

        if not prompt:
            await self._send(websocket, {"type": "error", "message": "no prompt to regenerate"})
            return

        log_event("GizmoServer", "REGENERATE", session=session_id[:8], prompt_len=len(prompt))
        await self._send(websocket, {"type": "thinking"})

        try:
            response = await llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=3000,
                temperature=0.0,
            )
        except Exception as e:
            await self._send(websocket, {"type": "error", "message": f"{type(e).__name__}: {e}"})
            return

        await self._send(websocket, {
            "type":     "regenerated",
            "response": response,
            "thinking": {},
        })

        log_event("GizmoServer", "REGENERATE_SENT", session=session_id[:8], words=len(response.split()))

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