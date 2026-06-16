"""
core/agent_simple.py

Mode router for Gizmo.

Modes:
  passive      — pipeline runs, no response (transcript ingestion, background listening)
  chat         — pipeline runs, Gizmo responds as companion (default)
  journal      — witness mode: gentle prompts, write-up on exit, approval gate
  brainstorm   — chaos volley + thinking partner; pipeline flushes at session end
  roleplay     — negotiation → scene generation with GM/NPC dual role
  aftercare    — warm presence, praise, honest check-in; requestable from anywhere

Mode switching:
  "passive mode"      → passive
  "chat mode"         → chat
  "journal mode"      → journal
  "brainstorm mode"   → brainstorm
  "roleplay mode"     → roleplay
  "aftercare"         → aftercare (also triggered automatically post-scene)

  Switching OUT of journal triggers write-up before releasing.
  Switching OUT of brainstorm flushes the pipeline.
  Switching OUT of roleplay saves and closes the scene log.

Keyphrases (any mode):
  "run wellness report"    → full wellness synthesis
  "run report for <name>"  → individual wellness synthesis
"""

import asyncio
import time
import json
from typing import AsyncGenerator, Optional

from core.log import log_event, log_error
from core.chunk_processor import ChunkProcessor
from core.responder import responder as _responder


# ── Mode state ────────────────────────────────────────────────────────────────

_mode:      str                        = "chat"
_processor: Optional[ChunkProcessor]  = None

_journal_session    = None
_brainstorm_session = None
_roleplay_session   = None
_aftercare_session  = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_processor(session_id: str, host: str, chunk_size: int, timeout_sec: float) -> ChunkProcessor:
    global _processor
    if _processor is None:
        _processor = ChunkProcessor(
            session_id=session_id,
            host=host,
            chunk_size=chunk_size,
            timeout_sec=timeout_sec,
        )
    if host and host != "unknown":
        _processor.host = host
    return _processor


async def _run_pipeline(
    user_message: str,
    session_id:   str,
    host:         str,
    chunk_size:   int,
    timeout_sec:  float,
    flush:        bool = False,
) -> Optional[dict]:
    processor = _get_processor(session_id, host, chunk_size, timeout_sec)
    lines = [l for l in user_message.splitlines() if l.strip()]
    chunk_result = None
    for line in lines:
        result = await processor.push_line(line)
        if result:
            chunk_result = result
    if flush:
        final = await processor.flush()
        chunk_result = final or chunk_result
    last = chunk_result or (processor.results[-1] if processor.results else None)
    return last


async def _close_journal(send_fn) -> None:
    """Close active journal session, waiting for write-up + approval to complete."""
    global _journal_session
    if _journal_session and not _journal_session._closed:
        await _journal_session.close()
        # Wait until approval flow completes (closed flag set)
        for _ in range(120):   # max 2 minutes of polling
            if _journal_session._closed:
                break
            await asyncio.sleep(1)
    _journal_session = None


async def _close_brainstorm(session_id: str, host: str, chunk_size: int, timeout_sec: float) -> None:
    """Close active brainstorm session, flushing pipeline."""
    global _brainstorm_session
    if _brainstorm_session and not _brainstorm_session._closed:
        processor = _get_processor(session_id, host, chunk_size, timeout_sec)
        await _brainstorm_session.close(chunk_processor=processor)
    _brainstorm_session = None


async def _close_roleplay(session_id: str, host: str, chunk_size: int, timeout_sec: float) -> None:
    """Close active roleplay session, flushing pipeline."""
    global _roleplay_session
    if _roleplay_session and not _roleplay_session._closed:
        processor = _get_processor(session_id, host, chunk_size, timeout_sec)
        await _roleplay_session.close(flush_pipeline=processor)
    _roleplay_session = None


async def _close_aftercare() -> None:
    global _aftercare_session
    if _aftercare_session and not _aftercare_session._closed:
        await _aftercare_session.close()
    _aftercare_session = None


async def _launch_aftercare(
    name:        str,
    session_id:  str,
    send_fn:     callable,
    scene_beats: list   = None,
    scene_note:  str    = "",
    reason:      str    = "",
) -> None:
    """Launch aftercare from any context — post-scene callback or manual request."""
    global _mode, _aftercare_session
    from core import aftercare
    _mode = "aftercare"
    _aftercare_session = await aftercare.start_session(
        name=name,
        session_id=session_id,
        on_message=send_fn,
        scene_beats=scene_beats or [],
        scene_note=scene_note,
        reason=reason,
    )


# ── Mode detection ────────────────────────────────────────────────────────────

_MODE_PHRASES = {
    "passive mode":     "passive",
    "chat mode":        "chat",
    "journal mode":     "journal",
    "brainstorm mode":  "brainstorm",
    "roleplay mode":    "roleplay",
    "aftercare":        "aftercare",
}

def _detect_mode_switch(msg: str) -> Optional[str]:
    for phrase, mode in _MODE_PHRASES.items():
        if phrase in msg:
            return mode
    return None


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentSimple:

    async def respond(
        self,
        user_message: str,
        history,
        session_id:   str   = "",
        context:      Optional[dict] = None,
        source:       str   = "user",
        chunk_size:   int   = 8,
        timeout_sec:  float = 10.0,
    ) -> AsyncGenerator[str, None]:
        global _mode, _journal_session, _brainstorm_session, _roleplay_session, _aftercare_session

        t_start  = time.monotonic()
        ctx      = context if context is not None else {}
        host     = ctx.get("current_host") or "unknown"
        ctx["session_id"] = session_id

        log_event("AgentSimple", "RECEIVE",
            session=session_id[:8],
            mode=_mode,
            preview=user_message[:60],
        )

        try:
            msg_lower = user_message.lower().strip()

            # ── Safeword handling (roleplay only) ────────────────────────────
            if ctx.get("safeword"):
                level = ctx["safeword"]
                if level == "red":
                    # Full stop — close scene, go straight to aftercare
                    if _roleplay_session and not _roleplay_session._closed:
                        beats = _roleplay_session.scene_log.get("beats", [])
                        await _close_roleplay(session_id, host, chunk_size, timeout_sec)
                        ac_parts = []
                        async def _ac_send(text: str):
                            ac_parts.append(text)
                        await _launch_aftercare(
                            name=host,
                            session_id=session_id,
                            send_fn=_ac_send,
                            scene_beats=beats,
                            scene_note="Red safeword called — full stop.",
                        )
                        for part in ac_parts:
                            yield part
                    return

                if level == "yellow":
                    # Pause — step out of scene, check in, can resume
                    if _roleplay_session and not _roleplay_session._closed:
                        _roleplay_session.state = "paused"
                        pause_parts = []
                        async def _pause_send(text: str):
                            pause_parts.append(text)
                        from core.llm import llm as _llm_client
                        raw = await _llm_client.generate(
                            messages=[{"role": "user", "content": f"{host} called yellow."}],
                            system_prompt=(
                                "You are Gizmo stepping out of a scene because the person called yellow — "
                                "pause, something's off. Step out warmly, check in, don't push. "
                                "One or two sentences. Let them lead."
                            ),
                            temperature=0.7,
                            max_new_tokens=120,
                        )
                        reply = raw.strip() if raw else "Hey — stepping out for a sec. What's up?"
                        yield reply
                    return

            # ── Wellness report keyphrases (any mode) ─────────────────────────
            if "run wellness report" in msg_lower:
                from core.wellness_synthesis import wellness_synthesis
                await wellness_synthesis.run()
                yield json.dumps({"status": "ok", "trigger": "wellness_report"})
                return

            if "run report for" in msg_lower:
                name = msg_lower.split("run report for")[-1].strip().split()[0].strip(".,!?")
                from core.wellness_synthesis import wellness_synthesis
                result = await wellness_synthesis.synthesize_one(name)
                yield json.dumps({"status": "ok", "trigger": "wellness_report", "name": name, "synthesized": bool(result)})
                return

            # ── Mode switch detection ─────────────────────────────────────────
            requested_mode = _detect_mode_switch(msg_lower)

            if requested_mode and requested_mode != _mode:
                # Journal intercepts mode switches — write-up first
                if _mode == "journal" and _journal_session and not _journal_session._closed:
                    # Push the message into the journal session first
                    # (it will detect the exit signal internally and begin write-up)
                    reply_text = None
                    async def _capture(text):
                        nonlocal reply_text
                        reply_text = text
                    await _journal_session.push(user_message)
                    # Hold — don't switch mode until journal is closed
                    # The journal session will call on_message when done
                    # actual mode switch happens after _closed is True
                    await _close_journal(_capture)
                    _mode = requested_mode
                    yield f"Switched to {_mode} mode."
                    return

                # Brainstorm flush on exit
                if _mode == "brainstorm" and _brainstorm_session:
                    await _close_brainstorm(session_id, host, chunk_size, timeout_sec)

                # Roleplay close on exit
                if _mode == "roleplay" and _roleplay_session:
                    await _close_roleplay(session_id, host, chunk_size, timeout_sec)

                # Aftercare close on exit
                if _mode == "aftercare" and _aftercare_session:
                    await _close_aftercare()

                _mode = requested_mode
                print(f"[AgentSimple] mode → {_mode}")

                # Start new session if entering journal, brainstorm, or roleplay
                if _mode == "journal":
                    reply_bucket = []
                    async def _journal_send(text: str):
                        reply_bucket.append(text)

                    from core import journal
                    _journal_session = journal.start_session(
                        name=host,
                        session_id=session_id,
                        on_message=_journal_send,
                    )
                    yield "Journal mode. What's on your mind?"
                    return

                if _mode == "brainstorm":
                    reply_bucket = []
                    async def _brainstorm_send(text: str):
                        reply_bucket.append(text)

                    from core import brainstorm
                    _brainstorm_session = brainstorm.start_session(
                        name=host,
                        session_id=session_id,
                        on_message=_brainstorm_send,
                    )
                    yield "Brainstorm mode. What are we working on?"
                    return

                if _mode == "roleplay":
                    reply_parts = []
                    async def _roleplay_send(text: str):
                        reply_parts.append(text)

                    # Aftercare callback — fired automatically post-scene if needed
                    async def _aftercare_cb(beats: list, note: str):
                        nonlocal reply_parts
                        ac_parts = []
                        async def _ac_send(text: str):
                            ac_parts.append(text)
                        await _launch_aftercare(
                            name=host,
                            session_id=session_id,
                            send_fn=_ac_send,
                            scene_beats=beats,
                            scene_note=note,
                        )
                        for part in ac_parts:
                            await _roleplay_send(part)

                    from core import roleplay
                    _roleplay_session = roleplay.start_session(
                        name=host,
                        session_id=session_id,
                        on_message=_roleplay_send,
                        aftercare_callback=_aftercare_cb,
                    )
                    await _roleplay_session.push("")
                    for part in reply_parts:
                        yield part
                    return

                if _mode == "aftercare":
                    ac_parts = []
                    async def _ac_send(text: str):
                        ac_parts.append(text)
                    # Extract reason from message if provided
                    reason = user_message.split("aftercare")[-1].strip().strip(".,!?") if "aftercare" in msg_lower else ""
                    await _launch_aftercare(
                        name=host,
                        session_id=session_id,
                        send_fn=_ac_send,
                        reason=reason,
                    )
                    for part in ac_parts:
                        yield part
                    return

                yield f"{_mode.capitalize()} mode."
                return

            # ── Active journal session ────────────────────────────────────────
            if _mode == "journal":
                if _journal_session is None or _journal_session._closed:
                    _mode = "chat"
                else:
                    reply_parts = []
                    async def _send(text: str):
                        reply_parts.append(text)
                    _journal_session.on_message = _send
                    await _journal_session.push(user_message)
                    for part in reply_parts:
                        yield part
                    return

            # ── Active brainstorm session ─────────────────────────────────────
            if _mode == "brainstorm":
                if _brainstorm_session is None or _brainstorm_session._closed:
                    _mode = "chat"
                else:
                    reply_parts = []
                    async def _send(text: str):
                        reply_parts.append(text)
                    _brainstorm_session.on_message = _send
                    await _brainstorm_session.push(user_message)
                    for part in reply_parts:
                        yield part
                    return

            # ── Active roleplay session ───────────────────────────────────────
            if _mode == "roleplay":
                if _roleplay_session is None or _roleplay_session._closed:
                    _mode = "chat"
                else:
                    reply_parts = []
                    async def _send(text: str):
                        reply_parts.append(text)
                    _roleplay_session.on_message = _send
                    await _roleplay_session.push(user_message)
                    for part in reply_parts:
                        yield part
                    return

            # ── Active aftercare session ──────────────────────────────────────
            if _mode == "aftercare":
                if _aftercare_session is None or _aftercare_session._closed:
                    _mode = "chat"
                else:
                    reply_parts = []
                    async def _send(text: str):
                        reply_parts.append(text)
                    _aftercare_session.on_message = _send
                    await _aftercare_session.push(user_message)
                    # If session closed itself (they said they're okay), drop back to chat
                    if _aftercare_session._closed:
                        _aftercare_session = None
                        _mode = "chat"
                    for part in reply_parts:
                        yield part
                    return

            # ── Pipeline (passive + chat) ─────────────────────────────────────
            if _mode == "passive":
                # Passive — pipeline must complete before we move on
                last_result = await _run_pipeline(
                    user_message=user_message,
                    session_id=session_id,
                    host=host,
                    chunk_size=chunk_size,
                    timeout_sec=timeout_sec,
                    flush=True,
                )
                yield ""
                return

            if _mode == "chat":
                # Chat — run pipeline in a separate thread with its own event loop
                # so the main loop stays free to respond immediately
                processor = _get_processor(session_id, host, chunk_size, timeout_sec)
                msg_to_process = user_message

                def _run_pipeline_thread():
                    """Run the pipeline in a thread with its own event loop."""
                    import asyncio as _asyncio
                    loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(loop)
                    try:
                        async def _inner():
                            try:
                                lines = [l for l in msg_to_process.splitlines() if l.strip()]
                                for line in lines:
                                    await processor.push_line(line)
                            except Exception as e:
                                log_error("AgentSimple", "threaded pipeline failed", exc=e)
                        loop.run_until_complete(_inner())
                    finally:
                        loop.close()

                import concurrent.futures
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                asyncio.get_event_loop().run_in_executor(executor, _run_pipeline_thread)

                # Respond immediately from existing file context
                last_result = processor.results[-1] if processor.results else {}

                duration_ms = round((time.monotonic() - t_start) * 1000)
                log_event("AgentSimple", "COMPLETE",
                    session=session_id[:8],
                    duration_ms=duration_ms,
                    mode=_mode,
                )

                response_text = await _responder.respond(
                    chunk_result=last_result,
                    context=ctx,
                    history=history or [],
                    user_message=user_message,
                )
                yield response_text or ""
                return

        except Exception as e:
            log_error("AgentSimple", "respond failed", exc=e)
            print(f"[AgentSimple] {type(e).__name__}: {e}", flush=True)
            yield json.dumps({"status": "error", "message": str(e)})


agent_simple = AgentSimple()
