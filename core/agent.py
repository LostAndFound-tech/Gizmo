"""
core/agent.py
The membrane. Coordinates all components. Presents output.

This file does not think. It does not retrieve. It does not generate.
It routes, sequences, and streams. That's all.

Flow:
  1. Receive input
  2. Archivist → brief (includes host change detection)
  3. Mind → facts        (parallel with direction stub)
     Ego  → direction   (sequential after facts — needs facts)
  4. Body → generate
  5. Archivist ← outgoing (close loop)
  6. Stream to caller

Interface (matches what server.py expects):
    async for chunk in agent.run(
        user_message=message,
        history=history,
        session_id=session_id,
        context=context,
    ):
        yield chunk
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

from core.log import log, log_event, log_error
from core.archivist import archivist, Brief
from core.llm import llm
from memory.history import get_session

# ── Tool registry ─────────────────────────────────────────────────────────────
# Imported here so Body can access it. Ego reads it for the system prompt.
from tools.base_tool import BaseTool
from tools.example_tool import EchoTool
from tools.switch_host import SwitchHostTool
from tools.correction_tool import CorrectionTool

# agent_tools module exposes TOOL_REGISTRY to Ego's system prompt builder
import core.agent_tools  # noqa — ensures TOOL_REGISTRY is populated before Ego runs

# ── Quiet mode ────────────────────────────────────────────────────────────────

_quiet: dict[str, Optional[float]] = {}


def set_quiet(session_id: str, duration_seconds: Optional[float] = None) -> None:
    expiry = time.time() + duration_seconds if duration_seconds else None
    _quiet[session_id] = expiry
    log_event("Agent", "QUIET_SET",
        session=session_id[:8],
        duration=duration_seconds or "indefinite",
    )


def clear_quiet(session_id: str) -> None:
    if session_id in _quiet:
        del _quiet[session_id]
        log_event("Agent", "QUIET_CLEARED", session=session_id[:8])


def is_quiet(session_id: str) -> bool:
    if session_id not in _quiet:
        return False
    expiry = _quiet[session_id]
    if expiry is not None and time.time() > expiry:
        clear_quiet(session_id)
        log_event("Agent", "QUIET_EXPIRED", session=session_id[:8])
        return False
    return True


# ── Pending insights queue ────────────────────────────────────────────────────

_pending: dict[str, asyncio.Queue] = {}


def _get_pending(session_id: str) -> asyncio.Queue:
    if session_id not in _pending:
        _pending[session_id] = asyncio.Queue()
    return _pending[session_id]


async def enqueue(
    session_id: str,
    message: str,
    source: str = "unknown",
    priority: int = 5,
    emergency: bool = False,
) -> None:
    queue = _get_pending(session_id)
    await queue.put({
        "message":   message,
        "source":    source,
        "priority":  priority,
        "emergency": emergency,
        "queued_at": time.time(),
    })
    log_event("Agent", "ENQUEUED",
        session=session_id[:8],
        source=source,
        emergency=emergency,
        preview=message[:40],
    )


# ── Push function ─────────────────────────────────────────────────────────────

_push_fn = None


def set_push_fn(fn) -> None:
    global _push_fn
    _push_fn = fn
    log("Agent", "push function registered")


async def push(message: str) -> None:
    if _push_fn is None:
        log("Agent", f"push skipped — no push function: {message[:40]}")
        return
    try:
        await _push_fn(message)
    except Exception as e:
        log_error("Agent", "push failed", exc=e)


# ── Emergency check ───────────────────────────────────────────────────────────

def _is_emergency(message: str) -> bool:
    import re
    _EMERGENCY_RE = re.compile(
        r"\b(storm|tornado|hurricane|flood|fire|emergency|911|ambulance|"
        r"hospital|accident|urgent|immediately|right now|danger|"
        r"medication|overdose|collapse|unconscious|bleeding)\b",
        re.IGNORECASE,
    )
    return bool(_EMERGENCY_RE.search(message))


# ── Quiet request detection ───────────────────────────────────────────────────

def _is_quiet_request(message: str) -> Optional[float]:
    import re
    msg = message.lower().strip()

    timed = re.search(
        r"(quiet|silence|shh|hush|stop talking).{0,20}(\d+)\s*(minute|min|hour|hr)",
        msg
    )
    if timed:
        amount = int(timed.group(2))
        unit = timed.group(3)
        seconds = amount * 3600 if "hour" in unit or unit == "hr" else amount * 60
        return float(seconds)

    indefinite = re.search(
        r"\b(quiet|silence|shh+|hush|stop|leave me alone|not now)\b", msg
    )
    if indefinite:
        return 0.0

    return None


# ── Core routing ──────────────────────────────────────────────────────────────

async def _get_facts(brief: Brief) -> dict:
    """Mind: retrieve relevant facts for this brief."""
    try:
        from core.mind import mind
        facts = await mind.query(brief)
        log_event("Agent", "FACTS_RETRIEVED",
            session=brief.session_id[:8],
            source=facts.get("source", "none"),
            confidence=facts.get("confidence", 0.0),
            topics=brief.topics,
        )
        return facts
    except Exception as e:
        log_error("Agent", "fact retrieval failed", exc=e)
        return {"synthesis": "", "confidence": 0.0, "source": "none", "chunks": []}


async def _get_direction(brief: Brief, facts: dict) -> dict:
    """
    Ego: assemble direction for Body.
    Calls ego.direct() — the real Ego component.
    Falls back to a minimal system prompt on error.
    """
    try:
        from core.ego import ego
        direction = await ego.direct(brief, facts)
        log_event("Agent", "DIRECTION_BUILT",
            session=brief.session_id[:8],
            headmate=brief.headmate or "unknown",
            register=brief.emotional_register,
            prompt_len=len(direction.system_prompt),
            tone=direction.tone[:40],
            corrections=len(direction.corrections),
            entity_question=bool(direction.entity_question),
        )
        return {"system_prompt": direction.system_prompt}
    except Exception as e:
        log_error("Agent", "Ego direction failed — falling back", exc=e)
        return {"system_prompt": "You are Gizmo, a persistent AI companion. Be helpful and warm."}


async def _generate(
    brief: Brief,
    direction: dict,
    history,
) -> AsyncGenerator[str, None]:
    """
    Body: generate response tokens.
    Tool loop — up to MAX_TOOL_CALLS tool executions before final response.
    """
    import json
    import re
    from core.agent_tools import TOOL_REGISTRY

    system_prompt = direction["system_prompt"]
    messages = history.as_messages_with_timestamps(brief.message)

    MAX_TOOL_CALLS = 3
    tool_calls = 0
    injected_results = ""

    while tool_calls < MAX_TOOL_CALLS:
        working = messages.copy()
        if injected_results:
            working[-1] = {
                "role": "user",
                "content": brief.message + injected_results,
            }

        response = await llm.generate(working, system_prompt=system_prompt)
        tool_call = _parse_tool_call(response)

        if tool_call is None:
            response = _strip_tool_calls(response)
            log_event("Body", "GENERATED",
                session=brief.session_id[:8],
                tokens=len(response.split()),
                tool_calls_made=tool_calls,
            )
            yield response
            return

        tool_name = tool_call.get("tool")
        tool_args  = tool_call.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            injected_results += f"\n[Tool Error: '{tool_name}' not found]\n"
            tool_calls += 1
            log_event("Body", "TOOL_NOT_FOUND", session=brief.session_id[:8], tool=tool_name)
            continue

        clean_args = {k: v for k, v in tool_args.items() if k != "session_id"}
        try:
            result = await TOOL_REGISTRY[tool_name].run(
                session_id=brief.session_id, **clean_args
            )
            injected_results += (
                f"\n[Tool: {tool_name}]\nResult: {result.output}\n"
                f"Task complete. Now respond to the user directly.\n"
            )
            log_event("Body", "TOOL_EXECUTED",
                session=brief.session_id[:8],
                tool=tool_name,
                success=True,
            )
        except Exception as e:
            injected_results += f"\n[Tool Error: {e}]\n"
            log_error("Body", f"tool execution failed: {tool_name}", exc=e)

        tool_calls += 1

        # One-shot tools respond immediately after execution
        if tool_name in ("switch_host", "log_correction", "alter_wheel"):
            working[-1] = {
                "role": "user",
                "content": brief.message + injected_results,
            }
            final = await llm.generate(working, system_prompt=system_prompt)
            final = _strip_tool_calls(final)
            log_event("Body", "ONE_SHOT_TOOL_RESPONSE",
                session=brief.session_id[:8],
                tool=tool_name,
            )
            yield final
            return

    # Exhausted tool calls — stream whatever we can
    log_event("Body", "MAX_TOOL_CALLS_REACHED", session=brief.session_id[:8])
    async for token in llm.stream(messages, system_prompt=system_prompt):
        yield token


def _parse_tool_call(text: str) -> Optional[dict]:
    import json, re
    text = text.strip()
    try:
        data = json.loads(text)
        if "tool" in data:
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*?"tool".*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _strip_tool_calls(text: str) -> str:
    import re
    text = re.sub(r'\{\s*"tool"\s*:.*?\}\s*', '', text, flags=re.DOTALL)
    return text.strip()


# ── Main entry point ──────────────────────────────────────────────────────────

class Agent:
    """The membrane. Routes between components. Streams output."""

    async def run(
        self,
        user_message: str,
        history,
        session_id: str = "",
        context: Optional[dict] = None,
        source: str = "user",
    ) -> AsyncGenerator[str, None]:
        """
        Main entry point. Called by server.py for every incoming message.
        Yields response chunks for streaming.
        """
        t_start = time.monotonic()
        ctx = context or {}

        log_event("Agent", "RECEIVE",
            session=session_id[:8],
            source=source,
            headmate=ctx.get("current_host") or "unknown",
            preview=user_message[:60],
        )

        # ── Quiet request ─────────────────────────────────────────────────────
        quiet_duration = _is_quiet_request(user_message)
        if quiet_duration is not None:
            if quiet_duration == 0.0:
                set_quiet(session_id)
                yield "got it"
            else:
                set_quiet(session_id, quiet_duration)
                mins = int(quiet_duration // 60)
                yield f"got it — quiet for {mins} minute{'s' if mins != 1 else ''}"
            return

        # ── 1. Archivist ──────────────────────────────────────────────────────
        brief = archivist.receive(
            message=user_message,
            session_id=session_id,
            history=history,
            context=ctx,
            source=source,
        )

        # ── 2. Mind + Ego ─────────────────────────────────────────────────────
        # Mind and Ego run sequentially — Ego needs facts.
        # Mind is the expensive async call; kick it off first.
        facts     = await _get_facts(brief)
        direction = await _get_direction(brief, facts)

        # ── 3. Body ───────────────────────────────────────────────────────────
        response_text = ""
        async for chunk in _generate(brief, direction, history):
            response_text += chunk

        # ── 4. Archivist — close loop ─────────────────────────────────────────
        if response_text:
            archivist.receive_outgoing(
                message=response_text,
                session_id=session_id,
                history=history,
                context=ctx,
                source="body",
            )

        # ── 5. Stream to caller ───────────────────────────────────────────────
        duration_ms = round((time.monotonic() - t_start) * 1000)
        log_event("Agent", "COMPLETE",
            session=session_id[:8],
            duration_ms=duration_ms,
            response_words=len(response_text.split()),
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]

        # ── 6. Drain pending insights ─────────────────────────────────────────
        await self._drain_pending(session_id)

    async def _drain_pending(self, session_id: str) -> None:
        queue = _get_pending(session_id)
        while not queue.empty():
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            emergency = item.get("emergency", False)
            message   = item.get("message", "")
            source    = item.get("source", "unknown")

            if is_quiet(session_id) and not emergency:
                log_event("Agent", "PENDING_HELD",
                    session=session_id[:8],
                    source=source,
                    reason="quiet_mode",
                )
                await queue.put(item)
                break

            log_event("Agent", "PENDING_SURFACING",
                session=session_id[:8],
                source=source,
                emergency=emergency,
            )
            await push(message)


# ── Singleton ─────────────────────────────────────────────────────────────────
agent = Agent()