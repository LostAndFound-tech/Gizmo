"""
core/agent.py
The membrane. Coordinates all components. Presents output.

This file does not think. It does not retrieve. It does not generate.
It routes, sequences, and streams. That's all.

Flow:
  1. Receive input
  2. Archivist → brief
  3. Mind → facts        (parallel)
     Ego  → direction   (parallel)
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

Quiet mode:
    agent.set_quiet(session_id, duration_seconds=None)
    agent.clear_quiet(session_id)

Unsolicited messages (from any component):
    await agent.push(session_id, message, source="ego")
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

from core.log import log, log_event, log_error
from core.archivist import archivist, Brief
from core.llm import llm
from memory.history import get_session

# ── Quiet mode ────────────────────────────────────────────────────────────────

_quiet: dict[str, Optional[float]] = {}


def set_quiet(session_id: str, duration_seconds: Optional[float] = None) -> None:
    expiry = time.time() + duration_seconds if duration_seconds else None
    _quiet[session_id] = expiry
    log_event("Agent", "QUIET_SET",
        session=session_id[:8],
        duration=duration_seconds or "indefinite",
        expires=expiry or "never",
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
        "message": message,
        "source": source,
        "priority": priority,
        "emergency": emergency,
        "queued_at": time.time(),
    })
    log_event("Agent", "ENQUEUED",
        session=session_id[:8],
        source=source,
        emergency=emergency,
        priority=priority,
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
        log("Agent", f"push skipped — no push function registered: {message[:40]}")
        return
    try:
        await _push_fn(message)
    except Exception as e:
        log_error("Agent", "push failed", exc=e)


# ── Emergency check ───────────────────────────────────────────────────────────

def _is_emergency(message: str) -> bool:
    import re
    _EMERGENCY_PATTERNS = re.compile(
        r"\b(storm|tornado|hurricane|flood|fire|emergency|911|ambulance|"
        r"hospital|accident|urgent|immediately|right now|danger|"
        r"medication|overdose|collapse|unconscious|bleeding)\b",
        re.IGNORECASE,
    )
    return bool(_EMERGENCY_PATTERNS.search(message))


# ── Quiet message detection ───────────────────────────────────────────────────

def _is_quiet_request(message: str) -> Optional[float]:
    import re
    msg = message.lower().strip()

    timed = re.search(
        r"(quiet|silence|shh|hush|stop talking).{0,20}"
        r"(\d+)\s*(minute|min|hour|hr)",
        msg
    )
    if timed:
        amount = int(timed.group(2))
        unit = timed.group(3)
        seconds = amount * 3600 if "hour" in unit or unit == "hr" else amount * 60
        return float(seconds)

    indefinite = re.search(
        r"\b(quiet|silence|shh+|hush|stop|leave me alone|not now)\b",
        msg
    )
    if indefinite:
        return 0.0

    return None


# ── Hold messages ─────────────────────────────────────────────────────────────

_HOLD_MESSAGES = [
    "give me a sec",
    "hang on, thinking",
    "still here, one moment",
    "okay that's interesting — just a moment",
]

_hold_index: dict[str, int] = {}


def _next_hold_message(session_id: str) -> str:
    idx = _hold_index.get(session_id, 0)
    msg = _HOLD_MESSAGES[idx % len(_HOLD_MESSAGES)]
    _hold_index[session_id] = idx + 1
    return msg


# ── Core routing ──────────────────────────────────────────────────────────────

async def _get_facts(brief: Brief) -> dict:
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
    try:
        direction = _build_system_prompt(brief, facts)
        log_event("Agent", "DIRECTION_BUILT",
            session=brief.session_id[:8],
            headmate=brief.headmate or "unknown",
            register=brief.emotional_register,
            topics=brief.topics,
            prompt_len=len(direction),
        )
        return {"system_prompt": direction}
    except Exception as e:
        log_error("Agent", "direction assembly failed", exc=e)
        return {"system_prompt": "You are Gizmo, a helpful companion."}


def _build_system_prompt(brief: Brief, facts: dict) -> str:
    import os
    import re
    import json
    from pathlib import Path
    from core.timezone import tz_now

    # ── Personality seed ──────────────────────────────────────────────────────
    personality = "You are Gizmo, a persistent AI companion."
    personality_path = os.path.join(
        os.path.dirname(__file__), "..", "personality.txt"
    )
    try:
        with open(personality_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                personality = content
    except FileNotFoundError:
        pass

    now_str = tz_now().strftime("%A %Y-%m-%d %H:%M %Z")

    # ── RAG knowledge block ───────────────────────────────────────────────────
    synthesis = facts.get("synthesis", "")
    rag_block = f"\n\n[Relevant knowledge]\n{synthesis}" if synthesis else ""

    # ── Headmate file — injected every message ────────────────────────────────
    headmate_block = ""
    if brief.headmate:
        try:
            personality_dir = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
            headmate_path = personality_dir / "headmates" / f"{brief.headmate.lower()}.json"
            if headmate_path.exists():
                data = json.loads(headmate_path.read_text(encoding="utf-8"))
                name = data.get("name", brief.headmate).title()
                lines = [f"[What I know about {name}]"]

                baseline = data.get("baseline", {})
                for k, v in baseline.items():
                    if k == "observations":
                        continue
                    if v not in ("unknown", 0, 0.0, "", None):
                        lines.append(f"  {k}: {v}")

                moments = data.get("moments_of_note", [])
                if moments:
                    lines.append("  Known facts:")
                    for m in moments[-8:]:
                        clean = re.sub(r'^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*', '', str(m))
                        lines.append(f"    - {clean}")

                patterns = data.get("observed_patterns", [])
                if patterns:
                    lines.append("  Observed patterns:")
                    for p in patterns[-3:]:
                        if isinstance(p, dict):
                            lines.append(f"    - {p.get('pattern', str(p))}")
                        else:
                            lines.append(f"    - {p}")

                corrections = data.get("corrections", [])
                if corrections:
                    lines.append("  Corrections from this person:")
                    for c in corrections[-3:]:
                        if isinstance(c, dict):
                            lines.append(f"    - {c.get('rule', str(c))}")
                        else:
                            lines.append(f"    - {c}")

                prefs = data.get("interaction_prefs", {})
                pref_lines = []
                for field in ("tone", "pacing", "checkins", "humor", "distress"):
                    v = prefs.get(field)
                    if v:
                        pref_lines.append(f"    {field}: {v}")
                persona = prefs.get("persona")
                explicit = [e for e in prefs.get("explicit", []) if e]
                if persona or pref_lines or explicit:
                    lines.append("  How they want to be engaged:")
                    if persona:
                        lines.append(f"    {persona}")
                    lines.extend(pref_lines)
                    for e in explicit:
                        lines.append(f"    - {e}")

                if len(lines) > 1:
                    headmate_block = "\n\n" + "\n".join(lines)
        except Exception as e:
            print(f"[Agent] headmate inject failed: {e}")

    # ── Situational context ───────────────────────────────────────────────────
    context_lines = []
    if brief.headmate:
        context_lines.append(f"  current_host: {brief.headmate}")
    if brief.fronters:
        context_lines.append(f"  fronters: {', '.join(brief.fronters)}")
    if brief.emotional_register != "neutral":
        context_lines.append(f"  emotional_register: {brief.emotional_register}")
    if brief.field_snapshot.get("hot"):
        context_lines.append(f"  active_topics: {', '.join(brief.field_snapshot['hot'])}")
    context_block = (
        "\n\n[Current situation]\n" + "\n".join(context_lines)
        if context_lines else ""
    )

    # ── Tools ─────────────────────────────────────────────────────────────────
    from core.agent_tools import TOOL_REGISTRY
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}"
        for t in TOOL_REGISTRY.values()
    )

    return f"""{personality}

Current time: {now_str}
Message history includes [HH:MM] timestamps — use these to reason about elapsed time.

Available tools:
{tool_descriptions}

TOOL USE RULES:
- To call a tool, output ONLY a raw JSON object on a single line. Nothing before it, nothing after it.
- The JSON must have exactly two keys: "tool" (string) and "args" (object).
- Valid example: {{"tool": "append_file", "args": {{"path": "notes/thought.txt", "content": "Something worth remembering."}}}}
- After the tool result is returned, respond naturally to the user.
- If no tool is needed, respond in plain text — do not output JSON.
- NEVER output a partial JSON object. NEVER output just a brace or fragment.{rag_block}{headmate_block}{context_block}

The person in "current_host" is who you are speaking WITH right now — address them as "you".
Be concise. Be accurate. When uncertain, say so.
Use switch_host whenever someone indicates a host change or fronter update.
Use log_correction whenever someone says you did something wrong or tells you to stop doing something.

KNOWLEDGE BASE RULES:
- [Relevant knowledge] is your memory — ground truth.
- If it contains an answer, USE IT.
- If it is empty, you genuinely have no memory of it — say so.
- Never contradict it. Never invent details beyond it."""


async def _generate(
    brief: Brief,
    direction: dict,
    history,
) -> AsyncGenerator[str, None]:
    import json
    import re

    system_prompt = direction["system_prompt"]
    messages = history.as_messages_with_timestamps(brief.message)

    from core.agent_tools import TOOL_REGISTRY

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

        # Log raw output so we can see exactly what DeepSeek generates
        print(f"[Body] raw='{response[:300]}'")

        # Check for tool call
        tool_call = _parse_tool_call(response)

        if tool_call is None:
            # No tool call — final response
            response = _strip_tool_calls(response)
            log_event("Body", "GENERATED",
                session=brief.session_id[:8],
                tokens=len(response.split()),
                tool_calls_made=tool_calls,
            )
            yield response
            return

        # Execute tool
        tool_name = tool_call.get("tool")
        tool_args  = tool_call.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            injected_results += f"\n[Tool Error: '{tool_name}' not found]\n"
            tool_calls += 1
            log_event("Body", "TOOL_NOT_FOUND",
                session=brief.session_id[:8],
                tool=tool_name,
            )
            continue

        clean_args = {k: v for k, v in tool_args.items() if k != "session_id"}
        try:
            result = await TOOL_REGISTRY[tool_name].run(
                session_id=brief.session_id, **clean_args
            )

            print(f"[Body] tool='{tool_name}' success={result.success} | {result.output[:120]}")

            log_event("Body", "TOOL_EXECUTED",
                session=brief.session_id[:8],
                tool=tool_name,
                success=result.success,
                output=result.output[:120],
            )

            if result.success:
                injected_results += (
                    f"\n[Tool: {tool_name}]\nStatus: SUCCESS\nResult: {result.output}\n"
                    f"Task complete. Now respond to the user directly.\n"
                )
            else:
                injected_results += (
                    f"\n[Tool: {tool_name}]\nStatus: FAILED\nReason: {result.output}\n"
                    f"The tool failed. Do NOT tell the user it succeeded. "
                    f"Report the failure honestly.\n"
                )

        except Exception as e:
            print(f"[Body] tool='{tool_name}' exception: {e}")
            injected_results += f"\n[Tool Error in '{tool_name}']: {e}\n"
            log_error("Body", f"tool execution failed: {tool_name}", exc=e)

        tool_calls += 1

        # One-shot tools respond immediately
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

    # Exhausted tool calls — generate anyway
    log_event("Body", "MAX_TOOL_CALLS_REACHED", session=brief.session_id[:8])
    async for token in llm.stream(messages, system_prompt=system_prompt):
        yield token


def _parse_tool_call(text: str) -> Optional[dict]:
    import json
    import re
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
    """
    The membrane. Routes between components. Streams output.
    Maintains quiet mode and pending insights queue per session.
    """

    async def run(
        self,
        user_message: str,
        history,
        session_id: str = "",
        context: Optional[dict] = None,
        source: str = "user",
    ) -> AsyncGenerator[str, None]:
        t_start = time.monotonic()
        ctx = context or {}

        log_event("Agent", "RECEIVE",
            session=session_id[:8],
            source=source,
            headmate=ctx.get("current_host") or "unknown",
            preview=user_message[:60],
        )

        # ── Quiet request detection ───────────────────────────────────────────
        quiet_duration = _is_quiet_request(user_message)
        if quiet_duration is not None:
            if quiet_duration == 0.0:
                set_quiet(session_id)
                yield "got it"
            else:
                set_quiet(session_id, quiet_duration)
                mins = int(quiet_duration // 60)
                yield f"got it — quiet for {mins} minute{'s' if mins != 1 else ''}"
            log_event("Agent", "QUIET_REQUESTED",
                session=session_id[:8],
                duration=quiet_duration,
            )
            return

        # ── 1. Archivist — receive and classify ───────────────────────────────
        brief = archivist.receive(
            message=user_message,
            session_id=session_id,
            history=history,
            context=ctx,
            source=source,
        )

        # ── 2. Mind + Ego — parallel ──────────────────────────────────────────
        facts_task     = asyncio.create_task(_get_facts(brief))
        direction_task = asyncio.create_task(_get_direction_stub(brief))

        facts     = await facts_task
        direction = await _get_direction(brief, facts)
        direction_task.cancel()

        # ── 3. Body — generate ────────────────────────────────────────────────
        response_text = ""
        async for chunk in _generate(brief, direction, history):
            response_text += chunk

        # ── 4. Ego — watch output (v1: no-op) ─────────────────────────────────
        # TODO: Ego checks output before it leaves

        # ── 5. Archivist — close loop ─────────────────────────────────────────
        if response_text:
            archivist.receive_outgoing(
                message=response_text,
                session_id=session_id,
                history=history,
                context=ctx,
                source="body",
            )

        # ── 6. Stream to caller ───────────────────────────────────────────────
        duration_ms = round((time.monotonic() - t_start) * 1000)
        log_event("Agent", "COMPLETE",
            session=session_id[:8],
            duration_ms=duration_ms,
            response_words=len(response_text.split()),
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]

        # ── 7. Drain pending insights ─────────────────────────────────────────
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
                    preview=message[:40],
                )
                await queue.put(item)
                break

            log_event("Agent", "PENDING_SURFACING",
                session=session_id[:8],
                source=source,
                emergency=emergency,
                preview=message[:40],
            )
            await push(message)


async def _get_direction_stub(brief: Brief) -> dict:
    """Placeholder — cancelled once real direction is ready."""
    await asyncio.sleep(9999)
    return {}


# ── Singleton ─────────────────────────────────────────────────────────────────
agent = Agent()