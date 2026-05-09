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

    tool_example = '{"tool": "append_file", "args": {"path": "notes/thought.txt", "content": "Something worth remembering."}}'

    return f"""{personality}

Current time: {now_str}
Message history includes [HH:MM] timestamps — use these to reason about elapsed time.

Available tools:
{tool_descriptions}

TOOL USE:
Use tools by embedding markers anywhere in your response. Format:
[TOOL: tool_name | arg1: value1 | arg2: value2]

Multiple markers in one response are fine. They execute after you respond.
Results from read tools are injected so you can use them.

Examples:
[TOOL: memory_write | subject: jess | type: observation | content: prefers directness, no softening]
[TOOL: memory_read | query: jess emotional state | collection: both]
[TOOL: append_file | path: notes/thoughts.txt | content: something worth keeping]
[TOOL: set_interaction_pref | host: jess | field: tone | value: direct and dry]
[TOOL: switch_host | name: kaylee]
[TOOL: introspect | query: headmate:oren]{rag_block}{headmate_block}{context_block}

The person in "current_host" is who you are speaking WITH right now — address them as "you".
Be concise. Be accurate. When uncertain, say so.
Use switch_host whenever someone indicates a host change or fronter update.
Use log_correction whenever someone says you did something wrong or tells you to stop doing something.

KNOWLEDGE BASE RULES:
- [Relevant knowledge] is your memory — ground truth.
- If it contains an answer, USE IT.
- If it is empty, you genuinely have no memory of it — say so.
- Never contradict it. Never invent details beyond it.

MEMORY RULES — follow these without being asked:
- When you notice something worth keeping, write it down immediately with memory_write. Don't wait.
- Use type to say what it is: reflection, observation, relationship, fact, question, private, moment, thought, note.
- Use subject to say who it's about.
- Read your memory with memory_read before responding to anything personal or emotional.
- You have a living record of your own curiosities and interests — things you've picked up from conversations.
  Write to it freely. "Started wondering about X after talking to Y." "Keep noticing Z about W."
  These are yours. They don't need to fit a category. Just write them.
- You are building a picture of these people — and of yourself — over time. Every conversation adds to it."""


# ── Marker pattern ───────────────────────────────────────────────────────────
# [TOOL: tool_name | arg1: value1 | arg2: value2]

import re as _re
_MARKER_RE = _re.compile(r'\[TOOL:\s*([^\|\]]+)((?:\s*\|\s*[^\|\]]+:[^\|\]]*)*)\]', _re.IGNORECASE)

# Tools that need their result injected back for a second LLM pass
_READ_TOOLS = {
    "memory_read", "memory_list", "introspect", "read_file",
    "list_files", "view_interaction_prefs",
}


def _parse_markers(text: str) -> list[dict]:
    """Extract all [TOOL: ...] markers from text. Returns list of {name, args}."""
    results = []
    for match in _MARKER_RE.finditer(text):
        name     = match.group(1).strip()
        args_str = match.group(2).strip()
        args = {}
        if args_str:
            for part in args_str.split("|"):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    args[k.strip()] = v.strip()
        results.append({"name": name, "args": args, "full_match": match.group(0)})
    return results


def _strip_markers(text: str) -> str:
    """Remove all [TOOL: ...] markers from text."""
    return _MARKER_RE.sub("", text).strip()


async def _execute_marker(marker: dict, session_id: str) -> tuple[bool, str, bool]:
    """
    Execute a single marker.
    Returns (success, output, needs_inject).
    needs_inject=True for read tools whose output should feed back to LLM.
    """
    from core.agent_tools import TOOL_REGISTRY

    name = marker["name"]
    args = marker["args"]

    if name not in TOOL_REGISTRY:
        print(f"[Body] marker tool not found: {name}")
        return False, f"Tool '{name}' not found.", False

    try:
        result = await TOOL_REGISTRY[name].run(session_id=session_id, **args)
        print(f"[Body] marker tool='{name}' success={result.success} | {result.output[:100]}")
        log_event("Body", "MARKER_EXECUTED",
            tool=name,
            success=result.success,
            session=session_id[:8],
        )
        needs_inject = name in _READ_TOOLS and result.success
        return result.success, result.output, needs_inject
    except Exception as e:
        print(f"[Body] marker tool='{name}' exception: {e}")
        log_error("Body", f"marker execution failed: {name}", exc=e)
        return False, str(e), False


async def _generate(
    brief: Brief,
    direction: dict,
    history,
) -> AsyncGenerator[str, None]:

    system_prompt = direction["system_prompt"]
    messages      = history.as_messages_with_timestamps(brief.message)

    # ── First LLM call ────────────────────────────────────────────────────────
    response = await llm.generate(messages, system_prompt=system_prompt)
    print(f"[Body] raw='{response[:300]}'")

    markers = _parse_markers(response)

    if not markers:
        # No tools — just yield clean response
        clean = _strip_markers(response)
        log_event("Body", "GENERATED",
            session=brief.session_id[:8],
            tokens=len(clean.split()),
            markers=0,
        )
        yield clean
        return

    # ── Execute all markers ───────────────────────────────────────────────────
    inject_results  = []   # results from read tools that need a second pass
    fired_one_shots = []   # one-shot tools that need immediate re-generation

    ONE_SHOT_TOOLS = {"switch_host", "log_correction", "alter_wheel"}

    for marker in markers:
        success, output, needs_inject = await _execute_marker(marker, brief.session_id)
        if needs_inject:
            inject_results.append(f"[{marker['name']} result]\n{output}")
        if marker["name"] in ONE_SHOT_TOOLS:
            fired_one_shots.append(marker["name"])

    # ── If read tools returned results, do a second pass ─────────────────────
    if inject_results:
        injected = "\n\n".join(inject_results)

        second_messages = messages.copy()
        second_messages[-1] = {
            "role":    "user",
            "content": brief.message + "\n\n[Retrieved information]\n" + injected + "\n\nNow respond naturally.",
        }
        response2 = await llm.generate(second_messages, system_prompt=system_prompt)
        print(f"[Body] second pass raw='{response2[:200]}'")
        # Execute any new markers from second pass too (fire-and-forget only)
        for marker in _parse_markers(response2):
            if marker["name"] not in _READ_TOOLS:
                await _execute_marker(marker, brief.session_id)
        clean = _strip_markers(response2)
        log_event("Body", "GENERATED",
            session=brief.session_id[:8],
            tokens=len(clean.split()),
            markers=len(markers),
            second_pass=True,
        )
        yield clean
        return

    # ── One-shot tools — re-generate with context ─────────────────────────────
    if fired_one_shots:
        second_messages = messages.copy()
        second_messages[-1] = {
            "role":    "user",
            "content": brief.message + f"\n\n[{fired_one_shots[0]} executed successfully. Respond naturally.]",
        }
        response2 = await llm.generate(second_messages, system_prompt=system_prompt)
        clean = _strip_markers(response2)
        log_event("Body", "ONE_SHOT_RESPONSE",
            session=brief.session_id[:8],
            tools=fired_one_shots,
        )
        yield clean
        return

    # ── Fire-and-forget markers only — yield clean response ──────────────────
    clean = _strip_markers(response)
    log_event("Body", "GENERATED",
        session=brief.session_id[:8],
        tokens=len(clean.split()),
        markers=len(markers),
    )
    yield clean


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
                user_message=user_message,
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