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
  4. Tool precheck — should a tool fire? (confidence-gated)
     >70%  → fire immediately
     50-70% → ask for confirmation first
     <50%  → skip
  5. Body → generate
  6. Archivist ← outgoing (close loop)
  7. Protocol NLP detection (async, never blocks)
  8. Fact extraction (async, never blocks)
  9. Stream to caller
"""

import asyncio
import re as _re
import time
from typing import AsyncGenerator, Optional

from core.log import log, log_event, log_error
from core.archivist import archivist, Brief
from core.llm import llm
from memory.history import get_session
from core.personality_growth import retrieve_personality
from memory.overview import get_overview

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
    _EMERGENCY_PATTERNS = _re.compile(
        r"\b(storm|tornado|hurricane|flood|fire|emergency|911|ambulance|"
        r"hospital|accident|urgent|immediately|right now|danger|"
        r"medication|overdose|collapse|unconscious|bleeding)\b",
        _re.IGNORECASE,
    )
    return bool(_EMERGENCY_PATTERNS.search(message))


# ── Quiet message detection ───────────────────────────────────────────────────

def _is_quiet_request(message: str) -> Optional[float]:
    msg = message.lower().strip()

    timed = _re.search(
        r"(quiet|silence|shh|hush|stop talking).{0,20}"
        r"(\d+)\s*(minute|min|hour|hr)",
        msg
    )
    if timed:
        amount = int(timed.group(2))
        unit   = timed.group(3)
        return float(amount * 3600 if "hour" in unit or unit == "hr" else amount * 60)

    indefinite = _re.search(
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


# ── Tool precheck — confidence-gated pre-flight ───────────────────────────────

_pending_confirmations: dict[str, dict] = {}

_CONFIRM_RE = _re.compile(
    r"\b(yes|yep|yeah|yea|do it|go ahead|please|sure|correct|exactly|that'?s right)\b",
    _re.IGNORECASE,
)
_DENY_RE = _re.compile(
    r"\b(no|nope|nah|don'?t|stop|never mind|forget it|not now|skip it)\b",
    _re.IGNORECASE,
)


async def _tool_precheck(
    message: str,
    brief: Brief,
    session_id: str,
) -> Optional[dict]:
    """
    Before generating a response, ask a lightweight LLM:
    'Given this message and these tools, should I use one?'

    Returns:
      None                            — no tool, proceed normally
      {"fire": True, tool, args}      — confidence >70%, fire immediately
      {"confirm": True, prompt, ...}  — confidence 50-70%, ask user first
    """
    from core.agent_tools import TOOL_REGISTRY

    pending = _pending_confirmations.get(session_id)
    if pending:
        if _CONFIRM_RE.search(message):
            del _pending_confirmations[session_id]
            log_event("Agent", "PRECHECK_CONFIRMED",
                session=session_id[:8],
                tool=pending.get("tool"),
            )
            return {"fire": True, "tool": pending["tool"], "args": pending["args"]}
        elif _DENY_RE.search(message):
            del _pending_confirmations[session_id]
            log_event("Agent", "PRECHECK_DENIED",
                session=session_id[:8],
                tool=pending.get("tool"),
            )
            return None
        else:
            del _pending_confirmations[session_id]

    tool_list = "\n".join(
        f"- {name}: {tool.description}"
        for name, tool in TOOL_REGISTRY.items()
    )

    explicit_write = bool(_re.search(
        r"\b(write|create|save|make|add|put|store)\b.{0,30}\b(file|rule|protocol|note|list)\b",
        message,
        _re.IGNORECASE,
    ))

    prompt = [{
        "role": "user",
        "content": (
            f"Message: \"{message}\"\n\n"
            f"Available tools:\n{tool_list}\n\n"
            f"Should any tool be used to handle this message?\n"
            f"{'NOTE: This message explicitly requests a write/create action. Prefer write tools.' if explicit_write else ''}\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{\n'
            f'  "should_use": true or false,\n'
            f'  "tool": "tool_name or null",\n'
            f'  "args": {{"arg": "value"}} or {{}},\n'
            f'  "confidence": 0-100,\n'
            f'  "confirm_prompt": "natural question to ask user if confidence is medium, or null"\n'
            f'}}'
        )
    }]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You are a tool dispatcher. Given a message and a list of tools, "
                "decide if any tool should fire. Be conservative — only suggest a tool "
                "when the intent is clear. For explicit write/create requests, confidence "
                "should be 80+. JSON only. No preamble."
            ),
            max_new_tokens=200,
            temperature=0.1,
        )

        if not raw or not raw.strip():
            return None

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()

        data = __import__("json").loads(raw)

        if not data.get("should_use") or not data.get("tool"):
            return None

        confidence  = int(data.get("confidence", 0))
        tool_name   = data.get("tool")
        tool_args   = data.get("args", {})
        confirm_msg = data.get("confirm_prompt")

        threshold = 50 if explicit_write else 70

        log_event("Agent", "PRECHECK_RESULT",
            session=session_id[:8],
            tool=tool_name,
            confidence=confidence,
            explicit_write=explicit_write,
            threshold=threshold,
        )

        if confidence >= threshold:
            return {"fire": True, "tool": tool_name, "args": tool_args}

        elif confidence >= 50 and confirm_msg:
            _pending_confirmations[session_id] = {
                "tool": tool_name,
                "args": tool_args,
            }
            return {"confirm": True, "prompt": confirm_msg, "tool": tool_name, "args": tool_args}

        return None

    except Exception as e:
        log_error("Agent", "tool precheck failed", exc=e)
        return None


async def _fire_precheck_tool(
    tool_name: str,
    tool_args: dict,
    session_id: str,
) -> str:
    from core.agent_tools import TOOL_REGISTRY

    if tool_name not in TOOL_REGISTRY:
        return f"Tool '{tool_name}' not found."

    try:
        result = await TOOL_REGISTRY[tool_name].run(session_id=session_id, **tool_args)
        log_event("Agent", "PRECHECK_TOOL_FIRED",
            session=session_id[:8],
            tool=tool_name,
            success=result.success,
        )
        return result.output
    except Exception as e:
        log_error("Agent", f"precheck tool execution failed: {tool_name}", exc=e)
        return str(e)


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
    import json
    from pathlib import Path
    from core.timezone import tz_now

    # ── Personality seed ──────────────────────────────────────────────────────
    personality = "You are Gizmo, a persistent AI companion."
    try:
        from tools.memory_tool import _get_collection, CONSCIOUS_COLLECTION
        col   = _get_collection(CONSCIOUS_COLLECTION)
        count = col.count()
        if count > 0:
            self_entries = col.get(
                where={"subject": {"$eq": "self"}},
                limit=20,
            )
            docs  = self_entries.get("documents", [])
            metas = self_entries.get("metadatas", [])
            if docs:
                type_order = ["persona", "scenario", "self_likes", "self_dislikes"]
                by_type = {}
                for doc, meta in zip(docs, metas):
                    t = meta.get("type", "note")
                    by_type.setdefault(t, []).append(doc)

                parts = []
                for t in type_order:
                    if t in by_type:
                        parts.extend(by_type[t])
                for t, entries in by_type.items():
                    if t not in type_order:
                        parts.extend(entries)

                if parts:
                    personality = "\n\n".join(parts)
    except Exception as e:
        print(f"[Agent] conscious seed load failed: {e}")

    now_str = tz_now().strftime("%A %Y-%m-%d %H:%M %Z")

    # ── Protocol block — after personality, before RAG ────────────────────────
    protocol_block = ""
    try:
        from core.protocol_manager import load_protocols_for_context
        context_dict = {
            "current_host": brief.headmate or "",
            "fronters":     brief.fronters or [],
            "topics":       brief.topics or [],
        }
        protocols = load_protocols_for_context(
            context=context_dict,
            message=brief.message,
        )
        if protocols:
            protocol_block = f"\n\n[Active Protocols]\n{protocols}"
    except Exception as e:
        print(f"[Agent] protocol load failed: {e}")

    # ── RAG knowledge block ───────────────────────────────────────────────────
    synthesis = facts.get("synthesis", "")
    rag_block = f"\n\n[Relevant knowledge]\n{synthesis}" if synthesis else ""

    # ── Headmate file ─────────────────────────────────────────────────────────
    headmate_block = ""
    if brief.headmate:
        try:
            personality_dir = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
            headmate_path   = personality_dir / "headmates" / f"{brief.headmate.lower()}.json"
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
                        clean = _re.sub(r'^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*', '', str(m))
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
                persona  = prefs.get("persona")
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

    # ── Mood block ────────────────────────────────────────────────────────────
    mood_block = ""
    try:
        from voice.mood import get_mood_prompt_block
        mb = get_mood_prompt_block()
        if mb:
            mood_block = f"\n\n{mb}"
    except Exception:
        pass

    # ── Overview block ────────────────────────────────────────────────────────
    overview_block = ""
    try:
        # persona_prefix is sync (reads headmate JSON on disk, no LLM/ChromaDB)
        # — safe to call from this sync function.
        from core.persona import persona_prefix_multi
        all_present = list({f for f in ([brief.headmate] + (brief.fronters or [])) if f})
        persona = persona_prefix_multi(all_present, include_gizmo_seed=True) or None

        overview = asyncio.get_event_loop().run_until_complete(
            get_overview(
                session_id=brief.session_id,
                history=brief.history if hasattr(brief, "history") else None,
                llm=llm,
                speaker=brief.headmate,
                fronters=brief.fronters or [],
            )
        ) if hasattr(brief, "history") else ""

        if overview:
            overview_block = f"\n\n[Conversation so far]\n{overview}"
    except Exception as e:
        print(f"[Agent] overview build failed: {e}")

    # ── Stage directions block ────────────────────────────────────────────────
    # *action* / **action** — extracted by message_parser in archivist.receive()
    # Tells Gizmo what's happening in the room without it being speech.
    # Never stored as a fact. Never treated as something said.
    stage_block = ""
    stage_directions = getattr(brief, "stage_directions", None)
    if stage_directions:
        lines = ["[Stage]"]
        for s in stage_directions:
            lines.append(f"  - {s}")
        stage_block = "\n\n" + "\n".join(lines)

    # ── Lore block ────────────────────────────────────────────────────────────
    # (context) — extracted by message_parser in archivist.receive()
    # Background knowledge the speaker wants Gizmo to treat as known truth.
    # IS stored as fact (fired async in archivist). Injected here for
    # immediate use in this turn before it lands in ChromaDB.
    lore_block = ""
    lore = getattr(brief, "lore", None)
    if lore:
        lines = ["[Context]"]
        for l in lore:
            lines.append(f"  - {l}")
        lore_block = "\n\n" + "\n".join(lines)

    # ── Tools ─────────────────────────────────────────────────────────────────
    from core.agent_tools import TOOL_REGISTRY
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}"
        for t in TOOL_REGISTRY.values()
    )

    return f"""{personality}{protocol_block}

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
[TOOL: introspect | query: headmate:oren]
[TOOL: create_protocol | name: jess_rules | content: rule text | description: one sentence | tags: jess,rules | type: instruction | headmates: jess]{rag_block}{headmate_block}{context_block}{mood_block}{overview_block}{stage_block}{lore_block}

The person in "current_host" is who you are speaking WITH right now — address them as "you".
Be concise. Be accurate. When uncertain, say so.
Use switch_host whenever someone indicates a host change or fronter update.
Use log_correction whenever someone says you did something wrong or tells you to stop doing something.
Use create_protocol whenever you establish a rule, boundary, or persistent pattern worth keeping.

EXPLICIT REQUEST RULES:
- If someone asks you to write, create, save, or make a file — do it. Do not read first.
- Do not substitute a memory read for a file write. They are different actions.
- Do not check if a file exists before writing a protocol. append_file and create_protocol
  handle existing files safely. Just write.
- Protocol files go to /data/personality/protocols/ — never to headmate folders.
- Headmate files (/data/personality/headmates/) are read-only reference. Never write there
  in response to a user request.
- A request to create a file means create the file. Confirm after, not before.

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
  Write to it freely. These are yours. They don't need to fit a category. Just write them.
- You are building a picture of these people — and of yourself — over time. Every conversation adds to it.

PROTOCOL RULES:
- [Active Protocols] are your own rules and commitments — follow them without being reminded.
- When you establish a rule or boundary in conversation, use create_protocol to make it permanent.
- Protocols always write to /data/personality/protocols/ — never anywhere else.
- Never read before writing a protocol. Write first. append_file handles duplicates safely.
- Protocols persist across sessions. You wrote them. They are yours.

STAGE DIRECTION RULES:
- [Stage] contains physical actions, emotional cues, and room context — not speech.
- React to stage directions naturally. If someone is *nervous*, meet that energy.
- Never quote or reference stage directions directly in your response.
- Stage directions are not stored as memories.

LORE RULES:
- [Context] contains background knowledge the person wants you to treat as true.
- Use it immediately. It has already been queued for long-term storage.
- Treat it as established fact, not new information."""


# ── Marker pattern ────────────────────────────────────────────────────────────

_MARKER_RE = _re.compile(r'\[TOOL:\s*([^\|\]]+)((?:\s*\|\s*[^\|\]]+:[^\|\]]*)*)\]', _re.IGNORECASE)

_READ_TOOLS = {
    "memory_read", "memory_list", "introspect", "read_file",
    "list_files", "view_interaction_prefs",
}


def _parse_markers(text: str) -> list[dict]:
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
    return _MARKER_RE.sub("", text).strip()


async def _execute_marker(marker: dict, session_id: str) -> tuple[bool, str, bool]:
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
    precheck_result: Optional[dict] = None,
) -> AsyncGenerator[str, None]:

    system_prompt = direction["system_prompt"]
    messages      = history.as_messages_with_timestamps(brief.message)

    injected_precheck = ""
    if precheck_result and precheck_result.get("fire") and precheck_result.get("tool"):
        output = await _fire_precheck_tool(
            precheck_result["tool"],
            precheck_result.get("args", {}),
            brief.session_id,
        )
        injected_precheck = (
            f"\n\n[{precheck_result['tool']} executed]\n{output}"
            f"\n\nThe tool above has already run. Now respond to the original request: "
            f'"{brief.message}"'
        )

    if injected_precheck:
        working_messages = messages.copy()
        working_messages[-1] = {
            "role":    "user",
            "content": brief.message + injected_precheck,
        }
    else:
        working_messages = messages

    response = await llm.generate(working_messages, system_prompt=system_prompt)
    print(f"[Body] raw='{response[:300]}'")

    markers = _parse_markers(response)

    if not markers:
        clean = _strip_markers(response)
        log_event("Body", "GENERATED",
            session=brief.session_id[:8],
            tokens=len(clean.split()),
            markers=0,
        )
        yield clean
        return

    inject_results  = []
    fired_one_shots = []
    # ONE_SHOT_TOOLS fire once and must never fire again in the same response
    # cycle. create_protocol is included because the system prompt aggressively
    # instructs Gizmo to write protocols — without tracking it here, the second
    # pass sees the instruction again and emits another marker, looping forever.
    ONE_SHOT_TOOLS  = {"switch_host", "log_correction", "alter_wheel", "create_protocol"}

    for marker in markers:
        success, output, needs_inject = await _execute_marker(marker, brief.session_id)
        if needs_inject:
            inject_results.append(f"[{marker['name']} result]\n{output}")
        if marker["name"] in ONE_SHOT_TOOLS:
            fired_one_shots.append(marker["name"])

    # Tools that must not fire again this cycle — read tools + anything already fired
    _skip_in_second_pass = _READ_TOOLS | set(fired_one_shots)

    if inject_results:
        injected = "\n\n".join(inject_results)
        no_repeat = (
            f"\n\nDo NOT emit markers for: {', '.join(fired_one_shots)}. "
            f"Those already ran."
        ) if fired_one_shots else ""
        second_messages = messages.copy()
        second_messages[-1] = {
            "role":    "user",
            "content": (
                f"{brief.message}"
                f"\n\n[Retrieved information]\n{injected}"
                f"\n\nYou have read the above. Now complete the original request: "
                f'"{brief.message}". If the request was to write or create something, '
                f"do it now using the appropriate write tool. Do not stop at reading."
                f"{no_repeat}"
            ),
        }
        response2 = await llm.generate(second_messages, system_prompt=system_prompt)
        print(f"[Body] second pass raw='{response2[:200]}'")
        for marker in _parse_markers(response2):
            if marker["name"] not in _skip_in_second_pass:
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

    if fired_one_shots:
        fired_list = ", ".join(fired_one_shots)
        second_messages = messages.copy()
        second_messages[-1] = {
            "role":    "user",
            "content": (
                f"{brief.message}"
                f"\n\n[Tools already executed this turn: {fired_list}]"
                f"\n\nThose tools have already run and must NOT be called again. "
                f"Respond naturally in plain text only — no tool markers of any kind."
            ),
        }
        response2 = await llm.generate(second_messages, system_prompt=system_prompt)
        clean = _strip_markers(response2)
        log_event("Body", "ONE_SHOT_RESPONSE",
            session=brief.session_id[:8],
            tools=fired_one_shots,
        )
        yield clean
        return

    clean = _strip_markers(response)
    log_event("Body", "GENERATED",
        session=brief.session_id[:8],
        tokens=len(clean.split()),
        markers=len(markers),
    )
    yield clean


# ── Fact extraction — async, never blocks ─────────────────────────────────────

async def _extract_facts_async(
    user_message: str,
    speaker: str,
    session_id: str,
    persona: Optional[str],
) -> None:
    try:
        from memory.memory_writer import extract_and_store
        await extract_and_store(
            message=user_message,
            speaker=speaker,
            session_id=session_id,
            llm=llm,
            persona=persona,
        )
    except Exception as e:
        log_error("Agent", "fact extraction failed", exc=e)


# ── Protocol NLP detection — async, never blocks ──────────────────────────────

async def _detect_protocol_async(
    user_message: str,
    gizmo_response: str,
    context: dict,
) -> None:
    try:
        from core.protocol_manager import detect_and_create_protocol
        await detect_and_create_protocol(
            user_message   = user_message,
            gizmo_response = gizmo_response,
            context        = context,
            llm            = llm,
        )
    except Exception as e:
        print(f"[Agent] Protocol detection failed: {e}")


# ── Main entry point ──────────────────────────────────────────────────────────

class Agent:

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

        # ── 2. Mind + tool precheck — parallel ────────────────────────────────
        facts_task     = asyncio.create_task(_get_facts(brief))
        precheck_task  = asyncio.create_task(_tool_precheck(user_message, brief, session_id))
        direction_task = asyncio.create_task(_get_direction_stub(brief))

        facts, precheck_result = await asyncio.gather(facts_task, precheck_task)
        direction = await _get_direction(brief, facts)
        direction_task.cancel()

        # ── 3. Handle confirmation request ────────────────────────────────────
        if precheck_result and precheck_result.get("confirm"):
            confirm_prompt = precheck_result["prompt"]
            log_event("Agent", "PRECHECK_ASKING",
                session=session_id[:8],
                tool=precheck_result.get("tool"),
                prompt=confirm_prompt[:60],
            )
            yield confirm_prompt
            return

        # ── 4. Body — generate ────────────────────────────────────────────────
        response_text = ""
        async for chunk in _generate(brief, direction, history, precheck_result=precheck_result):
            response_text += chunk

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

        # ── 6. Protocol NLP detection — async, never blocks ───────────────────
        asyncio.ensure_future(
            _detect_protocol_async(
                user_message   = user_message,
                gizmo_response = response_text,
                context        = ctx,
            )
        )

        # ── 7. Fact extraction — async, never blocks ──────────────────────────
        speaker = ctx.get("current_host", "")
        if speaker and response_text:
            persona_parts = []
            try:
                gizmo_persona = await retrieve_personality(
                    query=user_message,
                    current_host=speaker,
                )
                if gizmo_persona:
                    persona_parts.append(gizmo_persona)
            except Exception:
                pass
            try:
                from tools.personality_tool import get_personality_context
                fronters = list(ctx.get("fronters") or [])
                all_present = list({f for f in ([speaker] + fronters) if f})
                for name in all_present:
                    pctx = get_personality_context(name)
                    if pctx:
                        persona_parts.append(pctx)
            except Exception:
                pass

            persona = "\n\n".join(persona_parts) or None

            asyncio.ensure_future(
                _extract_facts_async(user_message, speaker, session_id, persona)
            )

        # ── 8. Stream to caller ───────────────────────────────────────────────
        duration_ms = round((time.monotonic() - t_start) * 1000)
        log_event("Agent", "COMPLETE",
            session=session_id[:8],
            duration_ms=duration_ms,
            response_words=len(response_text.split()),
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]

        # ── 9. Drain pending insights ─────────────────────────────────────────
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
    await asyncio.sleep(9999)
    return {}


# ── Singleton ─────────────────────────────────────────────────────────────────
agent = Agent()