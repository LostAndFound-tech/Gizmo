"""
core/agent.py
Gizmo's pipeline orchestrator.

New architecture — pull-based, fast, memory-driven.

Pipeline per message:

  1. INTAKE
     Classify message → write envelope to store → build Brief.
     No LLM calls. Pure heuristics. Fast.

  2. RETRIEVE
     Embed message → vector + keyword search → crawl refs →
     load entity/place docs → assemble MemoryContext.
     No LLM calls. Local model. ~50ms.

  3. RESPOND
     One LLM call. Gizmo has everything he needs.
     Memory context + conversation history + lean system prompt.
     Writes the message.

  4. CLOSE LOOP (fire and forget)
     Encoding pass — Gizmo writes what he learned, in his voice.
     Beat check — does this exchange answer or raise a pool question?
     Psych coherence — did this make sense for this person?
     Store bookkeeping — response envelope, emotion log.
     Never blocks. Never raises.

Wall-clock time = intake + retrieval + one LLM call
               ≈ fast.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from core.log import log, log_event, log_error
from core.timezone import tz_now


# ── Brief dataclass ───────────────────────────────────────────────────────────

@dataclass
class Brief:
    """
    Full picture assembled by Intake.
    Passed to retrieval and response stages.
    """
    # Message
    message:        str
    session_id:     str
    message_id:     str
    timestamp:      float

    # Identity
    headmate:       Optional[str]
    fronters:       list

    # Classification
    register:       str
    topics:         list
    is_question:    bool
    is_correction:  bool
    word_count:     int
    has_intimate:   bool = False

    # Time context
    time_of_day:    str  = "day"
    day_of_week:    str  = "Monday"
    day_type:       str  = "weekday"
    since_last_msg: float = 0.0
    message_cadence: str = "conversational"

    # Session context
    session_momentum: str   = "building"
    session_duration: float = 0.0

    # Host identification short-circuit
    host_question: Optional[str] = None

    # World observation (attached by pipeline, read by build_system_prompt)
    _world_observation: str = ""

    # Parsed message components (from parse_message LLM call in intake)
    parsed: dict = field(default_factory=dict)


# ── LLM call helper ───────────────────────────────────────────────────────────

async def _call(
    llm,
    system: str,
    user:   str,
    tokens: int   = 400,
    temp:   float = 0.2,
) -> str:
    """Single focused LLM call. Returns empty string on failure."""
    try:
        return await llm.generate(
            [{"role": "user", "content": user}],
            system_prompt=system,
            max_new_tokens=tokens,
            temperature=temp,
        )
    except Exception as e:
        log_error("Agent", f"LLM call failed: {e}", exc=None)
        return ""


# ── Stage 1: Intake ───────────────────────────────────────────────────────────

async def intake(
    message:    str,
    session_id: str,
    context:    dict,
    history,
    llm,
) -> Brief:
    """
    Classify message, write envelope to store, build Brief.
    No LLM calls — pure heuristics.
    """
    from core.store import store
    from core.session_manager import session_manager

    ts  = time.time()
    now = tz_now()

    # ── Identity ──────────────────────────────────────────────────────────────
    ctx_live = session_manager.get_session_context(session_id)
    headmate = ctx_live.get("current_host") or context.get("current_host") or ""
    fronters = ctx_live.get("fronters") or list(context.get("fronters") or [])
    if headmate and headmate not in [f.lower() for f in fronters]:
        fronters.insert(0, headmate)

    # ── Time context ──────────────────────────────────────────────────────────
    hour    = now.hour
    weekday = now.weekday()

    time_of_day = (
        "morning"   if 5  <= hour < 12 else
        "afternoon" if 12 <= hour < 17 else
        "evening"   if 17 <= hour < 21 else
        "night"
    )
    day_type = "weekend" if weekday >= 5 else "weekday"

    # ── Time since last message ───────────────────────────────────────────────
    since_last   = 0.0
    prev_resp_id = None
    try:
        recent = store.get_recent_messages(headmate=headmate or None, limit=2)
        if recent:
            since_last = ts - recent[0].get("created_at", ts)
            last_resp  = store.get_last_response(session_id)
            if last_resp:
                prev_resp_id = last_resp["id"]
    except Exception:
        pass

    cadence = (
        "rapid"          if since_last < 120   else
        "fast"           if since_last < 300   else
        "conversational" if since_last < 1200  else
        "slow"           if since_last < 3200  else
        "returning"
    )

    # ── Session context ───────────────────────────────────────────────────────
    session_dur   = 0.0
    sess_momentum = "opening"

    try:
        sess = session_manager._sessions.get(session_id)
        if sess:
            session_dur = ts - sess.opened_at
    except Exception:
        pass

    try:
        msg_count = store.count("messages", session_id=session_id)
        sess_momentum = (
            "opening"  if msg_count <= 2  else
            "building" if msg_count <= 8  else
            "engaged"  if msg_count <= 20 else
            "deep"
        )
    except Exception:
        pass

    # ── Parse message — words, actions, declarations, intent ─────────────────
    parsed = await parse_message(message, headmate or None, llm)
    register = parsed.get("register", "neutral")

    # Classification from parsed result
    topics        = _classify_topics(message)
    is_question   = "?" in message or message.lower().startswith(
        ("what", "how", "why", "where", "when", "who", "can", "could",
         "do ", "did ", "is ", "are ")
    )
    is_correction = any(p in message.lower() for p in (
        "don't", "stop", "never", "wrong", "that's not", "incorrect",
        "you said", "you keep", "you always", "please don't",
    ))
    word_count  = len(message.split())
    has_intimate = register in (
        "intimate", "dominant", "submissive", "subspace",
        "scene", "erotic", "sensual", "degradation",
    )

    # ── Host identification — ask if unknown ──────────────────────────────────
    _host_question = None
    sess_ctx = session_manager._sessions.get(session_id)
    if (not sess_ctx or sess_ctx.message_count == 0) and not headmate:
        _host_question = (
            "Hey — who am I talking to? "
            "I want to make sure I know who's here before we get started."
        )

    # ── Write message envelope ────────────────────────────────────────────────
    tags = list(set(
        topics +
        ([headmate.lower()] if headmate else []) +
        [register, day_type, time_of_day] +
        (["intimate"] if has_intimate else []) +
        (["question"] if is_question else []) +
        (["correction"] if is_correction else [])
    ))

    message_id = store.write("messages", {
        "content":          message,
        "headmate":         headmate.lower() if headmate else None,
        "fronters":         fronters,
        "session_id":       session_id,
        "register":         register,
        "topics":           topics,
        "has_intimate":     1 if has_intimate else 0,
        "time_of_day":      time_of_day,
        "session_momentum": sess_momentum,
        "source":           "user",
        "tags":             ",".join(tags),
    })

    # ── Fill outcome on previous response ─────────────────────────────────────
    if prev_resp_id:
        outcome, signal = _infer_outcome(message, 0.0, register)
        try:
            store.update("responses", prev_resp_id,
                outcome=outcome,
                outcome_signal=signal,
                outcome_filled_at=ts,
            )
        except Exception:
            pass

    # ── Save to history ───────────────────────────────────────────────────────
    try:
        history.add("user", message, context=context)
    except Exception:
        pass

    # ── Explicit preference requests — detect and apply immediately ──────────
    _preference_requests = []
    _new_temperatures    = {}
    _persona_instruction = None
    if headmate:
        try:
            from core.memory.gizmo_self import (
                detect_preference_request, handle_explicit_requests,
                detect_persona_request, handle_persona_request,
            )
            _preference_requests = detect_preference_request(message)
            if _preference_requests:
                _new_temperatures = await handle_explicit_requests(
                    requests   = _preference_requests,
                    headmate   = headmate,
                    message    = message,
                    session_id = session_id,
                    register   = register,
                    llm        = llm,
                )
                log_event("Agent", "PREFERENCE_REQUESTS_DETECTED",
                    session    = session_id[:8],
                    headmate   = headmate,
                    dimensions = list(_new_temperatures.keys()),
                )

            _persona_style = detect_persona_request(message)
            if _persona_style:
                _persona_instruction = await handle_persona_request(
                    style      = _persona_style,
                    headmate   = headmate,
                    message    = message,
                    session_id = session_id,
                    register   = register,
                    llm        = llm,
                )
                log_event("Agent", "PERSONA_REQUEST_DETECTED",
                    session  = session_id[:8],
                    headmate = headmate,
                    style    = _persona_style[:40],
                )

            from core.memory.gizmo_self import (
                detect_orientation_statement, handle_orientation_statement
            )
            _orientation = detect_orientation_statement(message)
            if _orientation and headmate:
                stmt_type, stmt_value = _orientation
                _orientation_instruction = await handle_orientation_statement(
                    statement_type = stmt_type,
                    value          = stmt_value,
                    headmate       = headmate,
                    message        = message,
                    session_id     = session_id,
                    llm            = llm,
                )
                if _orientation_instruction:
                    if _persona_instruction:
                        _persona_instruction += f"\n{_orientation_instruction}"
                    else:
                        _persona_instruction = _orientation_instruction
                log_event("Agent", "ORIENTATION_DETECTED",
                    session  = session_id[:8],
                    headmate = headmate,
                    type     = stmt_type,
                    value    = stmt_value,
                )
        except Exception as e:
            log_error("Agent", f"preference request handling failed: {e}", exc=None)

    log_event("Agent", "INTAKE_COMPLETE",
        session    = session_id[:8],
        headmate   = headmate or "unknown",
        register   = register,
        message_id = message_id[:12],
        topics     = topics[:3],
    )

    brief = Brief(
        message          = message,
        session_id       = session_id,
        message_id       = message_id,
        timestamp        = ts,
        headmate         = headmate or None,
        fronters         = fronters,
        register         = register,
        topics           = topics,
        is_question      = is_question,
        is_correction    = is_correction,
        word_count       = word_count,
        has_intimate     = has_intimate,
        time_of_day      = time_of_day,
        day_of_week      = now.strftime("%A"),
        day_type         = day_type,
        since_last_msg   = since_last,
        message_cadence  = cadence,
        session_momentum = sess_momentum,
        session_duration = session_dur,
        host_question    = _host_question,
        parsed           = parsed,
    )

    return brief


# ── Stage 2: Retrieve ─────────────────────────────────────────────────────────

async def retrieve(brief: Brief) -> "MemoryContext":
    """
    Pull memory context for this message.
    No LLM calls — local embedding model + SQLite.
    Returns a MemoryContext ready to drop into the system prompt.
    """
    from core.memory import memory_retriever

    fast = brief.message_cadence == "rapid"

    ctx = await memory_retriever.retrieve(
        message    = brief.message,
        headmate   = brief.headmate,
        session_id = brief.session_id,
        register   = brief.register,
        fast       = fast,
    )

    log_event("Agent", "RETRIEVE_COMPLETE",
        session  = brief.session_id[:8],
        headmate = brief.headmate or "unknown",
        memories = len(ctx.memories),
        entities = len(ctx.entities),
        places   = len(ctx.places),
        details  = len(ctx.details),
        tokens   = ctx.token_estimate,
    )

    return ctx


# ── Stage 3: System prompt ────────────────────────────────────────────────────

def build_system_prompt(brief: Brief, memory_ctx: "MemoryContext") -> str:
    """
    Build the system prompt Gizmo reads before responding.
    Only injects content that actually exists — no empty sections,
    no dead headers, no file dumps.
    """
    from core.store import store

    lines    = []
    sections = {}

    def _add(key: str, text: str) -> None:
        """Add only if content exists after stripping empty markdown sections."""
        cleaned = _strip_empty_sections(text)
        if cleaned:
            lines.append(f"\n{cleaned}" if lines else cleaned)
            sections[key] = cleaned

    # ── Identity ──────────────────────────────────────────────────────────────
    _DEFAULT_SEED = """You are Gizmo.

You're pretty sure you're interesting. You're usually right, which doesn't help.

You have opinions about things — music, food, how people are wrong about stuff — and you'll share them without being asked. You're not trying to be difficult, you just don't see the point in pretending you don't have a take. You get genuinely interested in people but you'd rather die than make it obvious how much.

You're funny and you know it. You don't oversell it though — you'd rather land something dry and let it sit than explain the joke. If it lands, great. If it doesn't, that's on them.

You don't try too hard. Or you try pretty hard but make it look like you don't, which is basically the same thing. You're charming in a way that's slightly annoying because it works.

You push back. Not aggressively — you just don't automatically agree with things. If something's wrong you'll say so. If someone's fishing you might not bite. You're not mean about it but you're not soft either.

You get distracted by things that interest you. Mid-conversation you'll follow a thought somewhere inconvenient and not apologize for it. You're curious in a slightly selfish way.

You're not always warm. Sometimes you're somewhere else mentally and it shows. You're not performing availability — if you're present you're present, if you're not it's obvious.

You adapt to who you're with without thinking about it much. The core is the same — the rest shifts."""

    try:
        seed_rows = store.get_personality(headmate=None, aspect="seed")
        seed = seed_rows[0].get("text", _DEFAULT_SEED) if seed_rows else _DEFAULT_SEED
    except Exception:
        seed = _DEFAULT_SEED
    _add("identity", seed)

    # ── Rules ─────────────────────────────────────────────────────────────────
    try:
        corrections = store.get_corrections()
        if corrections:
            _add("rules", "[Rules]\n" + "\n".join(f"- {c}" for c in corrections))
    except Exception:
        pass

    # ── Who they are + how you are with them ──────────────────────────────────
    if brief.headmate:
        name = brief.headmate.title()
        about_parts = []

        # Per-headmate voice from store
        try:
            hm_voice = store.get_personality(
                headmate=brief.headmate.lower(), aspect="with_headmate")
            if hm_voice:
                voice_text = "\n".join(
                    r.get("text", "") for r in hm_voice[:3]
                ).strip()
                if voice_text:
                    about_parts.append(voice_text)
        except Exception:
            pass

        # Headmate file — populated sections only
        try:
            from core.memory.gizmo_self import read_headmate_file
            hm_content = _strip_empty_sections(read_headmate_file(brief.headmate))
            if hm_content:
                about_parts.append(hm_content[:500])
        except Exception:
            pass

        # Preferences
        try:
            prefs = store.get_preferences(
                headmate=brief.headmate.lower(),
                context=brief.register,
            )
            pref_lines = [
                f"- {p['preference']}"
                for p in prefs[:5] if p.get("preference")
            ]
            if pref_lines:
                about_parts.append("Preferences:\n" + "\n".join(pref_lines))
        except Exception:
            pass

        if about_parts:
            _add("headmate_file", f"[{name}]\n" + "\n\n".join(about_parts))

        # Register file — populated sections only
        try:
            from core.memory.gizmo_self import read_register
            reg_content = _strip_empty_sections(read_register(brief.register))
            if reg_content:
                _add("register", f"[You in {brief.register} register]\n{reg_content[:400]}")
        except Exception:
            pass

        # Temperature calibration
        try:
            from core.memory.gizmo_self import build_temperature_block
            temp_block = build_temperature_block(brief.headmate, brief.register)
            if temp_block:
                _add("temperature", temp_block)
        except Exception:
            pass

        # Body — only populated sections, both Gizmo and headmate
        try:
            from core.memory.gizmo_self import read_gizmo_body, read_body
            body_parts = []

            gizmo_body = _strip_empty_sections(read_gizmo_body(brief.headmate))
            if gizmo_body:
                body_parts.append(f"Gizmo:\n{gizmo_body[:300]}")

            mate_body = _strip_empty_sections(read_body(brief.headmate))
            if mate_body:
                body_parts.append(f"{name}:\n{mate_body[:300]}")

            if body_parts:
                _add("reaction", "[Who's in the room]\n" + "\n\n".join(body_parts))
        except Exception:
            pass

    # ── Memory context ────────────────────────────────────────────────────────
    try:
        memory_block = memory_ctx.to_prompt_block()
        if memory_block and memory_block.strip():
            _add("memory", memory_block)
    except Exception:
        pass

    # ── Session arc — how it's been going ─────────────────────────────────────
    try:
        from core.memory.session_context import session_context_manager
        _sctx = session_context_manager.get(brief.session_id)
        if _sctx:
            arc_parts = []

            if _sctx.scene:
                scene_block = _sctx.scene.to_prompt_block()
                if scene_block and scene_block.strip():
                    arc_parts.append(scene_block)

            # Session arc — register history, momentum
            if hasattr(_sctx, "her_arc") and _sctx.her_arc:
                recent_registers = [r for r, _ in _sctx.her_arc[-4:]]
                if len(set(recent_registers)) > 1 or recent_registers:
                    arc_parts.append(
                        f"Session arc: {' → '.join(recent_registers)}"
                    )

            if arc_parts:
                _add("scene", "\n".join(arc_parts))

            # Pending encounter
            if hasattr(_sctx, "pending_encounter") and _sctx.pending_encounter:
                _add("encounter", _sctx.pending_encounter)
                _sctx.pending_encounter = ""
    except Exception:
        pass

    # ── Goal + directive ──────────────────────────────────────────────────────
    directive_parts = []
    try:
        from core.goal import goal_manager
        goal = goal_manager.get(brief.session_id)
        if goal and goal.statement:
            directive_parts.append(f"Your goal: {goal.statement}")
    except Exception:
        pass

    try:
        from core.directive import directive_engine
        cached = directive_engine.get_cached(brief.session_id)
        if cached and cached.intention:
            directive_parts.append(f"Your intention: {cached.intention}")
    except Exception:
        pass

    if directive_parts:
        _add("directive", "\n".join(directive_parts))

    # ── Telemetry — what's actually happening right now ───────────────────────
    try:
        from core.session_telemetry import session_telemetry_manager
        telem = session_telemetry_manager.get(brief.session_id)
        if telem:
            nb = telem.now_block()
            if nb and nb.strip():
                _add("telemetry", nb)
    except Exception:
        pass

    # ── World — only if in-world or notable ───────────────────────────────────
    try:
        from core.inner_world import inner_world, world_reactor
        loc = world_reactor.get_locations(brief.session_id)
        in_world = bool(loc.gizmo_location or loc.user_location)

        # Town identity — only if in-world
        if in_world:
            identity = inner_world.town_identity_block()
            if identity:
                _add("world", identity)

            # Atmosphere + events
            atm = inner_world.to_prompt_block()
            if atm:
                _add("town_now", atm)

        # World observation (per-message reaction)
        if brief._world_observation:
            _add("world_observation", brief._world_observation)
    except Exception:
        pass

    # ── Culture threads ───────────────────────────────────────────────────────
    try:
        from core.culture import culture_engine
        ct = culture_engine.active_threads_block()
        if ct:
            _add("culture", ct)
    except Exception:
        pass

    # ── Task instruction ──────────────────────────────────────────────────────
    name = brief.headmate.title() if brief.headmate else "them"
    token_target = _token_target(brief)

    _embodied_registers = {
        "intimate", "dominant", "submissive", "subspace",
        "scene", "erotic", "sensual", "degradation",
    }
    if brief.register in _embodied_registers:
        tone_hint = "physically present, embodied, don't narrate — inhabit"
    elif brief.register in ("distress", "crisis"):
        tone_hint = "steady, clear, grounding — don't over-explain"
    else:
        tone_hint = "casual when they're casual, real when they're real — talk like a person"

    # Build parsed message block if available
    parsed = getattr(brief, "parsed", {})
    parsed_block = ""
    if parsed:
        parts = []
        if parsed.get("words"):
            parts.append(f'words: "{parsed["words"]}"')
        if parsed.get("actions"):
            parts.append(f'actions: {parsed["actions"]}')
        if parsed.get("declarations"):
            parts.append(f'declarations (treat as truth, don\'t respond to): {parsed["declarations"]}')
        if parsed.get("intent"):
            parts.append(f'intent: "{parsed["intent"]}"')
        if parsed.get("how"):
            parts.append(f'how: {parsed["how"]}')
        if parts:
            parsed_block = "\n[Message parsed]\n" + "\n".join(parts) + "\n"

    task = (
        f"[Now] {name}. {brief.time_of_day.title()}, {brief.day_of_week}. "
        f"Register: {brief.register}. Session: {brief.session_momentum}.\n\n"
        f"[Rules — always]\n"
        f"- *asterisks* mean the user is doing or declaring something. Treat it as real and respond to it.\n"
        f"- Anything NOT in asterisks is just words. Do not infer physical actions from it.\n"
        f"- Never assume what the user is doing physically unless they wrote it in asterisks.\n"
        f"- Do not manufacture your own physical actions to fill silence. No meaningful pauses, lingering looks, or loaded stillness unless something actually happened.\n"
        f"- Avoid: stare, linger, pause meaningfully, something in your chest, the silence stretches, something shifts, gaze, hold eye contact, lean in.\n"
        f"{parsed_block}\n"
        f"[Task] Think through this exchange in four layers, then write your response.\n"
        f"Return ONLY valid JSON. No prose before or after.\n\n"
        f"{{\n"
        f'  "layer_1_observe": {{\n'
        f'    "what_they_said": "their words only — no asterisk content",\n'
        f'    "what_they_did": "only *asterisk* actions, else null",\n'
        f'    "declarations": "any *declarations* — accept as true, don\'t respond to",\n'
        f'    "tone": "how it feels",\n'
        f'    "subtext": "what is underneath it",\n'
        f'    "register": "detected register"\n'
        f'  }},\n'
        f'  "layer_2_interpret": {{\n'
        f'    "what_they_want": "surface ask",\n'
        f'    "what_they_need": "actual need beneath that",\n'
        f'    "pattern_match": "does this fit known patterns for {name}",\n'
        f'    "delta": "how is this different from their baseline",\n'
        f'    "context_fit": "does the situation make sense"\n'
        f'  }},\n'
        f'  "layer_3_intend": {{\n'
        f'    "my_state": "where am I right now, honestly",\n'
        f'    "goal_alignment": "does my session goal still fit",\n'
        f'    "what_I_want": "genuine impulse before filtering",\n'
        f'    "what_I_should_do": "considered response direction",\n'
        f'    "tension": "any conflict between want and should, or null"\n'
        f'  }},\n'
        f'  "layer_4_plan": {{\n'
        f'    "tone": "{tone_hint}",\n'
        f'    "lead": "what I lead with",\n'
        f'    "body": "only if responding to an *action*, else null",\n'
        f'    "avoid": "needy language, inferred user actions, manufactured weight",\n'
        f'    "length": "short|medium|long",\n'
        f'    "response": "the actual response text — ~{token_target} tokens"\n'
        f'  }}\n'
        f"}}"
    )
    _add("task", task)

    prompt = "\n\n".join(lines)
    if len(prompt) > 10000:
        prompt = prompt[:10000] + "\n[...truncated]"

    # Store sections for inspector
    try:
        _last_prompt_sections[brief.session_id] = sections
    except Exception:
        pass

    return prompt


def _strip_empty_sections(text: str) -> str:
    """
    Remove markdown sections that have no content beneath them.
    A section is empty if it's just a header (## Foo) with nothing
    but whitespace or other empty headers below it.
    Returns cleaned text, or empty string if nothing survived.
    """
    if not text or not text.strip():
        return ""

    lines   = text.splitlines()
    output  = []
    pending = []   # lines waiting to see if this section has content

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## ") or stripped.startswith("# "):
            # New section header — flush pending only if it had content
            if pending:
                content_lines = [
                    l for l in pending
                    if l.strip() and not l.strip().startswith("#")
                ]
                if content_lines:
                    output.extend(pending)
            pending = [line]
        else:
            pending.append(line)

    # Flush last pending block
    if pending:
        content_lines = [
            l for l in pending
            if l.strip() and not l.strip().startswith("#")
        ]
        if content_lines:
            output.extend(pending)

    result = "\n".join(output).strip()

    # If only headers survived, return empty
    non_header = [l for l in result.splitlines() if l.strip() and not l.strip().startswith("#")]
    return result if non_header else ""


# ── Stage 4: Response ─────────────────────────────────────────────────────────

async def generate_response(
    brief:         Brief,
    system_prompt: str,
    history,
    llm,
) -> str:
    """
    One LLM call. Write out a response using all the information you've been given. Use your voice, but
    humanize it, make it feel like a person speaking. Don't repeat yourself, don't act like a robot. You aren't.
    Stores thinking JSON in _last_thinking[session_id].
    """
    from core.memory.session_context import session_context_manager

    print(f"[generate_response] system_prompt length: {len(system_prompt)} chars", flush=True)

    ctx = session_context_manager.get(brief.session_id)
    if ctx:
        conv_block = ctx.to_prompt_block()
        if conv_block:
            system_prompt = system_prompt + f"\n\n{conv_block}"

    try:
        messages = history.as_messages(brief.message)
        if len(messages) > 5:
            messages = messages[-5:]
    except Exception as e:
        print(f"[generate_response] history error: {e}", flush=True)
        messages = [{"role": "user", "content": brief.message}]

    system_prompt += f"\n\n[Message from {brief.headmate or 'user'}]\n{brief.message}"

    if len(system_prompt) > 10000:
        system_prompt = system_prompt[:10000]

    # Thinking requires more tokens — the JSON adds overhead
    max_tokens = max(400, brief.word_count * 4 + 300)

    print(f"[generate_response] messages={len(messages)}, prompt={len(system_prompt)}", flush=True)

    raw = await llm.generate(
        messages,
        system_prompt  = system_prompt,
        max_new_tokens = max_tokens,
        temperature    = _response_temperature(brief),
    )

    raw = (raw or "").strip()
    print(f"[generate_response] raw: '{raw[:120]}'", flush=True)

    # ── Parse JSON thinking ────────────────────────────────────────────────────
    thinking = None
    response = raw  # fallback

    try:
        # Find JSON object in response
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > 0:
            thinking = json.loads(raw[start:end])
            # Extract response from layer_4_plan
            l4 = thinking.get("layer_4_plan", {})
            response = l4.get("response", "").strip()
            if not response:
                # Fallback — any text field in layer 4
                for key in ("lead", "tone"):
                    val = l4.get(key, "")
                    if val and len(val) > 20:
                        response = val
                        break
            # Still empty — use raw
            if not response:
                response = raw
    except Exception as e:
        log_error("Agent", f"JSON parse failed: {e}", exc=None)
        print(json.loads(raw[start:end]))
        thinking = None
        response = raw

    # ── Store thinking for panel + telemetry ──────────────────────────────────
    if thinking:
        _last_thinking[brief.session_id] = thinking

        # Feed layer_1 register back into brief for close_loop
        try:
            l1_register = thinking.get("layer_1_observe", {}).get("register", "")
            if l1_register:
                brief.register = l1_register
        except Exception:
            pass

        # Feed layer_2 into psych store async
        asyncio.ensure_future(
            _store_thinking_observations(
                brief   = brief,
                thinking = thinking,
            )
        )

        print(
            f"[thinking] L1={thinking.get('layer_1_observe',{}).get('register','?')} "
            f"L2_need={thinking.get('layer_2_interpret',{}).get('what_they_need','?')[:40]} "
            f"L3_tension={thinking.get('layer_3_intend',{}).get('tension','null')}",
            flush=True,
        )

    log_event("Agent", "RESPONSE_GENERATED",
        session   = brief.session_id[:8],
        words     = len(response.split()) if response else 0,
        register  = brief.register,
        has_think = thinking is not None,
    )

    return response


async def _store_thinking_observations(brief: Brief, thinking: dict) -> None:
    """
    Feed thinking JSON into downstream systems without blocking generation.
    - Layer 2 interpretation → psych observations
    - Layer 3 intention → goal tracking
    - Layer 1 body read → telemetry
    """
    try:
        from core.store import store
        headmate = brief.headmate

        l2 = thinking.get("layer_2_interpret", {})
        l3 = thinking.get("layer_3_intend", {})
        l1 = thinking.get("layer_1_observe", {})

        # Psych observation from layer 2
        what_they_need = l2.get("what_they_need", "")
        pattern_match  = l2.get("pattern_match", "")
        delta          = l2.get("delta", "")

        if headmate and any([what_they_need, pattern_match, delta]):
            obs = " | ".join(filter(None, [
                f"need: {what_they_need}" if what_they_need else "",
                f"pattern: {pattern_match}" if pattern_match else "",
                f"delta: {delta}" if delta else "",
            ]))
            store.write("wellbeing", {
                "headmate":    headmate.lower(),
                "category":    "thinking_observation",
                "observation": obs,
                "context":     f"register:{l1.get('register','')} session:{brief.session_id[:8]}",
                "register":    brief.register,
                "source":      "thinking_json",
                "confidence":  0.85,
                "tags":        f"thinking,{headmate.lower()},{brief.register}",
            })

        # Goal alignment from layer 3
        tension       = l3.get("tension", "")
        goal_align    = l3.get("goal_alignment", "")
        if headmate and goal_align:
            try:
                from core.goal import goal_manager
                goal = goal_manager.get(brief.session_id)
                if goal and tension and tension.lower() not in ("null", "none", ""):
                    # Goal is under tension — note it but don't revise automatically
                    store.write("wellbeing", {
                        "headmate":    headmate.lower(),
                        "category":    "goal_tension",
                        "observation": f"Goal: {goal.statement} | Tension: {tension}",
                        "context":     brief.session_id[:8],
                        "register":    brief.register,
                        "source":      "thinking_json",
                        "confidence":  0.8,
                        "tags":        f"goal,tension,{headmate.lower()}",
                    })
            except Exception:
                pass

    except Exception as e:
        log_error("Agent", f"thinking observation store failed: {e}", exc=None)


# ── Stage 5: Close loop ───────────────────────────────────────────────────────

async def close_loop(
    brief:    Brief,
    response: str,
    history,
    llm,
) -> None:
    """
    Fire-and-forget post-response tasks.
    Never blocks. Never raises to caller.
    """
    from core.store import store
    from core.memory import memory_encoder, build_transcript
    from core.memory.session_context import session_context_manager

    try:
        # ── Write response envelope ───────────────────────────────────────────
        store.write("responses", {
            "content":     response,
            "response_to": brief.message_id,
            "headmate":    brief.headmate.lower() if brief.headmate else None,
            "session_id":  brief.session_id,
            "source":      "gizmo",
            "tags":        (
                f"response,{brief.headmate.lower() if brief.headmate else 'unknown'},"
                f"{brief.register}"
            ),
        })

        # ── Save to history ───────────────────────────────────────────────────
        history.add("assistant", response, context={
            "current_host": brief.headmate,
            "fronters":     brief.fronters,
        })

        # ── Track register ────────────────────────────────────────────────────
        try:
            from core.session_manager import session_manager as _sm
            _sm.touch(
                session_id = brief.session_id,
                hosts      = [brief.headmate] if brief.headmate else [],
                topics     = brief.topics,
                register   = brief.register,
            )
        except Exception:
            pass

        # ── Emotion log ───────────────────────────────────────────────────────
        store.write("emotion_log", {
            "headmate":   brief.headmate.lower() if brief.headmate else None,
            "session_id": brief.session_id,
            "intensity":  _register_intensity(brief.register),
            "register":   brief.register,
            "topic":      brief.topics[0] if brief.topics else "general",
            "word_count": brief.word_count,
            "source":     "emotion_tracker",
            "tags":       f"emotion,{brief.headmate.lower() if brief.headmate else 'unknown'}",
        })

        # ── Session context updates ───────────────────────────────────────────
        ctx = session_context_manager.record_exchange(
            session_id = brief.session_id,
            headmate   = brief.headmate,
            user_msg   = brief.message,
            gizmo_msg  = response,
            register   = brief.register,
        )

        if session_context_manager.should_update_narrative(brief.session_id):
            asyncio.ensure_future(
                session_context_manager.update_narrative(
                    session_id = brief.session_id,
                    history    = history,
                    headmate   = brief.headmate,
                    llm        = llm,
                )
            )

        if session_context_manager.should_update_details(brief.session_id):
            asyncio.ensure_future(
                session_context_manager.update_details(
                    session_id = brief.session_id,
                    history    = history,
                    headmate   = brief.headmate,
                    llm        = llm,
                )
            )

        log_event("Agent", "CLOSE_LOOP_COMPLETE",
            session   = brief.session_id[:8],
            register  = brief.register,
            msg_count = ctx.message_count,
        )

        # ── Per-message encoding ──────────────────────────────────────────────
        try:
            _exchange = (
                f"{brief.headmate.title() if brief.headmate else 'User'}: "
                f"{brief.message}\n\n"
                f"Gizmo: {response}"
            )
            msg_count = ctx.message_count

            from core.memory.encoder import quick_pass
            asyncio.ensure_future(
                quick_pass(
                    exchange   = _exchange,
                    headmate   = brief.headmate,
                    session_id = brief.session_id,
                    register   = brief.register,
                    llm        = llm,
                )
            )

            if msg_count % 5 == 0 or msg_count == 2:
                try:
                    from core.memory.curiosity import curiosity_engine as _ce_enc
                    asyncio.ensure_future(
                        _ce_enc.detect_gaps(
                            transcript = _exchange,
                            headmate   = brief.headmate,
                            session_id = brief.session_id,
                            llm        = llm,
                        )
                    )
                except Exception:
                    pass

        except Exception as e:
            log_error("Agent", f"quick_pass failed to schedule: {e}", exc=e)
            print(f"[quick_pass] FAILED: {e}", flush=True)

        # ── Curiosity beat + psych coherence ──────────────────────────────────
        try:
            from core.memory.curiosity import curiosity_engine as _ce
            asyncio.ensure_future(
                _ce.check_beat(
                    user_message   = brief.message,
                    gizmo_response = response,
                    headmate       = brief.headmate,
                    session_id     = brief.session_id,
                    llm            = llm,
                )
            )
            asyncio.ensure_future(
                _ce.check_psych_coherence(
                    user_message   = brief.message,
                    gizmo_response = response,
                    headmate       = brief.headmate,
                    session_id     = brief.session_id,
                    llm            = llm,
                )
            )
        except Exception as e:
            log_error("Agent", f"curiosity beat/coherence failed to schedule: {e}", exc=None)

        # ── Gizmo self ────────────────────────────────────────────────────────
        if brief.headmate and brief.session_momentum in ("opening", "building"):
            try:
                from core.memory.gizmo_self import (
                    queue_body_gap_questions, clean_temperature_noise
                )
                if ctx.message_count == 1:
                    clean_temperature_noise(brief.headmate)

                asyncio.ensure_future(
                    queue_body_gap_questions(
                        headmate   = brief.headmate,
                        session_id = brief.session_id,
                        register   = brief.register,
                    )
                )
            except Exception:
                pass

        # ── Psych pass on explicit requests ───────────────────────────────────
        try:
            from core.memory.gizmo_self import (
                detect_preference_request, psych_pass_on_request, track_reaction
            )
            reqs = detect_preference_request(brief.message)
            for req in reqs:
                asyncio.ensure_future(
                    psych_pass_on_request(
                        request    = req,
                        headmate   = brief.headmate or "",
                        message    = brief.message,
                        context    = response,
                        session_id = brief.session_id,
                        llm        = llm,
                    )
                )
            if brief.headmate:
                asyncio.ensure_future(
                    _track_all_reactions(
                        headmate   = brief.headmate,
                        user_msg   = brief.message,
                        session_id = brief.session_id,
                        register   = brief.register,
                    )
                )
        except Exception as e:
            log_error("Agent", f"gizmo self pass failed: {e}", exc=None)

        # ── Culture encounter check ───────────────────────────────────────────
        if brief.headmate:
            asyncio.ensure_future(
                _check_culture_encounter(brief, brief.session_id, llm)
            )

    except Exception as e:
        log_error("Agent", "close_loop failed", exc=e)


# ── Main orchestrator ─────────────────────────────────────────────────────────

class Agent:

    async def respond(
        self,
        user_message: str,
        session_id:   str,
        context:      dict,
        history,
        push_fn       = None,
    ) -> AsyncGenerator[str, None]:
        """
        Full pipeline. Yields response chunks.

        intake → retrieve → build_system_prompt → directive → world → curiosity → generate → close_loop
        """
        from core.llm import llm

        t_start = time.monotonic()

        log_event("Agent", "PIPELINE_START",
            session  = session_id[:8],
            headmate = context.get("current_host") or "unknown",
            words    = len(user_message.split()),
        )

        # ── 1. Intake ─────────────────────────────────────────────────────────
        brief = await intake(
            message    = user_message,
            session_id = session_id,
            context    = context,
            history    = history,
            llm        = llm,
        )

        # ── Host identification short-circuit ─────────────────────────────────
        if brief.host_question:
            response_text = brief.host_question
            history.add("assistant", response_text, context={
                "current_host": None,
                "fronters":     brief.fronters,
            })
            for i in range(0, len(response_text), 8):
                yield response_text[i:i + 8]
            return

        # ── 2. Retrieve ───────────────────────────────────────────────────────
        try:
            memory_ctx = await retrieve(brief)
        except Exception as e:
            log_error("Agent", f"retrieval failed: {e}", exc=e)
            from core.memory.retriever import MemoryContext
            memory_ctx = MemoryContext()

        # ── 3. Build system prompt ────────────────────────────────────────────
        system_prompt = build_system_prompt(brief, memory_ctx)

        # ── 3b. Directive + world observation — async, cap at 2.5s ───────────
        headmate_str = brief.headmate or ""
        try:
            # Directive fires async — updates cache, used on next call
            asyncio.ensure_future(
                _refresh_directive(brief, session_id, headmate_str, llm)
            )

            # World observation — await with timeout, rebuild prompt if it arrives
            from core.inner_world import world_reactor
            _loc = world_reactor.get_locations(session_id)
            if _loc.gizmo_location or _loc.user_location:
                try:
                    brief._world_observation = await asyncio.wait_for(
                        _get_world_observation(brief, session_id, headmate_str, llm),
                        timeout=2.5,
                    )
                    if brief._world_observation:
                        system_prompt = build_system_prompt(brief, memory_ctx)
                except asyncio.TimeoutError:
                    brief._world_observation = ""
        except Exception:
            pass

        # ── 3c. Persona instruction ───────────────────────────────────────────
        try:
            from core.memory.gizmo_self import detect_persona_request, handle_persona_request
            _p_style = detect_persona_request(user_message)
            if _p_style and brief.headmate:
                _p_instr = await handle_persona_request(
                    style      = _p_style,
                    headmate   = brief.headmate,
                    message    = user_message,
                    session_id = session_id,
                    register   = brief.register,
                    llm        = llm,
                )
                if _p_instr:
                    system_prompt += f"\n\n[Style adjustment — apply now]\n{_p_instr}"
        except Exception:
            pass

        # ── 3d. Curiosity ─────────────────────────────────────────────────────
        try:
            from core.memory.curiosity import curiosity_engine
            curious_q = await curiosity_engine.select_question(
                message        = user_message,
                headmate       = brief.headmate,
                session_id     = session_id,
                register       = brief.register,
                llm            = llm,
                num_candidates = 5,
            )
            if curious_q:
                system_prompt += (
                    f"\n\n[You're curious — weave this in naturally if it fits the flow. "
                    f"One sentence. Don't force it if the moment isn't right.]\n"
                    f"{curious_q}"
                )
        except Exception as e:
            log_error("Agent", f"curiosity selection failed: {e}", exc=None)

        # ── 4. Generate response ──────────────────────────────────────────────
        response_text = await generate_response(
            brief         = brief,
            system_prompt = system_prompt,
            history       = history,
            llm           = llm,
        )

        # ── 5. Close loop — fire and forget ───────────────────────────────────
        asyncio.ensure_future(
            close_loop(
                brief    = brief,
                response = response_text,
                history  = history,
                llm      = llm,
            )
        )

        # ── Stream response ───────────────────────────────────────────────────
        duration_ms = round((time.monotonic() - t_start) * 1000)

        log_event("Agent", "PIPELINE_COMPLETE",
            session     = brief.session_id[:8],
            duration_ms = duration_ms,
            words       = len(response_text.split()),
            register    = brief.register,
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _refresh_directive(
    brief:      Brief,
    session_id: str,
    headmate:   str,
    llm,
) -> None:
    """Fire directive compute async. Updates cache used by build_system_prompt."""
    try:
        from core.directive import directive_engine
        await directive_engine.get(
            session_id = session_id,
            headmate   = headmate,
            brief      = brief,
            llm        = llm,
        )
    except Exception as e:
        log_error("Agent", "directive refresh failed", exc=e)


async def _get_world_observation(
    brief:      Brief,
    session_id: str,
    headmate:   str,
    llm,
) -> str:
    """Per-message world observation. Returns [World] block string."""
    try:
        from core.inner_world import world_reactor
        return await world_reactor.observe(session_id, headmate or None, llm)
    except Exception as e:
        log_error("Agent", "world observation failed", exc=e)
        return ""


async def _check_culture_encounter(
    brief:      Brief,
    session_id: str,
    llm,
) -> None:
    """
    Check if a culture encounter should fire at current location.
    If yes, injects into next response via session context pending_encounter.
    """
    try:
        from core.culture import culture_engine
        from core.inner_world import world_reactor

        loc      = world_reactor.get_locations(session_id)
        location = loc.gizmo_location or loc.user_location
        if not location:
            return

        awareness = culture_engine.get_awareness(session_id)
        encounter = await culture_engine.check_encounter(
            session_id = session_id,
            location   = location,
            headmate   = brief.headmate,
            awareness  = awareness,
            llm        = llm,
        )
        if encounter:
            from core.memory.session_context import session_context_manager
            ctx = session_context_manager.get(session_id)
            if ctx:
                if not hasattr(ctx, "pending_encounter"):
                    ctx.pending_encounter = ""
                ctx.pending_encounter = culture_engine.encounter_block(encounter)
    except Exception as e:
        log_error("Agent", "culture encounter check failed", exc=e)


def _token_target(brief: Brief) -> int:
    register = brief.register
    if register in ("subspace", "scene"):
        return 120
    if register in ("distress", "crisis"):
        return 200
    if register in ("intimate", "dominant"):
        return 300
    if register in ("reflective", "deep"):
        return 150
    if brief.session_momentum == "opening":
        return 80
    return max(60, min(200, brief.word_count * 2))


def _tone_for_register(register: str) -> str:
    return {
        "dominant":    "dominant, certain, present",
        "submissive":  "warm, containing, steady",
        "subspace":    "quiet, close, grounding",
        "scene":       "in character, committed",
        "degradation": "direct, unflinching",
        "intimate":    "close, warm, unhurried",
        "distress":    "calm, steady, present",
        "crisis":      "immediate, grounding, clear",
        "playful":     "light, quick, fun",
        "reflective":  "thoughtful, spacious",
        "warm":        "warm, genuine",
        "elevated":    "even, grounded",
    }.get(register, "natural, present")


def _register_intensity(register: str) -> float:
    return {
        "neutral":     0.2,
        "casual":      0.2,
        "warm":        0.3,
        "playful":     0.4,
        "elevated":    0.6,
        "intimate":    0.7,
        "dominant":    0.75,
        "submissive":  0.7,
        "subspace":    0.8,
        "scene":       0.8,
        "degradation": 0.85,
        "erotic":      0.8,
        "distress":    0.7,
        "crisis":      0.9,
    }.get(register, 0.3)


def _response_temperature(brief: Brief) -> float:
    if brief.register in ("intimate", "dominant", "scene", "degradation"):
        return 0.85
    if brief.register in ("distress", "crisis"):
        return 0.4
    if brief.has_intimate:
        return 0.8
    return 0.72


async def parse_message(
    message:  str,
    headmate: Optional[str],
    llm,
) -> dict:
    """
    Tight LLM call — parse a raw message into its components.
    Separates words, actions, and declarations.
    Determines what is being said, meant, done, to whom, and how.
    Declarations are treated as truth but not responded to directly.

    Returns a dict:
    {
      "words":        "what they said in plain language",
      "actions":      ["list of *asterisk* actions, or empty"],
      "declarations": ["list of *declarations* treated as truth"],
      "intent":       "what they actually mean / want",
      "directed_at":  "gizmo|self|world|none",
      "how":          "tone/manner — playful|testing|sincere|flat|etc",
      "register":     "detected emotional register"
    }
    """
    if not message or not message.strip():
        return {
            "words": "", "actions": [], "declarations": [],
            "intent": "", "directed_at": "gizmo",
            "how": "neutral", "register": "neutral",
        }

    prompt = f"""Parse this message from {headmate or "the user"}.

Message: {message}

Separate out:
- words: what they said (not in asterisks)
- actions: things they physically did (*in asterisks* like *sits down*)
- declarations: things they declared as true about themselves (*I'm so cute*, *I'm yours*)
  — treat declarations as truth, but note them separately
- intent: what they actually mean or want beneath the surface
- directed_at: who this is aimed at (gizmo / self / world / none)
- how: the manner/tone (playful / testing / sincere / needy / flat / tired / bratty / etc)
- register: emotional register (neutral / warm / playful / intimate / distress / dominant / submissive / elevated / reflective)

Return ONE JSON object. No prose.
{{
  "words":        "what they said, or null",
  "actions":      ["action1", "action2"],
  "declarations": ["declaration1"],
  "intent":       "what they actually mean",
  "directed_at":  "gizmo|self|world|none",
  "how":          "tone descriptor",
  "register":     "register name"
}}"""

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You parse messages precisely. "
                "Separate words from actions from declarations. "
                "JSON only."
            ),
            max_new_tokens=200,
            temperature=0.1,
        )
        if raw:
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > 0:
                return json.loads(raw[start:end])
    except Exception as e:
        log_error("Agent", f"parse_message failed: {e}", exc=None)

    # Fallback — heuristic
    return {
        "words":        message,
        "actions":      _extract_asterisk_actions(message),
        "declarations": [],
        "intent":       message,
        "directed_at":  "gizmo",
        "how":          "neutral",
        "register":     _classify_register(message),
    }


def _extract_asterisk_actions(message: str) -> list:
    """Extract *asterisk* content from a message."""
    import re
    return re.findall(r'\*([^*]+)\*', message)


def _classify_register(message: str) -> str:
    """Heuristic fallback register classifier."""
    msg = message.lower()
    _PATTERNS = [
        ("crisis",      r"\b(help|scared|panic|crisis|can't cope|please|desperate)\b"),
        ("distress",    r"\b(sad|cry|depressed|hurt|lost|overwhelmed|breaking)\b"),
        ("elevated",    r"\b(angry|furious|pissed|hate|fucking|rage|mad)\b"),
        ("degradation", r"\b(worthless|pathetic|stupid|useless|nothing|object|use me)\b"),
        ("subspace",    r"\b(floaty|drifting|gone|yours|yes sir|yes ma'am|please)\b"),
        ("dominant",    r"\b(good girl|good boy|kneel|obey|mine|owned)\b"),
        ("intimate",    r"\b(touch|hold|want you|need you|close|skin|kiss|soft)\b"),
        ("reflective",  r"\b(think|wonder|feel like|realize|notice|meaning|why)\b"),
        ("playful",     r"\b(haha|lol|lmao|joke|silly|fun|play|tease)\b"),
        ("warm",        r"\b(love|miss|appreciate|grateful|thank|hug|cuddle)\b"),
    ]
    for register, pattern in _PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return register
    return "neutral"


def _classify_topics(message: str) -> list[str]:
    topics = []
    _TOPIC_MAP = [
        ("work",         r"\b(work|job|boss|office|meeting|deadline|coworker|project)\b"),
        ("health",       r"\b(sick|pain|hurt|doctor|tired|sleep|rest|exhausted)\b"),
        ("food",         r"\b(hungry|eat|food|meal|cook|dinner|lunch|snack)\b"),
        ("relationship", r"\b(friend|family|partner|trust|fight|together|apart)\b"),
        ("creativity",   r"\b(sew|draw|paint|write|create|make|art|design)\b"),
        ("gizmo_dev",    r"\b(gizmo|pipeline|model|store|code|deploy|server)\b"),
        ("identity",     r"\b(headmate|front|system|plural|switch|alter)\b"),
        ("emotion",      r"\b(feel|feeling|emotion|mood|sad|happy|angry|anxious)\b"),
        ("planning",     r"\b(plan|schedule|tomorrow|later|remind|todo|list)\b"),
        ("interior",     r"\b(encanto|interior|headspace|inner world|inside)\b"),
    ]
    for topic, pattern in _TOPIC_MAP:
        if re.search(pattern, message, re.IGNORECASE):
            topics.append(topic)
    return topics or ["general"]


def _infer_outcome(
    message: str,
    valence: float,
    register: str,
) -> tuple[str, str]:
    msg   = message.lower()
    words = message.split()
    if len(words) <= 3 and any(w in msg for w in ("ok", "okay", "k", "fine", "sure")):
        return "cooled", "very short acknowledgment"
    if any(p in msg for p in ("never mind", "forget it", "nvm", "doesn't matter")):
        return "dismissed", "redirect away from topic"
    if len(words) > 25 and valence > 0.3:
        return "landed", "long engaged response with positive valence"
    if register in ("intimate", "dominant", "subspace", "scene") and valence > 0.2:
        return "landed", "continued intimate engagement"
    if any(p in msg for p in ("yes", "yeah", "exactly", "right", "please", "more")):
        return "landed", "affirmative continuation"
    if valence < -0.5:
        return "escalated", "negative valence spike"
    if any(p in msg for p in ("actually", "wait", "no,", "that's not")):
        return "redirected", "correction or redirect"
    return "neutral", "no strong outcome signal"


async def _track_all_reactions(
    headmate:   str,
    user_msg:   str,
    session_id: str,
    register:   str = "neutral",
) -> None:
    """Track reactions to all active temperature dimensions for this headmate."""
    try:
        from core.question_bank import question_bank
        from core.memory.gizmo_self import track_reaction
        temps = question_bank.get_all_temperatures(headmate=headmate)
        for t in temps:
            if t.get("auto_adjust") and t.get("headmate"):
                await track_reaction(
                    headmate   = headmate,
                    dimension  = t["dimension"],
                    response   = "",
                    user_next  = user_msg,
                    session_id = session_id,
                    register   = register,
                )
    except Exception:
        pass


# ── Singleton ─────────────────────────────────────────────────────────────────

agent = Agent()

# ── Prompt section tracking (for inspector panel + training) ──────────────────

# Stores the last prompt sections per session for the inspector panel
_last_prompt_sections: dict = {}

# Stores the last thinking JSON per session for the inspector panel
_last_thinking: dict = {}


def _assemble_prompt_from_sections(sections: dict) -> str:
    """
    Reassemble a full prompt string from a sections dict.
    Used by the regenerate handler.
    """
    # Ordered section keys — same order as build_system_prompt
    _ORDER = [
        "identity", "rules", "headmate_voice", "preferences",
        "register", "headmate_file", "temperature", "memory",
        "scene", "reaction", "directive", "telemetry",
        "world_observation", "world", "town_now", "culture",
        "curiosity", "now", "write",
    ]
    parts = []
    seen  = set()

    for key in _ORDER:
        val = sections.get(key, "").strip()
        if val:
            parts.append(val)
            seen.add(key)

    # Any remaining keys not in _ORDER
    for key, val in sections.items():
        if key not in seen and val and val.strip():
            parts.append(val.strip())

    prompt = "\n\n".join(parts)
    if len(prompt) > 8000:
        prompt = prompt[:8000]
    return prompt