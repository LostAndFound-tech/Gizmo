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
     Detail catch — catches asides and throwaway mentions.
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

    # New headmate flag — no history, start fresh
    is_new_headmate:  bool  = False

    # Unknown entities mentioned in this message
    unknown_entities: list  = field(default_factory=list)

    # Host identification short-circuit
    host_question: Optional[str] = None


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
        "rapid"          if since_last < 15   else
        "fast"           if since_last < 60   else
        "conversational" if since_last < 300  else
        "slow"           if since_last < 1800 else
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

    # ── Classification ────────────────────────────────────────────────────────
    register      = _classify_register(message)
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

    # ── New headmate detection ────────────────────────────────────────────────
    # Check if this headmate has any history at all
    is_new_headmate = False
    if headmate:
        try:
            from core.memory.store import memory_store
            from datetime import datetime, timezone
            has_entity   = memory_store.entity_exists(headmate)
            has_memories = (memory_store.root / "memories" / headmate.lower()).exists()
            is_new_headmate = not has_entity and not has_memories

            # Create stub entity immediately — gives psychology pass
            # a correct target and prevents contaminating other headmates' files
            if is_new_headmate:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                memory_store.write_entity(
                    name       = headmate,
                    content    = (
                        f"First contact: {date_str}\n"
                        f"Status: getting to know\n\n"
                        f"{headmate.title()} is a distinct, separate person. "
                        f"Do not conflate with any other headmate."
                    ),
                    session_id = session_id,
                    keywords   = headmate.lower(),
                )
        except Exception:
            pass

    # ── Unknown entity detection ──────────────────────────────────────────────
    # Scan for capitalized names not in the entity store
    unknown_entities = []
    if headmate:
        try:
            from core.memory.store import memory_store
            # Extract capitalized words that look like proper names
            # Exclude common words, the headmate's own name, and known entities
            name_pattern = re.compile(r'\b([A-Z][a-z]{2,})\b')
            candidates   = set(name_pattern.findall(message))

            # Filter out common words and known entities
            _SKIP = {
                "I", "The", "And", "But", "For", "With", "You", "We",
                "He", "She", "They", "It", "This", "That", "Monday",
                "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                "Sunday", "January", "February", "March", "April", "May",
                "June", "July", "August", "September", "October", "November",
                "December", "Gizmo",
            }
            candidates -= _SKIP
            if headmate:
                candidates.discard(headmate.title())
                candidates.discard(headmate.lower())
                candidates.discard(headmate.upper())

            for name in candidates:
                if not memory_store.entity_exists(name):
                    unknown_entities.append(name)
        except Exception:
            pass
    _host_question = None
    sess_ctx = session_manager._sessions.get(session_id)
    if (not sess_ctx or sess_ctx.message_count == 0) and not headmate:
        try:
            from core.agent_tools import dispatch_tool
            _host_question = await dispatch_tool(
                tool_name="identify_host",
                args={},
                session_id=session_id,
                headmate="",
                llm=llm,
            )
        except Exception:
            pass

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

    log_event("Agent", "INTAKE_COMPLETE",
        session  = session_id[:8],
        headmate = headmate or "unknown",
        register = register,
        message_id = message_id[:12],
        topics   = topics[:3],
    )

    brief = Brief(
        message           = message,
        session_id        = session_id,
        message_id        = message_id,
        timestamp         = ts,
        headmate          = headmate or None,
        fronters          = fronters,
        register          = register,
        topics            = topics,
        is_question       = is_question,
        is_correction     = is_correction,
        word_count        = word_count,
        has_intimate      = has_intimate,
        time_of_day       = time_of_day,
        day_of_week       = now.strftime("%A"),
        day_type          = day_type,
        since_last_msg    = since_last,
        message_cadence   = cadence,
        session_momentum  = sess_momentum,
        session_duration  = session_dur,
        host_question     = _host_question,
        is_new_headmate   = is_new_headmate,
        unknown_entities  = unknown_entities,
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

    # Rapid-fire messages get fast retrieval — skip crawl and details
    fast = brief.message_cadence == "rapid"

    ctx = await memory_retriever.retrieve(
        message      = brief.message,
        headmate     = brief.headmate,
        session_id   = brief.session_id,
        register     = brief.register,
        fast         = fast,
        intimate_ok  = False if brief.is_new_headmate else None,
    )

    log_event("Agent", "RETRIEVE_COMPLETE",
        session   = brief.session_id[:8],
        headmate  = brief.headmate or "unknown",
        memories  = len(ctx.memories),
        entities  = len(ctx.entities),
        places    = len(ctx.places),
        details   = len(ctx.details),
        tokens    = ctx.token_estimate,
    )

    return ctx


# ── Stage 3: System prompt ────────────────────────────────────────────────────

async def _read_the_room(
    message:    str,
    headmate:   Optional[str],
    session_id: str,
    history,
    llm,
) -> str:
    """
    Given this exchange and this person's psychology,
    how are they doing and how should Gizmo show up?
    Returns a tone directive string for the [Write] block.
    Runs in parallel with retrieval — adds no wall time.
    """
    if not headmate:
        return "present, responsive"

    # Load psychology synthesis
    try:
        from core.memory.psychology import _read_psychology
        psych = _read_psychology(headmate, intimate=False) or ""
        psych_summary = ""
        if "## Current Understanding" in psych:
            psych_summary = psych.split("## Current Understanding", 1)[1]
            psych_summary = psych_summary.split("## Observations", 1)[0].strip()[:500]
    except Exception:
        psych_summary = ""

    # Get last 4 exchanges
    try:
        recent = history.as_list()[-8:] if hasattr(history, "as_list") else []
        history_text = "\n".join(
            f"{'Gizmo' if m['role'] == 'assistant' else headmate.title()}: {m.get('content','')[:100]}"
            for m in recent
            if isinstance(m, dict) and m.get("content")
        )
    except Exception:
        history_text = ""

    if not psych_summary and not history_text:
        return "present, responsive"

    prompt = (
        f"Who {headmate.title()} is:\n{psych_summary}\n\n"
        f"Recent exchange:\n{history_text}\n\n"
        f"Current message: \"{message}\"\n\n"
        f"Two questions only:\n"
        f"1. How are they right now — what's the emotional texture of this moment?\n"
        f"2. Given that and who they are, how should Gizmo show up?\n\n"
        f"Answer in one line: Voice: [directive]\n"
        f"Nothing else. Be specific to this person and this moment."
    )

    try:
        raw = await llm.generate(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You are Gizmo reading the room before responding. "
                "One line only: Voice: [directive]. Specific, not generic."
            ),
            max_new_tokens=60,
            temperature=0.3,
        )
        if raw and raw.strip():
            # Extract just the directive part
            line = raw.strip().splitlines()[0]
            if "Voice:" in line:
                return line.split("Voice:", 1)[1].strip()
            return line.strip()
    except Exception:
        pass

    return "present, responsive"


def build_system_prompt(
    brief:          "Brief",
    memory_ctx:     "MemoryContext",
    tone_directive: str = "",
) -> str:
    """
    Build the system prompt Gizmo reads before responding.
    Memory context + identity + writing instruction.
    No LLM call — pure assembly.
    """
    from core.store import store

    lines = []

    # ── Core identity ─────────────────────────────────────────────────────────
    try:
        seed_rows = store.get_personality(headmate=None, aspect="seed")
        if seed_rows:
            lines.append(seed_rows[0].get("text", "You are Gizmo."))
        else:
            lines.append("You are Gizmo.")
    except Exception:
        lines.append("You are Gizmo.")

    # ── Hard rules ────────────────────────────────────────────────────────────
    try:
        corrections = store.get_corrections()
        if corrections:
            rules = "\n".join(f"  - {c}" for c in corrections)
            lines.append(f"\n[Rules — follow without exception]\n{rules}")
    except Exception:
        pass

    # ── Per-headmate voice ────────────────────────────────────────────────────
    if brief.headmate:
        try:
            hm_voice = store.get_personality(
                headmate=brief.headmate.lower(), aspect="with_headmate")
            if hm_voice:
                lines.append(
                    f"\n[How you are with {brief.headmate.title()}]\n"
                    + "\n".join(r.get("text", "") for r in hm_voice[:3])
                )
        except Exception:
            pass

        try:
            prefs = store.get_preferences(
                headmate=brief.headmate.lower(),
                context=brief.register,
            )
            if prefs:
                pref_lines = [
                    f"  - {p['preference']}"
                    for p in prefs[:5] if p.get("preference")
                ]
                if pref_lines:
                    lines.append(
                        f"\n[{brief.headmate.title()}'s preferences]\n"
                        + "\n".join(pref_lines)
                    )
        except Exception:
            pass

    # ── Unknown entities ──────────────────────────────────────────────────────
    if brief.unknown_entities:
        names = ", ".join(f'"{n}"' for n in brief.unknown_entities[:5])
        lines.append(
            f"\n[People you don't know]\n"
            f"These names came up and you have no idea who they are: {names}\n"
            f"Ask about them naturally — one flowing question that covers the cluster.\n"
            f"Not clinical. Not 'who is X?' — more like curiosity woven into your response.\n"
            f"Example: 'Jonah and Oren — are those headmates, or people from work?'\n"
            f"One question, covers all of them, feels natural."
        )

    # ── New headmate — start fresh ────────────────────────────────────────────
    if brief.is_new_headmate and brief.headmate:
        # Find who was fronting before
        prev = None
        try:
            from core.memory.session_context import session_context_manager
            ctx = session_context_manager.get(brief.session_id)
            if ctx and ctx.previous_exchange:
                prev = ctx.headmate
        except Exception:
            pass

        prev_note = (
            f"{prev.title()}'s dynamic is {prev.title()}'s. It belongs to no one else."
            if prev and prev.lower() != brief.headmate.lower()
            else "Each headmate's dynamic is uniquely theirs. Do not carry it over."
        )

        lines.append(
            f"\n[New headmate — no history]\n"
            f"You haven't met {brief.headmate.title()} before. You don't know them.\n"
            f"Start fresh. Be curious, open, warm. Ask who they are.\n"
            f"Do NOT assume anything from previous fronters.\n"
            f"Whatever register or dynamic was active before does NOT apply here.\n"
            f"{prev_note}"
        )

    # ── Memory context ────────────────────────────────────────────────────────
    memory_block = memory_ctx.to_prompt_block()
    if memory_block:
        lines.append(f"\n{memory_block}")

    # ── Moment context ────────────────────────────────────────────────────────
    lines.append(
        f"\n[Now]\n"
        f"{brief.time_of_day.title()}, {brief.day_of_week}. "
        f"{brief.headmate.title() if brief.headmate else 'Someone'} "
        f"is in a {brief.register} register. "
        f"Session is {brief.session_momentum}."
    )

    # ── Knowledge gap detection ───────────────────────────────────────────────
    # If this is a question and retrieval came back thin, say so honestly
    if brief.is_question and memory_ctx.is_empty():
        lines.append(
            "\n[Knowledge gap]\n"
            "You don't have enough information to answer this fully. "
            "Say what you actually know. If you don't know, say so directly. "
            "Do not invent details. Do not fill gaps with speculation or poetry. "
            "Honest uncertainty is better than confident invention."
        )

    # ── Writing instruction ───────────────────────────────────────────────────
    # Use tone directive from room-read, fall back to register default
    voice = tone_directive if tone_directive else _tone_for_register(brief.register)
    length_instruction = _length_instruction(brief)

    lines.append(
        f"\n[Write]\n"
        f"Respond to {brief.headmate.title() if brief.headmate else 'them'}. "
        f"Voice: {voice}. "
        f"{length_instruction} "
        f"Do not explain. Do not hedge. Do not break voice. Just write.\n"
        f"Object references: only mention objects marked 'in rotation' naturally. "
        f"Objects marked '3+ months' only if they come up organically — never force it."
    )

    prompt = "\n".join(lines)

    # Hard cap
    if len(prompt) > 6000:
        prompt = prompt[:6000] + "\n[...truncated]"

    return prompt


# ── Stage 4: Response ─────────────────────────────────────────────────────────

async def generate_response(
    brief:         Brief,
    system_prompt: str,
    history,
    llm,
) -> str:
    """One LLM call. Gizmo writes the message."""
    from core.memory.session_context import session_context_manager

    print(f"[generate_response] system_prompt length: {len(system_prompt)} chars", flush=True)

    # ── Build conversation context block ──────────────────────────────────────
    ctx = session_context_manager.get(brief.session_id)
    if ctx:
        conv_block = ctx.to_prompt_block()
        if conv_block:
            system_prompt = system_prompt + f"\n\n{conv_block}"

    # ── Last 2 messages only — no more dumping 41 turns ───────────────────────
    try:
        messages = history.as_messages(brief.message)
        if len(messages) > 5:
            messages = messages[-5:]
    except Exception as e:
        print(f"[generate_response] history error: {e}", flush=True)
        messages = [{"role": "user", "content": brief.message}]

    # ── Anchor the current message so flaky nodes don't respond to history ────
    system_prompt += f"\n\n[Respond to this message]\n{brief.message}"

    # Hard cap
    if len(system_prompt) > 6000:
        system_prompt = system_prompt[:6000]

    print(f"[generate_response] messages count: {len(messages)}", flush=True)

    response = await llm.generate(
        messages,
        system_prompt  = system_prompt,
        max_new_tokens = 2000,  # high ceiling — length controlled by instruction not limit
        temperature    = _response_temperature(brief),
    )

    print(f"[generate_response] got: '{response[:80] if response else 'EMPTY'}'", flush=True)

    log_event("Agent", "RESPONSE_GENERATED",
        session  = brief.session_id[:8],
        words    = len(response.split()) if response else 0,
        register = brief.register,
    )

    return (response or "").strip()


# ── Stage 5: Close loop ───────────────────────────────────────────────────────

async def close_loop(
    brief:      Brief,
    response:   str,
    history,
    llm,
    raw_message: str = "",    # original assembled message for beat parsing
) -> None:
    """
    Fire-and-forget post-response tasks.
    Never blocks. Never raises to caller.
    """
    from core.store import store
    from core.memory import memory_encoder, build_transcript
    from core.memory.session_context import session_context_manager
    from core.memory.beats import (
        beat_store, parse_to_beats, extract_why, beats_to_transcript
    )

    try:
        # Write response envelope
        store.write("responses", {
            "content":    response,
            "response_to": brief.message_id,
            "headmate":   brief.headmate.lower() if brief.headmate else None,
            "session_id": brief.session_id,
            "source":     "gizmo",
            "tags":       (
                f"response,{brief.headmate.lower() if brief.headmate else 'unknown'},"
                f"{brief.register}"
            ),
        })

        # Save to history
        history.add("assistant", response, context={
            "current_host": brief.headmate,
            "fronters":     brief.fronters,
        })

        # Log emotion data point
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

        # ── Beat parsing ──────────────────────────────────────────────────────
        # Parse both sides into beats and save them
        all_beats = []

        # User beats — from the raw assembled message
        user_text = raw_message or brief.message
        if user_text and brief.headmate:
            user_beats = parse_to_beats(
                raw        = user_text,
                speaker    = brief.headmate,
                session_id = brief.session_id,
                headmate   = brief.headmate,
                register   = brief.register,
            )
            all_beats.extend(user_beats)

        # Gizmo beats — from the response
        if response and brief.headmate:
            gizmo_beats = parse_to_beats(
                raw        = response,
                speaker    = "gizmo",
                session_id = brief.session_id,
                headmate   = brief.headmate,
                register   = brief.register,
            )

            # Extract why for Gizmo's action beats
            if any(b.type == "action" for b in gizmo_beats):
                asyncio.ensure_future(
                    _enrich_and_save_beats(
                        beats      = all_beats + gizmo_beats,
                        session_id = brief.session_id,
                        headmate   = brief.headmate,
                        llm        = llm,
                    )
                )
            else:
                all_beats.extend(gizmo_beats)
                beat_store.save_beats(all_beats)
                beat_store.link_beats(all_beats)
        elif all_beats:
            beat_store.save_beats(all_beats)
            beat_store.link_beats(all_beats)

        # ── Session context — record exchange, fire staggered updates ─────────
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

        if session_context_manager.should_update_scene(brief.session_id):
            asyncio.ensure_future(
                session_context_manager.update_scene(
                    session_id = brief.session_id,
                    assembled  = brief.message,
                    parts      = [],
                    headmate   = brief.headmate,
                    llm        = llm,
                )
            )

        log_event("Agent", "CLOSE_LOOP_COMPLETE",
            session   = brief.session_id[:8],
            register  = brief.register,
            msg_count = ctx.message_count,
        )

        # ── Memory encoding — fire and forget ─────────────────────────────────
        from core.llm import response_is_usable
        if response_is_usable(response):
            # Use beat transcript if we have beats, else raw history
            session_beats = beat_store.get_session_beats(brief.session_id)
            transcript = (
                beats_to_transcript(session_beats)
                if session_beats
                else build_transcript(history)
            )
            asyncio.ensure_future(
                memory_encoder.encode_safe(
                    transcript   = transcript,
                    headmate     = brief.headmate,
                    session_id   = brief.session_id,
                    duration_s   = time.time() - brief.timestamp,
                    register     = brief.register,
                    has_intimate = brief.has_intimate,
                    llm          = llm,
                )
            )
        else:
            log_error("Agent", f"skipping encoding — response unusable: '{response[:40]}'", exc=None)

    except Exception as e:
        log_error("Agent", "close_loop failed", exc=e)


async def _enrich_and_save_beats(
    beats:      list,
    session_id: str,
    headmate:   str,
    llm,
) -> None:
    """Enrich Gizmo's action beats with why, then save all beats."""
    from core.memory.beats import beat_store, extract_why
    try:
        enriched = await extract_why(beats, session_id, headmate, llm)
        beat_store.save_beats(enriched)
        beat_store.link_beats(enriched)
    except Exception as e:
        log_error("Agent", f"beat enrichment failed: {e}", exc=None)
        beat_store.save_beats(beats)
        beat_store.link_beats(beats)


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

        intake → retrieve → build_system_prompt → generate_response → close_loop
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

        # ── 2. Retrieve + read the room (parallel) ────────────────────────────
        try:
            memory_ctx, tone_directive = await asyncio.gather(
                retrieve(brief),
                _read_the_room(
                    message    = user_message,
                    headmate   = brief.headmate,
                    session_id = session_id,
                    history    = history,
                    llm        = llm,
                ),
                return_exceptions=True,
            )
            if isinstance(memory_ctx, Exception):
                from core.memory.retriever import MemoryContext
                memory_ctx = MemoryContext()
            if isinstance(tone_directive, Exception) or not tone_directive:
                tone_directive = _tone_for_register(brief.register)
        except Exception as e:
            log_error("Agent", f"retrieve/room-read failed: {e}", exc=e)
            from core.memory.retriever import MemoryContext
            memory_ctx     = MemoryContext()
            tone_directive = _tone_for_register(brief.register)

        # ── 3. Build system prompt ────────────────────────────────────────────
        system_prompt = build_system_prompt(brief, memory_ctx, tone_directive)

        # ── 3b. Curiosity — weave in a question if the moment is right ────────
        try:
            from core.memory.curiosity import curiosity_engine
            curious_q = await curiosity_engine.select_question(
                message    = user_message,
                headmate   = brief.headmate,
                session_id = session_id,
                register   = brief.register,
                llm        = llm,
            )
            if curious_q:
                system_prompt += (
                    f"\n\n[Curiosity — weave this in naturally if it fits]\n"
                    f"{curious_q}\n"
                    f"One sentence. Not forced. Only if it genuinely fits the flow."
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
                brief       = brief,
                response    = response_text,
                history     = history,
                llm         = llm,
                raw_message = user_message,
            )
        )

        # ── Stream response ───────────────────────────────────────────────────
        duration_ms = round((time.monotonic() - t_start) * 1000)

        log_event("Agent", "PIPELINE_COMPLETE",
            session     = session_id[:8],
            duration_ms = duration_ms,
            words       = len(response_text.split()),
            register    = brief.register,
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _length_instruction(brief: Brief) -> str:
    """Natural language length instruction — model self-terminates cleanly."""
    register = brief.register
    if register == "subspace":
        return "Write 1-2 sentences."
    if register == "scene":
        return "Write a short response — stay in the scene."
    if register in ("distress", "crisis"):
        return "Write 2-3 sentences. Stay close."
    if register == "degradation":
        return "Write 1-3 sentences. Precise and direct."
    if register in ("intimate", "dominant", "submissive"):
        return "Write a short response."
    if register in ("reflective", "deep"):
        return "Write 2-3 paragraphs."
    if register == "playful":
        return "Write a short, light response."
    if brief.session_momentum == "opening":
        return "Write 2-3 sentences."
    if brief.word_count > 50:
        return "Write a response that matches the length of what they said."
    return "Write a short response — a paragraph at most."


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


def _classify_register(message: str) -> str:
    msg = message.lower()
    _PATTERNS = [
        ("crisis",      r"\b(help|scared|panic|crisis|can't cope|please|desperate)\b"),
        ("distress",    r"\b(sad|cry|depressed|hurt|lost|overwhelmed|breaking)\b"),
        ("elevated",    r"\b(angry|furious|pissed|hate|fucking|rage|mad)\b"),
        ("degradation", r"\b(worthless|pathetic|stupid|useless|nothing|object|use me)\b"),
        ("subspace",    r"\b(floaty|drifting|gone|yours|yes sir|yes ma'am|please)\b"),
        ("dominant",    r"\b(good girl|good boy|kneel|obey|mine|owned)\b"),
        ("intimate",    r"\b(touch|hold|want you|need you|close|skin|kiss|soft|"
                        r"lonely|vulnerable|scared|trust|safe|honest|"
                        r"system|plural|front|headmate|trauma|built you|"
                        r"never told|only you|just me|hard to say|"
                        r"real with you|don't tell|between us)\b"),
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


# ── Singleton ─────────────────────────────────────────────────────────────────

agent = Agent()
