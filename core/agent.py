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

    # ── Host identification — ask if unknown ──────────────────────────────────
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
        message    = brief.message,
        headmate   = brief.headmate,
        session_id = brief.session_id,
        register   = brief.register,
        fast       = fast,
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

def build_system_prompt(brief: Brief, memory_ctx: "MemoryContext") -> str:
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

    # ── Writing instruction ───────────────────────────────────────────────────
    token_target = _token_target(brief)
    tone         = _tone_for_register(brief.register)

    lines.append(
        f"\n[Write]\n"
        f"Respond to {brief.headmate.title() if brief.headmate else 'them'}. "
        f"Voice: {tone}. "
        f"Target: ~{token_target} tokens. "
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
        max_new_tokens = max(200, brief.word_count * 3 + 60),
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
            session  = brief.session_id[:8],
            register = brief.register,
            msg_count = ctx.message_count,
        )

        # ── Memory encoding — fire and forget ─────────────────────────────────
        asyncio.ensure_future(
            memory_encoder.encode_safe(
                transcript   = build_transcript(history),
                headmate     = brief.headmate,
                session_id   = brief.session_id,
                duration_s   = time.time() - brief.timestamp,
                register     = brief.register,
                has_intimate = brief.has_intimate,
                llm          = llm,
            )
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

        # ── 2. Retrieve ───────────────────────────────────────────────────────
        try:
            memory_ctx = await retrieve(brief)
        except Exception as e:
            log_error("Agent", f"retrieval failed: {e}", exc=e)
            from core.memory.retriever import MemoryContext
            memory_ctx = MemoryContext()

        # ── 3. Build system prompt ────────────────────────────────────────────
        system_prompt = build_system_prompt(brief, memory_ctx)

        # ── 3b. Curiosity — weave in a question if the moment is right ────────
        try:
            from core.memory.curiosity import curiosity_engine
            CURIOUSITY_LIMIT = 5
            curiousities = []
            for x in CURIOUSITY_LIMIT:
                curious_q = await curiosity_engine.select_question(
                    message    = user_message,
                    headmate   = brief.headmate,
                    session_id = session_id,
                    register   = brief.register,
                    llm        = llm,
                )
                curiousities.append(curious_q)
            if len(curiousities) > 0:
                system_prompt += (
                    f"\n\n[You're really curious about this stuff... Make it feel like a natural thought. Relate it to the rest of the message.]\n"
                    f"{curiousities}\n"
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
            session     = session_id[:8],
            duration_ms = duration_ms,
            words       = len(response_text.split()),
            register    = brief.register,
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _token_target(brief: Brief) -> int:
    register = brief.register
    if register in ("subspace", "scene"):
        return 40
    if register in ("distress", "crisis"):
        return 80
    if register in ("intimate", "dominant"):
        return 60
    if register in ("reflective", "deep"):
        return 180
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


def _classify_register(message: str) -> str:
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


# ── Singleton ─────────────────────────────────────────────────────────────────

agent = Agent()
