"""
core/agent.py
Gizmo's pipeline orchestrator.

Coordinates all processing stages for every incoming message.
Does not think. Does not retrieve. Does not generate.
Routes, sequences, and streams. That's all.

Pipeline per message:

  1. INTAKE
     Receive message → extract intelligence → build full picture (brief)
     Writes inbound message envelope to store.

  2. PARALLEL AGENTS (all run simultaneously, all read from brief)
     a. Knowledge  — what do we know about this topic/person/moment
     b. Wellness   — current state, immediate needs, active flags
     c. Therapy    — longitudinal patterns, feed/break recommendation
     d. Narrative  — conversational arc, live thread, where this is going

  3. DIRECTOR
     Merges four agent outputs → single clear directive
     "This is what you say. This is what you do."
     Structured output: meaning, actions, tone, token target.

  4. PERSONALITY
     No LLM call. Reads store.
     Assembles the most precise system prompt ever written for this
     specific headmate in this specific moment.
     Determines HOW the directive is expressed, not WHAT.

  5. RESPONSE
     One LLM call. Writes the message.
     Given a perfect brief, even a small model does well here.
     Streams tokens to caller.

  6. CLOSE LOOP (fire and forget)
     Writes outbound response envelope to store.
     Pattern detector checks for firing patterns.
     Emotion log updated.
     Questions queued if gaps identified.

Wall-clock time = intake + max(parallel agents) + director + response
                ≈ three sequential hops, not seven.
"""

from __future__ import annotations

import asyncio
import json
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
    Passed to all four parallel agents unchanged.
    """
    # Message
    message:        str
    session_id:     str
    message_id:     str             # store ID of the written message
    timestamp:      float

    # Identity
    headmate:       Optional[str]
    fronters:       list

    # Classification
    register:       str
    topics:         list
    emotional_valence: float
    is_question:    bool
    is_correction:  bool
    word_count:     int

    # Time context
    time_of_day:    str
    day_of_week:    str
    day_type:       str             # weekday / weekend
    since_last_msg: float           # seconds
    message_cadence: str

    # Session context
    session_momentum: str
    session_duration: float         # seconds
    fronting_duration: float        # seconds headmate has been fronting

    # Extracted intelligence (from intake)
    subjects:       list = field(default_factory=list)
    new_facts:      list = field(default_factory=list)
    needs_active:   list = field(default_factory=list)
    has_intimate:   bool = False
    stress_level:   str  = "unknown"
    vibe:           list = field(default_factory=list)

    # Previous response ID (for outcome filling)
    prev_response_id: Optional[str] = None

    # Unknown entities — things mentioned Gizmo has never seen before
    # Each entry: {"name": str, "entity_type": str, "context": str}
    unknown_entities: list = field(default_factory=list)

    # Host identification — set when Gizmo needs to ask who's there
    # If set, agent.respond() returns this directly, skips full pipeline
    host_question: Optional[str] = None


# ── Directive dataclass ───────────────────────────────────────────────────────

@dataclass
class Directive:
    """
    Director output. What Gizmo says and does, precisely specified.
    Personality layer reads this and builds the final system prompt.
    """
    # What to say / do
    meaning:        str             # the core meaning to convey
    actions:        list            # specific things to do/reference/deploy
    suppress:       list            # things explicitly not to do

    # How to say it
    tone:           str             # dominant/warm/playful/grounding/etc.
    register:       str             # intimate/casual/serious/etc.
    token_target:   int             # how long the response should be

    # Pattern guidance
    pattern_action: Optional[str]   # feed/break/hold/None
    push_to:        Optional[float] # intensity target if feeding
    watch_for:      list            # edge indicators to log

    # Narrative thread
    thread:         str             # where this conversation is going
    knowledge_to_use: list          # specific facts to weave in

    # Flags
    check_in:       bool  = False
    check_in_style: str   = ""
    therapy_flag:   bool  = False
    log_conditions: bool  = False   # log what's happening for therapy model

    # Unknown entity asks — weave naturally into response
    # Each entry: {"name": str, "style": str}
    ask_about:      list  = field(default_factory=list)


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


def _parse_json(raw: str, fallback: dict) -> dict:
    """Parse JSON from LLM output. Returns fallback on failure."""
    if not raw:
        return fallback
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception:
        return fallback


# ── Stage 1: Intake ───────────────────────────────────────────────────────────

async def intake(
    message:    str,
    session_id: str,
    context:    dict,
    history,
    llm,
) -> Brief:
    """
    Receive message, extract intelligence, build full picture.
    Writes inbound envelope to store.
    Returns Brief passed to all downstream agents.
    """
    from core.store import store
    from core.timezone import tz_now, get_timezone
    from core.session_manager import session_manager

    ts  = time.time()
    now = tz_now()

    # ── Identity ──────────────────────────────────────────────────────────────
    # Pull from session manager first — it's the live source of truth
    ctx_live = session_manager.get_session_context(session_id)
    headmate = ctx_live.get("current_host") or context.get("current_host") or ""
    fronters = ctx_live.get("fronters") or list(context.get("fronters") or [])
    if headmate and headmate not in [f.lower() for f in fronters]:
        fronters.insert(0, headmate)

    # ── Time context ──────────────────────────────────────────────────────────
    hour     = now.hour
    weekday  = now.weekday()

    time_of_day = (
        "morning"   if 5  <= hour < 12 else
        "afternoon" if 12 <= hour < 17 else
        "evening"   if 17 <= hour < 21 else
        "night"
    )
    day_type = "weekend" if weekday >= 5 else "weekday"

    # Time since last message
    since_last   = 0.0
    prev_resp_id = None
    recent = store.get_recent_messages(headmate=headmate or None, limit=2)
    if recent:
        since_last = ts - recent[0].get("created_at", ts)
        last_resp  = store.get_last_response(session_id)
        if last_resp:
            prev_resp_id = last_resp["id"]

    cadence = (
        "rapid"         if since_last < 15   else
        "fast"          if since_last < 60   else
        "conversational" if since_last < 300 else
        "slow"          if since_last < 1800 else
        "returning"
    )

    # ── Session context ───────────────────────────────────────────────────────
    session_dur   = 0.0
    fronting_dur  = 0.0
    sess_momentum = "opening"

    try:
        sess = session_manager._sessions.get(session_id)
        if sess:
            session_dur  = ts - sess.opened_at
            fronting_dur = ts - sess.host_updated_at
    except Exception:
        pass

    # Momentum from message count
    try:
        msg_count = store.count("messages", session_id=session_id)
        sess_momentum = (
            "opening"    if msg_count <= 2  else
            "building"   if msg_count <= 8  else
            "engaged"    if msg_count <= 20 else
            "deep"
        )
    except Exception:
        pass

    # ── Heuristic classification ──────────────────────────────────────────────
    register    = _classify_register(message)
    topics      = _classify_topics(message)
    is_question = "?" in message or message.lower().startswith(
        ("what", "how", "why", "where", "when", "who", "can", "could", "do ", "did ", "is ", "are ")
    )
    is_correction = any(p in message.lower() for p in (
        "don't", "stop", "never", "wrong", "that's not", "incorrect",
        "you said", "you keep", "you always", "please don't",
    ))
    word_count    = len(message.split())
    has_intimate  = register in (
        "intimate", "dominant", "submissive", "subspace",
        "scene", "erotic", "sensual", "degradation",
    )

    # ── LLM extraction — runs async in close_loop after response ─────────────
    # Heuristic only during intake to keep pipeline fast
    extracted = {
        "subjects": [], "relationships": [], "new_facts": [],
        "needs_active": [], "vibe": [], "stress_level": "unknown",
        "valence": 0.0, "wellbeing_observations": [],
    }

    # ── Session context ───────────────────────────────────────────────────────
    sess_ctx = session_manager._sessions.get(session_id)

    # ── Host identification — ask if unknown ─────────────────────────────────
    _host_question = None
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

    # ── Auto-detect new arrivals from secondary boxes ─────────────────────────
    # Any name in the message parts that isn't in the store gets flagged
    # session_manager.process_message_entities handles this via _is_unknown
    # but we also handle explicit new names from multi-part exchanges here

    # ── Session manager entity processing ────────────────────────────────────
    # Skipping LLM entity extraction for now to reduce concurrent API calls
    # Heuristic classification in _classify_topics handles basic topic detection
    unknown_entities = []

    # ── Write message envelope to store ───────────────────────────────────────
    tags = list(set(
        topics +
        ([headmate.lower()] if headmate else []) +
        [register, day_type, time_of_day] +
        (["intimate"] if has_intimate else []) +
        (["question"] if is_question else []) +
        (["correction"] if is_correction else [])
    ))

    message_id = store.write("messages", {
        "content":           message,
        "headmate":          headmate.lower() if headmate else None,
        "fronters":          fronters,
        "session_id":        session_id,
        "register":          register,
        "topics":            topics,
        "emotional_valence": extracted.get("valence", 0.0),
        "subjects":          extracted.get("subjects", []),
        "new_facts":         extracted.get("new_facts", []),
        "needs_active":      extracted.get("needs_active", []),
        "has_intimate":      1 if has_intimate else 0,
        "vibe":              extracted.get("vibe", []),
        "stress_level":      extracted.get("stress_level", "unknown"),
        "time_of_day":       time_of_day,
        "session_momentum":  sess_momentum,
        "source":            "user",
        "tags":              ",".join(tags),
    })

    # ── Fill outcome on previous response ─────────────────────────────────────
    if prev_resp_id:
        outcome, signal = _infer_outcome(
            message, extracted.get("valence", 0.0), register
        )
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

    # ── Write facts from extraction ───────────────────────────────────────────
    for fact in extracted.get("new_facts", []):
        if fact and len(fact) > 5:
            store.write("facts", {
                "fact":       fact,
                "headmate":   headmate.lower() if headmate else None,
                "fact_type":  "observation",
                "register":   register,
                "context":    time_of_day,
                "session_id": session_id,
                "source":     "extractor",
                "tags":       f"fact,{headmate.lower() if headmate else 'unknown'},{register}",
            })

    # ── Write relationships ───────────────────────────────────────────────────
    for rel in extracted.get("relationships", []):
        try:
            store.write("relationships", {
                "speaker":                  rel.get("speaker", headmate or "unknown"),
                "entity":                   rel.get("entity", ""),
                "entity_type":              rel.get("entity_type", "unknown"),
                "relationship_label":       rel.get("label", ""),
                "relationship_category":    rel.get("category", "social_bond"),
                "confidence_type":          rel.get("confidence", "stated"),
                "hearsay_source":           rel.get("hearsay_source"),
                "hearsay_about":            rel.get("hearsay_about"),
                "intimate":                 1 if rel.get("intimate") else 0,
                "headmate":                 headmate.lower() if headmate else None,
                "session_id":               session_id,
                "source":                   "extractor",
                "tags":                     f"relationship,{rel.get('label', '')}",
            })
        except Exception:
            pass

    # ── Write wellbeing observations ──────────────────────────────────────────
    for obs in extracted.get("wellbeing_observations", []):
        try:
            store.write("wellbeing", {
                "headmate":   headmate.lower() if headmate else None,
                "category":   obs.get("category", "pattern"),
                "observation": obs.get("observation", ""),
                "context":    obs.get("context"),
                "register":   register,
                "session_id": session_id,
                "source":     "extractor",
                "confidence": 0.6,
                "tags":       f"wellbeing,{obs.get('category','pattern')},{headmate.lower() if headmate else 'unknown'}",
            })
        except Exception:
            pass

    # ── If we need to identify the host, return that question directly ────────
    # Skip the full pipeline — this is the only response that matters right now
    if _host_question:
        log_event("Agent", "INTAKE_ASKING_HOST",
            session=session_id[:8],
            question=_host_question,
        )
        # Still write the message envelope so it's in history
        # but return early — the brief is populated enough for this
        return Brief(
            message=message,
            session_id=session_id,
            message_id=message_id,
            timestamp=ts,
            headmate=None,
            fronters=fronters,
            register=register,
            topics=topics,
            emotional_valence=0.0,
            is_question=False,
            is_correction=False,
            word_count=word_count,
            time_of_day=time_of_day,
            day_of_week=now.strftime("%A"),
            day_type=day_type,
            since_last_msg=since_last,
            message_cadence=cadence,
            session_momentum="opening",
            session_duration=0.0,
            fronting_duration=0.0,
            host_question=_host_question,  # signal to agent.respond()
        )

    log_event("Agent", "INTAKE_COMPLETE",
        session=session_id[:8],
        headmate=headmate or "unknown",
        register=register,
        message_id=message_id[:12],
        topics=topics[:3],
    )

    return Brief(
        message=message,
        session_id=session_id,
        message_id=message_id,
        timestamp=ts,
        headmate=headmate or None,
        fronters=fronters,
        register=register,
        topics=topics,
        emotional_valence=extracted.get("valence", 0.0),
        is_question=is_question,
        is_correction=is_correction,
        word_count=word_count,
        time_of_day=time_of_day,
        day_of_week=now.strftime("%A"),
        day_type=day_type,
        since_last_msg=since_last,
        message_cadence=cadence,
        session_momentum=sess_momentum,
        session_duration=session_dur,
        fronting_duration=fronting_dur,
        subjects=extracted.get("subjects", []),
        new_facts=extracted.get("new_facts", []),
        needs_active=extracted.get("needs_active", []),
        has_intimate=has_intimate,
        stress_level=extracted.get("stress_level", "unknown"),
        vibe=extracted.get("vibe", []),
        prev_response_id=prev_resp_id,
        unknown_entities=unknown_entities,
    )


# ── Stage 2a: Knowledge agent ─────────────────────────────────────────────────

async def agent_knowledge(brief: Brief, llm) -> dict:
    """
    What do we know about this topic, this person, this moment?
    Reads facts, sessions, relationships, entity data from store.
    Entity summary comes from session manager cache — already warm.
    Returns synthesized knowledge block.
    """
    from core.store import store
    from core.session_manager import session_manager

    if not brief.headmate:
        return {"synthesis": "", "confidence": 0.0, "facts": []}

    # Entity summary from session manager cache — no store fetch needed
    entity_summary = session_manager.get_entity_summary(brief.session_id)

    # Hot facts pre-loaded by session manager
    sess_ctx  = session_manager._sessions.get(brief.session_id)
    hot_facts = sess_ctx.hot_facts[:8] if sess_ctx else []

    # Semantic search — only if we have data
    search_results = []
    try:
        fact_count = store.count("facts", headmate=brief.headmate.lower())
        if fact_count > 0:
            search_results = store.search(
                query=brief.message,
                tables=["facts", "messages", "reflections"],
                headmate=brief.headmate.lower(),
                limit=6,
            )
    except Exception:
        pass

    # Recent sessions context
    today_sessions = store.get_today_sessions(headmate=brief.headmate)

    # Relationships from store
    relationships = store.get_active("relationships",
        headmate=brief.headmate.lower(), limit=8)

    facts_text = "\n".join(f"- {f}" for f in hot_facts)
    search_text = "\n".join(
        f"- [{r.get('_table','?')}] {r.get('fact') or r.get('content','')[:100]}"
        for r in search_results[:5]
    )
    rel_text = "\n".join(
        f"- {r['speaker']} → {r['relationship_label']} → {r['entity']}"
        for r in relationships[:5]
        if r.get("relationship_label")
    )
    sessions_text = "\n".join(
        f"- {s.get('mood','?')} session: {s.get('summary','')[:100]}"
        for s in today_sessions[-3:]
        if s.get("summary")
    )

    if not any([facts_text, search_text, rel_text, sessions_text, entity_summary]):
        return {"synthesis": "", "confidence": 0.0, "facts": []}

    raw = await _call(llm,
        system=(
            "You synthesize knowledge about a person from memory fragments. "
            "Extract only what's directly relevant to the current message. "
            "Be specific and factual. No invented details. JSON only."
        ),
        user=f"""Current message: "{brief.message}"
Headmate: {brief.headmate}
Register: {brief.register}
Topics: {', '.join(brief.topics[:4])}

{entity_summary or ''}

Known facts:
{facts_text or '(none)'}

Search results:
{search_text or '(none)'}

Relationships:
{rel_text or '(none)'}

Today's sessions:
{sessions_text or '(none)'}

Return JSON:
{{
  "synthesis": "2-4 sentence synthesis of what's relevant right now",
  "key_facts": ["specific facts worth weaving into the response"],
  "confidence": 0.0-1.0,
  "gaps": ["things we don't know that would be useful"]
}}""",
        tokens=400,
        temp=0.1,
    )

    result = _parse_json(raw, {
        "synthesis": "", "key_facts": [], "confidence": 0.0, "gaps": []
    })

    log_event("Agent", "KNOWLEDGE_COMPLETE",
        session=brief.session_id[:8],
        confidence=result.get("confidence", 0.0),
        facts=len(result.get("key_facts", [])),
    )

    return result


# ── Stage 2b: Wellness agent ──────────────────────────────────────────────────

async def agent_wellness(brief: Brief, llm) -> dict:
    """
    How is she right now? What does she need? Any immediate flags?
    Current state only — not patterns over time (that's therapy's job).
    """
    from core.store import store

    if not brief.headmate:
        return {"state": "unknown", "needs": [], "flags": [], "note": None}

    # Current wellbeing observations
    needs_emotional = store.get_wellbeing(
        brief.headmate, category="emotional_need", limit=5)
    needs_physical  = store.get_wellbeing(
        brief.headmate, category="physical_need", limit=3)
    limits          = store.get_wellbeing(
        brief.headmate, category="limit", limit=10)
    what_works      = store.get_wellbeing(
        brief.headmate, category="works", limit=5)

    # Entity baseline for comparison
    entity = store.get_entity(brief.headmate)
    baseline_vibe = entity.get("baseline_vibe", []) if entity else []

    raw = await _call(llm,
        system=(
            "You assess someone's current emotional and physical state. "
            "You are direct, specific, and non-judgmental. "
            "Intimate data is valid data — don't sanitize. JSON only."
        ),
        user=f"""Headmate: {brief.headmate}
Current register: {brief.register}
Current vibe: {', '.join(brief.vibe) if brief.vibe else 'unknown'}
Baseline vibe: {', '.join(baseline_vibe) if baseline_vibe else 'unknown'}
Stress level: {brief.stress_level}
Valence: {brief.emotional_valence:+.2f}
Time of day: {brief.time_of_day} ({brief.day_of_week})
Message: "{brief.message}"

Known emotional needs: {', '.join(o['observation'] for o in needs_emotional) or 'none on file'}
Known physical needs:  {', '.join(o['observation'] for o in needs_physical) or 'none on file'}
Known limits:          {', '.join(o['observation'] for o in limits) or 'none on file'}
What works:            {', '.join(o['observation'] for o in what_works) or 'none on file'}

Return JSON:
{{
  "state": "one phrase describing current state",
  "needs_active": ["what she needs right now"],
  "flags": ["anything that warrants attention"],
  "what_works_now": ["approaches likely to land in this state"],
  "avoid_now": ["approaches to avoid in this state"],
  "therapy_note": "note for therapy model if anything significant, else null",
  "log_conditions": true/false
}}""",
        tokens=350,
        temp=0.1,
    )

    result = _parse_json(raw, {
        "state": "unknown", "needs_active": [], "flags": [],
        "what_works_now": [], "avoid_now": [],
        "therapy_note": None, "log_conditions": False,
    })

    log_event("Agent", "WELLNESS_COMPLETE",
        session=brief.session_id[:8],
        state=result.get("state", "?"),
        flags=len(result.get("flags", [])),
    )

    return result


# ── Stage 2c: Therapy agent ───────────────────────────────────────────────────

async def agent_therapy(brief: Brief, llm) -> dict:
    """
    Longitudinal view. Patterns, trends, feed/break recommendation.
    Also runs real-time monitoring for acute signals.
    """
    from core.store import store
    from core.session_manager import session_manager
    from core.therapy import therapy_agent

    if not brief.headmate:
        return {"recommendation": "hold", "patterns": [], "note": None}

    # Real-time monitoring — acute signals, edge proximity, limits
    therapy_note = await therapy_agent.monitor_session(
        session_id=brief.session_id,
        headmate=brief.headmate,
        brief_data={
            "register":    brief.register,
            "valence":     brief.emotional_valence,
            "stress_level": brief.stress_level,
            "intensity":   _register_intensity(brief.register),
            "message":     brief.message,
        },
        llm=llm,
    )

    # Pending nudges from therapy agent
    sess_ctx = session_manager._sessions.get(brief.session_id)
    pending_nudges = []
    if sess_ctx:
        pending_nudges = [
            q for q in sess_ctx.pending_questions
            if q.get("type") == "therapy_nudge"
            and q.get("expires", 0) > time.time()
        ]

    # Active patterns from session manager cache (already detected)
    active_patterns = sess_ctx.active_patterns if sess_ctx else []
    patterns_feed   = [p for p in active_patterns if p.get("action") == "feed"]
    patterns_break  = [p for p in active_patterns if p.get("action") == "break"]

    # Also pull from store for any not yet in cache
    stored_feed  = store.get_patterns(brief.headmate, action="feed")
    stored_break = store.get_patterns(brief.headmate, action="break")
    stored_hold  = store.get_patterns(brief.headmate, action="hold", min_confidence=0.3)

    # Merge — cache takes priority
    cache_ids    = {p.get("pattern_id") for p in active_patterns}
    extra_feed   = [p for p in stored_feed  if p["id"] not in cache_ids]
    extra_break  = [p for p in stored_break if p["id"] not in cache_ids]
    all_feed     = patterns_feed + extra_feed
    all_break    = patterns_break + extra_break

    # Emotion trend
    emotion_log = store.query("emotion_log",
        headmate=brief.headmate.lower(),
        order_by="created_at DESC",
        limit=20,
    )

    declining = [
        p for p in (all_feed + all_break)
        if p.get("outcome_quality_avg", 1.0) < 0.4
        and p.get("data_points", 0) >= 3
    ]

    def _pat_summary(p: dict) -> str:
        return (
            f"{p.get('pattern_type','?')} "
            f"(conf={p.get('confidence',0):.2f}, "
            f"pts={p.get('data_points',0)}, "
            f"quality={p.get('outcome_quality_avg',0):.2f}): "
            f"{p.get('approach','?')[:60]}"
        )

    feed_text    = "\n".join(_pat_summary(p) for p in all_feed[:3])
    break_text   = "\n".join(_pat_summary(p) for p in all_break[:3])
    hold_text    = "\n".join(_pat_summary(p) for p in stored_hold[:3])
    decline_text = "\n".join(_pat_summary(p) for p in declining[:3])

    nudge_text = ""
    if pending_nudges:
        nudge_text = "\nPending therapy nudges:\n" + "\n".join(
            f"- [{n.get('urgency','?')}] {n.get('message','')[:80]}"
            for n in pending_nudges[:2]
        )

    therapy_note_text = f"\nReal-time flag: {therapy_note}" if therapy_note else ""

    if len(emotion_log) >= 4:
        r_vals = [e.get("valence", 0) for e in emotion_log[:4]]
        o_vals = [e.get("valence", 0) for e in emotion_log[4:8]]
        r_avg  = sum(r_vals) / 4
        o_avg  = sum(o_vals) / max(len(o_vals), 1)
        valence_trend = (
            "improving"  if r_avg > o_avg + 0.1  else
            "declining"  if r_avg < o_avg - 0.1  else
            "stable"
        )
    else:
        valence_trend = "insufficient data"

    raw = await _call(llm,
        system=(
            "You are a longitudinal behavioral analyst. "
            "You assess patterns across sessions and make recommendations. "
            "You never moralize. Direct, specific, data-driven. "
            "Intimate patterns are valid data. JSON only."
        ),
        user=f"""Headmate: {brief.headmate}
Current register: {brief.register}
Current message: "{brief.message}"
Time: {brief.time_of_day}, {brief.day_of_week}
Fronting duration: {brief.fronting_duration/60:.0f} minutes
{therapy_note_text}
{nudge_text}

FEED patterns:
{feed_text or '(none confirmed)'}

BREAK patterns:
{break_text or '(none confirmed)'}

HOLD patterns:
{hold_text or '(none yet)'}

DECLINING:
{decline_text or '(none)'}

Emotion trend: {valence_trend}

Return JSON:
{{
  "recommendation": "feed|break|hold|flag_therapy",
  "active_pattern_id": "pattern id if one is firing, else null",
  "push_to": intensity target if feeding (float), else null,
  "approach": "specific approach",
  "reasoning": "why",
  "watch_for": ["edge indicators"],
  "thread_after": "what typically comes after",
  "therapy_flag": true/false,
  "therapy_note": "note for therapy report if flagged, else null",
  "surface_nudge": "nudge message to surface now if urgent, else null"
}}""",
        tokens=400,
        temp=0.1,
    )

    result = _parse_json(raw, {
        "recommendation":   "hold",
        "active_pattern_id": None,
        "push_to":          None,
        "approach":         "warm and present",
        "reasoning":        "insufficient pattern data",
        "watch_for":        [],
        "thread_after":     None,
        "therapy_flag":     False,
        "therapy_note":     None,
        "surface_nudge":    None,
    })

    # If therapy flagged, write note to store
    if result.get("therapy_flag") and result.get("therapy_note") and brief.headmate:
        try:
            from core.store import store as _store
            _store.write("wellbeing", {
                "headmate":    brief.headmate.lower(),
                "category":    "pattern",
                "observation": result["therapy_note"],
                "context":     f"{brief.register} | {brief.time_of_day} | {brief.day_of_week}",
                "register":    brief.register,
                "session_id":  brief.session_id,
                "source":      "therapy_agent",
                "confidence":  0.8,
                "tags":        f"therapy_flag,{brief.headmate.lower()}",
            })
        except Exception:
            pass

    log_event("Agent", "THERAPY_COMPLETE",
        session=brief.session_id[:8],
        recommendation=result.get("recommendation", "?"),
        therapy_flag=result.get("therapy_flag", False),
        real_time_flag=bool(therapy_note),
    )

    return result


# ── Stage 2d: Narrative agent ─────────────────────────────────────────────────

async def agent_narrative(brief: Brief, history, llm) -> dict:
    """
    Conversational arc. What thread is live? Where is this going?
    What would feel natural vs jarring?
    Keeps Gizmo present in a continuous conversation, not just
    responding to individual messages.
    """
    from core.store import store

    # Recent conversation history
    try:
        recent_msgs = history.as_list()[-8:]
    except Exception:
        recent_msgs = []

    # Today's arc
    today_sessions = store.get_today_sessions(headmate=brief.headmate) if brief.headmate else []

    # Unresolved threads from recent sessions
    unresolved = [
        s.get("unresolved") for s in today_sessions
        if s.get("unresolved")
    ]

    # Format recent history
    history_text = "\n".join(
        f"{'[{:.0f}m ago] '.format((brief.timestamp - m.get('timestamp', brief.timestamp)) / 60)}"
        f"{m['role'].upper()}: {m['content'][:120]}"
        for m in recent_msgs[-6:]
        if isinstance(m, dict) and m.get("content")
    ) if recent_msgs else "(no history)"

    raw = await _call(llm,
        system=(
            "You analyze conversational arcs and narrative threads. "
            "You understand where conversations are going and what feels natural. "
            "You are concise and specific. JSON only."
        ),
        user=f"""Headmate: {brief.headmate or 'unknown'}
Current message: "{brief.message}"
Register: {brief.register}
Session momentum: {brief.session_momentum}
Time since last message: {brief.since_last_msg/60:.1f} minutes

Recent conversation:
{history_text}

Unresolved threads from today:
{chr(10).join(f'- {u}' for u in unresolved) or '(none)'}

Return JSON:
{{
  "live_thread": "what thread is currently active in this conversation",
  "arc_position": "opening|building|peak|winding_down|returning",
  "natural_next": "what would feel natural to do/say next narratively",
  "avoid_narratively": "what would feel jarring or disruptive",
  "unresolved_relevant": "any unresolved thread relevant to this message, else null",
  "conversation_going_to": "where this conversation is heading",
  "tone_arc": "how the emotional tone has been moving"
}}""",
        tokens=300,
        temp=0.15,
    )

    result = _parse_json(raw, {
        "live_thread":          "general conversation",
        "arc_position":         "building",
        "natural_next":         "respond naturally",
        "avoid_narratively":    "",
        "unresolved_relevant":  None,
        "conversation_going_to": "unknown",
        "tone_arc":             "stable",
    })

    log_event("Agent", "NARRATIVE_COMPLETE",
        session=brief.session_id[:8],
        thread=result.get("live_thread", "?")[:40],
        arc=result.get("arc_position", "?"),
    )

    return result


# ── Stage 3: Director ─────────────────────────────────────────────────────────

async def director(
    brief:     Brief,
    knowledge: dict,
    wellness:  dict,
    therapy:   dict,
    narrative: dict,
    llm,
) -> Directive:
    """
    Merges four agent outputs into one clear directive.
    "This is what you say. This is what you do."
    Small model, structured output, fast.
    """

    # Calculate token target from room check signals
    token_target = _calculate_token_target(brief, therapy, narrative)

    # Build unknown entities block for director
    unknown_block = ""
    if brief.unknown_entities:
        unk_list = "\n".join(
            f"  - \"{u['name']}\" ({u['entity_type']}) — {u['context']}"
            for u in brief.unknown_entities[:3]
        )
        unknown_block = f"\nUNKNOWN ENTITIES (never seen before — ask naturally):\n{unk_list}"

    # Surface nudge if therapy flagged one as urgent
    nudge_block = ""
    if therapy.get("surface_nudge"):
        nudge_block = (
            f"\nURGENT NUDGE (from therapy agent — weave naturally):\n"
            f"  {therapy['surface_nudge']}"
        )

    raw = await _call(llm,
        system=(
            "You are a director. You receive four intelligence reports and "
            "produce one precise directive. You decide exactly what should be "
            "said and done. You are specific, not vague. JSON only."
        ),
        user=f"""Headmate: {brief.headmate or 'unknown'}
Message: "{brief.message}"
Register: {brief.register}
Token target: {token_target}
{unknown_block}
{nudge_block}

KNOWLEDGE REPORT:
{knowledge.get('synthesis') or '(nothing relevant retrieved)'}
Key facts to use: {', '.join(knowledge.get('key_facts', [])[:3]) or 'none'}
Gaps: {', '.join(knowledge.get('gaps', [])[:2]) or 'none'}

WELLNESS REPORT:
Current state: {wellness.get('state', 'unknown')}
Active needs: {', '.join(wellness.get('needs_active', [])[:3]) or 'none'}
What works now: {', '.join(wellness.get('what_works_now', [])[:3]) or 'none'}
Avoid now: {', '.join(wellness.get('avoid_now', [])[:3]) or 'none'}
Flags: {', '.join(wellness.get('flags', [])) or 'none'}

THERAPY REPORT:
Recommendation: {therapy.get('recommendation', 'hold')}
Approach: {therapy.get('approach', 'warm and present')}
Push to: {therapy.get('push_to') or 'N/A'}
Reasoning: {therapy.get('reasoning', '')}
Watch for: {', '.join(therapy.get('watch_for', [])) or 'none'}
Thread after: {therapy.get('thread_after') or 'unknown'}

NARRATIVE REPORT:
Live thread: {narrative.get('live_thread', 'general')}
Arc position: {narrative.get('arc_position', 'building')}
Natural next: {narrative.get('natural_next', '')}
Avoid: {narrative.get('avoid_narratively', '')}
Going to: {narrative.get('conversation_going_to', '')}

Return JSON:
{{
  "meaning": "the core meaning/intent of the response in one sentence",
  "actions": ["specific things to do, reference, deploy — concrete"],
  "suppress": ["specific things not to do"],
  "tone": "dominant|warm|playful|grounding|direct|tender|fierce|etc",
  "register": "{brief.register}",
  "token_target": {token_target},
  "pattern_action": "{therapy.get('recommendation','hold')}",
  "push_to": {therapy.get('push_to') or 'null'},
  "watch_for": {json.dumps(therapy.get('watch_for', []))},
  "thread": "{narrative.get('conversation_going_to', 'unknown')}",
  "knowledge_to_use": ["specific facts to weave in"],
  "ask_about": [
    {{
      "name": "entity name to ask about",
      "style": "how to ask — casual one sentence woven in, not clinical"
    }}
  ],
  "check_in": false,
  "check_in_style": "",
  "therapy_flag": {str(therapy.get('therapy_flag', False)).lower()},
  "log_conditions": {str(wellness.get('log_conditions', False)).lower()}
}}

For ask_about: only include unknown entities that are genuinely relevant
to this response. One ask maximum. Woven naturally, never clinical.
If no unknowns or they're not relevant, return empty array.""",
        tokens=450,
        temp=0.1,
    )

    data = _parse_json(raw, {
        "meaning":        "respond naturally",
        "actions":        [],
        "suppress":       [],
        "tone":           "warm",
        "register":       brief.register,
        "token_target":   token_target,
        "pattern_action": "hold",
        "push_to":        None,
        "watch_for":      [],
        "thread":         "unknown",
        "knowledge_to_use": [],
        "check_in":       False,
        "check_in_style": "",
        "therapy_flag":   False,
        "log_conditions": False,
    })

    directive = Directive(
        meaning=data.get("meaning", "respond naturally"),
        actions=data.get("actions", []),
        suppress=data.get("suppress", []),
        tone=data.get("tone", "warm"),
        register=data.get("register", brief.register),
        token_target=int(data.get("token_target", token_target)),
        pattern_action=data.get("pattern_action"),
        push_to=data.get("push_to"),
        watch_for=data.get("watch_for", []),
        thread=data.get("thread", ""),
        knowledge_to_use=data.get("knowledge_to_use", []),
        check_in=bool(data.get("check_in", False)),
        check_in_style=data.get("check_in_style", ""),
        therapy_flag=bool(data.get("therapy_flag", False)),
        log_conditions=bool(data.get("log_conditions", False)),
        ask_about=data.get("ask_about", []),
    )

    log_event("Agent", "DIRECTOR_COMPLETE",
        session=brief.session_id[:8],
        meaning=directive.meaning[:60],
        tone=directive.tone,
        tokens=directive.token_target,
        pattern=directive.pattern_action,
    )

    return directive


# ── Stage 4: Personality layer ────────────────────────────────────────────────

def personality_layer(brief: Brief, directive: Directive) -> str:
    """
    No LLM call. Reads store. Builds the final system prompt.
    Determines HOW the directive is expressed, not WHAT.
    Gizmo's voice + this headmate's specific relationship with him
    filters the directive into a precise writing instruction.
    """
    from core.store import store

    lines = []

    # ── Seed (Gizmo's core identity) ──────────────────────────────────────────
    seed_rows = store.get_personality(headmate=None, aspect="seed")
    if seed_rows:
        lines.append(seed_rows[0].get("text", "You are Gizmo."))
    else:
        lines.append("You are Gizmo.")

    # ── Hard corrections (always at top, non-negotiable) ──────────────────────
    corrections = store.get_corrections()
    if corrections:
        rules = "\n".join(f"  - {c}" for c in corrections)
        lines.append(f"\n[RULES — follow without exception]\n{rules}")

    # ── Per-headmate personality (Gizmo-with-this-person) ────────────────────
    if brief.headmate:
        hm_voice = store.get_personality(
            headmate=brief.headmate.lower(), aspect="with_headmate")
        if hm_voice:
            lines.append(
                f"\n[How you are with {brief.headmate.title()}]\n"
                + "\n".join(r.get("text", "") for r in hm_voice[:3])
            )

        # Change requests from this headmate
        changes = store.get_personality(
            headmate=brief.headmate.lower(), aspect="change_request")
        if changes:
            change_text = "\n".join(f"  - {r.get('text','')}" for r in changes[:5])
            lines.append(f"\n[Requested changes from {brief.headmate.title()}]\n{change_text}")

        # Preferences (context-filtered)
        prefs = store.get_preferences(
            headmate=brief.headmate.lower(),
            context=brief.register,
        )
        if prefs:
            pref_lines = []
            for p in prefs[:6]:
                override_note = ""
                if p.get("gizmo_override") and p.get("override_note"):
                    override_note = f" [override: {p['override_note']}]"
                pref_lines.append(
                    f"  - {p['preference']} "
                    f"(context: {', '.join(p.get('default_context') or ['any'])})"
                    f"{override_note}"
                )
            lines.append(
                f"\n[Preferences for {brief.headmate.title()}]\n"
                + "\n".join(pref_lines)
            )

        # Interaction prefs
        iprefs = store.get_active("interaction_prefs",
            headmate=brief.headmate.lower(), limit=10)
        if iprefs:
            itext = "\n".join(f"  - {p['content']}" for p in iprefs
                             if p.get("content"))
            lines.append(f"\n[Interaction style]\n{itext}")

    # ── Directive (the what) ──────────────────────────────────────────────────
    lines.append(f"\n[DIRECTIVE]")
    lines.append(f"Meaning: {directive.meaning}")

    if directive.actions:
        lines.append("Do:")
        for a in directive.actions:
            lines.append(f"  - {a}")

    if directive.suppress:
        lines.append("Do NOT:")
        for s in directive.suppress:
            lines.append(f"  - {s}")

    if directive.knowledge_to_use:
        lines.append("Weave in:")
        for k in directive.knowledge_to_use[:3]:
            lines.append(f"  - {k}")

    if directive.check_in:
        lines.append(
            f"Check in — style: {directive.check_in_style or 'gentle, non-clinical'}"
        )

    if directive.watch_for:
        lines.append(
            f"Watch for (log if seen): {', '.join(directive.watch_for)}"
        )

    # Unknown entity asks — woven naturally, never clinical
    if directive.ask_about:
        ask = directive.ask_about[0]  # one ask maximum per response
        lines.append(
            f"You don't know who/what \"{ask['name']}\" is. "
            f"Ask about it naturally — one sentence, woven in, not a big deal. "
            f"Style: {ask.get('style', 'casual and curious')}. "
            f"Never say \"I don't have information about\" or anything database-sounding."
        )

    # ── Writing instruction ───────────────────────────────────────────────────
    lines.append(f"\n[WRITE]")
    lines.append(
        f"Write a response in {brief.headmate.title() if brief.headmate else 'their'}'s "
        f"register ({directive.register})."
    )
    lines.append(f"Voice: {directive.tone}.")
    lines.append(
        f"Target length: approximately {directive.token_target} tokens. "
        f"This is a target, not a hard limit — land it right."
    )
    lines.append(
        "Do not explain. Do not hedge. Do not break voice. Just write."
    )

    return "\n".join(lines)


# ── Stage 5: Response ─────────────────────────────────────────────────────────

async def generate_response(
    brief:         Brief,
    system_prompt: str,
    history,
    llm,
) -> str:
    """
    One LLM call. Writes the actual message.
    Given a perfect brief, the model just writes.
    """
    # Hard cap on system prompt to avoid token overflows
    if len(system_prompt) > 6000:
        system_prompt = system_prompt[:6000] + "\n[...truncated]"

    print(f"[generate_response] system_prompt length: {len(system_prompt)} chars", flush=True)

    # Use as_messages() — clean role/content pairs, appends current message
    try:
        messages = history.as_messages(brief.message)
    except Exception as e:
        print(f"[generate_response] history error: {e}", flush=True)
        messages = [{"role": "user", "content": brief.message}]

    print(f"[generate_response] messages count: {len(messages)}", flush=True)

    response = await llm.generate(
        messages,
        system_prompt=system_prompt,
        max_new_tokens=max(200, brief.word_count * 3 + 60),
        temperature=_response_temperature(brief),
    )

    print(f"[generate_response] got: '{response[:80] if response else 'EMPTY'}'", flush=True)

    log_event("Agent", "RESPONSE_GENERATED",
        session=brief.session_id[:8],
        words=len(response.split()) if response else 0,
        register=brief.register,
    )

    return (response or "").strip()


def directive_token_target(brief: Brief) -> int:
    """Fallback token target if directive not available."""
    if brief.register in ("subspace", "scene", "intimate"):
        return 60
    if brief.register in ("distress", "crisis"):
        return 80
    if brief.register in ("reflective", "deep"):
        return 200
    return max(60, brief.word_count * 2)


def _response_temperature(brief: Brief) -> float:
    if brief.register in ("intimate", "dominant", "scene"):
        return 0.85
    if brief.register in ("distress", "crisis"):
        return 0.4
    if brief.has_intimate:
        return 0.8
    return 0.72


# ── Stage 6: Close loop ───────────────────────────────────────────────────────

async def close_loop(
    brief:     Brief,
    directive: Directive,
    response:  str,
    history,
    llm,
) -> None:
    """
    Fire-and-forget post-response tasks.
    Never blocks. Never raises to caller.
    """
    from core.store import store

    try:
        # Write response envelope
        store.write("responses", {
            "content":      response,
            "response_to":  brief.message_id,
            "headmate":     brief.headmate.lower() if brief.headmate else None,
            "session_id":   brief.session_id,
            "approach":     directive.tone,
            "why":          directive.meaning,
            "brief_snapshot": (
                f"{brief.headmate} | {brief.register} | "
                f"{brief.time_of_day} | {brief.day_of_week} | "
                f"pattern={directive.pattern_action}"
            ),
            "what_i_knew":  directive.knowledge_to_use,
            "source":       "gizmo",
            "tags":         (
                f"response,{brief.headmate.lower() if brief.headmate else 'unknown'},"
                f"{brief.register},{directive.tone}"
            ),
        })

        # Save to history
        history.add("assistant", response, context={
            "current_host": brief.headmate,
            "fronters": brief.fronters,
        })

        # Log emotion data point
        store.write("emotion_log", {
            "headmate":   brief.headmate.lower() if brief.headmate else None,
            "session_id": brief.session_id,
            "valence":    brief.emotional_valence,
            "intensity":  _register_intensity(brief.register),
            "chaos":      0.3 if brief.stress_level in ("high", "crisis") else 0.1,
            "register":   brief.register,
            "topic":      brief.topics[0] if brief.topics else "general",
            "word_count": brief.word_count,
            "source":     "emotion_tracker",
            "tags":       f"emotion,{brief.headmate.lower() if brief.headmate else 'unknown'}",
        })

        # Log therapy conditions if flagged
        if directive.log_conditions and brief.headmate:
            store.write("wellbeing", {
                "headmate":   brief.headmate.lower(),
                "category":   "pattern",
                "observation": (
                    f"Session conditions: {brief.register} register, "
                    f"{brief.stress_level} stress, {brief.time_of_day} "
                    f"on {brief.day_of_week}. Pattern: {directive.pattern_action}."
                ),
                "context":    brief.message[:100],
                "register":   brief.register,
                "session_id": brief.session_id,
                "source":     "therapy_log",
                "confidence": 0.9,
                "tags":       f"therapy_log,{brief.headmate.lower()},conditions",
            })

        # Log pattern instance if one fired
        if (directive.pattern_action in ("feed", "break")
                and brief.headmate):
            asyncio.ensure_future(
                _log_pattern_instance(brief, directive, response)
            )

        # Update pattern instance outcome from response outcome
        if brief.prev_response_id and brief.headmate:
            asyncio.ensure_future(
                _update_pattern_outcome(brief, directive)
            )

        # Check for gaps → queue questions
        # Disabled for now — too many concurrent API calls
        # Re-enable once rate limiting is resolved
        # if brief.headmate:
        #     asyncio.ensure_future(
        #         _check_and_queue_questions(brief, directive, llm)
        #     )

        log_event("Agent", "CLOSE_LOOP_COMPLETE",
            session=brief.session_id[:8],
            therapy_logged=directive.log_conditions,
            pattern=directive.pattern_action,
        )

        # ── Async extraction — runs after response sent, never blocks ─────────
        # Extracts facts, relationships, wellbeing from the exchange
        from core.memory import memory_encoder, build_transcript

        asyncio.ensure_future(
            memory_encoder.encode_safe(
                transcript = build_transcript(history),
                headmate   = brief.headmate,
                session_id = brief.session_id,
                duration_s = time.time() - brief.timestamp,
                register   = brief.register,
                llm        = llm,
            )
        )

    except Exception as e:
        log_error("Agent", "close_loop failed", exc=e)


async def _update_pattern_outcome(brief: Brief, directive: Directive) -> None:
    """Route response outcome to pattern engine for instance quality update."""
    try:
        from core.store import store
        from core.pattern_engine import pattern_engine

        # Get the outcome we filled in during this intake
        resp = store.get_last_response(brief.session_id)
        if not resp or not resp.get("outcome"):
            return

        await pattern_engine.update_instance_outcome(
            session_id=brief.session_id,
            headmate=brief.headmate,
            outcome=resp["outcome"],
            outcome_signal=resp.get("outcome_signal", ""),
            post_pattern=None,
        )
    except Exception as e:
        log_error("Agent", "pattern outcome update failed", exc=e)


async def _async_extract(brief: Brief, response: str, llm) -> None:
    """
    Extract intelligence from the exchange after response is sent.
    Writes facts, relationships, wellbeing to store.
    Never blocks the response — fire and forget from close_loop.
    """
    try:
        from core.store import store

        extracted = await _extract_intelligence(
            message=brief.message,
            headmate=brief.headmate or "",
            register=brief.register,
            has_intimate=brief.has_intimate,
            llm=llm,
        )

        if not extracted:
            return

        # Write facts
        for fact in extracted.get("new_facts", []):
            if fact and len(fact) > 5:
                store.write("facts", {
                    "fact":       fact,
                    "headmate":   brief.headmate.lower() if brief.headmate else None,
                    "fact_type":  "observation",
                    "register":   brief.register,
                    "context":    brief.time_of_day,
                    "session_id": brief.session_id,
                    "source":     "extractor",
                    "tags":       f"fact,{brief.headmate.lower() if brief.headmate else 'unknown'},{brief.register}",
                })

        # Write relationships
        for rel in extracted.get("relationships", []):
            if rel.get("entity") and rel.get("label"):
                store.write("relationships", {
                    "speaker":               rel.get("speaker", brief.headmate or "unknown"),
                    "entity":                rel.get("entity", ""),
                    "entity_type":           rel.get("entity_type", "unknown"),
                    "relationship_label":    rel.get("label", ""),
                    "relationship_category": rel.get("category", "social_bond"),
                    "confidence_type":       rel.get("confidence", "stated"),
                    "intimate":              1 if rel.get("intimate") else 0,
                    "headmate":              brief.headmate.lower() if brief.headmate else None,
                    "session_id":            brief.session_id,
                    "source":               "extractor",
                    "tags":                 f"relationship,{rel.get('label','')}",
                })

        # Write wellbeing
        for obs in extracted.get("wellbeing_observations", []):
            if obs.get("observation"):
                store.write("wellbeing", {
                    "headmate":    brief.headmate.lower() if brief.headmate else None,
                    "category":    obs.get("category", "pattern"),
                    "observation": obs.get("observation", ""),
                    "context":     obs.get("context", ""),
                    "register":    brief.register,
                    "session_id":  brief.session_id,
                    "source":      "extractor",
                    "confidence":  0.6,
                    "tags":        f"wellbeing,{obs.get('category','pattern')},{brief.headmate.lower() if brief.headmate else 'unknown'}",
                })

        log_event("Agent", "EXTRACTION_COMPLETE",
            session=brief.session_id[:8],
            facts=len(extracted.get("new_facts", [])),
            relationships=len(extracted.get("relationships", [])),
            wellbeing=len(extracted.get("wellbeing_observations", [])),
        )

        # Print detailed extraction results
        print(f"\n[EXTRACTION] headmate={brief.headmate} register={brief.register}", flush=True)
        print(f"[EXTRACTION] message: {brief.message[:80]}", flush=True)

        facts_written = extracted.get("new_facts", [])
        if facts_written:
            print(f"[EXTRACTION] facts ({len(facts_written)}):", flush=True)
            for f in facts_written:
                print(f"  → fact: {f}", flush=True)
        else:
            print(f"[EXTRACTION] facts: none", flush=True)

        rels_written = extracted.get("relationships", [])
        if rels_written:
            print(f"[EXTRACTION] relationships ({len(rels_written)}):", flush=True)
            for r in rels_written:
                print(f"  → {r.get('speaker','?')} --[{r.get('label','?')}]--> {r.get('entity','?')} ({r.get('confidence','?')})", flush=True)
        else:
            print(f"[EXTRACTION] relationships: none", flush=True)

        wb_written = extracted.get("wellbeing_observations", [])
        if wb_written:
            print(f"[EXTRACTION] wellbeing ({len(wb_written)}):", flush=True)
            for w in wb_written:
                print(f"  → [{w.get('category','?')}] {w.get('observation','')[:80]}", flush=True)
        else:
            print(f"[EXTRACTION] wellbeing: none", flush=True)

        vibe = extracted.get("vibe", [])
        stress = extracted.get("stress_level", "unknown")
        valence = extracted.get("valence", 0.0)
        print(f"[EXTRACTION] vibe={vibe} stress={stress} valence={valence:+.2f}\n", flush=True)

    except Exception as e:
        log_error("Agent", "async extraction failed", exc=e)


async def _log_pattern_instance(
    brief: Brief, directive: Directive, response: str
) -> None:
    """Log that a pattern fired this exchange."""
    from core.store import store
    try:
        # Find the active pattern
        patterns = store.get_patterns(
            brief.headmate, action=directive.pattern_action
        )
        if not patterns:
            return

        # Use highest confidence pattern
        pattern = patterns[0]
        store.log_pattern_instance(
            pattern_id=pattern["id"],
            headmate=brief.headmate,
            session_id=brief.session_id,
            intensity_in=_register_intensity(brief.register),
            intensity_out=float(directive.push_to or _register_intensity(brief.register)),
            gizmo_pushed=directive.pattern_action == "feed",
            push_type=directive.tone if directive.pattern_action == "feed" else None,
            post_pattern=None,  # filled in next session
            outcome_quality=0.5,  # placeholder, updated when outcome fills
            notes=directive.meaning,
        )
    except Exception as e:
        log_error("Agent", "pattern instance log failed", exc=e)


async def _check_and_queue_questions(
    brief: Brief, directive: Directive, llm
) -> None:
    """
    Identify gaps in knowledge that would improve future responses.
    Queue a question if one is worth asking.
    Non-intrusive — only queues, doesn't ask immediately.
    Shameless — asks directly when the moment comes.
    """
    from core.store import store

    try:
        # Don't queue if there are already pending questions
        pending = store.get_pending_questions(brief.headmate)
        if len(pending) >= 2:
            return

        # Only look for gaps during certain registers
        if brief.register in ("distress", "crisis"):
            return

        raw = await _call(llm,
            system=(
                "You identify gaps in knowledge about someone that would "
                "meaningfully improve how you show up for them. "
                "You only flag gaps worth asking about. "
                "You never ask things that are invasive or premature. "
                "JSON only."
            ),
            user=f"""Headmate: {brief.headmate}
Current exchange register: {brief.register}
What we know: {', '.join(directive.knowledge_to_use[:3]) or 'limited'}
What worked: {', '.join(directive.actions[:2]) or 'unclear'}
What was suppressed: {', '.join(directive.suppress[:2]) or 'nothing'}

Is there ONE specific thing worth knowing about {brief.headmate} that would
meaningfully improve how you show up for them?

Return JSON:
{{
  "gap_found": true/false,
  "question": "the question to ask — direct, conversational, not clinical",
  "gap_identified": "what knowing this would improve",
  "timing": "when to ask — casual moment / after intimate scene / when she's reflective / etc"
}}

Only return gap_found=true if the question is genuinely useful and not already
likely known. If nothing is needed, return gap_found=false.""",
            tokens=200,
            temp=0.2,
        )

        data = _parse_json(raw, {"gap_found": False})
        if data.get("gap_found") and data.get("question"):
            store.queue_question(
                headmate=brief.headmate,
                question=data["question"],
                gap_identified=data.get("gap_identified", ""),
                context=data.get("timing", ""),
                session_id=brief.session_id,
            )
            log_event("Agent", "QUESTION_QUEUED",
                headmate=brief.headmate,
                question=data["question"][:60],
            )

    except Exception as e:
        log_error("Agent", "question queue check failed", exc=e)


# ── Main orchestrator ─────────────────────────────────────────────────────────

class Agent:

    async def respond(
        self,
        user_message: str,
        session_id:   str,
        context:      dict,
        history,
        push_fn       = None,   # optional: async fn(str) for streaming
    ) -> AsyncGenerator[str, None]:
        """
        Full pipeline. Yields response chunks.
        """
        from core.llm import llm

        t_start = time.monotonic()

        log_event("Agent", "PIPELINE_START",
            session=session_id[:8],
            headmate=context.get("current_host") or "unknown",
            words=len(user_message.split()),
        )

        # ── 1. Intake ─────────────────────────────────────────────────────────
        brief = await intake(
            message=user_message,
            session_id=session_id,
            context=context,
            history=history,
            llm=llm,
        )

        # ── Host identification short-circuit ─────────────────────────────────
        # If Gizmo doesn't know who he's talking to, ask before anything else.
        # The answer to this question re-enters as a normal message and the
        # session manager picks up the name, sets the host, and we proceed.
        if brief.host_question:
            response_text = brief.host_question
            history.add("assistant", response_text, context={
                "current_host": None,
                "fronters":     brief.fronters,
            })
            for i in range(0, len(response_text), 8):
                yield response_text[i:i + 8]
            return

        # ── 2. Sequential agents (rate limiting workaround) ───────────────────
        # Run sequentially until we have proper rate limiting
        knowledge = await agent_knowledge(brief, llm)
        wellness  = await agent_wellness(brief, llm)
        therapy   = await agent_therapy(brief, llm)
        narrative = await agent_narrative(brief, history, llm)

        # ── 3. Director ───────────────────────────────────────────────────────
        directive = await director(
            brief=brief,
            knowledge=knowledge,
            wellness=wellness,
            therapy=therapy,
            narrative=narrative,
            llm=llm,
        )

        # ── 4. Personality layer ──────────────────────────────────────────────
        try:
            system_prompt = personality_layer(brief, directive)
        except Exception as e:
            log_error("Agent", f"personality_layer failed: {e}", exc=e)
            system_prompt = "You are Gizmo, a warm and present companion. Respond naturally."

        # Hard cap
        if len(system_prompt) > 6000:
            system_prompt = system_prompt[:6000]

        # ── 5. Response ───────────────────────────────────────────────────────
        response_text = await generate_response(
            brief=brief,
            system_prompt=system_prompt,
            history=history,
            llm=llm,
        )

        # ── 6. Close loop (fire and forget) ───────────────────────────────────
        asyncio.ensure_future(
            close_loop(
                brief=brief,
                directive=directive,
                response=response_text,
                history=history,
                llm=llm,
            )
        )

        # ── Stream response ───────────────────────────────────────────────────
        duration_ms = round((time.monotonic() - t_start) * 1000)

        log_event("Agent", "PIPELINE_COMPLETE",
            session=session_id[:8],
            duration_ms=duration_ms,
            words=len(response_text.split()),
            pattern=directive.pattern_action,
            tone=directive.tone,
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _calculate_token_target(
    brief: Brief,
    therapy: dict,
    narrative: dict,
) -> int:
    """Determine response length from context signals."""
    register = brief.register
    arc      = narrative.get("arc_position", "building")

    if register in ("subspace", "scene"):
        return 40
    if register in ("distress", "crisis"):
        return 80
    if register in ("intimate", "dominant"):
        return 60 if arc != "deep" else 100
    if register in ("reflective", "deep"):
        return 180
    if arc == "peak":
        return 120
    if brief.session_momentum == "opening":
        return 80

    # Mirror message length
    return max(60, min(200, brief.word_count * 2))


def _register_intensity(register: str) -> float:
    """Map register to intensity float."""
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


def _classify_register(message: str) -> str:
    """Heuristic register classification."""
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

    import re
    for register, pattern in _PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return register

    return "neutral"


def _classify_topics(message: str) -> list[str]:
    """Heuristic topic classification."""
    import re
    topics = []

    _TOPIC_MAP = [
        ("work",        r"\b(work|job|boss|office|meeting|deadline|coworker|project)\b"),
        ("health",      r"\b(sick|pain|hurt|doctor|tired|sleep|rest|exhausted)\b"),
        ("food",        r"\b(hungry|eat|food|meal|cook|dinner|lunch|snack)\b"),
        ("relationship",r"\b(friend|family|partner|trust|fight|together|apart)\b"),
        ("creativity",  r"\b(sew|draw|paint|write|create|make|art|design)\b"),
        ("gizmo_dev",   r"\b(gizmo|pipeline|model|store|code|deploy|server)\b"),
        ("identity",    r"\b(headmate|front|system|plural|switch|alter)\b"),
        ("emotion",     r"\b(feel|feeling|emotion|mood|sad|happy|angry|anxious)\b"),
        ("planning",    r"\b(plan|schedule|tomorrow|later|remind|todo|list)\b"),
    ]

    for topic, pattern in _TOPIC_MAP:
        if re.search(pattern, message, re.IGNORECASE):
            topics.append(topic)

    return topics or ["general"]


async def _extract_intelligence(
    message:      str,
    headmate:     str,
    register:     str,
    has_intimate: bool,
    llm,
) -> dict:
    """
    Lightweight LLM extraction of structured intelligence from a message.
    Returns subjects, relationships, facts, needs, vibe, stress, valence.
    Falls back to empty dict on failure — never blocks.
    """
    intimate_instruction = ""
    if has_intimate:
        intimate_instruction = (
            "\nAlso extract wellbeing_observations for intimate data: "
            "kinks noticed, dynamics that landed, emotional needs active, limits. "
            "category: emotional_need/physical_need/works/pulled_away/pattern/limit"
        )

    raw = await _call(llm,
        system=(
            "Extract structured intelligence from a message. "
            "Use the speaker's actual vocabulary — never sanitize. "
            "Intimate data is valid data. JSON only."
        ),
        user=f"""Speaker: {headmate or 'unknown'}
Register: {register}
Message: "{message}"

Return JSON:
{{
  "subjects": [{{"entity": "name", "entity_type": "headmate/external/object/concept", "owner": null, "speaker_is_owner": true}}],
  "relationships": [{{"speaker": "{headmate}", "entity": "", "label": "", "category": "", "confidence": "stated", "intimate": false}}],
  "new_facts": ["concrete facts stated"],
  "needs_active": ["needs present in this message"],
  "vibe": ["2-4 words describing current energy"],
  "stress_level": "none|low|medium|high|crisis",
  "valence": -1.0 to 1.0,
  "wellbeing_observations": [{{"category": "", "observation": "", "context": ""}}]{intimate_instruction}
}}
JSON only. No markdown.""",
        tokens=500,
        temp=0.1,
    )

    return _parse_json(raw, {
        "subjects": [], "relationships": [], "new_facts": [],
        "needs_active": [], "vibe": [], "stress_level": "unknown",
        "valence": 0.0, "wellbeing_observations": [],
    })


def _infer_outcome(
    message: str,
    valence: float,
    register: str,
) -> tuple[str, str]:
    """Infer how previous response landed from next message."""
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