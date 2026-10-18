"""
Add these routes to server.py inside the start() method,
alongside the existing route definitions.

Place after the handle_session_detail route.
"""

# ── Session beats endpoint ────────────────────────────────────────────────────

async def handle_session_beats(request):
    """GET /sessions/{session_id}/beats — returns beat sequence as JSON"""
    sid = request.match_info.get("session_id", "")
    try:
        from core.memory.beats import beat_store
        beats = beat_store.get_session_beats(sid)
        items = [
            {
                "id":          b.id,
                "speaker":     b.speaker,
                "type":        b.type,
                "content":     b.content,
                "register":    b.register,
                "directed_at": b.directed_at,
                "why":         b.why,
                "landed":      b.landed,
                "timestamp":   b.timestamp,
            }
            for b in beats
        ]
        return web.Response(
            text         = json.dumps({"session_id": sid, "beats": items}),
            content_type = "application/json",
        )
    except Exception as e:
        return web.Response(
            text         = json.dumps({"error": str(e)}),
            content_type = "application/json",
            status       = 500,
        )


# ── Session narrative endpoint ────────────────────────────────────────────────

async def handle_session_narrative(request):
    """
    GET /sessions/{session_id}/narrative
    Returns cached narrative if available, generates if not.
    Query param: headmate (required)
    Query param: regenerate=true to force regeneration
    """
    sid        = request.match_info.get("session_id", "")
    headmate   = request.rel_url.query.get("headmate", "")
    regenerate = request.rel_url.query.get("regenerate", "false").lower() == "true"

    if not headmate:
        return web.Response(
            text         = json.dumps({"error": "headmate required"}),
            content_type = "application/json",
            status       = 400,
        )

    try:
        from core.memory.narrative import get_cached_narrative, render_session_narrative
        from core.llm import llm

        # Try cache first
        if not regenerate:
            cached = get_cached_narrative(sid, headmate)
            if cached:
                return web.Response(
                    text         = json.dumps({"session_id": sid, "narrative": cached, "cached": True}),
                    content_type = "application/json",
                )

        # Generate fresh
        narrative = await render_session_narrative(
            session_id = sid,
            headmate   = headmate,
            llm        = llm,
        )

        if not narrative:
            return web.Response(
                text         = json.dumps({"error": "no beats found for session"}),
                content_type = "application/json",
                status       = 404,
            )

        return web.Response(
            text         = json.dumps({"session_id": sid, "narrative": narrative, "cached": False}),
            content_type = "application/json",
        )

    except Exception as e:
        return web.Response(
            text         = json.dumps({"error": str(e)}),
            content_type = "application/json",
            status       = 500,
        )


# ── Reimagine endpoint ────────────────────────────────────────────────────────

async def handle_reimagine(request):
    """
    POST /sessions/reimagine
    Body: {
      "session_id": "sess_abc",
      "source_headmate": "jess",
      "target_headmate": "oren",
      "setup_beats": 3          (optional, default 3)
    }

    EPHEMERAL — generated, returned, never stored.
    """
    try:
        body           = await request.json()
        session_id     = body.get("session_id", "")
        source_headmate = body.get("source_headmate", "")
        target_headmate = body.get("target_headmate", "")
        setup_beats    = int(body.get("setup_beats", 3))

        if not session_id or not source_headmate or not target_headmate:
            return web.Response(
                text         = json.dumps({"error": "session_id, source_headmate, and target_headmate required"}),
                content_type = "application/json",
                status       = 400,
            )

        from core.memory.narrative import reimagine_session
        from core.llm import llm

        narrative = await reimagine_session(
            session_id       = session_id,
            source_headmate  = source_headmate,
            target_headmate  = target_headmate,
            setup_beats      = setup_beats,
            llm              = llm,
        )

        if not narrative:
            return web.Response(
                text         = json.dumps({"error": "could not generate reimagining"}),
                content_type = "application/json",
                status       = 404,
            )

        return web.Response(
            text         = json.dumps({
                "session_id":       session_id,
                "source_headmate":  source_headmate,
                "target_headmate":  target_headmate,
                "narrative":        narrative,
                "ephemeral":        True,   # reminder: this is never stored
            }),
            content_type = "application/json",
        )

    except Exception as e:
        return web.Response(
            text         = json.dumps({"error": str(e)}),
            content_type = "application/json",
            status       = 500,
        )


# ── Route registrations ───────────────────────────────────────────────────────
# Add these to app.router.add_* section in start():

# app.router.add_get("/sessions/{session_id}/beats",     handle_session_beats)
# app.router.add_get("/sessions/{session_id}/narrative", handle_session_narrative)
# app.router.add_post("/sessions/reimagine",             handle_reimagine)
