"""
PATCH for core/agent.py
Shows how to handle the directed queue drain loop with TTS output,
and how to inject Gizmo's current mood into the system prompt.

── MOOD INJECTION ────────────────────────────────────────────────────────────
In build_system_prompt(), add mood block at the end:

    from voice.mood import get_mood_prompt_block

    def build_system_prompt(...) -> str:
        # ... existing code ...
        mood_block = ""
        try:
            from voice.mood import get_mood_prompt_block
            mood_block = get_mood_prompt_block()
            if mood_block:
                mood_block = f"\n\n{mood_block}"
        except Exception:
            pass

        return f\"\"\"{personality}
    ... rest of prompt ...
    {mood_block}\"\"\"

── QUEUE DRAIN LOOP ──────────────────────────────────────────────────────────
In your main entrypoint where you drain the directed_queue:

    from voice.tts import speak

    queue = get_directed_queue()
    while not queue.empty():
        item = await queue.get()
        item_type = item.get("type", "user")

        if item_type == "gizmo_voice":
            utterance = item["transcript"].replace("[GIZMO_VOICE] ", "")
            await speak(utterance)

        elif item_type == "conflict_alert":
            response = ""
            async for chunk in agent.run(
                user_message=item["transcript"],
                history=history,
                session_id=session_id,
                use_rag=False,
                context=item.get("context"),
            ):
                response += chunk
            await speak(response)

        elif item_type == "personality_contradiction":
            response = ""
            async for chunk in agent.run(
                user_message=item["transcript"],
                history=history,
                session_id=session_id,
                use_rag=False,
                context=item.get("context"),
            ):
                response += chunk
            await speak(response)

        elif item_type == "reminder":
            response = ""
            async for chunk in agent.run(
                user_message=item["transcript"],
                history=history,
                session_id=session_id,
                context=item.get("context"),
            ):
                response += chunk
            await speak(response)

        else:
            response = ""
            async for chunk in agent.run(
                user_message=item["transcript"],
                history=history,
                session_id=session_id,
                context=item.get("context"),
            ):
                response += chunk
            await speak(response)

── TOOL REGISTRY ─────────────────────────────────────────────────────────────
Add to TOOL_REGISTRY in agent.py:

    from tools.personality_tool import PersonalityResolveTool, PersonalityQueryTool
    from tools.chattiness_tool import ChattinessTool

    TOOL_REGISTRY = {
        tool.name: tool for tool in [
            EchoTool(),
            SwitchHostTool(),
            CorrectionTool(),
            PersonalityResolveTool(),
            PersonalityQueryTool(),
            ChattinessTool(),
        ]
    }

── STARTUP SEQUENCE ──────────────────────────────────────────────────────────
GPU box (add to main entrypoint):

    from voice.tts import TTSServer
    from voice.receiver import start_receiver

    tts_server = TTSServer()
    asyncio.ensure_future(tts_server.start())
    await start_receiver(llm, context_fn=context_fn, directed_queue=directed_queue)

Pi (run separately):

    GPU_HOST=192.168.1.x python -m voice.tts        # audio playback
    GPU_HOST=192.168.1.x python -m voice.streamer   # mic capture
"""
