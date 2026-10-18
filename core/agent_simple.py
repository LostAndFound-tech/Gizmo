"""
core/agent_simple.py
Bare-bones agent: message + history in, reply out. One LLM call via core.llm.

No chunk pipeline, no modes, no lessons, no keyphrases — add those back
one at a time once this loop is solid.
"""

import time
from typing import AsyncGenerator, Optional

from core.llm import llm
from core.log import log_event, log_error

MAX_TOKENS  = 1024
TEMPERATURE = 0.8

SYSTEM_PROMPT = """You are Gizmo, a warm, attentive AI companion for the BloodeStream, a plural system.
Several headmates share one body and may take turns speaking with you.
Speak naturally and concisely. Address whoever is currently talking by name.
Don't assume you remember things that aren't in this conversation."""


def _build_system(ctx: dict) -> str:
    system = SYSTEM_PROMPT
    host = ctx.get("current_host")
    if host:
        system += f"\n\nCurrently speaking: {host.title()}."
    fronters = [f for f in ctx.get("fronters", []) if f and f != host]
    if fronters:
        system += f" Also present: {', '.join(f.title() for f in fronters)}."
    return system


class AgentSimple:

    async def respond(
        self,
        user_message: str,
        history,
        session_id: str = "",
        context: Optional[dict] = None,
        **_ignored,
    ) -> AsyncGenerator[str, None]:
        t_start = time.monotonic()
        ctx = context if context is not None else {}
        log_event("AgentSimple", "RECEIVE", session=session_id[:8], preview=user_message[:60])

        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in (history or [])
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        messages.append({"role": "user", "content": user_message})

        text = await llm.generate(
            messages,
            system_prompt=_build_system(ctx),
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        # llm.generate returns "" after exhausting retries — surface it as an error
        if not text.strip():
            log_error("AgentSimple", "llm returned empty after retries", exc=None)
            raise RuntimeError("LLM returned an empty response")

        log_event("AgentSimple", "COMPLETE", session=session_id[:8],
                  duration_ms=round((time.monotonic() - t_start) * 1000))
        yield text.strip()

    def end_session(self, session_id: str) -> None:
        """Called by the server on disconnect. Nothing to clean up yet."""
        pass


agent_simple = AgentSimple()
