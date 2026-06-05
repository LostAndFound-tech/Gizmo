"""
core/agent_simple.py
Bypass agent — one module, no orchestration.

Flow:
  1. Receive message + history
  2. Call the target module
  3. Stream response back to caller

Swap TARGET_MODULE and how you call it as you test each engine.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

from core.log import log_event, log_error
from core.llm import llm
from core.context_deductor import content_deductor as CD


async def _call_module(
    message: str,
    history: list,
    session_id: str,
    context: dict,
) -> str:
    context_data = await CD.extract(message, "", "unknown", session_id)
    if context_data is None:
        print("Context extraction returned nothing")

    print("---------------")
    print(f"Input: {message}")
    print(f"Context: {context_data}")

    messages = [
        {"role": entry["role"], "content": entry["content"]}
        for entry in (history or [])
    ]
    messages.append({"role": "user", "content": message})

    return await llm.generate(messages)


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentSimple:

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

        log_event("AgentSimple", "RECEIVE",
            session=session_id[:8],
            preview=user_message[:60],
        )

        try:
            response_text = await _call_module(
                message=user_message,
                history=history,
                session_id=session_id,
                context=ctx,
            )
        except Exception as e:
            log_error("AgentSimple", "module call failed", exc=e)
            yield f"Something went wrong: {type(e).__name__}: {e}"
            return

        duration_ms = round((time.monotonic() - t_start) * 1000)
        log_event("AgentSimple", "COMPLETE",
            session=session_id[:8],
            duration_ms=duration_ms,
            words=len(response_text.split()),
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]


# ── Singleton ─────────────────────────────────────────────────────────────────
agent_simple = AgentSimple()