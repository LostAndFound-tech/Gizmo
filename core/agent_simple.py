"""
core/agent_simple.py
Bypass agent — one module, no orchestration.

Current target: context_deductor output only.
Returns prompt via context dict so server can send it to the inspector.
"""

import time
from typing import AsyncGenerator, Optional

from core.log import log_event, log_error
from core.context_deductor import content_deductor as CD, _build_prompt


async def _call_module(
    message: str,
    history: list,
    session_id: str,
    context: dict,
) -> tuple[str, str]:
    subject = context.get("current_host") or "unknown"
    prompt  = _build_prompt(message, subject)

    context_data = await CD.extract(message, "", subject, session_id)
    response = str(context_data) if context_data else "Context extraction returned nothing."

    return prompt, response


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
        ctx = context if context is not None else {}

        log_event("AgentSimple", "RECEIVE",
            session=session_id[:8],
            preview=user_message[:60],
        )

        try:
            prompt, response_text = await _call_module(
                message=user_message,
                history=history,
                session_id=session_id,
                context=ctx,
            )
            ctx["_last_prompt"] = prompt
        except Exception as e:
            log_error("AgentSimple", "module call failed", exc=e)
            yield f"Something went wrong: {type(e).__name__}: {e}"
            return

        duration_ms = round((time.monotonic() - t_start) * 1000)
        log_event("AgentSimple", "COMPLETE",
            session=session_id[:8],
            duration_ms=duration_ms,
        )

        chunk_size = 8
        for i in range(0, len(response_text), chunk_size):
            yield response_text[i:i + chunk_size]


# ── Singleton ─────────────────────────────────────────────────────────────────
agent_simple = AgentSimple()