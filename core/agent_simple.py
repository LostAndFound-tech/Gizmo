"""
core/agent_simple.py
Bypass agent — passive pipeline + optional chat mode.

Passive mode: runs full chunk pipeline, writes all data, yields metadata JSON.
Chat mode:    runs full chunk pipeline, writes all data, calls responder,
              yields response text.

Keyphrases:
  "chat mode"           → switch to chat mode
  "passive mode"        → switch to passive mode
  "run wellness report" → trigger wellness synthesis
"""

import time
import json
from typing import AsyncGenerator, Optional

from core.log import log_event, log_error
from core.chunk_processor import ChunkProcessor
from core.responder import responder as _responder


# ── Mode state ────────────────────────────────────────────────────────────────

_chat_mode: bool                       = False
_processor: Optional["ChunkProcessor"] = None


class AgentSimple:

    async def respond(
        self,
        user_message: str,
        history,
        session_id:  str   = "",
        context:     Optional[dict] = None,
        source:      str   = "user",
        chunk_size:  int   = 8,
        timeout_sec: float = 10.0,
    ) -> AsyncGenerator[str, None]:
        global _chat_mode, _processor
        t_start = time.monotonic()
        ctx  = context if context is not None else {}
        host = ctx.get("current_host") or "unknown"
        print(f"[DEBUG] agent_simple host: {repr(host)}, ctx: {ctx}")

        log_event("AgentSimple", "RECEIVE",
            session=session_id[:8],
            preview=user_message[:60],
        )

        try:
            msg_lower = user_message.lower().strip()

            # ── Keyphrase triggers ────────────────────────────────────────────
            if "run wellness report" in msg_lower:
                print("[AgentSimple] wellness report triggered")
                from core.wellness_synthesis import wellness_synthesis
                await wellness_synthesis.run()
                yield json.dumps({"status": "ok", "trigger": "wellness_report"})
                return

            if "run report for" in msg_lower:
                name = msg_lower.split("run report for")[-1].strip().split()[0].strip(".,!?")
                print(f"[AgentSimple] individual wellness report triggered for {name}")
                from core.wellness_synthesis import wellness_synthesis
                result = await wellness_synthesis.synthesize_one(name)
                yield json.dumps({"status": "ok", "trigger": "wellness_report", "name": name, "synthesized": bool(result)})
                return

            if "chat mode" in msg_lower:
                _chat_mode = True
                print("[AgentSimple] switched to chat mode")
                yield "Chat mode on."
                return

            if "passive mode" in msg_lower:
                _chat_mode = False
                print("[AgentSimple] switched to passive mode")
                yield "Passive mode on."
                return

            # ── Chunk pipeline ────────────────────────────────────────────────
            if _processor is None:
                _processor = ChunkProcessor(
                    session_id=session_id,
                    host=host,
                    chunk_size=chunk_size,
                    timeout_sec=timeout_sec,
                )
            processor = _processor
            processor = _processor
            if host and host != "unknown":
                processor.host = host
                print("DEBUG: processor.host:", processor.host)

            lines = [l for l in user_message.splitlines() if l.strip()]

            chunk_result = None
            for line in lines:
                result = await processor.push_line(line)
                if result:
                    chunk_result = result

            # Always flush — chat mode needs immediate response on single messages,
            # passive mode needs complete processing of transcript chunks
            final_chunk = await processor.flush()
            print(f"[DEBUG] flush result: {final_chunk}", flush=True)
            print(f"[DEBUG] chunk_result: {chunk_result}", flush=True)
            print(f"[DEBUG] processor.results: {len(processor.results)}", flush=True)
            chunk_result = final_chunk or chunk_result
            chunk_result = final_chunk or chunk_result

            last_result = chunk_result or (processor.results[-1] if processor.results else None)

            ctx["session_id"] = session_id

            duration_ms = round((time.monotonic() - t_start) * 1000)
            log_event("AgentSimple", "COMPLETE",
                session=session_id[:8],
                duration_ms=duration_ms,
                mode="chat" if _chat_mode else "passive",
            )

            # ── Chat mode ─────────────────────────────────────────────────────
            if _chat_mode and last_result:
                response_text = await _responder.respond(
                    chunk_result=last_result,
                    context=ctx,
                    history=history or [],
                    user_message=user_message,
                )

                yield response_text or ""
                return

            # ── Passive mode ──────────────────────────────────────────────────
            yield ""

        except Exception as e:
            log_error("AgentSimple", "respond failed", exc=e)
            print(f"[AgentSimple] {type(e).__name__}: {e}", flush=True)
            yield json.dumps({"status": "error", "message": str(e)})



agent_simple = AgentSimple()