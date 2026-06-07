"""
core/agent_simple.py
Bypass agent — one module, no orchestration.

Current target: context_deductor + descriptor_catcher pipeline.
Returns combined output as a streamed response.
"""

import os
import time
import json
from typing import AsyncGenerator, Optional

from core.log import log_event, log_error
from core.context_deductor import content_deductor as CD, _build_prompt, _SYSTEM
from core.Descriptor_catcher import descriptor_catcher as describer
from core.BehaviorCatcher import behaviorcatcher as behavior
from core.vision import visioncatcher as vision
from core.story_gen import story_gen as story


async def _call_module(
    message: str,
    session_id: str,
    context: dict,
) -> tuple[str, str]:
    print("CALL MODULE ENTERED", flush=True)
    subject     = context.get("current_host") or "unknown"
    user_prompt = _build_prompt(message, subject)
    full_prompt = f"{_SYSTEM}\n\n{user_prompt}"

    context_data = await CD.extract(message, "", subject, session_id)
    response     = str(context_data) if context_data else "Context extraction returned nothing."

    return full_prompt, response


class memSaver:
    def write(
            self,
            path:str, content:str
    ):
        write_path = os.environ.get("DATA_DIR")+f"/{path}"
        if not os.path.exists(write_path):
            os.makedirs(write_path, exist_ok=True)
        with open (os.environ.get("DATA_DIR")+f"/{path}/"+ "TEST.txt", "w", encoding="utf-8") as file:
            file.write(content)
        pass


class AgentSimple:

    async def respond(
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
                session_id=session_id,
                context=ctx,
            )
            ctx["_last_prompt"] = prompt

            descriptors = "NONE"
            behaviors = "NONE"
            try:
                parsed = json.loads(response_text)
                print("PARSED PASS 1:", parsed, flush=True)
                thread = parsed.get("thread", [])
                descriptors = await describer.extract(
                    user_message=user_message,
                    thread=str(thread),
                    subject=ctx.get("current_host") or "unknown",
                    session_file=session_id,
                )
                behaviors = await behavior.extract(
                    user_message=str(thread),
                    thread=str(thread),
                    subject=ctx.get("current_host") or "unknown",
                    session_file=session_id
                )

                print("DESCRIPTORS:", descriptors, flush=True)
                visions = await vision.extract(
                    actions = thread,
                    body = descriptors
                )
                visions = json.loads(visions)
                print(visions)
                story_written = await story.extract(thread, "", "", [f"Kinks:{visions["kinks"]}", f"fetishes:{visions["fetishes"]}", f"Try this: {visions["possible connections"]}"])
            except Exception as e:
                print("Descriptors failed:", e, flush=True)
                descriptors = "NONE"

        except Exception as e:
            log_error("AgentSimple", "module call failed", exc=e)
            print(f"OUTER EXCEPT: {type(e).__name__}: {e}", flush=True)
            yield f"Something went wrong: {type(e).__name__}: {e}"
            return

        duration_ms = round((time.monotonic() - t_start) * 1000)
        log_event("AgentSimple", "COMPLETE",
            session=session_id[:8],
            duration_ms=duration_ms,
        )

        final = f"RESPONSE:\n{response_text}\n\n---\n\nDESCRIPTORS\n{descriptors}\n\n---\n\nBEHAVIORS\n{behaviors}\n\n---\n\nsexuality:\n{visions}\n\n---\n\nsexuality:\n{story_written}"
        mem = memSaver()
        mem.write("final", final)
        yield final


agent_simple = AgentSimple()