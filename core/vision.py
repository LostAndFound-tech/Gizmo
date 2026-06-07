"""
core/vision.py
Context extraction from conversational statements.

Parses each sentence individually, identifies subjects, and extracts
whatever that sentence actually says about them — freeform attributes
under a typed envelope (person, place, action).

Returns raw JSON string for inspection and downstream ingestion.
"""
import os
import asyncio
import json
import re
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """

In this exchange, what can you gather about the user's sexuality? Any kinks? Any fetishes? Any abberant behavior? No judgement, just an idea of their sexuality.
follow this scheme exactly:
{
    "name":"jess",
    "kinks":["slavery", "service", "exhibitionism"],
    "fetishes":["infantalism"]
    "possible connections":"try humilating her to see how she reacts (confidence:.7)"
}

Rules:
 - Give a non-judgmental idea of kinks and fetishes.
 - Do not judge the user.
 - Be complete, include every fetish over .4 confidence. 
""".strip()


def _build_prompt(actions: str, body: str) -> str:
    print(f"The user looks like {body}"
          f"They did: {actions}")
    return (
        (f"The user looks like {body}"
          f"They did: {actions}")
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm

        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SYSTEM,
            temperature=0.0,
            max_new_tokens=3000,
        )

        if not raw or not raw.strip():
            log_event("ContextDeductor", "EMPTY_RESPONSE")
            return None

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return clean

    except Exception as e:
        log_error("ContextDeductor", "LLM call failed", exc=e)
        print(f"[ContextDeductor] LLM call failed: {type(e).__name__}: {e}")
        return None


# ── File write ────────────────────────────────────────────────────────────────

def _actions_file(subject: str) -> Path:
    date_str = tz_now().strftime("%Y-%m-%d")
    path = DATA_DIR / "testing" / subject.lower() / "actions" / f"{date_str}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_to_file(record: str, subject: str) -> None:
    try:
        path = _actions_file(subject)
        with path.open("a", encoding="utf-8") as f:
            f.write(record + "\n")
    except Exception as e:
        log_error("ContextDeductor", "file write failed", exc=e)


# ── Public API ────────────────────────────────────────────────────────────────
        
class VisionCatcher:
    async def extract(
        self,
        actions:   str,
        body:         str,
    ) -> Optional[str]:
        if not str(actions).strip():
            return None
        try:
            prompt  = _build_prompt(str(actions), body)
            context = await _call_llm(prompt)

            return context

        except Exception as e:
            log_error("ContextDeductor", "extract failed", exc=e)
            print(f"[ContextDeductor] extract failed: {type(e).__name__}: {e}")
            return None


visioncatcher = VisionCatcher()