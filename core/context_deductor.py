"""
core/context_deductor.py
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
You extract structured context from conversational statements.
Return ONLY valid JSON. No markdown fences. No explanation. No preamble.

Split the input into individual sentences. For each sentence, extract who said it,
what subjects are referenced, what type each subject is, and whatever that sentence
actually says about them. If it's a descriptor, note what it's describing. If more than one word is in *asterisks*,
it is in action or a declaration, and is not heard.

Use exactly this structure:

{
  "topic": "Willow's whereabouts and plans",
  "primary_subjects": ["Willow", "the dog"],
  "speakers": ["Ember", "Kaylee"],
  "scene": "Ember is asking about where Willow is going. Kaylee is answering.",
  "thread": [
    "Ember is curious about Willow's current activity",
    "Kaylee clarifies Willow is taking the dog out",
    "Ember presses for what happens after",
    "Kaylee clarifies Willow is going to physical therapy"
  ]
}

Rules:
- speaker is always the person whose statement this is
- type is one of: person, place, action
- Under type, include only what this sentence actually expresses — no invented details
- If a sentence references multiple subjects, each gets their own block
- If a sentence has no clear subjects, still include the sentence key with speaker and empty subjects
- Actions include who did it (verb) and who or what received it (recipient) if present
""".strip()


def _build_prompt(user_message: str, subject: str) -> str:
    return (
        f"The person speaking is: {subject}\n\n"
        f"Statement:\n{user_message}"
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


# ── Public API ────────────────────────────────────────────────────────────────

class ContentDeductor:

    async def extract(
        self,
        user_message:   str,
        gizmo_response: str,
        subject:        str,
        session_file:   str,
    ) -> Optional[str]:
        if not user_message.strip():
            return None

        try:
            prompt  = _build_prompt(user_message, subject)
            context = await _call_llm(prompt)

            if not context:
                log_event("ContextDeductor", "NO_CONTEXT_EXTRACTED",
                    subject=subject,
                    session=session_file,
                )
                return None

            return context

        except Exception as e:
            log_error("ContextDeductor", "extract failed", exc=e)
            print(f"[ContextDeductor] extract failed: {type(e).__name__}: {e}")
            print(context)
            return None


content_deductor = ContentDeductor()