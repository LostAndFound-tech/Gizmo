"""
core/Descriptor_catcher.py
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
You gather descriptive datapoints about people and objects from conversational statements.
You will receive the original message and a thread summary.
Return ONLY a valid JSON array. No markdown fences. No explanation. No preamble.
If nothing is being described, return [].

Capture EVERYTHING the text says about each object — appearance, personality, behavior, possessions, relationships, conditions. One rich object per entity. Never invent. Never use null, Unknown, or empty lists. Omit fields not supported by the text.

Example of a rich Person entry:
[{
  "Object": "Ara",
  "Type": "Person",
  "Hair": ["hay-colored", "stringy"],
  "Personality": ["self-deprecating", "deflects compliments"],
  "Relationships": ["close with Honey", "Jess admires her hair"]
}]

Example of a Clothing entry:
[{
  "Object": "jacket",
  "Type": "Clothing",
  "Owner": "Jess",
  "Color": "green",
  "Style": ["puffy sleeves"]
}]

Example of multiple rich objects in one exchange:
[{
  "Object": "Ara",
  "Type": "Person",
  "Hair": ["hay-colored", "stringy"],
  "Personality": ["self-deprecating"]
},
{
  "Object": "Honey",
  "Type": "Person",
  "Hair": ["widely considered the best"],
  "Reputation": ["praised by Jess and Ara"]
}]

Rules:
- One object per entity, as many fields as the text supports.
- If two speakers describe the same thing differently, include both in the same list.
- Objects can be people, body parts, clothing, animals, places, or things.
- Preferences and wishes about oneself are not descriptors of that person — skip them.
""".strip()


def _build_prompt(user_message: str, thread: str) -> str:
    print(f"The original message: {user_message}"
          f"The threads: {thread}")
    return (
        f"The original message: {user_message}"
        f"The threads: {thread}"
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

class DescriptorCatcher:

    async def extract(
        self,
        user_message:   str,
        thread:         str,
        subject:        str,
        session_file:   str,
    ) -> Optional[str]:
        print("DESCRIPTOR NUDGED")
        if not user_message.strip():
            return None
        try:
            prompt  = _build_prompt(user_message, thread)
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
            return None


descriptor_catcher = DescriptorCatcher()