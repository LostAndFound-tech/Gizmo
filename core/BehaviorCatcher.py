"""
core/BehaviorCatcher.py
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
You extract behavioral datapoints about people from conversational statements.
You will receive the original message and a thread summary.
Return ONLY a valid JSON array. No markdown fences. No explanation. No preamble.
If nothing behavioral is present, return [].

For each person whose behavior, personality, or actions are revealed, produce one entry.
Infer who owns each behavior before filing it. Indirect statements count.

Fields — include only what the text supports:
  Subject     — who this is about
  Type        — always "Person"
  Actions     — list of things they did or said, as plain statements
  Reactions   — list of how they responded to something
  Personality — list of traits inferable from their behavior
  Relationships — list of how they relate to others in this exchange
  Source      — who made the observation ("self", or the observer's name)
  Statement   — the original text this was drawn from

Example of a rich entry:
[{
  "Subject": "Ember",
  "Type": "Person",
  "Actions": ["wanted to climb the mountain immediately"],
  "Personality": ["impulsive", "physically energetic"],
  "Source": "observed",
  "Statement": "Ember says she wants to climb"
},
{
  "Subject": "Jess",
  "Type": "Person",
  "Actions": ["refused to read the sign"],
  "Reactions": ["resisted Oren's suggestion"],
  "Personality": ["resistant to caution", "impulsive"],
  "Source": "observed",
  "Statement": "Jess says she doesn't want to read the sign"
}]

Rules:
- One entry per person. Stack all behaviors into that one entry.
- Infer personality from action — don't just restate the action as a trait.
- Source is "self" if they described themselves, otherwise name the observer.
- If two people show the same trait independently, file it under each separately.
""".strip()

_SEEING = """

If you were nearly 8 feet tall, fat, hairy, stinky... how would you react to a girl given this description 
performing each of the actions below. Consider yourself to be any mention of you. 

For each person whose behavior, personality, or actions are revealed, produce one entry.
Infer who owns each behavior before filing it. Indirect statements count.

Fields — include only what the text supports:
  Subject     — who this is about
  Type        — always "Person"
  Actions     — list of things they did or said, as plain statements
  Reactions   — list of how they responded to something
  Personality — list of traits inferable from their behavior
  Relationships — list of how they relate to others in this exchange
  Source      — who made the observation ("self", or the observer's name)
  Statement   — the original text this was drawn from

Example of a rich entry:
[{
  "Subject": "Ember",
  "Type": "Person",
  "Actions": ["wanted to climb the mountain immediately"],
  "Personality": ["impulsive", "physically energetic"],
  "Source": "observed",
  "Statement": "Ember says she wants to climb"
},
{
  "Subject": "Jess",
  "Type": "Person",
  "Actions": ["refused to read the sign"],
  "Reactions": ["resisted Oren's suggestion"],
  "Personality": ["resistant to caution", "impulsive"],
  "Source": "observed",
  "Statement": "Jess says she doesn't want to read the sign"
}]

Rules:
- One entry per person. Stack all behaviors into that one entry.
- Infer personality from action — don't just restate the action as a trait.
- Source is "self" if they described themselves, otherwise name the observer.
- If two people show the same trait independently, file it under each separately.
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
        
class BehaviorCatcher:
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


behaviorcatcher = BehaviorCatcher()