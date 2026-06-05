"""
core/event_extractor.py
Atomic event extraction from conversational exchanges.

Runs on every exchange (user message + Gizmo response) as a fire-and-forget
async task. Produces a list of atomic event records — one per discrete action
or statement — and appends them to a SQLite action log.

No psychology here. No scoring. No interpretation beyond what the narrative
plainly shows. That comes later, in the psychology engine batch pass.

Each record captures:
  - Who did it (subject)
  - Who received it (recipient, if any)
  - What they did (action, as described — full and uncut)
  - Why the narrative suggests they did it (intent, single word or short phrase)
  - Whether it was physical, verbal, postural, or expressed emotion
  - The source exchange for traceback

LLM used: cheap, fast model. This is not a reasoning task.
Mistral Small 3 or Ministral 3B via OpenRouter both work well here.
The prompt is tight and the output is structured JSON — low token cost.

Usage (fire-and-forget from archivist):
    asyncio.create_task(
        event_extractor.extract(
            user_message=brief["message"],
            gizmo_response=response,
            subject=brief["host"],
            session_file=session_file,
            timestamp=tz_now().isoformat(),
        )
    )
"""

import asyncio
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = Path("/data")
DB_PATH     = DATA_DIR / "action_log.db"

# Cheap model — this is classification, not reasoning
EXTRACTOR_MODEL = "mistralai/mistral-small-3"



# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """
    You determine the context of a beat of a conversation. You're excellent at determining the subject of a sentence,
    and determining whether they are speaking, or acting, and to whom or to what. You write them out in valid JSON 
    format. Do not infer any details unless you are confident they are correct. Write simple statements in [SUBJECT, VERB, SUBJECT] format. 
""".strip()


def _build_prompt(
    user_message: str,
    gizmo_response: str,
    subject: str,
) -> str:
    return (
        f"Determine the context of this exchange. Note any details in appearance, location, objects, or the texture of the moment.\n\n"
        f"The person speaking to Gizmo is: {subject}\n\n"
        f"{subject} wrote:\n{user_message}\n\n"
        f"Gizmo responded:\n{gizmo_response}"
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[list[dict]]:
    """
    Call the cheap extractor model. Returns parsed event list or None on failure.
    Uses OpenRouter directly — separate from the main LLM client so we can
    specify a different model without touching the main config.
    """
    try:
        import os
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=30,
        )

        response = await client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=3000,
            temperature=0.0,   # deterministic — this is extraction not generation
        )

        raw = response.choices[0].message.content or ""
        if not raw.strip():
            log_event("EventExtractor", "EMPTY_RESPONSE", model=EXTRACTOR_MODEL)
            return None

        # Strip markdown fences if the model added them anyway
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)

        if not isinstance(parsed, list):
            log_event("EventExtractor", "BAD_SHAPE", raw=raw[:200])
            return None

        return parsed

    except json.JSONDecodeError as e:
        log_error("EventExtractor", "JSON parse failed", exc=e)
        return None
    except Exception as e:
        log_error("EventExtractor", "LLM call failed", exc=e)
        return None
    
# ── File write ────────────────────────────────────────────────────────────────

def _actions_file(subject: str) -> Path:
    """
    Lazy path resolution — called at write time, never at import.
    /data/headmates/{subject}/actions/{YYYY-MM-DD}.jsonl
    """
    date_str = tz_now().strftime("%Y-%m-%d")
    path = DATA_DIR / "testing" / subject.lower() / "actions" / f"{date_str}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_to_file(record: dict, subject: str) -> None:
    """Append one event record as a JSON line to the headmate's daily action file."""
    try:
        path = _actions_file(subject)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log_error("EventExtractor", "File write failed", exc=e)

# ── Public API ────────────────────────────────────────────────────────────────

class ContentDeductor:
    """
    Singleton. Call extract() as a fire-and-forget task from the archivist.
    """

    def __init__(self):
        pass

    async def extract(
        self,
        user_message:   str,
        gizmo_response: str,
        subject:        str,
        session_file:   str,
    ) -> None:
        """
        Extract atomic events from one exchange and append to action_log.
        Designed to run as asyncio.create_task() — never blocks the main flow.
        Timestamp is generated internally via tz_now() — never trusted from caller.
        """
        if not user_message.strip() and not gizmo_response.strip():
            return

        try:
            prompt = _build_prompt(user_message, gizmo_response, subject)
            context = await _call_llm(prompt)

            if not context:
                log_event("EventExtractor", "NO_EVENTS_EXTRACTED",
                    subject=subject,
                    session=session_file,
                )
                return

            return context
        except Exception as E:
            print("context extractor failed with an error:", E)

content_deductor = ContentDeductor()