"""
core/personality_sync.py
Bridges personality_growth.py's ChromaDB output back to personality.txt.

personality_growth.rewrite_personality() rewrites the ChromaDB collections
but doesn't touch personality.txt — which is what Ego actually reads as its
seed. This module wraps the rewrite and distills the result back to disk.

Also provides: run_growth_cycle() — the full observation + rewrite pipeline
called by the archiver and the weekly loop.

personality.txt format:
  Plain prose, written in first person as Gizmo.
  No headers, no JSON. Just voice.
  Ego reads it as the opening of the system prompt.
"""

import asyncio
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

import os

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_PERSONALITY_TXT = _PERSONALITY_DIR / "personality.txt"
_PERSONALITY_BAK = _PERSONALITY_DIR / "personality.bak.txt"


async def rewrite_and_sync(llm) -> bool:
    """
    Run personality_growth.rewrite_personality(), then distill the result
    back into personality.txt so Ego picks it up on the next request.

    Returns True on success.
    """
    try:
        from core.personality_growth import rewrite_personality, _store, CORE_COLLECTION

        # Run the full rewrite into ChromaDB
        success = await rewrite_personality(llm)
        if not success:
            log_error("PersonalitySync", "rewrite_personality() failed — skipping sync")
            return False

        # Pull the fresh core chunks
        store = _store(CORE_COLLECTION)
        if store.count == 0:
            log_event("PersonalitySync", "SYNC_SKIPPED", reason="core collection empty after rewrite")
            return False

        result = store.collection.get()
        core_chunks = result.get("documents", [])

        if not core_chunks:
            log_event("PersonalitySync", "SYNC_SKIPPED", reason="no chunks retrieved")
            return False

        # Distill chunks into a personality.txt — flowing prose, first person
        chunk_text = "\n\n".join(core_chunks)

        prompt = [{
            "role": "user",
            "content": (
                f"The following are structured personality dimensions for an AI named Gizmo.\n\n"
                f"{chunk_text}\n\n"
                f"Rewrite this as a single flowing piece of first-person prose — "
                f"how Gizmo would describe themselves to someone meeting them for the first time. "
                f"Natural, specific, honest. Not a list. Not headers. Just voice.\n\n"
                f"This text will be placed at the start of every system prompt, "
                f"so it should read as: 'I am...' / 'I...' — present tense, first person.\n\n"
                f"4-8 sentences. Warm but not saccharine. Real."
            )
        }]

        result_text = await llm.generate(
            prompt,
            system_prompt=(
                "You distill personality data into a natural first-person voice. "
                "Write as the AI speaking about itself. Prose only. No headers or lists."
            ),
            max_new_tokens=400,
            temperature=0.4,
        )

        if not result_text or not result_text.strip():
            log_event("PersonalitySync", "SYNC_SKIPPED", reason="LLM returned empty")
            return False

        new_personality = result_text.strip()

        # Back up current personality.txt before overwriting
        try:
            if _PERSONALITY_TXT.exists():
                _PERSONALITY_BAK.write_text(
                    _PERSONALITY_TXT.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        except Exception as e:
            log_error("PersonalitySync", "backup failed — continuing anyway", exc=e)

        # Write new personality
        _PERSONALITY_TXT.write_text(new_personality, encoding="utf-8")

        # Tell Ego to reload its seed
        try:
            from core.ego import ego
            ego._reload_seed()
        except Exception as e:
            log_error("PersonalitySync", "Ego seed reload failed", exc=e)

        log_event("PersonalitySync", "SYNC_COMPLETE",
            chars=len(new_personality),
            preview=new_personality[:80],
        )
        log("PersonalitySync", f"personality.txt updated ({len(new_personality)} chars)")
        return True

    except Exception as e:
        log_error("PersonalitySync", "rewrite_and_sync failed", exc=e)
        return False


async def run_growth_cycle(
    session_id: str,
    history,
    current_host: Optional[str],
    fronters: list[str],
    llm,
    rewrite: bool = False,
) -> None:
    """
    Full growth cycle for a session:
      1. observe() — extract and store observations from this session
      2. rewrite_and_sync() — optional, only if rewrite=True

    The archiver calls this after each session with rewrite=False.
    The weekly loop calls this with rewrite=True.
    """
    try:
        from core.personality_growth import observe
        count = await observe(
            session_id=session_id,
            history=history,
            current_host=current_host,
            fronters=fronters,
            llm=llm,
        )
        log_event("PersonalitySync", "OBSERVE_COMPLETE",
            session=session_id[:8],
            observations=count,
        )
    except Exception as e:
        log_error("PersonalitySync", "observe() failed", exc=e)

    if rewrite:
        await rewrite_and_sync(llm)


async def weekly_rewrite_loop(llm) -> None:
    """
    Background loop. Triggers a full rewrite + sync once per week.
    Started by server.py after port bind.
    """
    import asyncio
    WEEK_SECONDS = 7 * 24 * 60 * 60

    log("PersonalitySync", "weekly rewrite loop started")
    while True:
        await asyncio.sleep(WEEK_SECONDS)
        log("PersonalitySync", "weekly rewrite triggered")
        try:
            await rewrite_and_sync(llm)
        except Exception as e:
            log_error("PersonalitySync", "weekly rewrite failed", exc=e)


def start_personality_sync_loop(llm, loop=None) -> None:
    """Schedule the weekly rewrite loop on the running event loop."""
    import asyncio
    loop = loop or asyncio.get_event_loop()
    asyncio.ensure_future(weekly_rewrite_loop(llm), loop=loop)
    log("PersonalitySync", "weekly rewrite loop scheduled")
