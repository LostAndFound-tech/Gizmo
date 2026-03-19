"""
ambient/pipeline.py
Orchestrates the full ambient listening pipeline:
  listener → transcriber → tagger → RAG ingest

Also handles the "directed at Gizmo" case — if speech is directed at Gizmo,
it queues the transcript as a user message for the agent to respond to.

Usage:
    from ambient.pipeline import start_ambient, stop_ambient, get_directed_queue

    # Start background ambient listening
    start_ambient(llm, loop=asyncio.get_event_loop())

    # In your main agent loop, drain the directed queue
    queue = get_directed_queue()
    while not queue.empty():
        transcript = await queue.get()
        # handle as if user sent a message
"""

import asyncio
import time
from datetime import datetime
from typing import Optional

from ambient.listener import listen
from ambient.transcriber import transcribe
from ambient.tagger import tag
from ambient.reminders import store_reminder, start_reminder_checker

AMBIENT_COLLECTION = "ambient_log"

# Queue for transcripts directed at Gizmo — agent drains this
_directed_queue: asyncio.Queue = asyncio.Queue()
_ambient_task: Optional[asyncio.Task] = None


def get_directed_queue() -> asyncio.Queue:
    return _directed_queue


async def _ingest_ambient_chunk(
    summary: str,
    topics: list[str],
    raw_transcript: str,
    llm,
    context: Optional[dict] = None,
) -> None:
    """Ingest a tagged ambient chunk into the ambient_log collection."""
    from core.rag import RAGStore

    now = datetime.now()
    metadata = {
        "source": "ambient_mic",
        "type": "ambient_transcript",
        "date": now.strftime("%Y-%m-%d"),
        "hour": now.strftime("%H"),           # for "earlier today" queries
        "time": now.strftime("%H:%M"),
        "timestamp": str(int(time.time())),
        "topics": ", ".join(topics),          # comma-sep string (ChromaDB limitation)
        "raw_transcript": raw_transcript[:500],  # keep original, capped
        "collection": AMBIENT_COLLECTION,
    }

    # Add fronter context if available
    if context:
        host = context.get("current_host", "")
        if host:
            metadata["fronter"] = host

    try:
        store = RAGStore(collection_name=AMBIENT_COLLECTION)
        store.ingest_texts(
            [summary],
            metadatas=[metadata],
        )
        print(f"[Pipeline] Ingested ambient chunk | topics: {topics}")
    except Exception as e:
        print(f"[Pipeline] Ingest failed: {e}")


async def _ambient_loop(
    llm,
    context_fn=None,  # optional callable that returns current context dict
    device_index: int = None,
) -> None:
    """
    Main ambient loop. Runs forever until cancelled.
    Pulls utterances from the listener, transcribes, tags, ingests.
    """
    print("[Pipeline] Ambient pipeline starting...")

    async for audio_chunk in listen(device_index=device_index):
        try:
            # 1. Transcribe
            result = await transcribe(audio_chunk)
            if not result.meaningful:
                continue

            # 2. Get current context (fronter info etc.) if available
            context = context_fn() if context_fn else None

            # 3. Tag
            tag_result = await tag(result.text, llm=llm, context=context)

            # 4. Store reminder if one was detected
            if tag_result.reminder:
                r = tag_result.reminder
                fronter = (context or {}).get("current_host", "")
                store_reminder(
                    due_iso=r["due_iso"],
                    due_date=r["due_date"],
                    due_hour=r["due_hour"],
                    due_minute=r["due_minute"],
                    message=r["message"],
                    set_by=fronter,
                    raw_transcript=result.text,
                )

            # 5. If directed at Gizmo, queue for agent response
            if tag_result.directed_at_gizmo:
                print(f"[Pipeline] Directed at Gizmo: '{result.text[:60]}'")
                await _directed_queue.put({
                    "transcript": result.text,
                    "context": context,
                })

            # 6. Always ingest into ambient_log (even directed speech is worth remembering)
            await _ingest_ambient_chunk(
                summary=tag_result.summary,
                topics=tag_result.topics,
                raw_transcript=result.text,
                llm=llm,
                context=context,
            )

        except Exception as e:
            print(f"[Pipeline] Error processing audio chunk: {e}")
            # Don't crash the loop — keep listening
            continue


def start_ambient(
    llm,
    loop: asyncio.AbstractEventLoop = None,
    context_fn=None,
    device_index: int = None,
) -> None:
    """
    Schedule the ambient pipeline on the running event loop.
    Call once at startup. Safe to call multiple times — won't double-start.
    """
    global _ambient_task

    if _ambient_task is not None and not _ambient_task.done():
        print("[Pipeline] Ambient pipeline already running.")
        return

    loop = loop or asyncio.get_event_loop()
    _ambient_task = asyncio.ensure_future(
        _ambient_loop(llm, context_fn=context_fn, device_index=device_index),
        loop=loop,
    )
    start_reminder_checker(_directed_queue, loop=loop)
    print("[Pipeline] Ambient pipeline started.")


def stop_ambient() -> None:
    """Cancel the ambient pipeline."""
    global _ambient_task
    if _ambient_task and not _ambient_task.done():
        _ambient_task.cancel()
        print("[Pipeline] Ambient pipeline stopped.")
    _ambient_task = None