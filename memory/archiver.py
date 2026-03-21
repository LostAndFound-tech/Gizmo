"""
memory/archiver.py
Background archiver — watches for inactive sessions, distills conversations
into attributed memory chunks, and ingests them into the right collections.

Flow:
  - Runs every 60 seconds checking all active sessions
  - If a session has been inactive for INACTIVITY_THRESHOLD seconds → archive
  - Slices history into WINDOW_SIZE message windows
  - For each window: identifies who was present, summarizes, ingests
  - Archives go into each fronter's collection + main
  - Session marked as archived to avoid re-processing
"""

import asyncio
import time
from datetime import datetime
from typing import Optional

INACTIVITY_THRESHOLD = 60   # 15 minutes in seconds
WINDOW_SIZE = 4                   # messages per chunk
CHECK_INTERVAL = 60               # check every 60 seconds


async def _summarize_window(
    messages: list[dict],
    fronters: set,
    session_id: str,
    window_index: int,
    llm,
) -> Optional[str]:
    """
    Ask the LLM to distill a 4-message window into a tight memory chunk.
    """
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content']}"
        for m in messages
        if m["role"] in ("user", "assistant")
    )

    if not transcript.strip():
        return None

    fronter_list = ", ".join(sorted(fronters)) if fronters else "unknown"

    prompt = [
        {
            "role": "user",
            "content": (
                f"Summarize this conversation excerpt into a concise memory. "
                f"Write it as a factual paragraph, past tense, as if recording what was discussed. "
                f"Include who said what where perspectives differ. "
                f"Participants: {fronter_list}.\n\n"
                f"{transcript}\n\n"
                f"Be specific about any facts, ideas, or conclusions reached. "
                f"2-4 sentences maximum. No bullet points."
            )
        }
    ]

    try:
        summary = await llm.generate(
            prompt,
            system_prompt=(
                "You distill conversations into precise, factual memory entries. "
                "Past tense. Specific. Attributed where relevant. Never vague."
            ),
            max_new_tokens=200,
            temperature=0.3,
        )
        return summary.strip()
    except Exception as e:
        print(f"[Archiver] Summarization failed for window {window_index}: {e}")
        return None


async def _archive_session(session_id: str, history, llm) -> None:
    """
    Archive a single session — slice, summarize, ingest.
    """
    from core.rag import RAGStore

    messages = history.as_list()
    if not messages:
        return

    print(f"[Archiver] Archiving session {session_id[:8]}... ({len(messages)} messages)")

    # Slice into windows of WINDOW_SIZE
    windows = [
        messages[i:i + WINDOW_SIZE]
        for i in range(0, len(messages), WINDOW_SIZE)
    ]

    archived_count = 0
    session_date = datetime.fromtimestamp(
        messages[0].get("timestamp", time.time())
    ).strftime("%Y-%m-%d")

    for i, window in enumerate(windows):
        # Who was present during this window?
        presence = history.get_fronters_for_window(window)
        fronters = presence["fronters"]
        collections = presence["collections"]  # always includes "main"

        # Summarize this window
        summary = await _summarize_window(
            window,
            fronters=fronters,
            session_id=session_id,
            window_index=i,
            llm=llm,
        )

        if not summary:
            continue

        # Build metadata
        metadata = {
            "source": f"conversation:{session_id[:8]}",
            "type": "archived_conversation",
            "date": session_date,
            "session_id": session_id,
            "window": i,
            "fronters_present": ", ".join(sorted(fronters)) if fronters else "unknown",
        }

        # Raw window text — embedded separately so specific details are searchable
        raw_window = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content']}"
            for m in window
            if m["role"] in ("user", "assistant")
        ).strip()

        # Ingest into every relevant collection
        for collection_name in collections:
            try:
                store = RAGStore(collection_name=collection_name)
                docs = [summary]
                metas = [{**metadata, "collection": collection_name}]

                # Ingest raw window alongside summary so retrieval isn't
                # limited to what survived summarization
                if raw_window:
                    docs.append(raw_window)
                    metas.append({**metadata, "collection": collection_name, "type": "archived_raw"})

                store.ingest_texts(docs, metadatas=metas)
                print(f"[Archiver] → ingested window {i} into '{collection_name}'")
            except Exception as e:
                print(f"[Archiver] Failed to ingest into '{collection_name}': {e}")

        archived_count += 1
        # Small pause between windows to avoid hammering the LLM
        await asyncio.sleep(1)

    print(f"[Archiver] Session {session_id[:8]} archived — {archived_count} chunks across {len(collections)} collections")
    history.archived = True


async def archiver_loop(llm) -> None:
    """
    Background loop. Runs every CHECK_INTERVAL seconds.
    Finds inactive unarchived sessions and archives them.
    """
    from memory.history import get_all_sessions

    print("[Archiver] Background archiver started")

    while True:
        await asyncio.sleep(CHECK_INTERVAL)

        sessions = get_all_sessions()
        for session_id, history in list(sessions.items()):
            if history.archived:
                continue
            if len(history) == 0:
                continue
            if history.seconds_since_active() >= INACTIVITY_THRESHOLD:
                try:
                    await _archive_session(session_id, history, llm)
                except Exception as e:
                    print(f"[Archiver] Error archiving {session_id[:8]}: {e}")


def start_archiver(llm, loop: asyncio.AbstractEventLoop) -> None:
    """Schedule the archiver loop on the running event loop."""
    asyncio.ensure_future(archiver_loop(llm), loop=loop)
