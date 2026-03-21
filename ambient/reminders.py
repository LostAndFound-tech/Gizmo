"""
ambient/reminders.py
Reminder system backed by ChromaDB.

Flow:
  - detect_reminder_intent() — called by tagger on every transcript
  - If intent found, parse_reminder() extracts due datetime + message via LLM
  - store_reminder() persists to 'reminders' ChromaDB collection
  - reminder_checker_loop() runs every 30s, fires due reminders into directed queue
  - Delivered reminders marked status="delivered" — they stay as memory

ChromaDB metadata schema per reminder:
  {
    "due_iso":      "2026-03-16T17:00:00",   # ISO datetime string
    "due_date":     "2026-03-16",
    "due_hour":     "17",
    "due_minute":   "00",
    "message":      "call mom",              # extracted reminder body
    "fronter":      "princess",             # who set it
    "status":       "pending" | "delivered",
    "created_at":   "2026-03-16T14:23:00",
    "source":       "ambient_reminder",
    "type":         "reminder",
  }
"""

import asyncio
import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

REMINDERS_COLLECTION = "reminders"
CHECK_INTERVAL = 30  # seconds between checks

# Kept as a fast pre-filter for the most explicit cases only —
# catches obvious phrases without an LLM call. If this misses,
# detect_reminder_intent() falls through to an LLM check.
_REMINDER_PATTERN = re.compile(
    r"\b(remind me|reminder|don't let me forget|make sure i|"
    r"tell me (at|when|to)|note (that|to)|remember to|set a timer)\b",
    re.IGNORECASE,
)

# Module-level LLM reference — set by the pipeline on startup
_llm = None


def set_llm(llm) -> None:
    """Called by the pipeline so reminder detection can use the shared LLM."""
    global _llm
    _llm = llm


async def detect_reminder_intent(transcript: str) -> bool:
    """
    Two-stage check for reminder/timer intent.

    Stage 1: fast regex for explicit trigger phrases — no LLM cost.
    Stage 2: if regex misses, ask the LLM with a tight yes/no prompt.
             Catches natural phrasing like "I put a pizza in for 20 minutes"
             or "laundry's running for another 45" without pattern maintenance.

    Returns True if either stage fires.
    """
    # Stage 1 — free
    if _REMINDER_PATTERN.search(transcript):
        return True

    # Stage 2 — LLM fallback
    if _llm is None:
        return False

    try:
        prompt = [
            {
                "role": "user",
                "content": (
                    f"Does this statement imply something time-bound that a person "
                    f"might want to be notified about when the time is up? "
                    f"Examples that should return YES: "
                    f"'I put a pizza in for 20 minutes', 'laundry's running for 45', "
                    f"'bread needs another half hour', 'I started the timer for 10 minutes'. "
                    f"Examples that should return NO: "
                    f"'I had pizza yesterday', 'I do laundry on Sundays', 'that took forever'. "
                    f"Respond with only YES or NO.\n\n"
                    f"Statement: \"{transcript}\""
                )
            }
        ]
        result = await _llm.generate(
            prompt,
            system_prompt="You detect time-bound activity intent. Respond only YES or NO.",
            max_new_tokens=5,
            temperature=0.0,
        )
        answer = result.strip().upper()
        detected = answer.startswith("YES")
        if detected:
            print(f"[Reminders] LLM detected implicit timer intent: '{transcript[:60]}'")
        return detected
    except Exception as e:
        print(f"[Reminders] Intent detection LLM call failed: {e}")
        return False


async def parse_reminder(
    transcript: str,
    llm,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """
    Use LLM to extract due datetime and message from a natural language transcript.
    Returns dict with 'due_iso' and 'message', or None if parsing fails.

    Handles:
      - "remind me at 5" → today at 17:00 (assumes PM if ambiguous and hour < current)
      - "remind me at 5am" → today at 05:00
      - "remind me in two hours" → now + 2h
      - "remind me Friday at 3" → next Friday at 15:00
      - "remind me tomorrow morning" → tomorrow at 09:00
    """
    now = now or datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M")
    weekday = now.strftime("%A")

    prompt = [
        {
            "role": "user",
            "content": (
                f"Current date and time: {now_str} ({weekday})\n\n"
                f"Extract the reminder from this transcript and respond with ONLY valid JSON — "
                f"no markdown, no explanation:\n\n"
                f'"{transcript}"\n\n'
                f"Respond with exactly:\n"
                f'{{"due_iso": "YYYY-MM-DDTHH:MM:00", "message": "what to remind"}}\n\n'
                f"Rules:\n"
                f"- due_iso must be a full ISO datetime string\n"
                f"- If only a time is given (e.g. 'at 5'), assume today. "
                f"  If that time has already passed today, assume tomorrow.\n"
                f"- If AM/PM is ambiguous, prefer the next occurrence "
                f"  (e.g. if it's 2pm and they say 'at 5', use 5pm today)\n"
                f"- 'in two hours' means {now_str} + 2 hours\n"
                f"- 'tomorrow morning' = tomorrow at 09:00\n"
                f"- 'Friday' = the next upcoming Friday\n"
                f"- message should be concise — just what needs to be remembered, "
                f"  not the whole transcript. Strip 'remind me to', 'remind me at X to', etc.\n"
                f"- If you cannot find a clear reminder in the transcript, "
                f'  respond with {{"due_iso": null, "message": null}}'
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You extract reminder intent from natural language. "
                "Always respond with valid JSON only. No markdown. No explanation."
            ),
            max_new_tokens=100,
            temperature=0.1,
        )

        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        due_iso = parsed.get("due_iso")
        message = parsed.get("message")

        if not due_iso or not message:
            print("[Reminders] LLM found no reminder intent in transcript")
            return None

        # Validate the datetime parses correctly
        due_dt = datetime.fromisoformat(due_iso)

        # Sanity check — don't accept reminders more than 1 year out
        if due_dt > now + timedelta(days=365):
            print(f"[Reminders] Parsed datetime too far in future: {due_iso}")
            return None

        # If due time is in the past, bump to same time tomorrow
        if due_dt < now:
            print(f"[Reminders] Due time {due_iso} is in the past — bumping to tomorrow")
            due_dt = due_dt + timedelta(days=1)

        return {
            "due_iso": due_dt.isoformat(timespec="seconds"),
            "due_date": due_dt.strftime("%Y-%m-%d"),
            "due_hour": due_dt.strftime("%H"),
            "due_minute": due_dt.strftime("%M"),
            "message": message.strip(),
        }

    except Exception as e:
        print(f"[Reminders] Parse failed: {e}")
        return None


def store_reminder(
    due_iso: str,
    due_date: str,
    due_hour: str,
    due_minute: str,
    message: str,
    set_by: Optional[str] = None,       # who created the reminder
    raw_transcript: Optional[str] = None,
) -> str:
    """
    Persist a reminder to the 'reminders' ChromaDB collection.
    Returns the reminder ID.

    set_by: the fronter who set the reminder (captured at creation time).
    At delivery time, the current host is read from context separately —
    if they differ, the delivery message attributes the reminder to set_by.
    """
    from core.rag import RAGStore

    reminder_id = f"reminder_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now().isoformat(timespec="seconds")

    doc_text = f"Reminder set for {due_iso}: {message}"

    metadata = {
        "due_iso": due_iso,
        "due_date": due_date,
        "due_hour": due_hour,
        "due_minute": due_minute,
        "message": message,
        "set_by": set_by or "unknown",
        "status": "pending",
        "created_at": now_iso,
        "source": "ambient_reminder",
        "type": "reminder",
    }
    if raw_transcript:
        metadata["raw_transcript"] = raw_transcript[:300]

    try:
        store = RAGStore(collection_name=REMINDERS_COLLECTION)
        store.ingest_texts(
            [doc_text],
            metadatas=[metadata],
            ids=[reminder_id],
        )
        print(f"[Reminders] Stored: '{message}' due {due_iso} (id={reminder_id})")
        return reminder_id
    except Exception as e:
        print(f"[Reminders] Failed to store reminder: {e}")
        return ""


def _mark_delivered(reminder_id: str) -> None:
    """Update a reminder's status to 'delivered' in ChromaDB."""
    from core.rag import RAGStore
    try:
        store = RAGStore(collection_name=REMINDERS_COLLECTION)
        # ChromaDB update requires fetching first
        result = store.collection.get(ids=[reminder_id])
        if not result["ids"]:
            return
        existing_meta = result["metadatas"][0]
        existing_meta["status"] = "delivered"
        existing_meta["delivered_at"] = datetime.now().isoformat(timespec="seconds")
        store.collection.update(
            ids=[reminder_id],
            metadatas=[existing_meta],
        )
        print(f"[Reminders] Marked delivered: {reminder_id}")
    except Exception as e:
        print(f"[Reminders] Failed to mark delivered: {e}")


def _get_due_reminders() -> list[dict]:
    """
    Fetch all pending reminders whose due_iso <= now.
    Returns list of {id, message, fronter, due_iso}.
    """
    from core.rag import RAGStore
    import chromadb
    from core.rag import CHROMA_PERSIST_DIR

    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = [c.name for c in client.list_collections()]
        if REMINDERS_COLLECTION not in existing:
            return []

        store = RAGStore(collection_name=REMINDERS_COLLECTION)
        if store.count == 0:
            return []

        # Get all pending reminders
        results = store.collection.get(
            where={
                "$and": [
                    {"status": {"$eq": "pending"}},
                    {"type": {"$eq": "reminder"}},
                ]
            }
        )

        if not results["ids"]:
            return []

        now = datetime.now()
        due = []

        for reminder_id, meta in zip(results["ids"], results["metadatas"]):
            due_iso = meta.get("due_iso", "")
            if not due_iso:
                continue
            try:
                due_dt = datetime.fromisoformat(due_iso)
                if due_dt <= now:
                    due.append({
                        "id": reminder_id,
                        "message": meta.get("message", "something"),
                        "set_by": meta.get("set_by", ""),
                        "due_iso": due_iso,
                    })
            except ValueError:
                continue

        return due

    except Exception as e:
        print(f"[Reminders] Error fetching due reminders: {e}")
        return []


async def reminder_checker_loop(directed_queue: asyncio.Queue) -> None:
    """
    Background loop. Checks for due reminders every CHECK_INTERVAL seconds.
    Fires due reminders into the directed queue for agent delivery.
    """
    print("[Reminders] Checker loop started")

    while True:
        await asyncio.sleep(CHECK_INTERVAL)

        try:
            due = _get_due_reminders()
            for reminder in due:
                print(f"[Reminders] Firing: '{reminder['message']}' (due {reminder['due_iso']})")

                set_by = reminder.get("set_by", "")
                known_set_by = set_by and set_by != "unknown"

                # current_host will be resolved by the agent at delivery time
                # We embed set_by into the delivery text so the agent can
                # attribute naturally if the current host differs
                if known_set_by:
                    delivery_text = (
                        f"[REMINDER DELIVERY] "
                        f"Hey, this is a reminder that {set_by.capitalize()} set: "
                        f"{reminder['message']}"
                    )
                else:
                    delivery_text = (
                        f"[REMINDER DELIVERY] "
                        f"Hey, just a reminder: {reminder['message']}"
                    )

                await directed_queue.put({
                    "transcript": delivery_text,
                    "context": {},          # agent reads current host from live context
                    "type": "reminder",
                    "reminder_id": reminder["id"],
                    "set_by": set_by,
                })

                _mark_delivered(reminder["id"])

        except Exception as e:
            print(f"[Reminders] Checker error: {e}")


def start_reminder_checker(
    directed_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop = None,
) -> None:
    """Schedule the reminder checker on the running event loop."""
    loop = loop or asyncio.get_event_loop()
    asyncio.ensure_future(reminder_checker_loop(directed_queue), loop=loop)
    print("[Reminders] Checker scheduled.")
