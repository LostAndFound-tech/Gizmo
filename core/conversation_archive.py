"""
core/conversation_archive.py
Writes closed sessions to disk as dated conversation logs.

Transcripts are written incrementally — one exchange at a time via
append_exchange() — so the file is always current and session close
just needs to run the summary pass on what's already there.

Structure:
  /data/conversations/
    YYYY-MM-DD/
      index.json                       — all sessions today, searchable
      session_<timestamp>.txt          — raw transcript (written incrementally)
      session_<timestamp>_summary.json — summary + metadata

index.json schema:
  {
    "date": "YYYY-MM-DD",
    "sessions": [
      {
        "session_id": str,
        "timestamp": int,
        "filename": str,
        "summary_file": str,
        "hosts": [str],
        "topics": [str],
        "message_count": int,
        "opened_at": str,
        "closed_at": str,
        "mood": str,
        "summary": str,
        "unresolved": str | null
      }
    ]
  }
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

_CONVERSATIONS_DIR = Path(os.getenv("CONVERSATIONS_DIR", "/data/conversations"))


# ── Path helpers ──────────────────────────────────────────────────────────────

def _date_dir(date_str: str) -> Path:
    d = _CONVERSATIONS_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _index_path(date_str: str) -> Path:
    return _CONVERSATIONS_DIR / date_str / "index.json"


def _load_index(date_str: str) -> dict:
    path = _index_path(date_str)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"date": date_str, "sessions": []}
    except Exception as e:
        log_error("Archive", f"failed to load index for {date_str}", exc=e)
        return {"date": date_str, "sessions": []}


def _save_index(date_str: str, index: dict) -> None:
    path = _index_path(date_str)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except Exception as e:
        log_error("Archive", f"failed to save index for {date_str}", exc=e)


# ── Transcript path helpers ───────────────────────────────────────────────────

def _transcript_path(session_id: str, opened_at: float) -> Path:
    """Deterministic path for a session's transcript."""
    date_str  = datetime.fromtimestamp(opened_at).strftime("%Y-%m-%d")
    timestamp = int(opened_at)
    return _date_dir(date_str) / f"session_{timestamp}.txt"


# ── Incremental write — called per exchange ───────────────────────────────────

def append_exchange(
    session_id: str,
    opened_at: float,
    user_message: str,
    gizmo_response: str,
    host: str,
    timestamp: float = None,
) -> None:
    """
    Append a single exchange to the transcript file.
    Creates the file with a header on first call.
    Called by archivist.receive_outgoing() after every exchange.
    Never raises — errors logged and swallowed.
    """
    if not user_message and not gizmo_response:
        return

    ts = timestamp or time.time()
    time_str = datetime.fromtimestamp(ts).strftime("%H:%M")
    speaker  = (host or "User").title()
    path     = _transcript_path(session_id, opened_at)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write header if file is new
        if not path.exists():
            date_str   = datetime.fromtimestamp(opened_at).strftime("%Y-%m-%d")
            opened_str = datetime.fromtimestamp(opened_at).strftime("%H:%M:%S")
            header = (
                f"Session: {session_id}\n"
                f"Date: {date_str}\n"
                f"Opened: {opened_str}\n"
                f"Host: {speaker}\n"
                f"\n{'─' * 60}\n\n"
            )
            path.write_text(header, encoding="utf-8")

        # Append the exchange
        existing = path.read_text(encoding="utf-8")
        exchange = (
            f"[{time_str}] {speaker}: {user_message.strip()}\n"
            f"[{time_str}] Gizmo: {gizmo_response.strip()}\n\n"
        )
        path.write_text(existing + exchange, encoding="utf-8")

    except Exception as e:
        log_error("Archive", f"append_exchange failed for {session_id[:8]}", exc=e)


# ── Summary generation ────────────────────────────────────────────────────────

async def _generate_summary(
    transcript: str,
    hosts: list,
    topics: list,
    llm,
) -> dict:
    """
    Run a summary pass on the transcript.
    Returns dict with summary, mood, unresolved, notable.
    """
    # Use last 6000 chars — enough for a real summary without blowing context
    excerpt = transcript[-6000:] if len(transcript) > 6000 else transcript

    if not excerpt.strip():
        return {
            "summary": "No transcript content.",
            "mood": "unknown",
            "unresolved": None,
            "notable": [],
        }

    host_str = ", ".join(h.title() for h in hosts) if hosts else "unknown"

    prompt = [{
        "role": "user",
        "content": (
            f"This is a conversation transcript involving {host_str}.\n\n"
            f"{excerpt}\n\n"
            f"Summarize this conversation in detail. What was actually discussed? "
            f"What did each person say, share, or feel? What matters here?\n\n"
            f"Respond with ONLY valid JSON, no markdown:\n"
            f'{{\n'
            f'  "summary": "detailed summary of what was actually discussed and what mattered",\n'
            f'  "mood": "overall emotional tone",\n'
            f'  "unresolved": "anything left hanging or unfinished, or null",\n'
            f'  "notable": ["specific facts, moments, or things worth remembering long-term"]\n'
            f'}}'
        )
    }]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You summarize conversation transcripts in detail. "
                "Capture what was actually said and what mattered. "
                "JSON only. No markdown. No preamble."
            ),
            max_new_tokens=600,
            temperature=0.2,
        )
        raw = raw.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as e:
        log_error("Archive", f"summary generation failed: {e}", exc=None)
        return {
            "summary": "Summary generation failed.",
            "mood": "unknown",
            "unresolved": None,
            "notable": [],
        }


# ── Session close — finalize and summarize ────────────────────────────────────

async def finalize_session(
    session_id: str,
    opened_at: float,
    closed_at: float,
    hosts: list,
    topics: list,
    message_count: int,
    llm,
) -> Optional[str]:
    """
    Called by session_manager when a session closes.
    Reads the already-written transcript, generates a summary, updates the index.
    Returns the transcript filename on success, None on failure.
    """
    path = _transcript_path(session_id, opened_at)

    if not path.exists():
        log_event("Archive", "SKIP_NO_TRANSCRIPT", session=session_id[:8])
        return None

    transcript = path.read_text(encoding="utf-8")
    if not transcript.strip():
        log_event("Archive", "SKIP_EMPTY_TRANSCRIPT", session=session_id[:8])
        return None

    date_str  = datetime.fromtimestamp(opened_at).strftime("%Y-%m-%d")
    timestamp = int(opened_at)
    transcript_filename = f"session_{timestamp}.txt"

    # Append closing line to transcript
    try:
        closed_str = datetime.fromtimestamp(closed_at).strftime("%H:%M:%S")
        path.write_text(
            transcript + f"{'─' * 60}\nClosed: {closed_str}\n",
            encoding="utf-8"
        )
    except Exception as e:
        log_error("Archive", "failed to write closing line", exc=e)

    # Generate summary
    summary_data = await _generate_summary(transcript, hosts, topics, llm)
    summary_data.update({
        "session_id":      session_id,
        "timestamp":       timestamp,
        "hosts":           hosts,
        "topics":          topics,
        "message_count":   message_count,
        "opened_at":       datetime.fromtimestamp(opened_at).isoformat(),
        "closed_at":       datetime.fromtimestamp(closed_at).isoformat(),
        "transcript_file": transcript_filename,
    })

    # Write summary file
    summary_filename = f"session_{timestamp}_summary.json"
    summary_path = _date_dir(date_str) / summary_filename
    try:
        summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        log_event("Archive", "SUMMARY_WRITTEN",
            session=session_id[:8],
            mood=summary_data.get("mood", "unknown"),
            summary_preview=summary_data.get("summary", "")[:80],
        )
    except Exception as e:
        log_error("Archive", "failed to write summary", exc=e)

    # Update daily index
    index = _load_index(date_str)

    # Remove any existing entry for this session
    index["sessions"] = [
        s for s in index["sessions"]
        if s.get("session_id") != session_id
    ]

    index["sessions"].append({
        "session_id":    session_id,
        "timestamp":     timestamp,
        "filename":      transcript_filename,
        "summary_file":  summary_filename,
        "hosts":         hosts,
        "topics":        topics,
        "message_count": message_count,
        "opened_at":     datetime.fromtimestamp(opened_at).strftime("%H:%M"),
        "closed_at":     datetime.fromtimestamp(closed_at).strftime("%H:%M"),
        "summary":       summary_data.get("summary", ""),
        "mood":          summary_data.get("mood", "unknown"),
        "unresolved":    summary_data.get("unresolved"),
        "notable":       summary_data.get("notable", []),
    })

    _save_index(date_str, index)

    # ── Ingest summary into each host's ChromaDB collection ──────────────────
    # This is what makes conversations retrievable by Mind.
    # Each host gets an entry in their personal collection AND in memory.
    summary_text = summary_data.get("summary", "")
    if summary_text and summary_text not in ("Summary unavailable.", "No transcript content.", "Summary generation failed."):
        try:
            from tools.memory_tool import _get_collection, MEMORY_COLLECTION
            import uuid as _uuid

            opened_str = datetime.fromtimestamp(opened_at).strftime("%H:%M")
            closed_str = datetime.fromtimestamp(closed_at).strftime("%H:%M")
            hosts_str  = ", ".join(h.title() for h in hosts)
            transcript_link = f"/data/conversations/{date_str}/{transcript_filename}"

            # Build the entry content — summary + link
            entry_content = (
                f"Conversation with {hosts_str} on {date_str} ({opened_str}–{closed_str})
"
                f"{summary_text}"
            )
            if summary_data.get("unresolved"):
                entry_content += f"
Unresolved: {summary_data['unresolved']}"

            entry_metadata = {
                "type":             "conversation_summary",
                "subject":          hosts[0].lower() if hosts else "",
                "hosts":            ", ".join(hosts),
                "date":             date_str,
                "session_id":       session_id,
                "transcript_file":  transcript_link,
                "mood":             summary_data.get("mood", "unknown"),
                "written_at":       datetime.fromtimestamp(closed_at).isoformat(timespec="seconds"),
                "tags":             f"conversation,{','.join(hosts)}",
            }

            # Write to memory collection (primary)
            mem_col = _get_collection(MEMORY_COLLECTION)
            mem_col.add(
                documents=[entry_content],
                metadatas=[entry_metadata],
                ids=[f"conv_{session_id[:16]}"],
            )

            # Also write to each host's personal collection
            try:
                from core.rag import RAGStore
                for host in hosts:
                    host_store = RAGStore(collection_name=host.lower())
                    host_store.ingest_texts(
                        [entry_content],
                        metadatas=[entry_metadata],
                        ids=[f"conv_{session_id[:16]}_{host.lower()}"],
                    )
            except Exception as e:
                log_error("Archive", f"failed to ingest into host collections: {e}", exc=None)

            # Write notable facts to memory too
            for notable in summary_data.get("notable", []):
                if notable and len(notable) > 10:
                    try:
                        mem_col.add(
                            documents=[notable],
                            metadatas=[{
                                "type":       "fact",
                                "subject":    hosts[0].lower() if hosts else "",
                                "date":       date_str,
                                "session_id": session_id,
                                "written_at": datetime.fromtimestamp(closed_at).isoformat(timespec="seconds"),
                                "tags":       f"notable,conversation,{','.join(hosts)}",
                                "source":     "conversation_summary",
                            }],
                            ids=[f"notable_{session_id[:12]}_{_uuid.uuid4().hex[:8]}"],
                        )
                    except Exception:
                        pass

            log_event("Archive", "SUMMARY_INGESTED",
                session=session_id[:8],
                hosts=hosts,
                collections=["memory"] + [h.lower() for h in hosts],
            )

        except Exception as e:
            log_error("Archive", f"failed to ingest summary into ChromaDB: {e}", exc=None)

    log_event("Archive", "SESSION_FINALIZED",
        session=session_id[:8],
        date=date_str,
        hosts=hosts,
        messages=message_count,
        transcript=transcript_filename,
    )

    return transcript_filename


# ── Read helpers for Mind ─────────────────────────────────────────────────────

def get_today_index() -> dict:
    return _load_index(_today_str())


def get_index_for_date(date_str: str) -> dict:
    return _load_index(date_str)


def read_transcript(date_str: str, filename: str) -> Optional[str]:
    path = _CONVERSATIONS_DIR / date_str / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error("Archive", f"failed to read transcript {filename}", exc=e)
        return None


def get_today_summary_for_prompt() -> str:
    """
    Compact summary of today's sessions for rumination prompt.
    """
    index = get_today_index()
    sessions = index.get("sessions", [])
    if not sessions:
        return ""

    lines = [f"Today's conversations ({len(sessions)} session{'s' if len(sessions) != 1 else ''}):"]
    for s in sessions:
        hosts      = ", ".join(h.title() for h in s.get("hosts", []))
        time_range = f"{s.get('opened_at', '?')}–{s.get('closed_at', '?')}"
        summary    = s.get("summary", "")
        unresolved = s.get("unresolved")
        line = f"  [{time_range}] {hosts}: {summary}"
        if unresolved:
            line += f" (unresolved: {unresolved})"
        lines.append(line)

    return "\n".join(lines)