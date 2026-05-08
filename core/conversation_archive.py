"""
core/conversation_archive.py
Writes closed sessions to disk as dated conversation logs.

Structure:
  /data/conversations/
    YYYY-MM-DD/
      index.json                      — all sessions today, searchable
      session_<timestamp>.txt         — raw transcript
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
        "summary": str       — one-line snippet for quick recall
      }
    ]
  }

Mind reads index.json first on temporal queries — cheap.
Full transcripts only loaded when specifically needed.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.log import log, log_event, log_error

_CONVERSATIONS_DIR = Path(os.getenv("CONVERSATIONS_DIR", "/data/conversations"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _today_dir() -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    d = _CONVERSATIONS_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d


def _date_dir(date_str: str) -> Path:
    return _CONVERSATIONS_DIR / date_str


def _index_path(date_str: str) -> Path:
    return _date_dir(date_str) / "index.json"


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


# ── Summary generation ────────────────────────────────────────────────────────

async def _generate_summary(
    transcript: str,
    hosts: list[str],
    topics: list[str],
    llm,
) -> dict:
    """
    Run a summary pass on the closed session transcript.
    Returns a dict with summary, mood, unresolved, notable.
    """
    # Trim transcript if very long — last 4000 chars is enough for summary
    excerpt = transcript[-4000:] if len(transcript) > 4000 else transcript

    host_str = ", ".join(h.title() for h in hosts) if hosts else "unknown"

    prompt = [{
        "role": "user",
        "content": (
            f"This is a conversation transcript involving {host_str}.\n\n"
            f"{excerpt}\n\n"
            f"Summarize this conversation. Respond with ONLY valid JSON, no markdown:\n"
            f'{{\n'
            f'  "summary": "2-3 sentence plain summary of what was discussed",\n'
            f'  "mood": "overall emotional tone of the conversation",\n'
            f'  "unresolved": "anything left hanging or unfinished, or null",\n'
            f'  "notable": ["any facts or moments worth remembering long-term"]\n'
            f'}}'
        )
    }]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You summarize conversation transcripts concisely and factually. "
                "JSON only. No markdown. No preamble."
            ),
            max_new_tokens=300,
            temperature=0.2,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        log_error("Archive", "summary generation failed", exc=e)
        return {
            "summary": "Summary unavailable.",
            "mood": "unknown",
            "unresolved": None,
            "notable": [],
        }


# ── Main archive function ─────────────────────────────────────────────────────

async def archive_session(
    session_id: str,
    history,                    # ConversationHistory
    hosts: list[str],
    topics: list[str],
    opened_at: float,
    closed_at: float,
    llm,
) -> Optional[str]:
    """
    Archive a closed session to disk.
    Returns the transcript filename on success, None on failure.

    Called by session_manager when a session closes.
    """
    messages = history.as_list() if hasattr(history, "as_list") else []
    if not messages:
        log_event("Archive", "SKIP_EMPTY", session=session_id[:8])
        return None

    now = datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    timestamp = int(closed_at)

    # ── Build transcript ──────────────────────────────────────────────────────
    lines = [
        f"Session: {session_id}",
        f"Date: {date_str}",
        f"Opened: {datetime.fromtimestamp(opened_at).strftime('%H:%M:%S')}",
        f"Closed: {datetime.fromtimestamp(closed_at).strftime('%H:%M:%S')}",
        f"Hosts: {', '.join(h.title() for h in hosts) if hosts else 'unknown'}",
        f"Topics: {', '.join(topics) if topics else 'none'}",
        f"Messages: {len(messages)}",
        "",
        "─" * 60,
        "",
    ]

    for msg in messages:
        role    = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        ts      = msg.get("timestamp", "")

        if role == "user":
            speaker = hosts[0].title() if hosts else "User"
        else:
            speaker = "Gizmo"

        time_str = ""
        if ts:
            try:
                time_str = f"[{datetime.fromisoformat(str(ts)).strftime('%H:%M')}] "
            except Exception:
                pass

        lines.append(f"{time_str}{speaker}: {content}")
        lines.append("")

    transcript = "\n".join(lines)

    # ── Write transcript ──────────────────────────────────────────────────────
    transcript_filename = f"session_{timestamp}.txt"
    transcript_path = _date_dir(date_str) / transcript_filename
    try:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(transcript, encoding="utf-8")
        log_event("Archive", "TRANSCRIPT_WRITTEN",
            session=session_id[:8],
            file=transcript_filename,
            messages=len(messages),
        )
    except Exception as e:
        log_error("Archive", "failed to write transcript", exc=e)
        return None

    # ── Generate summary ──────────────────────────────────────────────────────
    summary_data = await _generate_summary(transcript, hosts, topics, llm)
    summary_data.update({
        "session_id":    session_id,
        "timestamp":     timestamp,
        "hosts":         hosts,
        "topics":        topics,
        "message_count": len(messages),
        "opened_at":     datetime.fromtimestamp(opened_at).isoformat(),
        "closed_at":     datetime.fromtimestamp(closed_at).isoformat(),
        "transcript_file": transcript_filename,
    })

    summary_filename = f"session_{timestamp}_summary.json"
    summary_path = _date_dir(date_str) / summary_filename
    try:
        summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        log_event("Archive", "SUMMARY_WRITTEN",
            session=session_id[:8],
            file=summary_filename,
            mood=summary_data.get("mood", "unknown"),
        )
    except Exception as e:
        log_error("Archive", "failed to write summary", exc=e)

    # ── Update daily index ────────────────────────────────────────────────────
    index = _load_index(date_str)
    index["sessions"].append({
        "session_id":    session_id,
        "timestamp":     timestamp,
        "filename":      transcript_filename,
        "summary_file":  summary_filename,
        "hosts":         hosts,
        "topics":        topics,
        "message_count": len(messages),
        "opened_at":     datetime.fromtimestamp(opened_at).strftime("%H:%M"),
        "closed_at":     datetime.fromtimestamp(closed_at).strftime("%H:%M"),
        "summary":       summary_data.get("summary", ""),
        "mood":          summary_data.get("mood", "unknown"),
        "unresolved":    summary_data.get("unresolved"),
    })
    _save_index(date_str, index)

    log_event("Archive", "SESSION_ARCHIVED",
        session=session_id[:8],
        date=date_str,
        hosts=hosts,
        topics=topics,
        messages=len(messages),
    )

    return transcript_filename


# ── Read helpers for Mind ─────────────────────────────────────────────────────

def get_today_index() -> dict:
    """Return today's session index. Used by Mind for temporal queries."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return _load_index(date_str)


def get_index_for_date(date_str: str) -> dict:
    return _load_index(date_str)


def read_transcript(date_str: str, filename: str) -> Optional[str]:
    """Read a full transcript. Only called when Mind needs the full text."""
    path = _date_dir(date_str) / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error("Archive", f"failed to read transcript {filename}", exc=e)
        return None


def get_today_summary_for_prompt() -> str:
    """
    Return a compact summary of today's sessions for injection into
    the rumination prompt or system prompt.
    Returns empty string if no sessions today.
    """
    index = get_today_index()
    sessions = index.get("sessions", [])
    if not sessions:
        return ""

    lines = [f"Today's conversations ({len(sessions)} session{'s' if len(sessions) != 1 else ''}):"]
    for s in sessions:
        hosts   = ", ".join(h.title() for h in s.get("hosts", []))
        time_range = f"{s.get('opened_at', '?')}–{s.get('closed_at', '?')}"
        summary = s.get("summary", "")
        unresolved = s.get("unresolved")
        line = f"  [{time_range}] {hosts}: {summary}"
        if unresolved:
            line += f" (unresolved: {unresolved})"
        lines.append(line)

    return "\n".join(lines)
