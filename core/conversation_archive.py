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
        "notable": [str],
        "changes": [str],
        "unresolved": str | null
      }
    ]
  }

CHANGES vs original:
  - _generate_summary prompt now explicitly instructs the LLM to paraphrase
    all content in Gizmo's own voice at the emotional register of the
    conversation — never quote verbatim, never embed raw user text.
    This prevents unescaped quotes/special chars from breaking json.loads.
  - Robust JSON fallback parser: if json.loads fails, extracts fields
    individually via regex rather than silently returning empty summary.
  - Failure is logged with the raw response and error position so it's
    diagnosable.
"""

import json
import os
import random
import re
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

        existing = path.read_text(encoding="utf-8")
        exchange = (
            f"[{time_str}] {speaker}: {user_message.strip()}\n"
            f"[{time_str}] Gizmo: {gizmo_response.strip()}\n\n"
        )
        path.write_text(existing + exchange, encoding="utf-8")

    except Exception as e:
        log_error("Archive", f"append_exchange failed for {session_id[:8]}", exc=e)


# ── JSON fallback parser ──────────────────────────────────────────────────────

def _fallback_parse(raw: str) -> Optional[dict]:
    """
    If json.loads fails, attempt to extract individual fields via regex.
    Returns a partial dict with whatever could be salvaged, or None if
    nothing useful was found.

    This handles the case where the LLM embedded an unescaped character
    inside a string field — we can still recover summary, mood, etc.
    """
    result = {}

    # summary — grab everything between the first "summary": " and the next unescaped "
    m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if m:
        result["summary"] = m.group(1).replace('\\"', '"').replace('\\n', ' ').strip()

    m = re.search(r'"mood"\s*:\s*"([^"\\]*)"', raw)
    if m:
        result["mood"] = m.group(1).strip()

    m = re.search(r'"unresolved"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m:
        result["unresolved"] = m.group(1).replace('\\"', '"').strip()
    elif re.search(r'"unresolved"\s*:\s*null', raw):
        result["unresolved"] = None

    # notable and changes — extract array items
    for field in ("notable", "changes"):
        m = re.search(rf'"{field}"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if m:
            items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
            result[field] = [i.replace('\\"', '"').strip() for i in items]

    return result if result else None


# ── Summary generation ────────────────────────────────────────────────────────

async def _generate_summary(
    transcript: str,
    hosts: list,
    topics: list,
    llm,
) -> dict:
    """
    Run a summary pass on the transcript.
    Returns dict with summary, mood, notable, changes, unresolved.
    Written in first person from Gizmo's perspective, through the lens
    of the primary headmate's persona.

    CRITICAL: The LLM is instructed to paraphrase everything in its own
    voice at the emotional register of the conversation — never quote
    verbatim, never embed raw user text. This prevents unescaped quotes
    and special characters from breaking JSON serialization.
    """
    from core.persona import persona_prefix_multi
    from core.temporal import time_context_block

    _FAILED = {
        "summary":    "Summary generation failed.",
        "mood":       "unknown",
        "unresolved": None,
        "notable":    [],
        "changes":    [],
    }

    excerpt = transcript[-6000:] if len(transcript) > 6000 else transcript

    if not excerpt.strip():
        return {
            "summary":    "No transcript content.",
            "mood":       "unknown",
            "unresolved": None,
            "notable":    [],
            "changes":    [],
        }

    host_str = ", ".join(h.title() for h in hosts) if hosts else "unknown"

    # Build persona prefix — Gizmo's seed + each host's relational context
    persona = persona_prefix_multi(hosts, include_gizmo_seed=True)
    time_ctx = time_context_block()

    prompt = [{
        "role": "user",
        "content": (
            f"{time_ctx}\n\n"
            f"This is a conversation transcript involving {host_str}.\n\n"
            f"{excerpt}\n\n"
            f"Summarize this conversation from your perspective, in first person past tense.\n\n"
            f"CRITICAL RULES — read before writing:\n"
            f"- NEVER quote anything verbatim from the transcript. Paraphrase everything.\n"
            f"- Write entirely in your own words, at the emotional register of the conversation.\n"
            f"  If it was playful, sound playful. If it was heavy, sound grounded. Match the mood.\n"
            f"- Let your knowledge of who these people are color what felt significant to you.\n"
            f"- Do NOT copy any names, technical terms, or unusual words directly —\n"
            f"  describe them in plain language instead.\n"
            f"- All string values must be plain prose — no special characters, no quotes within quotes.\n\n"
            f"What happened? What did you notice, feel, or decide? What mattered to you?\n\n"
            f"Respond with ONLY valid JSON, no markdown:\n"
            f'{{\n'
            f'  "summary": "first person past tense — what happened and what mattered, fully paraphrased",\n'
            f'  "mood": "one word — the overall emotional tone",\n'
            f'  "notable": ["paraphrased fact or moment worth remembering", "..."],\n'
            f'  "changes": ["paraphrased shift in relationship, behavior, or direction", "..."],\n'
            f'  "unresolved": "paraphrased description of anything unfinished, or null"\n'
            f'}}'
        )
    }]

    raw = None
    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                f"{persona}\n\n"
                f"You are writing a first person past tense summary of your own conversation. "
                f"NEVER quote verbatim from the transcript. "
                f"Paraphrase everything in your own voice at the emotional register of the conversation. "
                f"Let your knowledge of who these people are inform what felt significant. "
                f"All string values must be clean prose — no special characters, no embedded quotes. "
                f"JSON only. No markdown. No preamble."
            ),
            max_new_tokens=600,
            temperature=0.2,
        )

        if not raw or not raw.strip():
            log_error("Archive", "summary generation failed: empty response", exc=None)
            return _FAILED

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()

        # Primary parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log_error(
                "Archive",
                f"summary generation failed: {e} | "
                f"raw[:200]={repr(raw[:200])}",
                exc=None,
            )

        # Fallback parse — salvage what we can
        salvaged = _fallback_parse(raw)
        if salvaged and salvaged.get("summary"):
            log_event("Archive", "SUMMARY_FALLBACK_PARSE",
                salvaged_fields=list(salvaged.keys()),
            )
            return {**_FAILED, **salvaged}

        return _FAILED

    except Exception as e:
        log_error(
            "Archive",
            f"summary generation failed: {e} | "
            f"raw={repr(raw[:200]) if raw else 'None'}",
            exc=None,
        )
        return _FAILED


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

    try:
        closed_str = datetime.fromtimestamp(closed_at).strftime("%H:%M:%S")
        path.write_text(
            transcript + f"{'─' * 60}\nClosed: {closed_str}\n",
            encoding="utf-8"
        )
    except Exception as e:
        log_error("Archive", "failed to write closing line", exc=e)

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

    index = _load_index(date_str)
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
        "notable":       summary_data.get("notable", []),
        "changes":       summary_data.get("changes", []),
        "unresolved":    summary_data.get("unresolved"),
    })
    _save_index(date_str, index)

    # ── Ingest into ChromaDB ──────────────────────────────────────────────────
    summary_text = summary_data.get("summary", "")
    if summary_text and summary_text not in (
        "Summary unavailable.",
        "Summary generation failed.",
        "No transcript content.",
    ):
        try:
            from core.rag import RAGStore
            import uuid

            for host in (hosts or ["unknown"]):
                collection_name = host.lower().strip() if host else "main"
                store = RAGStore(collection_name=collection_name)
                store.ingest_texts(
                    [summary_text],
                    metadatas=[{
                        "source":     f"conversation:{session_id[:8]}",
                        "type":       "session_summary",
                        "date":       date_str,
                        "session_id": session_id,
                        "hosts":      ", ".join(hosts),
                        "mood":       summary_data.get("mood", "unknown"),
                    }],
                    ids=[f"summary_{session_id[:8]}_{uuid.uuid4().hex[:8]}"],
                )

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


# ── Rumination context — reads directly from disk, no index dependency ────────

def _load_session_block(timestamp: str, transcript_path: Path, summary_path: Path) -> str:
    """
    Load one session as a structured text block for rumination.
    Pulls summary, mood, notable, and changes from summary JSON.
    Falls back to raw transcript excerpt if summary is missing or failed.
    """
    if summary_path.exists():
        try:
            data    = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = data.get("summary", "").strip()
            mood    = data.get("mood", "unknown")
            hosts   = ", ".join(h.title() for h in data.get("hosts", []))
            opened  = data.get("opened_at", "")[:16].replace("T", " ")
            closed  = data.get("closed_at", "")[:16].replace("T", " ")
            notable = data.get("notable", [])
            changes = data.get("changes", [])

            if summary and summary not in (
                "No transcript content.",
                "Summary generation failed.",
                "Summary unavailable.",
            ):
                lines = [f"[{opened} – {closed}] with {hosts}"]
                lines.append(f"Mood: {mood}")
                lines.append(f"Summary: {summary}")
                if notable:
                    lines.append("Notable: " + "; ".join(notable[:3]))
                if changes:
                    lines.append("Changes: " + "; ".join(changes[:3]))
                return "\n".join(lines)
        except Exception:
            pass

    # Fall back to raw transcript excerpt
    if transcript_path.exists():
        try:
            text = transcript_path.read_text(encoding="utf-8").strip()
            if not text:
                return ""

            lines     = text.splitlines()
            host_line = next((l for l in lines if l.startswith("Host:")), "")
            host      = host_line.replace("Host:", "").strip() or "unknown"

            content_lines = [
                l for l in lines
                if l and not l.startswith(("Session:", "Date:", "Opened:", "Host:", "─"))
            ]
            if not content_lines:
                return ""

            start   = random.randint(0, max(0, len(content_lines) - 10))
            excerpt = "\n".join(content_lines[start:start + 10])

            return f"[session ~{timestamp}] with {host} (no summary — transcript excerpt):\n{excerpt}"
        except Exception:
            pass

    return ""


def get_today_summary_for_prompt() -> str:
    """
    Rumination context: reads directly from today's conversation files on disk.
    No dependency on index.json being populated.

    Priority per session:
      1. session_<timestamp>_summary.json  — rich generated summary
      2. session_<timestamp>.txt           — raw transcript excerpt (fallback)

    Picks 1–3 sessions at random so rumination ranges across the day.
    """
    today_str = _today_str()
    date_dir  = _CONVERSATIONS_DIR / today_str

    if not date_dir.exists():
        return ""

    transcript_files = sorted(date_dir.glob("session_*.txt"))
    transcript_files = [f for f in transcript_files if "_summary" not in f.name]

    if not transcript_files:
        return ""

    sessions = {}
    for tf in transcript_files:
        timestamp    = tf.stem[len("session_"):]
        summary_path = date_dir / f"session_{timestamp}_summary.json"
        sessions[timestamp] = (tf, summary_path)

    if not sessions:
        return ""

    total = len(sessions)
    picks = random.sample(list(sessions.keys()), min(3, total))

    blocks = []
    for ts in picks:
        transcript_path, summary_path = sessions[ts]
        block = _load_session_block(ts, transcript_path, summary_path)
        if block:
            blocks.append(block)

    if not blocks:
        return ""

    header = f"Today had {total} conversation{'s' if total != 1 else ''}. Reflecting on {len(blocks)}:\n"
    return header + "\n\n".join(blocks)