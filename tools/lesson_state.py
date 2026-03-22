"""
tools/lesson_state.py
Persistent state for active teaching sessions.

A lesson is a collaborative edit of a tool's name, description, and behavior.
It lives in memory while the session is active, and checkpoints to disk
after every round so nothing is lost mid-lesson.

State keys:
  tool_name       — snake_case identifier
  source_file     — absolute path to the tool's .py file (None if new)
  description     — current working description (what the agent reads)
  behavior        — current working behavior spec (what run() does)
  trigger_phrases — current list of when-to-call phrases
  run_source      — current run() source code (for core tools)
  settled         — list of fields marked done
  rounds          — list of {question, answer, changed} dicts
  status          — 'active' | 'saved' | 'abandoned'
  started_by      — who opened the lesson
  last_touched    — ISO timestamp
  is_wip          — True if saved mid-lesson
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

_CHECKPOINT_DIR = Path(__file__).parent / "generated" / ".lessons"

# In-memory store: session_id -> lesson dict
_active_lessons: dict[str, dict] = {}


def start_lesson(
    session_id: str,
    tool_name: str,
    source_file: Optional[str],
    description: str,
    behavior: str,
    trigger_phrases: list[str],
    run_source: str,
    started_by: str = "",
) -> dict:
    lesson = {
        "tool_name": tool_name,
        "source_file": source_file,
        "description": description,
        "behavior": behavior,
        "trigger_phrases": trigger_phrases,
        "run_source": run_source,
        "settled": [],
        "rounds": [],
        "status": "active",
        "started_by": started_by,
        "last_touched": datetime.now().isoformat(),
        "is_wip": False,
    }
    _active_lessons[session_id] = lesson
    _checkpoint(session_id, lesson)
    return lesson


def get_lesson(session_id: str) -> Optional[dict]:
    if session_id in _active_lessons:
        return _active_lessons[session_id]
    # Try to restore from disk
    path = _checkpoint_path(session_id)
    if path.exists():
        try:
            lesson = json.loads(path.read_text())
            if lesson.get("status") == "active":
                _active_lessons[session_id] = lesson
                return lesson
        except Exception:
            pass
    return None


def update_lesson(session_id: str, updates: dict) -> Optional[dict]:
    lesson = get_lesson(session_id)
    if not lesson:
        return None
    lesson.update(updates)
    lesson["last_touched"] = datetime.now().isoformat()
    _active_lessons[session_id] = lesson
    _checkpoint(session_id, lesson)
    return lesson


def add_round(session_id: str, question: str, answer: str, changed: dict) -> Optional[dict]:
    lesson = get_lesson(session_id)
    if not lesson:
        return None
    lesson["rounds"].append({
        "question": question,
        "answer": answer,
        "changed": changed,
        "timestamp": datetime.now().isoformat(),
    })
    lesson["last_touched"] = datetime.now().isoformat()
    _active_lessons[session_id] = lesson
    _checkpoint(session_id, lesson)
    return lesson


def settle_field(session_id: str, field: str) -> None:
    lesson = get_lesson(session_id)
    if lesson and field not in lesson["settled"]:
        lesson["settled"].append(field)
        _active_lessons[session_id] = lesson
        _checkpoint(session_id, lesson)


def close_lesson(session_id: str, status: str = "saved", is_wip: bool = False) -> None:
    lesson = get_lesson(session_id)
    if lesson:
        lesson["status"] = status
        lesson["is_wip"] = is_wip
        lesson["last_touched"] = datetime.now().isoformat()
        _active_lessons.pop(session_id, None)
        _checkpoint(session_id, lesson)


def list_open_lessons() -> list[dict]:
    """Return all lessons currently checkpointed as active."""
    results = []
    if not _CHECKPOINT_DIR.exists():
        return results
    for path in _CHECKPOINT_DIR.glob("*.json"):
        try:
            lesson = json.loads(path.read_text())
            if lesson.get("status") == "active":
                results.append({
                    "session_id": path.stem,
                    "tool_name": lesson.get("tool_name"),
                    "started_by": lesson.get("started_by"),
                    "last_touched": lesson.get("last_touched"),
                    "rounds": len(lesson.get("rounds", [])),
                    "is_wip": lesson.get("is_wip", False),
                })
        except Exception:
            pass
    return results


# ── Internal ──────────────────────────────────────────────────────────────────

def _checkpoint_path(session_id: str) -> Path:
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return _CHECKPOINT_DIR / f"{session_id}.json"


def _checkpoint(session_id: str, lesson: dict) -> None:
    try:
        _checkpoint_path(session_id).write_text(
            json.dumps(lesson, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"[LessonState] Checkpoint failed: {e}")
