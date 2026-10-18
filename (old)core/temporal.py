"""
core/temporal.py
Time reasoning utility for Gizmo.

Gizmo struggles with time because:
  1. LLM calls have no inherent sense of "now"
  2. Timestamps in history are numbers, not natural language
  3. "Earlier today", "last week", "a few minutes ago" require anchoring
     to a real clock, which the model doesn't have natively
  4. Session boundaries mean Gizmo can't feel elapsed time between sessions

This module provides:
  - Natural language time descriptions from timestamps
  - Relative time strings ("3 minutes ago", "yesterday at 2pm")
  - A time context block for injection into system prompts
  - Session gap detection ("last time we spoke was 4 days ago")
  - Time-of-day awareness ("it's late", "good morning" context)

All output is in the active timezone from core.timezone.

Usage:
    from core.temporal import time_context_block, relative_time, session_gap

    # Inject into system prompt
    prompt = time_context_block() + "\\n\\n" + your_prompt

    # Describe a timestamp naturally
    label = relative_time(some_unix_timestamp)   # "about 2 hours ago"

    # Session gap for greetings
    gap = session_gap(last_session_timestamp)    # "3 days ago"
"""

from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


def _now() -> datetime:
    from core.timezone import tz_now
    return tz_now()


def _tz() -> ZoneInfo:
    from core.timezone import get_timezone
    return get_timezone()


# ── Relative time ─────────────────────────────────────────────────────────────

def relative_time(
    ts: float,
    now: Optional[datetime] = None,
    brief: bool = False,
) -> str:
    """
    Convert a unix timestamp to a natural language relative time string.

    brief=False (default): "about 3 hours ago", "yesterday at 2:30pm"
    brief=True: "3h ago", "yesterday", "Mon"

    Examples:
        45 seconds ago   → "just now" / "just now"
        4 minutes ago    → "4 minutes ago" / "4m ago"
        2 hours ago      → "about 2 hours ago" / "2h ago"
        yesterday 14:30  → "yesterday at 2:30pm" / "yesterday"
        3 days ago       → "3 days ago" / "3d ago"
        10 days ago      → "last Tuesday" / "last Tue"
        45 days ago      → "about 6 weeks ago" / "6w ago"
        13 months ago    → "about a year ago" / "~1yr ago"
    """
    if now is None:
        now = _now()

    try:
        then = datetime.fromtimestamp(ts, tz=_tz())
    except (OSError, ValueError, OverflowError):
        return "unknown time"

    delta = now - then
    total_seconds = delta.total_seconds()

    if total_seconds < 0:
        return "just now"

    seconds = int(total_seconds)
    minutes = seconds // 60
    hours   = minutes // 60
    days    = delta.days

    if seconds < 60:
        return "just now"

    if minutes < 60:
        if brief:
            return f"{minutes}m ago"
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

    if hours < 6:
        if brief:
            return f"{hours}h ago"
        return f"about {hours} hour{'s' if hours != 1 else ''} ago"

    # Same calendar day
    if days == 0:
        time_str = then.strftime("%-I:%M%p").lower()
        if brief:
            return f"today {time_str}"
        return f"earlier today at {time_str}"

    if days == 1:
        time_str = then.strftime("%-I:%M%p").lower()
        if brief:
            return "yesterday"
        return f"yesterday at {time_str}"

    if days < 7:
        day_name = then.strftime("%A")
        time_str = then.strftime("%-I:%M%p").lower()
        if brief:
            return then.strftime("%a")
        return f"last {day_name} at {time_str}"

    weeks = days // 7
    if days < 60:
        if brief:
            return f"{weeks}w ago"
        return f"about {weeks} week{'s' if weeks != 1 else ''} ago"

    months = days // 30
    if days < 365:
        if brief:
            return f"{months}mo ago"
        return f"about {months} month{'s' if months != 1 else ''} ago"

    years = days // 365
    if brief:
        return f"~{years}yr ago"
    return f"about {'a year' if years == 1 else f'{years} years'} ago"


# ── Time of day ───────────────────────────────────────────────────────────────

def time_of_day(now: Optional[datetime] = None) -> str:
    """
    Return a natural label for the current time of day.
    "early morning", "morning", "midday", "afternoon",
    "evening", "night", "late night"
    """
    if now is None:
        now = _now()
    h = now.hour
    if h < 5:
        return "late night"
    if h < 9:
        return "early morning"
    if h < 12:
        return "morning"
    if h < 14:
        return "midday"
    if h < 17:
        return "afternoon"
    if h < 20:
        return "evening"
    if h < 23:
        return "night"
    return "late night"


def is_late(now: Optional[datetime] = None) -> bool:
    """True if it's after 10pm or before 6am."""
    if now is None:
        now = _now()
    return now.hour >= 22 or now.hour < 6


def is_weekend(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = _now()
    return now.weekday() >= 5


# ── Session gap ───────────────────────────────────────────────────────────────

def session_gap(last_ts: Optional[float]) -> Optional[str]:
    """
    Describe the gap since the last session in natural language.
    Returns None if last_ts is None (first session ever).

    Used by greeter to set the right tone:
      None          → first time, no gap language needed
      "just now"    → same session, reconnect
      "20 minutes ago" → short break
      "3 days ago"  → meaningful gap, worth acknowledging
    """
    if last_ts is None:
        return None
    return relative_time(last_ts)


def gap_is_significant(last_ts: Optional[float], threshold_hours: float = 4.0) -> bool:
    """
    True if the gap since last_ts is large enough to warrant acknowledgment.
    Default threshold: 4 hours.
    """
    if last_ts is None:
        return True
    now   = _now()
    delta = now - datetime.fromtimestamp(last_ts, tz=_tz())
    return delta.total_seconds() > threshold_hours * 3600


# ── Time context block ────────────────────────────────────────────────────────

def time_context_block(
    last_session_ts: Optional[float] = None,
    include_gap: bool = True,
) -> str:
    """
    Build a compact time context block for injection into any system prompt.

    Gives the LLM everything it needs to reason about time naturally:
      - Current date and time
      - Day of week
      - Time of day label
      - Whether it's late / a weekend
      - How long since the last session (if provided)

    Example output:
        [Time context]
        Now: Tuesday 2026-05-12 11:34pm MDT
        Time of day: night (late)
        Last session: about 6 hours ago

    Keep it compact — this goes into every prompt.
    """
    now = _now()
    lines = ["[Time context]"]
    lines.append(f"Now: {now.strftime('%A %Y-%m-%d %I:%M%p %Z').replace('  ', ' ')}")

    tod = time_of_day(now)
    late_note = " (late)" if is_late(now) else ""
    weekend_note = " (weekend)" if is_weekend(now) else ""
    lines.append(f"Time of day: {tod}{late_note}{weekend_note}")

    if include_gap and last_session_ts is not None:
        gap = relative_time(last_session_ts, now=now)
        if gap != "just now":
            lines.append(f"Last session: {gap}")

    return "\n".join(lines)


def annotate_timestamps(
    messages: list[dict],
    now: Optional[datetime] = None,
) -> list[dict]:
    """
    Take a list of history messages with unix `timestamp` fields and
    add a `time_label` field to each with a relative time string.

    Useful for building richer context when feeding history to an LLM
    that needs to reason about when things were said.

    Input:  [{"role": "user", "content": "...", "timestamp": 1234567890}, ...]
    Output: same list with "time_label": "about 2 hours ago" added
    """
    if now is None:
        now = _now()
    result = []
    for msg in messages:
        ts = msg.get("timestamp")
        annotated = dict(msg)
        if ts:
            annotated["time_label"] = relative_time(ts, now=now)
        result.append(annotated)
    return result


def format_timestamp(ts: float, fmt: str = "%I:%M%p %Z") -> str:
    """
    Format a unix timestamp as a clock string in the active timezone.
    Default: "2:34pm MDT"
    """
    try:
        dt = datetime.fromtimestamp(ts, tz=_tz())
        return dt.strftime(fmt).lstrip("0")
    except Exception:
        return "unknown"


def duration_natural(seconds: float) -> str:
    """
    Convert a duration in seconds to natural language.
    "45 seconds", "3 minutes", "2 hours", "4 days"
    """
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"
