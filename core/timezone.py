"""
core/timezone.py
Single source of truth for timezone-aware datetimes.

All files should use tz_now() instead of datetime.now().
Timezone is set by the client on connect and stored here.
Falls back to UTC if never set.

Usage:
    from core.timezone import tz_now, set_timezone

    now = tz_now()
    now.strftime("%Y-%m-%d %H:%M")
"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_timezone: ZoneInfo = ZoneInfo("UTC")


def set_timezone(tz_name: str) -> None:
    """
    Set the active timezone from an IANA name e.g. 'America/Denver'.
    Called by the server when a client connects with a timezone in context.
    Silently falls back to UTC on invalid names.
    """
    global _timezone
    try:
        _timezone = ZoneInfo(tz_name)
        print(f"[Timezone] Set to {tz_name}")
    except (ZoneInfoNotFoundError, KeyError):
        print(f"[Timezone] Unknown timezone '{tz_name}', keeping {_timezone.key}")


def get_timezone() -> ZoneInfo:
    return _timezone


def tz_now() -> datetime:
    """Return current datetime in the active timezone."""
    return datetime.now(_timezone)
