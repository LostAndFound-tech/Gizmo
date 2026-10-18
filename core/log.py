"""
core/log.py
Structured per-component logging — reflection-ready.

Each component gets its own log file. Logs are:
  - Timestamped with local time (via core.timezone)
  - Structured with an event type so reflection can parse them
  - Human readable, grep-able, and machine parseable

Log files live in {DATA_DIR}/logs/{component}.log
A system.log captures cross-component events, startup, errors.

Usage:
    from core.log import log, log_event

    log("Archivist", "starting up")

    log_event("Archivist", "CLASSIFY",
        input="I'm hungry",
        result="food_mention",
        confidence=0.87,
        headmate="alastor"
    )

Format:
    [YYYY-MM-DD HH:MM:SS TZ] [Component] MESSAGE
    [YYYY-MM-DD HH:MM:SS TZ] [Component] EVENT_TYPE | key=value | key=value
"""

import os
import sys
from pathlib import Path
from typing import Any

# ── Log directory ─────────────────────────────────────────────────────────────
# Defaults to ./data so it works locally without a /data mount (matches
# server.py). On Render, set DATA_DIR=/data to use the persistent disk.

_DATA_DIR = os.getenv("DATA_DIR", "./data")
_LOG_DIR = Path(_DATA_DIR) / "logs"

# Known components — each gets its own file.
# Unknown components fall back to system.log.
COMPONENTS = {
    "Archivist",
    "Mind",
    "Ego",
    "Body",
    "Id",
    "Empath",
    "Librarian",
    "System",
    "Server",
    "Pipeline",
    "Archiver",
    "Reflection",
    # Bare-bones rebuild
    "GizmoServer",
    "AgentSimple",
    "Sessions",
}

# File handles — opened lazily, one per component.
_handles: dict[str, Any] = {}


def _ensure_log_dir() -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[Log] WARNING: could not create log dir {_LOG_DIR}: {e}", file=sys.stderr)


def _get_handle(component: str):
    """Return (and cache) a file handle for this component's log."""
    if component not in _handles:
        _ensure_log_dir()
        path = _LOG_DIR / (component.lower() + ".log")
        try:
            # line-buffered so logs flush on each write
            _handles[component] = open(path, "a", buffering=1, encoding="utf-8")
        except Exception as e:
            print(f"[Log] WARNING: could not open {path}: {e}", file=sys.stderr)
            _handles[component] = None
    return _handles[component]


def _now_str() -> str:
    """Current local time as a log-friendly string."""
    try:
        from core.timezone import tz_now
        return tz_now().strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _resolve_component(component: str) -> str:
    """Normalise component name. Unknown components → System log."""
    for known in COMPONENTS:
        if component.lower() == known.lower():
            return known
    return "System"


def _write(component: str, line: str, echo: bool = True) -> None:
    """Write a line to the component's log file (and stdout if echo)."""
    resolved = _resolve_component(component)
    handle = _get_handle(resolved)

    # Print to stdout so Render/systemd/terminal capture it
    if echo:
        print(line, flush=True)

    if handle is not None:
        try:
            handle.write(line + "\n")
        except Exception as e:
            print(f"[Log] WARNING: write failed for {resolved}: {e}", file=sys.stderr)


# ── Public API ────────────────────────────────────────────────────────────────

def log(component: str, message: str) -> None:
    """Simple log line."""
    _write(component, f"[{_now_str()}] [{component}] {message}")


def log_event(component: str, event: str, **kwargs) -> None:
    """
    Structured event log — reflection-parseable.

    kwargs are serialised as key=value pairs.
    String values with spaces are quoted. Numbers and bools are bare.
    """
    parts = [f"[{_now_str()}] [{component}]", event]

    for k, v in kwargs.items():
        if isinstance(v, str) and (" " in v or not v):
            parts.append(f'{k}="{v}"')
        elif isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, (list, tuple, set)):
            joined = ",".join(str(i) for i in v)
            parts.append(f"{k}=[{joined}]")
        else:
            parts.append(f"{k}={v}")

    _write(component, " | ".join(parts))


def log_error(component: str, message: str, exc: Exception = None) -> None:
    """Error log — goes to both the component log and system.log."""
    exc_str = f"{type(exc).__name__}: {exc}" if exc else ""
    line = f'[{_now_str()}] [{component}] ERROR | msg="{message}"'
    if exc_str:
        line += f" | exc={exc_str}"

    _write(component, line)

    # Copy to system.log (file only — already echoed to stdout above)
    if _resolve_component(component) != "System":
        _write("System", line, echo=False)


def get_log_path(component: str) -> Path:
    """Return the path to a component's log file. Useful for reflection."""
    resolved = _resolve_component(component)
    return _LOG_DIR / (resolved.lower() + ".log")


def read_todays_log(component: str) -> str:
    """Read today's entries from a component's log, ready for a reflection prompt."""
    try:
        from core.timezone import tz_now
        today = tz_now().strftime("%Y-%m-%d")
    except Exception:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    path = get_log_path(component)
    if not path.exists():
        return f"[No log found for {component} on {today}]"

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        todays = [l for l in lines if l.startswith(f"[{today}")]
        if not todays:
            return f"[No entries for {component} on {today}]"
        return "\n".join(todays)
    except Exception as e:
        return f"[Error reading log for {component}: {e}]"


def close_all() -> None:
    """Close all open log file handles. Call on shutdown."""
    for handle in _handles.values():
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
    _handles.clear()
