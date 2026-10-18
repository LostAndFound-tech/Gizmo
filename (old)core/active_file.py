"""
core/active_file.py

Session-scoped active file tracking.

A session can have:
  - An active WRITE file: every response Gizmo generates is auto-appended
  - An active READ file: contents are injected into the system prompt each turn
  - Both can be the same file, or different files, or only one at a time

Usage:
    from core.active_file import set_active_file, get_active_read, get_active_write, clear_active_file

    # Open same file for both reading and writing
    set_active_file(session_id, path="notes/session.md", label="our brainstorm", mode="both")

    # Open a reference doc for reading only
    set_active_file(session_id, path="refs/spec.md", label="the spec", mode="read")

    # Open a separate output file for writing
    set_active_file(session_id, path="notes/output.md", label="notes", mode="write")

    # Close just the write side
    clear_active_file(session_id, mode="write")

    # Close everything
    clear_active_file(session_id, mode="both")
"""

from core.log import log_event

# session_id → {"read": {path, label} | None, "write": {path, label} | None}
_active_files: dict[str, dict] = {}


def _ensure(session_id: str) -> dict:
    if session_id not in _active_files:
        _active_files[session_id] = {"read": None, "write": None}
    return _active_files[session_id]


def set_active_file(
    session_id: str,
    path: str,
    label: str = "",
    mode: str = "both",
) -> None:
    """
    Set an active file for this session.

    mode:
      "read"  — inject file contents into system prompt each turn
      "write" — auto-append Gizmo's response to this file each turn
      "both"  — do both (default)
    """
    state = _ensure(session_id)
    entry = {"path": path, "label": label or path}

    if mode in ("read", "both"):
        state["read"] = entry
    if mode in ("write", "both"):
        state["write"] = entry

    log_event("ActiveFile", "SET",
        session=session_id[:8],
        path=path,
        label=label or path,
        mode=mode,
    )


def get_active_read(session_id: str) -> dict | None:
    """Return the active read file for this session, or None."""
    return _active_files.get(session_id, {}).get("read")


def get_active_write(session_id: str) -> dict | None:
    """Return the active write file for this session, or None."""
    return _active_files.get(session_id, {}).get("write")


def clear_active_file(session_id: str, mode: str = "both") -> None:
    """
    Close the active file(s) for this session.

    mode:
      "read"  — stop injecting into prompt
      "write" — stop auto-appending
      "both"  — close everything (default)
    """
    if session_id not in _active_files:
        return

    state = _active_files[session_id]

    if mode in ("read", "both") and state.get("read"):
        log_event("ActiveFile", "CLOSED",
            session=session_id[:8],
            mode="read",
            path=state["read"]["path"],
        )
        state["read"] = None

    if mode in ("write", "both") and state.get("write"):
        log_event("ActiveFile", "CLOSED",
            session=session_id[:8],
            mode="write",
            path=state["write"]["path"],
        )
        state["write"] = None

    # Clean up empty state
    if not state["read"] and not state["write"]:
        del _active_files[session_id]


def get_status(session_id: str) -> dict:
    """Return a summary of active file state for this session."""
    state = _active_files.get(session_id, {"read": None, "write": None})
    return {
        "read":  state.get("read"),
        "write": state.get("write"),
    }
