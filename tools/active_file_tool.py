"""
tools/active_file_tool.py

Tools for Gizmo to open and close session-scoped active files.

set_active_file  — open a file for reading, writing, or both
close_active_file — stop reading/writing one or both sides
"""

from tools.base_tool import BaseTool, ToolResult


class SetActiveFileTool(BaseTool):
    name        = "set_active_file"
    description = (
    "Open a working document for this session — use this when someone wants to "
    "draft, build, brainstorm, or review something together in real time. "
    "NOT for behavioral rules or persistent instructions (use create_protocol for those). "
    "Mode 'write': auto-append my responses each turn. "
    "Mode 'read': inject file contents into my context each turn. "
    "Mode 'both': do both — default for collaborative work."
)
    args_schema = {
        "path":  {"type": "string", "description": "File path (e.g. notes/session.md)"},
        "label": {"type": "string", "description": "Human-readable name (e.g. 'our brainstorm')", "default": ""},
        "mode":  {"type": "string", "description": "'read', 'write', or 'both'", "default": "both"},
    }

    async def run(
        self,
        session_id: str,
        path: str,
        label: str = "",
        mode: str = "both",
        **kwargs,
    ) -> ToolResult:
        try:
            from core.active_file import set_active_file
            set_active_file(session_id, path, label, mode)

            # For write/both modes, create the file with a header if it doesn't exist
            if mode in ("write", "both"):
                from pathlib import Path
                import os
                base_dir = os.getenv("DATA_DIR", "/data")
                full_path = Path(base_dir) / path
                if not full_path.exists():
                    from tools.file_tool import WriteFileTool
                    writer = WriteFileTool()
                    header = f"# {label or path}\n\n"
                    result = await writer.run(
                        session_id=session_id,
                        path=path,
                        content=header,
                    )
                    if not result.success:
                        return ToolResult(
                            success=False,
                            output=f"Couldn't create '{path}': {result.output}",
                        )

            mode_desc = {
                "both":  "reading and writing",
                "write": "writing",
                "read":  "reading",
            }.get(mode, mode)

            return ToolResult(
                success=True,
                output=f"Active file set to '{path}' ({mode_desc}). {_mode_note(mode)}",
            )
        except Exception as e:
            return ToolResult(success=False, output=str(e))


class CloseActiveFileTool(BaseTool):
    name        = "close_active_file"
    description = (
        "Stop using the active working file for this session. "
        "Mode 'write': stop auto-appending my responses. "
        "Mode 'read': stop injecting file contents into context. "
        "Mode 'both': close everything (default)."
    )
    args_schema = {
        "mode": {"type": "string", "description": "'read', 'write', or 'both'", "default": "both"},
    }

    async def run(
        self,
        session_id: str,
        mode: str = "both",
        **kwargs,
    ) -> ToolResult:
        try:
            from core.active_file import get_status, clear_active_file
            status = get_status(session_id)

            if not status["read"] and not status["write"]:
                return ToolResult(success=True, output="No active file was open.")

            clear_active_file(session_id, mode)

            closed = []
            if mode in ("read", "both") and status.get("read"):
                closed.append(f"read ('{status['read']['path']}')")
            if mode in ("write", "both") and status.get("write"):
                closed.append(f"write ('{status['write']['path']}')")

            return ToolResult(
                success=True,
                output=f"Closed {' and '.join(closed)}. No longer active.",
            )
        except Exception as e:
            return ToolResult(success=False, output=str(e))


def _mode_note(mode: str) -> str:
    if mode == "both":
        return "I'll read from it each turn and append my responses as we go."
    if mode == "write":
        return "I'll append my responses to it each turn."
    if mode == "read":
        return "Its contents will be in my context each turn."
    return ""
