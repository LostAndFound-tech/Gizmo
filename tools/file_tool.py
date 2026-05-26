"""tools/file_tool.py — File read/write tools."""
import os
from pathlib import Path
from core.store import store
from core.log import log_event

_BASE = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_ALLOWED_ROOTS = [_BASE, Path("/data/reports")]


def _safe_path(path_str: str) -> Path:
    """Resolve path and verify it's under an allowed root."""
    p = Path(path_str).resolve()
    for root in _ALLOWED_ROOTS:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise PermissionError(f"path not allowed: {path_str}")


class ReadFileTool:
    name        = "read_file"
    description = "Read a file from the personality or reports directory."

    async def execute(self, args, session_id, headmate, llm) -> str:
        path_str = args.get("path", "")
        if not path_str:
            return "no path provided"
        try:
            p = _safe_path(path_str)
            if not p.exists():
                return f"file not found: {path_str}"
            return p.read_text(encoding="utf-8")
        except PermissionError as e:
            return str(e)
        except Exception as e:
            return f"error reading file: {e}"


class WriteFileTool:
    name        = "write_file"
    description = "Write content to a file. Creates directories if needed."

    async def execute(self, args, session_id, headmate, llm) -> str:
        path_str = args.get("path", "")
        content  = args.get("content") or args.get("text", "")
        if not path_str or not content:
            return "need path and content"
        try:
            p = _safe_path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            store.write("files", {
                "path":        str(p),
                "description": content[:80],
                "file_type":   args.get("file_type", "note"),
                "headmate":    headmate.lower() if headmate else None,
                "source":      "gizmo",
                "tags":        f"file,{args.get('file_type','note')}",
            })
            log_event("WriteFileTool", "FILE_WRITTEN", path=str(p))
            return f"written: {p.name}"
        except PermissionError as e:
            return str(e)
        except Exception as e:
            return f"error writing file: {e}"


class AppendFileTool:
    name        = "append_file"
    description = "Append content to an existing file."

    async def execute(self, args, session_id, headmate, llm) -> str:
        path_str = args.get("path", "")
        content  = args.get("content") or args.get("text", "")
        if not path_str or not content:
            return "need path and content"
        try:
            p = _safe_path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(f"\n{content}")
            return f"appended to: {p.name}"
        except PermissionError as e:
            return str(e)
        except Exception as e:
            return f"error appending: {e}"


class ListFilesTool:
    name        = "list_files"
    description = "List files in a directory."

    async def execute(self, args, session_id, headmate, llm) -> str:
        path_str = args.get("path", str(_BASE))
        try:
            p = _safe_path(path_str)
            if not p.exists():
                return f"directory not found: {path_str}"
            files = sorted(p.iterdir())
            lines = [str(f.relative_to(_BASE.parent)) for f in files]
            return "\n".join(lines) if lines else "(empty)"
        except PermissionError as e:
            return str(e)
        except Exception as e:
            return f"error listing: {e}"


class DeleteFileTool:
    name        = "delete_file"
    description = "Delete a file."

    async def execute(self, args, session_id, headmate, llm) -> str:
        path_str = args.get("path", "")
        if not path_str:
            return "no path provided"
        try:
            p = _safe_path(path_str)
            if not p.exists():
                return f"file not found: {path_str}"
            p.unlink()
            return f"deleted: {p.name}"
        except PermissionError as e:
            return str(e)
        except Exception as e:
            return f"error deleting: {e}"


# Legacy stubs — kept for registry compatibility
class ConfirmWriteTool:
    name        = "confirm_write"
    description = "Confirm a pending file write."
    async def execute(self, args, session_id, headmate, llm) -> str:
        return "confirmed"

class CancelWriteTool:
    name        = "cancel_write"
    description = "Cancel a pending file write."
    async def execute(self, args, session_id, headmate, llm) -> str:
        return "cancelled"
