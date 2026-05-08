"""
tools/file_tool.py
Direct file access for Gizmo — read, write, append, and list
within his own data directory.

Scoped strictly to /data/personality/ and /data/notes/.
Cannot touch code, ChromaDB, or anything outside these paths.

Operations:
  read_file    — read any file in scope
  write_file   — write (create or overwrite) a file in scope
  append_file  — append to an existing file (or create if missing)
  list_files   — list files in a directory in scope
  delete_file  — delete a file in scope (requires confirm=True)

Use cases:
  - Mid-conversation: notice something worth remembering, write it down
  - Read a headmate's file to check what's known before asking
  - Append a note to someone's moments_of_note directly
  - Update personality.txt intentionally
  - Create a scratch note for something that doesn't fit existing structure
  - List what files exist for a headmate or directory

The /data/notes/ directory is Gizmo's own scratchpad —
unstructured, freeform, his to use however makes sense.
"""

import json
import os
from pathlib import Path
from typing import Optional

from tools.base_tool import BaseTool, ToolResult

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_NOTES_DIR       = Path(os.getenv("NOTES_DIR", "/data/notes"))

# All allowed root paths — nothing outside these
_ALLOWED_ROOTS = [_PERSONALITY_DIR, _NOTES_DIR]

_MAX_READ_BYTES  = 32_000   # ~8k tokens, enough for any single file
_MAX_WRITE_BYTES = 64_000   # reasonable cap on writes


def _resolve_and_validate(path_str: str) -> Optional[Path]:
    """
    Resolve a path and confirm it's within an allowed root.
    Returns None if the path is outside scope.
    """
    # Strip leading slashes so relative paths work naturally
    path_str = path_str.lstrip("/")

    # Try each allowed root
    for root in _ALLOWED_ROOTS:
        candidate = (root / path_str).resolve()
        try:
            candidate.relative_to(root.resolve())
            return candidate
        except ValueError:
            continue

    # Also accept absolute paths that are within allowed roots
    try:
        absolute = Path(path_str).resolve()
        for root in _ALLOWED_ROOTS:
            try:
                absolute.relative_to(root.resolve())
                return absolute
            except ValueError:
                continue
    except Exception:
        pass

    return None


def _safe_read(path: Path) -> tuple[str, str]:
    """Read a file. Returns (content, error). One will be empty."""
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_READ_BYTES:
            raw = raw[:_MAX_READ_BYTES]
            return raw.decode("utf-8", errors="replace") + "\n[truncated]", ""
        return raw.decode("utf-8", errors="replace"), ""
    except FileNotFoundError:
        return "", f"File not found: {path}"
    except Exception as e:
        return "", f"Read failed: {e}"


# ── Read tool ─────────────────────────────────────────────────────────────────

class ReadFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a file from your data directory. "
            "Use this to check what you already know about someone before asking, "
            "review your own personality or notes, or inspect any data file. "
            "Paths are relative to /data/personality/ or /data/notes/. "
            "Examples: 'headmates/oren.json', 'personality.txt', 'notes/reminders.txt'. "
            "Args: path (str) — file path to read."
        )

    async def run(self, path: str = "", session_id: str = "", **kwargs) -> ToolResult:
        if not path:
            return ToolResult(success=False, output="Need a file path.")

        resolved = _resolve_and_validate(path)
        if resolved is None:
            return ToolResult(
                success=False,
                output=f"Path '{path}' is outside my data directory. I can only read within /data/personality/ and /data/notes/."
            )

        content, error = _safe_read(resolved)
        if error:
            return ToolResult(success=False, output=error)

        # Pretty-print JSON if applicable
        if resolved.suffix == ".json":
            try:
                parsed = json.loads(content.replace("\n[truncated]", ""))
                content = json.dumps(parsed, indent=2)
            except Exception:
                pass  # not valid JSON or truncated — return as-is

        return ToolResult(
            success=True,
            output=content,
            data={"path": str(resolved), "size": len(content)},
        )


# ── Write tool ────────────────────────────────────────────────────────────────

class WriteFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write (create or overwrite) a file in your data directory. "
            "Use for updating personality.txt, writing a new note, "
            "or updating structured data you've read and modified. "
            "WARNING: overwrites existing content — use append_file to add without replacing. "
            "For JSON files, content must be valid JSON. "
            "Paths are relative to /data/personality/ or /data/notes/. "
            "Args: "
            "path (str) — file path to write. "
            "content (str) — content to write. "
            "confirm (bool) — must be true to actually write (prevents accidental overwrites)."
        )

    async def run(
        self,
        path: str = "",
        content: str = "",
        confirm: bool = False,
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not path:
            return ToolResult(success=False, output="Need a file path.")
        if not content:
            return ToolResult(success=False, output="Need content to write.")
        if not confirm:
            return ToolResult(
                success=False,
                output="Set confirm=true to write. This will overwrite the file if it exists."
            )

        resolved = _resolve_and_validate(path)
        if resolved is None:
            return ToolResult(
                success=False,
                output=f"Path '{path}' is outside my data directory."
            )

        if len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
            return ToolResult(
                success=False,
                output=f"Content too large ({len(content)} chars). Max is {_MAX_WRITE_BYTES} bytes."
            )

        # Validate JSON if writing a .json file
        if resolved.suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return ToolResult(
                    success=False,
                    output=f"Invalid JSON: {e}. File not written."
                )

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except Exception as e:
            return ToolResult(success=False, output=f"Write failed: {e}")

        return ToolResult(
            success=True,
            output=f"Written: {resolved} ({len(content)} chars)",
            data={"path": str(resolved), "size": len(content)},
        )


# ── Append tool ───────────────────────────────────────────────────────────────

class AppendFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "append_file"

    @property
    def description(self) -> str:
        return (
            "Append text to a file without overwriting it. "
            "Use for adding notes, logging observations, or adding to a list. "
            "Creates the file if it doesn't exist. "
            "For structured JSON files, use write_file instead "
            "(read → modify → write is safer for JSON). "
            "Paths are relative to /data/personality/ or /data/notes/. "
            "Args: "
            "path (str) — file path to append to. "
            "content (str) — text to append. A newline is added before content if the file is non-empty."
        )

    async def run(
        self,
        path: str = "",
        content: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not path:
            return ToolResult(success=False, output="Need a file path.")
        if not content:
            return ToolResult(success=False, output="Need content to append.")

        resolved = _resolve_and_validate(path)
        if resolved is None:
            return ToolResult(
                success=False,
                output=f"Path '{path}' is outside my data directory."
            )

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            existing = ""
            if resolved.exists():
                existing = resolved.read_text(encoding="utf-8")

            separator = "\n" if existing and not existing.endswith("\n") else ""
            resolved.write_text(existing + separator + content, encoding="utf-8")
        except Exception as e:
            return ToolResult(success=False, output=f"Append failed: {e}")

        return ToolResult(
            success=True,
            output=f"Appended to {resolved} ({len(content)} chars)",
            data={"path": str(resolved), "appended": len(content)},
        )


# ── List tool ─────────────────────────────────────────────────────────────────

class ListFilesTool(BaseTool):
    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return (
            "List files in a directory within your data directory. "
            "Use to see what headmate files exist, what notes you've written, "
            "or what's in any directory you have access to. "
            "Paths are relative to /data/personality/ or /data/notes/. "
            "Pass an empty path or '.' to list the top-level personality directory. "
            "Args: path (str) — directory path to list (default: personality root)."
        )

    async def run(self, path: str = "", session_id: str = "", **kwargs) -> ToolResult:
        # Default to personality dir
        if not path or path in (".", ""):
            target = _PERSONALITY_DIR
        else:
            target = _resolve_and_validate(path)
            if target is None:
                return ToolResult(
                    success=False,
                    output=f"Path '{path}' is outside my data directory."
                )

        if not target.exists():
            return ToolResult(
                success=True,
                output=f"Directory doesn't exist yet: {target}",
                data={"files": [], "dirs": []}
            )

        if not target.is_dir():
            return ToolResult(
                success=False,
                output=f"'{path}' is a file, not a directory. Use read_file to read it."
            )

        try:
            entries = sorted(target.iterdir())
            files = [e.name for e in entries if e.is_file()]
            dirs  = [e.name + "/" for e in entries if e.is_dir()]
            all_entries = dirs + files

            if not all_entries:
                output = f"{target}: (empty)"
            else:
                output = f"{target}:\n" + "\n".join(f"  {e}" for e in all_entries)

            return ToolResult(
                success=True,
                output=output,
                data={"files": files, "dirs": dirs, "path": str(target)},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"List failed: {e}")


# ── Delete tool ───────────────────────────────────────────────────────────────

class DeleteFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "delete_file"

    @property
    def description(self) -> str:
        return (
            "Delete a file from your data directory. "
            "Use with care — this is permanent. "
            "Requires confirm=true to actually delete. "
            "Cannot delete directories, only files. "
            "Paths are relative to /data/personality/ or /data/notes/. "
            "Args: "
            "path (str) — file path to delete. "
            "confirm (bool) — must be true to actually delete."
        )

    async def run(
        self,
        path: str = "",
        confirm: bool = False,
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not path:
            return ToolResult(success=False, output="Need a file path.")
        if not confirm:
            return ToolResult(
                success=False,
                output="Set confirm=true to delete. This is permanent."
            )

        resolved = _resolve_and_validate(path)
        if resolved is None:
            return ToolResult(
                success=False,
                output=f"Path '{path}' is outside my data directory."
            )

        if not resolved.exists():
            return ToolResult(success=False, output=f"File not found: {resolved}")

        if resolved.is_dir():
            return ToolResult(
                success=False,
                output="That's a directory. I can only delete individual files."
            )

        try:
            resolved.unlink()
        except Exception as e:
            return ToolResult(success=False, output=f"Delete failed: {e}")

        return ToolResult(
            success=True,
            output=f"Deleted: {resolved}",
            data={"path": str(resolved)},
        )
