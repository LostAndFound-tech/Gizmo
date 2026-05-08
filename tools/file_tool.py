"""
tools/file_tool.py
Direct file access for Gizmo — read, write, append, and list
within his data directory (/data/).

No confirm gate on writes. Instead, write_file uses a smart subject-match:
  - New file            → write immediately
  - Existing file       → LLM checks if incoming content is same subject
      Same subject      → overwrite silently
      Different subject → hold the write, return a confirmation request
                          so Gizmo can ask the user before proceeding

Append never needs a check — it never destroys existing content.
Delete still requires confirm=True — it's genuinely irreversible.

Path resolution:
  Relative → resolved under /data/  ('notes/x.txt' → /data/notes/x.txt)
  Absolute → must be within /data/
"""

import json
import os
from pathlib import Path
from typing import Optional

from tools.base_tool import BaseTool, ToolResult

_DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))

_MAX_READ_BYTES  = 32_000
_MAX_WRITE_BYTES = 64_000
_SUBJECT_SAMPLE  = 300   # chars sent to subject-match LLM call


# ── Path validation ───────────────────────────────────────────────────────────

def _resolve(path_str: str) -> tuple[Optional[Path], str]:
    """
    Resolve path to absolute within /data/.
    Returns (resolved, error). On success error is "".
    """
    if not path_str or not path_str.strip():
        return None, "Empty path."

    path_str = path_str.strip()

    candidate = (
        Path(path_str).resolve()
        if path_str.startswith("/")
        else (_DATA_ROOT / path_str).resolve()
    )

    try:
        candidate.relative_to(_DATA_ROOT.resolve())
        return candidate, ""
    except ValueError:
        return None, (
            f"'{path_str}' is outside /data/. "
            f"Use relative paths like 'notes/x.txt' or "
            f"'personality/headmates/oren.json'."
        )


# ── Read helpers ──────────────────────────────────────────────────────────────

def _read_raw(path: Path) -> tuple[str, str]:
    """Returns (content, error)."""
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


# ── Write helpers ─────────────────────────────────────────────────────────────

def _write_raw(path: Path, content: str) -> tuple[bool, str]:
    """Actually write. Returns (success, message)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True, f"Written: {path} ({len(content)} chars)"
    except Exception as e:
        return False, f"Write failed: {e}"


# ── Subject match ─────────────────────────────────────────────────────────────

async def _same_subject(existing: str, incoming: str) -> bool:
    """
    Cheap LLM call — are these two content blocks about the same subject?
    Returns True (safe to overwrite) or False (different subject, hold write).
    Defaults to True on any failure so we never silently block a write.
    """
    existing_sample = existing[:_SUBJECT_SAMPLE].strip()
    if not existing_sample:
        return True  # empty file — always safe

    incoming_sample = incoming[:_SUBJECT_SAMPLE].strip()

    try:
        from core.llm import llm
        result = await llm.generate(
            [{
                "role": "user",
                "content": (
                    f"File A (existing):\n\"\"\"\n{existing_sample}\n\"\"\"\n\n"
                    f"File B (incoming):\n\"\"\"\n{incoming_sample}\n\"\"\"\n\n"
                    f"Are these about the same subject? One word: YES or NO."
                )
            }],
            system_prompt=(
                "Compare two content blocks. "
                "Reply with exactly one word: YES or NO."
            ),
            max_new_tokens=5,
            temperature=0.0,
        )
        return result.strip().upper().startswith("Y")
    except Exception:
        return True  # on error, don't block


# ── Pending write store ───────────────────────────────────────────────────────
# Holds writes that need user confirmation (subject mismatch detected).
# Keyed by session_id.

_pending: dict[str, dict] = {}


def release_pending(session_id: str) -> tuple[bool, str]:
    """User confirmed — execute the held write."""
    if session_id not in _pending:
        return False, "No pending write to confirm."
    p = _pending.pop(session_id)
    return _write_raw(p["path"], p["content"])


def discard_pending(session_id: str) -> bool:
    """User declined — drop the held write."""
    return bool(_pending.pop(session_id, None))


# ── Read tool ─────────────────────────────────────────────────────────────────

class ReadFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a file from your data directory (/data/). "
            "ALWAYS use this before writing to check what's already there. "
            "Paths are relative to /data/ — e.g. 'notes/thoughts.txt', "
            "'personality/headmates/oren.json', 'personality/personality.txt'. "
            "Args: path (str) — file path to read."
        )

    async def run(self, path: str = "", session_id: str = "", **kwargs) -> ToolResult:
        if not path:
            return ToolResult(success=False, output="Need a file path.")

        resolved, error = _resolve(path)
        if error:
            return ToolResult(success=False, output=error)

        content, read_error = _read_raw(resolved)
        if read_error:
            return ToolResult(success=False, output=read_error)

        # Pretty-print JSON
        if resolved.suffix == ".json":
            try:
                parsed = json.loads(content.replace("\n[truncated]", ""))
                content = json.dumps(parsed, indent=2)
            except Exception:
                pass

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
            "Write content to a file under /data/. "
            "New files are written immediately. "
            "Existing files: content is checked against what's already there — "
            "if same subject, overwrites silently. "
            "If different subject, write is held and you must ask the user to confirm. "
            "For JSON files, content must be valid JSON. "
            "ALWAYS read the file first if you think it might already exist. "
            "Paths relative to /data/ — e.g. 'notes/x.txt', "
            "'personality/headmates/oren.json'. "
            "Args: "
            "path (str) — file path. "
            "content (str) — full content to write."
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
            return ToolResult(success=False, output="Need content to write.")

        resolved, error = _resolve(path)
        if error:
            return ToolResult(success=False, output=error)

        if len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
            return ToolResult(
                success=False,
                output=f"Content too large. Max {_MAX_WRITE_BYTES} bytes."
            )

        # JSON validation
        if resolved.suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return ToolResult(
                    success=False,
                    output=f"Invalid JSON: {e}. Not written."
                )

        # New file — write immediately
        if not resolved.exists():
            ok, msg = _write_raw(resolved, content)
            return ToolResult(success=ok, output=msg, data={"path": str(resolved)})

        # Existing file — subject check
        existing, read_err = _read_raw(resolved)
        if read_err:
            # Can't read existing — write anyway, something's wrong with the file
            ok, msg = _write_raw(resolved, content)
            return ToolResult(success=ok, output=msg, data={"path": str(resolved)})

        same = await _same_subject(existing, content)

        if same:
            ok, msg = _write_raw(resolved, content)
            return ToolResult(success=ok, output=msg, data={"path": str(resolved)})
        else:
            # Hold the write — different subject detected
            _pending[session_id] = {"path": resolved, "content": content}
            return ToolResult(
                success=False,
                output=(
                    f"The file at '{path}' already exists and appears to contain "
                    f"different content. I've held the write. "
                    f"Tell the user what you wanted to write and ask if they want to overwrite it. "
                    f"If yes, call confirm_write. If no, call cancel_write."
                ),
                data={"path": str(resolved), "pending": True},
            )


# ── Confirm / cancel pending write tools ──────────────────────────────────────

class ConfirmWriteTool(BaseTool):
    @property
    def name(self) -> str:
        return "confirm_write"

    @property
    def description(self) -> str:
        return (
            "Confirm a held file write after the user has approved overwriting. "
            "Only call this after write_file returned a 'held' result AND "
            "the user has explicitly said yes to overwriting. "
            "Args: (none beyond session_id)"
        )

    async def run(self, session_id: str = "", **kwargs) -> ToolResult:
        ok, msg = release_pending(session_id)
        return ToolResult(success=ok, output=msg)


class CancelWriteTool(BaseTool):
    @property
    def name(self) -> str:
        return "cancel_write"

    @property
    def description(self) -> str:
        return (
            "Cancel a held file write — user declined the overwrite. "
            "Args: (none beyond session_id)"
        )

    async def run(self, session_id: str = "", **kwargs) -> ToolResult:
        discarded = discard_pending(session_id)
        return ToolResult(
            success=True,
            output="Write cancelled." if discarded else "No pending write to cancel.",
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
            "Creates the file and any parent directories if missing. "
            "Safe — never destroys existing content. "
            "Prefer this over write_file when adding to an existing document. "
            "For JSON files use write_file (read → modify → write is safer). "
            "Paths relative to /data/ — e.g. 'notes/thoughts.txt'. "
            "Args: "
            "path (str) — file path. "
            "content (str) — text to append."
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

        resolved, error = _resolve(path)
        if error:
            return ToolResult(success=False, output=error)

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            existing = resolved.read_text(encoding="utf-8") if resolved.exists() else ""
            separator = "\n" if existing and not existing.endswith("\n") else ""
            resolved.write_text(existing + separator + content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Appended {len(content)} chars to {resolved}",
                data={"path": str(resolved), "appended": len(content)},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Append failed: {e}")


# ── List tool ─────────────────────────────────────────────────────────────────

class ListFilesTool(BaseTool):
    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return (
            "List files and directories under /data/. "
            "Pass empty path or '.' to list /data/ itself. "
            "Args: path (str) — directory to list (relative to /data/ or absolute)."
        )

    async def run(self, path: str = "", session_id: str = "", **kwargs) -> ToolResult:
        if not path or path.strip() in (".", ""):
            target = _DATA_ROOT
            err = ""
        else:
            target, err = _resolve(path)
            if err:
                return ToolResult(success=False, output=err)

        if not target.exists():
            return ToolResult(
                success=True,
                output=f"{target}: (does not exist yet)",
                data={"files": [], "dirs": []},
            )
        if not target.is_dir():
            return ToolResult(
                success=False,
                output=f"'{path}' is a file — use read_file to read it.",
            )

        try:
            entries  = sorted(target.iterdir())
            dirs     = [e.name + "/" for e in entries if e.is_dir()]
            files    = [e.name for e in entries if e.is_file()]
            listing  = dirs + files
            output   = f"{target}:\n" + "\n".join(f"  {e}" for e in listing) if listing else f"{target}: (empty)"
            return ToolResult(
                success=True,
                output=output,
                data={"dirs": dirs, "files": files, "path": str(target)},
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
            "Delete a file under /data/. Permanent — cannot be undone. "
            "Requires confirm=True. Cannot delete directories. "
            "Args: "
            "path (str) — file to delete. "
            "confirm (bool) — must be true."
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
                output="Set confirm=true to delete. This cannot be undone.",
            )

        resolved, error = _resolve(path)
        if error:
            return ToolResult(success=False, output=error)

        if not resolved.exists():
            return ToolResult(success=False, output=f"File not found: {resolved}")
        if resolved.is_dir():
            return ToolResult(success=False, output="That's a directory — can only delete files.")

        try:
            resolved.unlink()
            return ToolResult(
                success=True,
                output=f"Deleted: {resolved}",
                data={"path": str(resolved)},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Delete failed: {e}")