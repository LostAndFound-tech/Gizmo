"""
tools/rewrite_file_tool.py

Gives Gizmo the ability to rewrite an existing file cleanly —
read the current content, understand what's there, replace it entirely.

Used for updating protocols, notes, or any file where stacking new
content below old content would cause confusion.
"""

import os
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult

_BASE_DIR = Path(os.getenv("DATA_DIR", "/data"))


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _BASE_DIR / p


class RewriteFileTool(BaseTool):
    name        = "rewrite_file"
    description = (
        "Read an existing file and replace its contents entirely with new content. "
        "Use this when updating something already written — a protocol, a note, a rule — "
        "where you want to replace the old version, not stack on top of it. "
        "Always read the file first so you know what you're replacing."
    )
    args_schema = {
        "path":    {"type": "string", "description": "File path to rewrite"},
        "content": {"type": "string", "description": "New content to write (replaces everything)"},
    }

    async def run(
        self,
        session_id: str,
        path: str,
        content: str,
        **kwargs,
    ) -> ToolResult:
        try:
            full_path = _resolve(path)

            # Read existing content so Gizmo can confirm what he's replacing
            existing = ""
            existed  = full_path.exists()
            if existed:
                try:
                    existing = full_path.read_text(encoding="utf-8")
                except Exception as e:
                    return ToolResult(
                        success=False,
                        output=f"Couldn't read '{path}' before rewriting: {e}",
                    )

            # Write new content
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")

            if existed:
                old_lines = len(existing.splitlines())
                new_lines = len(content.splitlines())
                return ToolResult(
                    success=True,
                    output=(
                        f"Rewrote '{path}'. "
                        f"Was {old_lines} lines, now {new_lines} lines."
                    ),
                )
            else:
                return ToolResult(
                    success=True,
                    output=f"Created '{path}' ({len(content.splitlines())} lines).",
                )

        except Exception as e:
            return ToolResult(success=False, output=str(e))
