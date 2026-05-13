"""
tools/recall_read_tool.py

Fetch the full exchange for a specific message id.
Called after recall_search identifies a promising stub.

Returns the complete user message and Gizmo's response,
plus all context metadata.
"""

import json
from tools.base_tool import BaseTool, ToolResult


class RecallReadTool(BaseTool):
    name        = "recall_read"
    description = (
        "Fetch the full exchange for a message id from recall_search. "
        "Returns the complete user message, Gizmo's full response, "
        "and all context. Use this after recall_search finds something relevant."
    )
    args_schema = {
        "id": {
            "type": "string",
            "description": "Message id from recall_search results e.g. 'msg_abc123_1715123456000'",
        },
    }

    async def run(
        self,
        session_id: str,
        id: str,
        **kwargs,
    ) -> ToolResult:
        try:
            from core.message_store import get_by_id

            row = get_by_id(id)
            if not row:
                return ToolResult(
                    success=False,
                    output=f"No message found with id '{id}'.",
                )

            ts       = row["timestamp"][:16].replace("T", " ")
            host     = (row.get("host") or "unknown").title()
            fronters = json.loads(row.get("fronters") or "[]")
            topics   = json.loads(row.get("topics") or "[]")
            mood     = row.get("mood", "")
            register = row.get("emotional_register", "")
            lore     = json.loads(row.get("lore") or "[]")

            fronter_str = ", ".join(f.title() for f in fronters) if fronters else host
            topic_str   = ", ".join(topics) if topics else "—"

            lines = [
                f"[{ts}] with {fronter_str}",
                f"mood: {mood or '—'}  register: {register or '—'}  topics: {topic_str}",
                "",
                f"They said:",
                row.get("user_message", "").strip(),
                "",
                f"I said:",
                row.get("gizmo_response", "").strip(),
            ]

            if lore:
                lines += ["", "Context provided:", *[f"  - {l}" for l in lore]]

            return ToolResult(
                success=True,
                output="\n".join(lines),
            )

        except Exception as e:
            return ToolResult(success=False, output=str(e))
