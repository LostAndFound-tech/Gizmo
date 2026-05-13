"""
tools/recall_search_tool.py

Fast metadata search over the message store.
Returns lightweight stubs — id, timestamp, host, topics, mood,
and a short preview of the user message.

Gizmo uses this to find relevant past exchanges without pulling
full response text. Once he finds a promising stub, he calls
recall_read with the id to get the full exchange.
"""

import json
from tools.base_tool import BaseTool, ToolResult


class RecallSearchTool(BaseTool):
    name        = "recall_search"
    description = (
        "Search past conversations by topic, host, mood, keyword, or time. "
        "Returns short stubs — timestamp, who was there, topics, mood, and a "
        "preview of what was said. Use this first to find relevant exchanges. "
        "Then use recall_read with a message id to get the full exchange. "
        "Examples: find when Princess mentioned dresses, what we talked about "
        "this morning, any anxious conversations this week."
    )
    args_schema = {
        "topics": {
            "type": "array",
            "description": "Topic keywords to search for e.g. ['fashion', 'dress', 'clothing']",
            "default": [],
        },
        "host": {
            "type": "string",
            "description": "Filter by headmate name",
            "default": "",
        },
        "keyword": {
            "type": "string",
            "description": "Freetext search in message content",
            "default": "",
        },
        "mood": {
            "type": "string",
            "description": "Filter by mood e.g. 'anxious', 'playful', 'warm'",
            "default": "",
        },
        "since": {
            "type": "string",
            "description": "Date to search from e.g. '2026-05-12' or 'today' or 'yesterday'",
            "default": "",
        },
        "limit": {
            "type": "integer",
            "description": "Max results to return (default 10, max 30)",
            "default": 10,
        },
    }

    async def run(
        self,
        session_id: str,
        topics: list = None,
        host: str = "",
        keyword: str = "",
        mood: str = "",
        since: str = "",
        limit: int = 10,
        **kwargs,
    ) -> ToolResult:
        try:
            from core.message_store import search
            from datetime import datetime, timedelta

            # Resolve 'today' / 'yesterday' to date strings
            since_str = None
            if since:
                s = since.strip().lower()
                if s == "today":
                    since_str = datetime.now().strftime("%Y-%m-%d")
                elif s == "yesterday":
                    since_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    since_str = since

            limit = min(max(1, limit), 30)

            rows = search(
                host    = host or None,
                topics  = topics or None,
                keyword = keyword or None,
                mood    = mood or None,
                since   = since_str,
                limit   = limit,
                order   = "DESC",
            )

            if not rows:
                return ToolResult(
                    success=True,
                    output="No matching messages found.",
                )

            lines = [f"{len(rows)} result(s):\n"]
            for row in rows:
                ts      = row["timestamp"][:16].replace("T", " ")
                h       = (row.get("host") or "unknown").title()
                topics_  = json.loads(row.get("topics") or "[]")
                mood_    = row.get("mood", "")
                preview  = (row.get("user_message") or "")[:80].strip()
                if len(row.get("user_message", "")) > 80:
                    preview += "..."
                notable = " ★" if row.get("notable") else ""

                topic_str = ", ".join(topics_[:4]) if topics_ else "—"
                lines.append(
                    f"[{ts}] {h}{notable}\n"
                    f"  id: {row['id']}\n"
                    f"  topics: {topic_str}\n"
                    f"  mood: {mood_ or '—'}\n"
                    f"  said: \"{preview}\""
                )

            return ToolResult(
                success=True,
                output="\n\n".join(lines),
            )

        except Exception as e:
            return ToolResult(success=False, output=str(e))
