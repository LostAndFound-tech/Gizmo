"""
memory/history.py
Conversation history per session with timestamps and context snapshots.
Each message stores who was present when, enabling per-fronter archiving.
"""

import time
from collections import deque
from typing import Literal, Optional


class ConversationHistory:
    def __init__(self, max_turns: int = 20):
        self.max_messages = max_turns * 2
        self._messages: deque[dict] = deque(maxlen=self.max_messages)
        self.last_active: float = time.time()
        self.archived: bool = False

    def add(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        context: Optional[dict] = None,
    ) -> None:
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "context": context or {},
        })
        self.last_active = time.time()

    def as_messages(self, new_user_message: str) -> list[dict]:
        """Clean messages list for LLM — strips metadata, keeps role/content."""
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages
        ]
        messages.append({"role": "user", "content": new_user_message})
        return messages

    def as_list(self) -> list[dict]:
        """Full message list including timestamps and context snapshots."""
        return list(self._messages)

    def format_for_prompt(self, new_user_message: str) -> str:
        lines = []
        for msg in self._messages:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {msg['content']}")
        lines.append(f"User: {new_user_message}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def seconds_since_active(self) -> float:
        return time.time() - self.last_active

    def get_fronters_for_window(self, messages: list[dict]) -> dict:
        """
        Given a window of messages, return who was present.
        Returns: {
            "host": set of hosts seen,
            "fronters": set of all fronters seen,
            "collections": set of all collections to ingest into (fronters + main)
        }
        """
        hosts = set()
        fronters = set()

        for msg in messages:
            ctx = msg.get("context", {})
            host = ctx.get("current_host", "")
            if host:
                hosts.add(host.lower().strip())
                fronters.add(host.lower().strip())
            for f in ctx.get("fronters", []):
                name = f.lower().strip() if isinstance(f, str) else str(f).lower().strip()
                if name:
                    fronters.add(name)

        collections = fronters | {"main"}
        return {
            "hosts": hosts,
            "fronters": fronters,
            "collections": collections,
        }


# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict[str, ConversationHistory] = {}


def get_session(session_id: str) -> ConversationHistory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationHistory()
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    if session_id in _sessions:
        _sessions[session_id].clear()


def get_all_sessions() -> dict[str, ConversationHistory]:
    return _sessions