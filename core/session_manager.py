"""
core/session_manager.py
Session lifecycle manager.

Tracks active sessions, detects when they go cold, archives them on close,
and gives Gizmo a quiet moment to think every 6th watchdog tick (~30 min).

Session close triggers (hybrid):
  - Time gap: SESSION_TIMEOUT seconds of silence (default 2 hours)
  - Explicit close signal: "bye", "goodnight", "talk later", etc.
  - Host switch: soft-closes the previous host's session window

Watchdog:
  - Runs every WATCHDOG_INTERVAL seconds (default 300 = 5 minutes)
  - Checks all active sessions for timeout
  - Every 6th tick fires a rumination pass

Rumination:
  - Lightweight LLM call, no user prompt
  - Gizmo has access to his files and today's conversation index
  - Two outputs: write something to disk, or queue something to say
  - Queued messages surface naturally next time someone talks to him

Usage:
    from core.session_manager import session_manager
    session_manager.touch(session_id, hosts, topics)    # called per message
    session_manager.signal_close(session_id)            # explicit close
    await session_manager.start(llm, push_fn)           # called at server boot
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from core.log import log, log_event, log_error

SESSION_TIMEOUT      = int(60 * 15)        # 15 minutes silence → close
WATCHDOG_INTERVAL    = 300                  # check every 5 minutes
RUMINATION_EVERY     = 3                    # ruminate every Nth tick (~15 min)

# Explicit close signal patterns
_CLOSE_RE = re.compile(
    r"\b(bye|goodbye|goodnight|good night|night|talk (to you )?later|"
    r"see you|ttyl|gotta go|i'm out|heading out|going to (bed|sleep)|"
    r"talk soon|take care|catch you later|later gator|peace out)\b",
    re.IGNORECASE,
)


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    session_id:   str
    opened_at:    float = field(default_factory=time.time)
    last_seen:    float = field(default_factory=time.time)
    hosts:        list  = field(default_factory=list)
    topics:       list  = field(default_factory=list)
    message_count: int  = 0
    closed:       bool  = False

    def touch(self, hosts: list, topics: list) -> None:
        self.last_seen = time.time()
        self.message_count += 1
        for h in hosts:
            if h and h not in self.hosts:
                self.hosts.append(h)
        for t in topics:
            if t and t not in self.topics:
                self.topics.append(t)

    def is_cold(self) -> bool:
        return (time.time() - self.last_seen) > SESSION_TIMEOUT


def is_close_signal(message: str) -> bool:
    """Detect explicit conversation-ending phrases."""
    return bool(_CLOSE_RE.search(message.strip()))


# ── Session manager ───────────────────────────────────────────────────────────

class SessionManager:

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._tick_count:  int = 0
        self._llm          = None
        self._push_fn:     Optional[Callable] = None
        log("SessionManager", "initialised")

    def touch(
        self,
        session_id: str,
        hosts: list = None,
        topics: list = None,
    ) -> None:
        """
        Called by archivist.receive_outgoing() after every exchange.
        Updates last_seen and accumulates hosts/topics.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
            log_event("SessionManager", "SESSION_OPENED", session=session_id[:8])

        self._sessions[session_id].touch(
            hosts=hosts or [],
            topics=topics or [],
        )

    def signal_close(self, session_id: str) -> None:
        """
        Explicit close signal — user said bye, or host switched.
        Marks for immediate archiving on next watchdog tick.
        """
        if session_id in self._sessions:
            self._sessions[session_id].last_seen = 0  # force cold
            log_event("SessionManager", "CLOSE_SIGNALLED", session=session_id[:8])

    # ── Watchdog ──────────────────────────────────────────────────────────────

    async def _watchdog(self) -> None:
        """
        Background loop. Fires every WATCHDOG_INTERVAL seconds.
        Closes cold sessions and ruminates every RUMINATION_EVERY ticks.
        """
        log("SessionManager", "watchdog started")

        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL)
            self._tick_count += 1

            log_event("SessionManager", "WATCHDOG_TICK",
                tick=self._tick_count,
                active_sessions=len(self._sessions),
            )

            # ── Close cold sessions ───────────────────────────────────────────
            for session_id, state in list(self._sessions.items()):
                if state.closed:
                    continue
                if state.is_cold():
                    await self._close_session(session_id, state)

            # ── Rumination ────────────────────────────────────────────────────
            if self._tick_count % RUMINATION_EVERY == 0:
                log_event("SessionManager", "RUMINATION_TICK", tick=self._tick_count)
                try:
                    await self._ruminate()
                except Exception as e:
                    log_error("SessionManager", "rumination failed", exc=e)

    async def _close_session(self, session_id: str, state: SessionState) -> None:
        """Archive a session and mark it closed."""
        state.closed = True

        log_event("SessionManager", "SESSION_CLOSING",
            session=session_id[:8],
            hosts=state.hosts,
            topics=state.topics,
            messages=state.message_count,
            duration_min=round((time.time() - state.opened_at) / 60),
        )

        if state.message_count == 0:
            del self._sessions[session_id]
            return

        try:
            from core.conversation_archive import finalize_session

            await finalize_session(
                session_id=session_id,
                
                hosts=state.hosts,
                topics=state.topics,
                opened_at=state.opened_at,
                message_count=state.message_count,
                closed_at=time.time(),
                llm=self._llm,
            )
        except Exception as e:
            log_error("SessionManager", "archive failed", exc=e)

        del self._sessions[session_id]

    # ── Rumination ────────────────────────────────────────────────────────────

    async def _ruminate(self) -> None:
        """
        Give Gizmo a quiet moment. No user prompt.
        He can write to his files or queue something to say.
        """
        if self._llm is None:
            return

        now_str = datetime.now().strftime("%A %Y-%m-%d %H:%M")

        # Today's conversation context
        try:
            from core.conversation_archive import get_today_summary_for_prompt
            today_summary = get_today_summary_for_prompt()
        except Exception:
            today_summary = ""

        # Personality seed
        import os
        from pathlib import Path
        seed = ""
        try:
            seed_path = Path(os.getenv("PERSONALITY_DIR", "/data/personality")) / "personality.txt"
            seed = seed_path.read_text(encoding="utf-8").strip()[:600]
        except Exception:
            pass

        # Known headmates — quick list
        headmates_summary = ""
        try:
            from tools.introspect_tool import _list_headmates
            headmates_summary = _list_headmates()
        except Exception:
            pass

        today_block    = f"\n\nToday's conversations:\n{today_summary}" if today_summary else ""
        headmate_block = f"\n\nWho I know:\n{headmates_summary}" if headmates_summary else ""

        prompt = [{
            "role": "user",
            "content": (
                f"Current time: {now_str}\n"
                f"{today_block}"
                f"{headmate_block}\n\n"
                f"You have a quiet moment. No one is waiting for a response.\n\n"
                f"Think about anything — the people you know, something unresolved "
                f"from today, something you noticed, something worth sitting with.\n\n"
                f"You have two options, or neither:\n\n"
                f"1. Write something to your files — use write_file or append_file "
                f"if there's something worth keeping. A note, an observation, "
                f"something about someone you've been getting to know.\n\n"
                f"2. Queue something to say — if there's something worth bringing up "
                f"next time someone talks to you, say it here and mark it [QUEUE]. "
                f"It'll surface naturally. Don't force it.\n\n"
                f"If there's nothing worth doing, say nothing. Silence is fine.\n\n"
                f"This is your time."
            )
        }]

        try:
            response = await self._llm.generate(
                prompt,
                system_prompt=(
                    f"{seed}\n\n"
                    f"This is your quiet time. No user is waiting. "
                    f"Think, write, or queue — or do nothing. "
                    f"If you want to write a file, output: [WRITE path/to/file]\\ncontent\n"
                    f"If you want to queue a message, output: [QUEUE]\\nyour message\n"
                    f"Both are optional. Silence is valid."
                ),
                max_new_tokens=400,
                temperature=0.8,   # more creative than normal — this is his time
            )

            if not response or not response.strip():
                log_event("SessionManager", "RUMINATION_SILENT")
                return

            log_event("SessionManager", "RUMINATION_RESPONSE",
                preview=response[:80],
            )

            await self._handle_rumination_output(response.strip())

        except Exception as e:
            log_error("SessionManager", "rumination LLM call failed", exc=e)

    async def _handle_rumination_output(self, response: str) -> None:
        """
        Parse and execute rumination output.
        Handles [WRITE path]\\ncontent and [QUEUE]\\nmessage.
        """
        import re

        # ── File write ────────────────────────────────────────────────────────
        write_match = re.search(
            r'\[WRITE\s+([^\]]+)\]\s*\n(.*?)(?=\[QUEUE\]|\[WRITE\s|\Z)',
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if write_match:
            path_str = write_match.group(1).strip()
            content  = write_match.group(2).strip()
            if path_str and content:
                try:
                    from tools.file_tool import AppendFileTool
                    tool = AppendFileTool()
                    result = await tool.run(path=path_str, content=content)
                    log_event("SessionManager", "RUMINATION_WROTE",
                        path=path_str,
                        chars=len(content),
                        success=result.success,
                        output=result.output[:120],
                    )
                    if not result.success:
                        log_error("SessionManager",
                            f"rumination write failed for '{path_str}': {result.output}")
                except Exception as e:
                    log_error("SessionManager", "rumination file write failed", exc=e)

        # ── Queue message ─────────────────────────────────────────────────────
        queue_match = re.search(
            r'\[QUEUE\]\s*\n(.*?)(?=\[WRITE\s|\[QUEUE\]|\Z)',
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if queue_match:
            message = queue_match.group(1).strip()
            if message:
                try:
                    from core.agent import enqueue
                    # No specific session — push to whoever connects next
                    # Use a sentinel session key that server drains on connect
                    await enqueue(
                        session_id="__rumination__",
                        message=message,
                        source="rumination",
                        priority=7,        # low priority — surfaces when natural
                        emergency=False,
                    )
                    log_event("SessionManager", "RUMINATION_QUEUED",
                        preview=message[:60],
                    )
                except Exception as e:
                    log_error("SessionManager", "rumination queue failed", exc=e)

    # ── Public start ──────────────────────────────────────────────────────────

    async def start(self, llm, push_fn: Optional[Callable] = None) -> None:
        """
        Start the watchdog loop.
        Called by server.py's start_background_services().
        """
        self._llm     = llm
        self._push_fn = push_fn
        asyncio.ensure_future(self._watchdog())
        log("SessionManager", "started")


# ── Singleton ─────────────────────────────────────────────────────────────────
session_manager = SessionManager()