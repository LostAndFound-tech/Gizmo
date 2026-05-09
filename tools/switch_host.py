"""
tools/switch_host.py
Handles host and fronter switches mid-conversation.

When called:
- Updates session context (current_host, fronters)
- Triggers host change detection
- Logs the switch to main RAG collection only (not personal collections)
- Returns a warm, personal greeting for the incoming host
"""

import time
from datetime import datetime
from tools.base_tool import BaseTool, ToolResult

# Session context store — mirrors what the agent tracks
# keyed by session_id, holds live context dict
_session_contexts: dict[str, dict] = {}


def get_session_context(session_id: str) -> dict:
    return _session_contexts.get(session_id, {})


def update_session_context(session_id: str, updates: dict) -> dict:
    if session_id not in _session_contexts:
        _session_contexts[session_id] = {}
    _session_contexts[session_id].update(updates)
    return _session_contexts[session_id]


class SwitchHostTool(BaseTool):
    @property
    def name(self) -> str:
        return "switch_host"

    @property
    def description(self) -> str:
        return (
            "Switch the current host or update who is co-fronting. "
            "Use when someone says they are taking over, stepping back, "
            "or another headmate is joining or leaving the front. "
            "Args: new_host (str) — who is now hosting. "
            "staying_fronters (list[str], optional) — who else is still present. "
            "session_id (str) — current session id."
        )

    async def run(
        self,
        new_host: str = "",
        staying_fronters: list = None,
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not new_host:
            return ToolResult(success=False, output="No new host specified.")

        new_host       = new_host.strip()
        new_host_lower = new_host.lower()
        staying_fronters = staying_fronters or []

        # Build updated fronters list
        fronters = list({new_host_lower} | {
            f.lower().strip() for f in staying_fronters
            if isinstance(f, str) and f.strip()
        })

        # Get previous context
        prev_context = get_session_context(session_id)
        prev_host    = prev_context.get("current_host", "")

        # Update session context
        update_session_context(session_id, {
            "current_host": new_host,
            "fronters":     fronters,
            "last_switch":  datetime.now().isoformat(),
        })

        # Log switch — main only, not personal collections
        await _log_switch(
            session_id=session_id,
            new_host=new_host,
            prev_host=prev_host,
            fronters=fronters,
        )

        # Update Alter Wheel — non-fatal, Pi may not be running
        try:
            from tools.alter_wheel_tool import AlterWheelTool
            import asyncio
            wheel = AlterWheelTool()
            await asyncio.wait_for(
                wheel.run(
                    action="switch",
                    new_host=new_host,
                    staying_fronters=[f for f in fronters if f != new_host_lower],
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            print("[SwitchHost] Wheel timeout — Pi not reachable, continuing anyway")
        except Exception as e:
            print(f"[SwitchHost] Wheel update failed (non-fatal): {e}")

        # Build result summary
        fronters_str  = ", ".join(f for f in fronters if f != new_host_lower)
        co_front_note = f" Co-fronting: {fronters_str}." if fronters_str else ""
        prev_note     = f" Previously: {prev_host}." if prev_host and prev_host.lower() != new_host_lower else ""

        return ToolResult(
            success=True,
            output=(
                f"Host switched to {new_host}.{prev_note}{co_front_note} "
                f"Greet {new_host} warmly and personally based on what you know about them. "
                f"Do not announce the switch mechanically — just address them directly."
            ),
            data={
                "new_host":      new_host,
                "previous_host": prev_host,
                "fronters":      fronters,
                "session_id":    session_id,
            }
        )


async def _log_switch(
    session_id: str,
    new_host: str,
    prev_host: str,
    fronters: list,
) -> None:
    """
    Log a host switch.
    Writes to main RAG only — never to personal headmate collections.
    Personal collections are for facts and memories, not system events.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        prev_note = (
            f" {prev_host} stepped back."
            if prev_host and prev_host.lower() != new_host.lower()
            else ""
        )
        co_note = (
            f" Also present: {', '.join(f for f in fronters if f != new_host.lower())}."
            if len(fronters) > 1 else ""
        )

        note = f"On {timestamp}, {new_host} took over as host.{prev_note}{co_note}"

        metadata = {
            "source":          "host_switch",
            "type":            "system_event",
            "subject":         new_host.lower(),
            "date":            timestamp,
            "session_id":      session_id,
            "new_host":        new_host.lower(),
            "previous_host":   prev_host.lower() if prev_host else "",
        }

        # Main RAG only — system log, not personal memory
        try:
            from core.rag import RAGStore
            store = RAGStore(collection_name="main")
            store.ingest_texts([note], metadatas=[metadata])
            print(f"[SwitchHost] Logged switch to main")
        except Exception as e:
            print(f"[SwitchHost] RAG log failed (non-fatal): {e}")

    except Exception as e:
        print(f"[SwitchHost] _log_switch failed: {e}")