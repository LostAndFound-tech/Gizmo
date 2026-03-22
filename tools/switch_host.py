"""
tools/switch_host.py
Handles host and fronter switches mid-conversation.

Switch types (classified by LLM from natural language):
  swap        — X takes over, current host moves to fronters
                "I'm swapping out with Oren", "X and I are switching"
  join        — X added to fronters only, host unchanged
                "Kaylee just joined", "I'm here too"
  depart      — current host leaving, BLOCKED until new host is named
                "I'm heading to my office", "stepping back"
  host_only   — change host, fronters untouched
                "X is taking over"
  add_fronter — add someone to fronters without changing host
  remove_fronter — remove someone from fronters

Departure rule:
  A hostless front is not allowed. If the host indicates they're leaving
  without naming a replacement, the tool blocks the switch and asks who
  is taking over before completing anything.

Host/fronters are fully separate:
  current_host = who is driving / being addressed
  fronters = others present — NOT including host

When a switch completes, the server should push a context_update
message to the client so the UI boxes update automatically.
"""

import json
from datetime import datetime
from tools.base_tool import BaseTool, ToolResult

_session_contexts: dict[str, dict] = {}

# Track pending departures per session — waiting for a new host to be named
_pending_departures: dict[str, dict] = {}


def get_session_context(session_id: str) -> dict:
    return _session_contexts.get(session_id, {})


def update_session_context(session_id: str, updates: dict) -> dict:
    if session_id not in _session_contexts:
        _session_contexts[session_id] = {}
    _session_contexts[session_id].update(updates)
    return _session_contexts[session_id]


# ── Switch classification ─────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """
Classify this fronting switch statement. Return ONLY valid JSON, no markdown.

{
  "action": "swap|join|depart|host_only|add_fronter|remove_fronter",
  "new_host": "name if someone is becoming host, or null",
  "joining": ["names joining fronters only, not as host"],
  "leaving": ["names leaving fronters"],
  "is_departure": true/false,
  "confidence": 0.0-1.0
}

Action definitions:
  swap         — current host trading places with someone (current host → fronters, new person → host)
  join         — someone joining co-front without changing host
  depart       — current host leaving, no replacement named yet
  host_only    — new host named explicitly, fronters unchanged
  add_fronter  — adding someone to fronters only
  remove_fronter — someone leaving fronters

Examples:
  "I'm swapping out with Oren"       → swap, new_host: Oren
  "Kaylee's joining us"              → join, joining: [Kaylee]
  "I'm heading to my office"         → depart, is_departure: true
  "I'm going to Corter"              → depart, is_departure: true
  "Oren's taking over"               → host_only, new_host: Oren
  "Princess and I are switching"     → swap, new_host: Princess
  "everyone's leaving except Honey"  → host_only, new_host: Honey
  "I'm stepping back"                → depart, is_departure: true

Current context: {context}
Statement: {statement}
"""


async def _classify_switch(statement: str, context: dict, llm) -> dict:
    """Ask the LLM to classify what kind of switch this is."""
    context_str = f"current_host={context.get('current_host', 'unknown')}, fronters={context.get('fronters', [])}"
    prompt_text = _CLASSIFY_PROMPT.replace("{context}", context_str).replace("{statement}", statement)

    try:
        from core.llm import llm as _llm
        _llm = llm or _llm
        raw = await _llm.generate(
            [{"role": "user", "content": prompt_text}],
            system_prompt="Classify fronting switch statements. JSON only. No markdown.",
            max_new_tokens=150,
            temperature=0.1,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[SwitchHost] Classification failed: {e}")
        return {"action": "host_only", "new_host": None, "confidence": 0.0}


# ── Main tool ─────────────────────────────────────────────────────────────────

class SwitchHostTool(BaseTool):
    @property
    def name(self) -> str:
        return "switch_host"

    @property
    def description(self) -> str:
        return (
            "Handle any fronting switch — host changes, co-fronters joining or leaving, "
            "or someone stepping back. Understands natural language switch statements. "
            "Call whenever someone indicates a host change, swap, departure, or fronter update. "
            "The tool classifies the switch type and handles blocking if host would go empty. "
            "Args: "
            "statement (str) — the original natural language statement about the switch. "
            "new_host (str, optional) — override if new host is already known. "
            "staying_fronters (list[str], optional) — explicit list of who stays in fronters. "
            "session_id (str) — current session id."
        )

    async def run(
        self,
        statement: str = "",
        new_host: str = "",
        staying_fronters: list = None,
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        from core.llm import llm

        prev_context = get_session_context(session_id)
        prev_host = prev_context.get("current_host", "")
        current_fronters = list(prev_context.get("fronters") or [])

        # Check for a pending departure — if we have one and new_host is now known, complete it
        if session_id in _pending_departures and new_host:
            return await self._complete_departure(
                session_id=session_id,
                new_host=new_host.strip(),
                staying_fronters=staying_fronters or current_fronters,
                prev_host=prev_host,
            )

        # Classify the switch from natural language
        classification = await _classify_switch(
            statement=statement or new_host,
            context=prev_context,
            llm=llm,
        )

        action = classification.get("action", "host_only")
        classified_new_host = classification.get("new_host") or new_host or ""
        joining = classification.get("joining", [])
        leaving = classification.get("leaving", [])
        is_departure = classification.get("is_departure", False)

        print(f"[SwitchHost] Classified '{statement}' → action={action}, new_host={classified_new_host}")

        # ── Departure — block until new host named ────────────────────────────
        if action == "depart" or (is_departure and not classified_new_host):
            _pending_departures[session_id] = {
                "departing": prev_host,
                "fronters":  current_fronters,
                "timestamp": datetime.now().isoformat(),
            }
            return ToolResult(
                success=True,
                output=(
                    f"{prev_host or 'The current host'} is stepping back. "
                    f"Ask who is taking over before completing the switch — "
                    f"a hostless front isn't allowed. "
                    f"Do not complete the switch until a new host is confirmed."
                ),
                data={"action": "depart_blocked", "departing": prev_host},
            )

        # ── Swap — current host moves to fronters, new host takes over ────────
        if action == "swap" and classified_new_host:
            new_fronters = list(current_fronters)
            if prev_host and prev_host.lower() not in [f.lower() for f in new_fronters]:
                new_fronters.append(prev_host)
            # Remove new host from fronters if they were there
            new_fronters = [f for f in new_fronters if f.lower() != classified_new_host.lower()]
            return await self._execute_switch(
                session_id=session_id,
                new_host=classified_new_host,
                new_fronters=new_fronters,
                prev_host=prev_host,
                action=action,
            )

        # ── Join — add to fronters only ───────────────────────────────────────
        if action in ("join", "add_fronter") and joining:
            new_fronters = list(current_fronters)
            for name in joining:
                if name.lower() not in [f.lower() for f in new_fronters]:
                    new_fronters.append(name)
            update_session_context(session_id, {"fronters": new_fronters})
            await _log_switch_to_rag(session_id, prev_host, prev_host, new_fronters, action="join")
            joined_str = ", ".join(joining)
            return ToolResult(
                success=True,
                output=f"{joined_str} joined the front. Host is still {prev_host}.",
                data={"action": "join", "new_host": prev_host, "fronters": new_fronters},
            )

        # ── Remove fronter ────────────────────────────────────────────────────
        if action == "remove_fronter" and leaving:
            new_fronters = [f for f in current_fronters if f.lower() not in [l.lower() for l in leaving]]
            update_session_context(session_id, {"fronters": new_fronters})
            await _log_switch_to_rag(session_id, prev_host, prev_host, new_fronters, action="remove")
            left_str = ", ".join(leaving)
            return ToolResult(
                success=True,
                output=f"{left_str} stepped back from the front. Host is still {prev_host}.",
                data={"action": "remove_fronter", "new_host": prev_host, "fronters": new_fronters},
            )

        # ── Host only — change host, fronters untouched ───────────────────────
        if classified_new_host:
            new_fronters = staying_fronters if staying_fronters is not None else current_fronters
            # Ensure new host isn't in fronters
            new_fronters = [f for f in new_fronters if f.lower() != classified_new_host.lower()]
            return await self._execute_switch(
                session_id=session_id,
                new_host=classified_new_host,
                new_fronters=new_fronters,
                prev_host=prev_host,
                action="host_only",
            )

        return ToolResult(
            success=False,
            output="Couldn't figure out who's switching. Can you clarify who's taking over?",
        )

    async def _execute_switch(
        self,
        session_id: str,
        new_host: str,
        new_fronters: list,
        prev_host: str,
        action: str,
    ) -> ToolResult:
        new_host = new_host.strip()

        update_session_context(session_id, {
            "current_host": new_host,
            "fronters":     new_fronters,
            "last_switch":  datetime.now().isoformat(),
        })

        await _log_switch_to_rag(session_id, new_host, prev_host, new_fronters, action=action)

        # Push UI update via server if reachable
        try:
            from core.push import _connected
            import json as _json
            payload = _json.dumps({
                "type":         "context_update",
                "current_host": new_host,
                "fronters":     new_fronters,
            })
            for ws in list(_connected):
                try:
                    await ws.send(payload)
                except Exception:
                    pass
        except Exception:
            pass

        # Update Alter Wheel — non-fatal
        try:
            from tools.alter_wheel_tool import AlterWheelTool
            import asyncio
            wheel = AlterWheelTool()
            await asyncio.wait_for(
                wheel.run(
                    action="switch",
                    new_host=new_host,
                    staying_fronters=new_fronters,
                ),
                timeout=3.0,
            )
        except Exception as e:
            print(f"[SwitchHost] Wheel update failed (non-fatal): {e}")

        fronters_str = ", ".join(new_fronters)
        co_front_note = f" Co-fronting: {fronters_str}." if fronters_str else ""
        prev_note = f" Previously: {prev_host}." if prev_host and prev_host.lower() != new_host.lower() else ""

        return ToolResult(
            success=True,
            output=(
                f"Host switched to {new_host}.{prev_note}{co_front_note} "
                f"Greet {new_host} warmly and personally based on what you know about them. "
                f"Do not announce the switch mechanically — just address them directly."
            ),
            data={
                "action":          action,
                "new_host":        new_host,
                "previous_host":   prev_host,
                "fronters":        new_fronters,
                "session_id":      session_id,
            },
        )

    async def _complete_departure(
        self,
        session_id: str,
        new_host: str,
        staying_fronters: list,
        prev_host: str,
    ) -> ToolResult:
        """Complete a previously blocked departure now that we have a new host."""
        pending = _pending_departures.pop(session_id, {})
        departing = pending.get("departing", prev_host)

        # Departing person does NOT move to fronters — they're gone
        new_fronters = [f for f in staying_fronters if f.lower() != new_host.lower()
                        and f.lower() != (departing or "").lower()]

        return await self._execute_switch(
            session_id=session_id,
            new_host=new_host,
            new_fronters=new_fronters,
            prev_host=departing,
            action="depart_complete",
        )


# ── RAG logging ───────────────────────────────────────────────────────────────

async def _log_switch_to_rag(
    session_id: str,
    new_host: str,
    prev_host: str,
    fronters: list,
    action: str = "switch",
) -> None:
    try:
        from core.rag import RAGStore

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        prev_note = f" {prev_host} stepped back." if prev_host and prev_host.lower() != new_host.lower() else ""
        co_note = f" Also present: {', '.join(fronters)}." if fronters else ""
        note = f"On {timestamp}, {new_host} took over as host.{prev_note}{co_note}"

        metadata = {
            "source":          "host_switch",
            "type":            "system_event",
            "action":          action,
            "date":            timestamp,
            "session_id":      session_id,
            "new_host":        new_host.lower(),
            "previous_host":   prev_host.lower() if prev_host else "",
        }

        collections = {"main", new_host.lower()}
        if prev_host:
            collections.add(prev_host.lower())

        for collection in collections:
            store = RAGStore(collection_name=collection)
            store.ingest_texts([note], metadatas=[{**metadata, "collection": collection}])
            print(f"[SwitchHost] Logged switch to '{collection}'")

    except Exception as e:
        print(f"[SwitchHost] Failed to log switch to RAG: {e}")
