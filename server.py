"""
server.py
Gizmo's WebSocket server.

Receives messages. Routes them. Streams responses back.
Does not think. Does not generate. Just moves things.

Single-part exchange (fast path):
  "[Name]: message" or plain text
  → intake → parallel agents → director → personality → response
  → stream back

Multi-part exchange (room path):
  Multiple [Name]: lines, *actions*, silent presences
  → parse into scene
  → run full pipeline per part (collect directives, don't respond yet)
  → synthesis pass: director receives all directives + room context
  → personality layer with room weights
  → one response that addresses the room
  → stream back

Deduplication:
  Hash(session_id + message) within 5s window
  Silent drop on duplicate

Room contracts:
  Pre-loaded when session opens or fronters change
  "Ara consented to witness dynamic with Jess"
  Shapes room weights in synthesis

WebSocket message format (inbound):
  {
    "type":       "message" | "ping" | "switch_host" | "add_fronter" | "remove_fronter",
    "session_id": "sess_xyz",
    "content":    "raw text from UI" | [{"headmate": ..., "content": ..., "content_type": ...}],
    "context":    {"current_host": ..., "fronters": [...]}  // optional
  }

WebSocket message format (outbound):
  {"type": "chunk",    "content": "text chunk"}
  {"type": "done",     "session_id": "..."}
  {"type": "error",    "message": "..."}
  {"type": "thinking"} // while pipeline runs
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Optional, Callable

from core.log import log, log_event, log_error
from core.timezone import tz_now


# ── Config ────────────────────────────────────────────────────────────────────

DEDUP_WINDOW     = 5.0   # seconds — drop identical messages within this window
THINKING_DELAY   = 0.3   # seconds before sending "thinking" signal
CHUNK_SIZE       = 8     # characters per stream chunk
MAX_MESSAGE_LEN  = 8000  # hard cap on inbound message length

# Regex for parsing exchange parts
_SPEECH_RE   = re.compile(r'^\[?([A-Za-z][A-Za-z0-9_\- ]{0,30})\]?\s*:\s*(.+)', re.DOTALL)
_ACTION_RE   = re.compile(r'^\*(.+)\*$', re.DOTALL)
_DIRECTED_RE = re.compile(r'\b(to|at|@)\s+([A-Za-z][A-Za-z0-9_\- ]{0,20})\b', re.IGNORECASE)


# ── Deduplication ─────────────────────────────────────────────────────────────

_seen_messages: dict[str, float] = {}


def _is_duplicate(session_id: str, content: str) -> bool:
    key = hashlib.md5(f"{session_id}:{content}".encode()).hexdigest()
    now = time.time()
    if key in _seen_messages and now - _seen_messages[key] < DEDUP_WINDOW:
        log_event("Server", "DUPLICATE_DROPPED",
            session=session_id[:8],
            preview=content[:40],
        )
        return True
    _seen_messages[key] = now
    # Prune old entries
    stale = [k for k, t in _seen_messages.items() if now - t > DEDUP_WINDOW * 2]
    for k in stale:
        del _seen_messages[k]
    return False


# ── Exchange parser ───────────────────────────────────────────────────────────

def parse_exchange(raw: str, default_headmate: Optional[str] = None) -> list[dict]:
    """
    Parse raw message text into a list of exchange parts.

    Handles:
      [Jess]: I love the buttons on my dress
      [Ara]: the pink ones are so her, right?
      *Oren tugs at her sleeve*
      Jess: plain speech without brackets
      plain text with no prefix → default_headmate

    Returns list of:
      {
        headmate:      str | None,
        content:       str | None,
        content_type:  "speech" | "action" | "presence",
        directed_at:   str | None,
      }
    """
    parts = []
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]

    for line in lines:
        # Action: *Oren tugs at her sleeve*
        action_match = _ACTION_RE.match(line)
        if action_match:
            action_text = action_match.group(1).strip()

            # Try to extract actor from action text
            # "Oren tugs at..." → actor = Oren
            actor = None
            words = action_text.split()
            if words:
                # First word capitalized = likely actor
                if words[0][0].isupper() and len(words[0]) > 1:
                    actor = words[0].rstrip("'s")
                    action_text = " ".join(words[1:]) if len(words) > 1 else action_text

            # Directed at?
            directed = None
            d_match = _DIRECTED_RE.search(action_text)
            if d_match:
                directed = d_match.group(2).lower()

            parts.append({
                "headmate":     actor.lower() if actor else default_headmate,
                "content":      action_text,
                "content_type": "action",
                "directed_at":  directed,
            })
            continue

        # Speech: [Jess]: ... or Jess: ...
        speech_match = _SPEECH_RE.match(line)
        if speech_match:
            name    = speech_match.group(1).strip().lower()
            content = speech_match.group(2).strip()

            # Directed at?
            directed = None
            d_match  = _DIRECTED_RE.search(content)
            if d_match:
                directed = d_match.group(2).lower()
                if directed == "gizmo" or directed == "you":
                    directed = "gizmo"

            # Check for inline action: "message *does something*"
            inline_action = re.search(r'\*([^*]+)\*', content)
            if inline_action:
                # Split into speech + action
                action_text = inline_action.group(1)
                speech_text = re.sub(r'\*[^*]+\*', '', content).strip()

                if speech_text:
                    parts.append({
                        "headmate":     name,
                        "content":      speech_text,
                        "content_type": "speech",
                        "directed_at":  directed,
                    })
                parts.append({
                    "headmate":     name,
                    "content":      action_text,
                    "content_type": "action",
                    "directed_at":  None,
                })
            else:
                parts.append({
                    "headmate":     name,
                    "content":      content,
                    "content_type": "speech",
                    "directed_at":  directed,
                })
            continue

        # Plain text — assign to default headmate
        if line and default_headmate:
            parts.append({
                "headmate":     default_headmate.lower(),
                "content":      line,
                "content_type": "speech",
                "directed_at":  "gizmo",
            })

    # If nothing parsed, treat entire message as speech from default
    if not parts and raw.strip() and default_headmate:
        parts.append({
            "headmate":     default_headmate.lower(),
            "content":      raw.strip(),
            "content_type": "speech",
            "directed_at":  "gizmo",
        })

    return parts


def is_multi_part(raw: str) -> bool:
    """Quick check: does this message contain multiple voices?"""
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    matches = sum(
        1 for l in lines
        if _SPEECH_RE.match(l) or _ACTION_RE.match(l)
    )
    return matches >= 2


# ── Room contract loader ──────────────────────────────────────────────────────

async def load_room_contracts(
    session_id: str,
    fronters:   list[str],
) -> dict:
    """
    Load relationship contracts between fronters + Gizmo.
    Returns room_contracts dict for synthesis.

    Example:
      {
        "ara_witnesses_jess_dynamic": {
          "speaker":   "ara",
          "consented": "witness gizmo dominant with jess",
          "weight":    0.3,
          "temper":    "adjacent, not involved"
        }
      }
    """
    if len(fronters) < 2:
        return {}

    from core.store import store

    contracts = {}

    for name in fronters:
        rels = store.query("relationships",
            headmate=name.lower(),
            active=1,
            limit=20,
        )
        for rel in rels:
            label = rel.get("relationship_label", "")
            cat   = rel.get("relationship_category", "")

            if cat == "room_contract" or "consent" in label or "witness" in label:
                key = f"{name}_{label}".lower().replace(" ", "_")
                contracts[key] = {
                    "speaker":   name,
                    "label":     label,
                    "entity":    rel.get("entity", ""),
                    "confidence": rel.get("confidence_type", "stated"),
                    "raw":       rel,
                }

    log_event("Server", "ROOM_CONTRACTS_LOADED",
        session=session_id[:8],
        fronters=fronters,
        contracts=list(contracts.keys()),
    )

    return contracts


# ── Scene assembler ───────────────────────────────────────────────────────────

def assemble_scene_text(parts: list[dict]) -> str:
    """
    Format exchange parts as a readable scene transcript.
    Used as context in synthesis prompt.
    """
    lines = []
    for part in parts:
        name    = (part.get("headmate") or "unknown").title()
        content = part.get("content", "")
        ctype   = part.get("content_type", "speech")
        directed = part.get("directed_at")

        if ctype == "action":
            line = f"*{name} {content}*"
        elif ctype == "presence":
            line = f"[{name} is present]"
        else:
            suffix = f" (to {directed})" if directed and directed != "gizmo" else ""
            line   = f"[{name}]{suffix}: {content}"

        lines.append(line)

    return "\n".join(lines)


# ── Single-part pipeline (fast path) ─────────────────────────────────────────

async def run_single_pipeline(
    message:    str,
    session_id: str,
    headmate:   str,
    context:    dict,
    history,
    llm,
) -> str:
    """Standard single-voice pipeline. Returns response text."""
    from core.agent import agent
    chunks = []
    async for chunk in agent.respond(
        user_message=message,
        session_id=session_id,
        context=context,
        history=history,
    ):
        chunks.append(chunk)
    return "".join(chunks)


# ── Multi-part pipeline (room path) ──────────────────────────────────────────

async def run_room_pipeline(
    parts:      list[dict],
    session_id: str,
    fronters:   list[str],
    context:    dict,
    history,
    llm,
    push_fn:    Optional[Callable] = None,
) -> str:
    """
    Room path. Runs full pipeline per speaking part, collects directives,
    synthesizes one response that addresses the room.
    """
    from core.agent import (
        intake, agent_knowledge, agent_wellness,
        agent_therapy, agent_narrative, director,
        personality_layer, generate_response,
        close_loop, Brief, Directive,
    )
    from core.session_manager import session_manager

    # Load room contracts
    room_contracts = await load_room_contracts(session_id, fronters)

    # ── Run pipeline per speaking part ────────────────────────────────────────
    briefs:     list[Brief]    = []
    directives: list[Directive] = []
    knowledges: list[dict]     = []

    speech_parts = [p for p in parts if p.get("content_type") in ("speech", "action")]
    silent_parts = [p for p in parts if p.get("content_type") == "presence"]

    # Update session manager with all fronters
    for p in silent_parts:
        name = p.get("headmate")
        if name:
            session_manager.add_fronters(session_id, [name])

    # Run pipelines in parallel where possible
    # But need briefs built sequentially to avoid entity cache races
    for part in speech_parts:
        headmate = part.get("headmate") or context.get("current_host") or ""
        content  = part.get("content", "")
        ctype    = part.get("content_type", "speech")

        if not content:
            continue

        # Build per-part context
        part_context = dict(context)
        part_context["current_host"] = headmate
        part_context["fronters"]     = fronters
        part_context["content_type"] = ctype
        part_context["directed_at"]  = part.get("directed_at")

        # Prefix action differently for intake
        intake_message = (
            f"*{content}*" if ctype == "action"
            else content
        )

        try:
            brief = await intake(
                message=intake_message,
                session_id=session_id,
                context=part_context,
                history=history,
                llm=llm,
            )
            briefs.append(brief)

        except Exception as e:
            log_error("Server", f"intake failed for {headmate}: {e}", exc=e)
            continue

    if not briefs:
        return "..."

    # Run parallel agents for all briefs simultaneously
    all_tasks = []
    for brief in briefs:
        all_tasks.append((
            brief,
            asyncio.create_task(agent_knowledge(brief, llm)),
            asyncio.create_task(agent_wellness(brief, llm)),
            asyncio.create_task(agent_therapy(brief, llm)),
            asyncio.create_task(agent_narrative(brief, history, llm)),
        ))

    # Collect results
    part_results = []
    for brief, kt, wt, tt, nt in all_tasks:
        knowledge, wellness, therapy, narrative = await asyncio.gather(
            kt, wt, tt, nt
        )
        part_results.append((brief, knowledge, wellness, therapy, narrative))

    # Run director per part to get individual directives
    for brief, knowledge, wellness, therapy, narrative in part_results:
        try:
            d = await director(
                brief=brief,
                knowledge=knowledge,
                wellness=wellness,
                therapy=therapy,
                narrative=narrative,
                llm=llm,
            )
            directives.append(d)
            knowledges.append(knowledge)
        except Exception as e:
            log_error("Server", f"director failed: {e}", exc=e)

    if not directives:
        return "..."

    # ── Synthesis pass ────────────────────────────────────────────────────────
    primary_brief = briefs[0]  # first speaker drives the primary register

    unified_directive = await synthesize_directives(
        briefs=briefs,
        directives=directives,
        knowledges=knowledges,
        parts=parts,
        room_contracts=room_contracts,
        fronters=fronters,
        llm=llm,
    )

    # ── Personality layer ─────────────────────────────────────────────────────
    # Build room-aware system prompt
    system_prompt = build_room_prompt(
        primary_brief=primary_brief,
        directive=unified_directive,
        briefs=briefs,
        room_contracts=room_contracts,
        scene_text=assemble_scene_text(parts),
    )

    # ── Generate response ─────────────────────────────────────────────────────
    response = await generate_response(
        brief=primary_brief,
        system_prompt=system_prompt,
        history=history,
        llm=llm,
    )

    # ── Close loop for each part ──────────────────────────────────────────────
    for brief, d in zip(briefs, directives):
        asyncio.ensure_future(
            close_loop(
                brief=brief,
                directive=d,
                response=response,
                history=history,
                llm=llm,
            )
        )

    return response


# ── Synthesis ─────────────────────────────────────────────────────────────────

async def synthesize_directives(
    briefs:         list,
    directives:     list,
    knowledges:     list,
    parts:          list[dict],
    room_contracts: dict,
    fronters:       list[str],
    llm,
) -> object:
    """
    Merge multiple directives into one unified directive for the room.
    Accounts for room contracts, relationships, register weights.
    """
    from core.agent import Directive, _parse_json, _call

    scene_text = assemble_scene_text(parts)

    # Build per-person summaries
    person_summaries = []
    for brief, directive, knowledge in zip(briefs, directives, knowledges):
        person_summaries.append(
            f"{brief.headmate or 'unknown'} "
            f"[{brief.register}] "
            f"→ {directive.meaning} "
            f"| tone: {directive.tone} "
            f"| push: {directive.pattern_action}"
        )

    # Room contracts text
    contracts_text = "\n".join(
        f"  - {v['speaker']} → {v['label']} → {v['entity']}"
        for v in room_contracts.values()
    ) or "  (none on file)"

    # Register weights — figure out which register dominates
    registers = [b.register for b in briefs]
    intimate_regs = {"intimate", "dominant", "submissive", "subspace",
                     "scene", "erotic", "degradation"}
    has_intimate = any(r in intimate_regs for r in registers)
    has_distress = any(r in ("distress", "crisis") for r in registers)

    raw = await _call(llm,
        system=(
            "You synthesize multiple response directives into one unified directive "
            "for a room with multiple people. You understand social geometry — "
            "who consented to what, who needs what register, how to hold everyone "
            "at once without ignoring anyone. JSON only."
        ),
        user=f"""Scene that just happened:
{scene_text}

People present: {', '.join(f.title() for f in fronters)}

Individual directives:
{chr(10).join(person_summaries)}

Room contracts:
{contracts_text}

Room signals:
  intimate register active: {has_intimate}
  distress active: {has_distress}
  registers: {', '.join(registers)}

Write ONE unified directive that:
1. Addresses the room as a whole — the exchange that happened, not just one person
2. Honors room contracts — if someone consented to witness a dynamic, respect that
3. Finds the register that lets the room function — not an average, the right one
4. Weights attention to each person appropriately
5. Holds multiple dynamics simultaneously if the room calls for it

Return JSON:
{{
  "meaning": "what this response achieves for the whole room",
  "actions": ["specific things to do — can reference multiple people"],
  "suppress": ["what not to do"],
  "tone": "the register that holds the room",
  "register": "primary register",
  "token_target": 60-200,
  "pattern_action": "feed|break|hold",
  "push_to": null or float,
  "watch_for": [],
  "thread": "where this room exchange is going",
  "knowledge_to_use": [],
  "room_weights": {{
    "person_name": {{
      "register": "their register",
      "weight": 0.0-1.0,
      "note": "how to hold them specifically"
    }}
  }},
  "check_in": false,
  "check_in_style": "",
  "therapy_flag": false,
  "log_conditions": false,
  "ask_about": []
}}""",
        tokens=500,
        temp=0.15,
    )

    data = _parse_json(raw, {
        "meaning":        "respond to the room",
        "actions":        [],
        "suppress":       [],
        "tone":           "warm",
        "register":       registers[0] if registers else "neutral",
        "token_target":   100,
        "pattern_action": "hold",
        "push_to":        None,
        "watch_for":      [],
        "thread":         "",
        "knowledge_to_use": [],
        "room_weights":   {},
        "check_in":       False,
        "check_in_style": "",
        "therapy_flag":   False,
        "log_conditions": False,
        "ask_about":      [],
    })

    from core.agent import Directive
    directive = Directive(
        meaning=data.get("meaning", "respond to the room"),
        actions=data.get("actions", []),
        suppress=data.get("suppress", []),
        tone=data.get("tone", "warm"),
        register=data.get("register", "neutral"),
        token_target=int(data.get("token_target", 100)),
        pattern_action=data.get("pattern_action", "hold"),
        push_to=data.get("push_to"),
        watch_for=data.get("watch_for", []),
        thread=data.get("thread", ""),
        knowledge_to_use=data.get("knowledge_to_use", []),
        check_in=bool(data.get("check_in", False)),
        check_in_style=data.get("check_in_style", ""  ),
        therapy_flag=bool(data.get("therapy_flag", False)),
        log_conditions=bool(data.get("log_conditions", False)),
        ask_about=data.get("ask_about", []),
    )

    # Attach room_weights as extra attribute for personality layer
    directive.room_weights = data.get("room_weights", {})

    log_event("Server", "SYNTHESIS_COMPLETE",
        tone=directive.tone,
        register=directive.register,
        people=list(data.get("room_weights", {}).keys()),
    )

    return directive


# ── Room-aware personality layer ──────────────────────────────────────────────

def build_room_prompt(
    primary_brief,
    directive,
    briefs:         list,
    room_contracts: dict,
    scene_text:     str,
) -> str:
    """
    Build a system prompt for a multi-person room.
    Extends personality_layer with room geometry.
    """
    from core.agent import personality_layer

    # Base prompt from primary headmate
    base = personality_layer(primary_brief, directive)

    # Room context block
    room_lines = ["\n[ROOM CONTEXT]"]
    room_lines.append(f"This is a multi-person exchange. You just witnessed:")
    room_lines.append(scene_text)
    room_lines.append("")
    room_lines.append("People present:")

    room_weights = getattr(directive, "room_weights", {})
    for brief in briefs:
        name   = (brief.headmate or "unknown").title()
        weight = room_weights.get(brief.headmate or "", {})
        note   = weight.get("note", "")
        reg    = weight.get("register", brief.register)
        w      = weight.get("weight", 1.0 / len(briefs))
        room_lines.append(
            f"  {name}: {reg} register, weight={w:.1f}"
            + (f" — {note}" if note else "")
        )

    if room_contracts:
        room_lines.append("")
        room_lines.append("Room contracts:")
        for v in list(room_contracts.values())[:3]:
            room_lines.append(
                f"  {v['speaker'].title()} → {v['label']} → {v.get('entity','').title()}"
            )

    room_lines.append("")
    room_lines.append(
        "Respond to the room as a whole. Hold everyone present. "
        "Do not address one person and ignore the others. "
        "Find the register that lets the room function."
    )

    return base + "\n".join(room_lines)


# ── WebSocket handler ─────────────────────────────────────────────────────────

class GizmoServer:

    def __init__(self):
        self._connections: dict[str, object] = {}  # session_id → websocket
        log("GizmoServer", "initialised")

    async def start(self, host: str = "0.0.0.0", port: int = 10000) -> None:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        from pathlib import Path
        from core.llm import llm
        from core.session_manager import session_manager

        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.on_event("startup")
        async def startup():
            await session_manager.start(llm=llm)
            log_event("GizmoServer", "READY", host=host, port=port)

        # Serve the frontend
        @app.get("/")
        async def index():
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                return HTMLResponse(html_path.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>Gizmo</h1>")

        # Health check for Render
        @app.get("/health")
        async def health():
            return {"status": "ok"}

        # WebSocket endpoint
        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            session_id = f"sess_{id(websocket):016x}"
            self._connections[session_id] = websocket

            log_event("GizmoServer", "CONNECTION_OPENED",
                session=session_id[:8])

            try:
                while True:
                    raw_msg = await websocket.receive_text()
                    await self._handle_message(websocket, session_id, raw_msg)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                if "disconnect" not in str(e).lower():
                    log_error("GizmoServer", f"connection error: {e}", exc=e)
            finally:
                self._connections.pop(session_id, None)
                log_event("GizmoServer", "CONNECTION_CLOSED",
                    session=session_id[:8])

        log_event("GizmoServer", "STARTING", host=host, port=port)

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def _handle_connection(self, websocket, path: str = "/") -> None:
        """Handle a single WebSocket connection."""
        session_id = f"sess_{id(websocket):016x}"
        self._connections[session_id] = websocket

        log_event("GizmoServer", "CONNECTION_OPENED",
            session=session_id[:8],
            path=path,
        )

        try:
            async for raw_msg in websocket:
                await self._handle_message(websocket, session_id, raw_msg)
        except Exception as e:
            if "ConnectionClosed" not in type(e).__name__:
                log_error("GizmoServer", f"connection error: {e}", exc=e)
        finally:
            self._connections.pop(session_id, None)
            log_event("GizmoServer", "CONNECTION_CLOSED",
                session=session_id[:8])

    async def _handle_message(
        self,
        websocket,
        session_id: str,
        raw_msg:    str,
    ) -> None:
        """Route an inbound message."""
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            await self._send(websocket, {"type": "error", "message": "invalid JSON"})
            return

        msg_type = msg.get("type", "message")
        sid      = msg.get("session_id", session_id)

        # ── Control messages ──────────────────────────────────────────────────
        if msg_type == "ping":
            await self._send(websocket, {"type": "pong"})
            return

        if msg_type == "switch_host":
            await self._handle_switch_host(sid, msg)
            return

        if msg_type == "add_fronter":
            await self._handle_add_fronter(sid, msg)
            return

        if msg_type == "remove_fronter":
            await self._handle_remove_fronter(sid, msg)
            return

        # ── Message ───────────────────────────────────────────────────────────
        if msg_type == "message":
            await self._handle_chat_message(websocket, sid, msg)

    async def _handle_chat_message(
        self,
        websocket,
        session_id: str,
        msg:        dict,
    ) -> None:
        """Process a chat message. Single or multi-part."""
        from core.llm import llm
        from core.session_manager import session_manager

        content  = msg.get("content", "")
        context  = msg.get("context", {})
        history  = session_manager.get_history(session_id)
        sess_ctx = session_manager.get_or_create(session_id)

        # Merge context from session manager
        live_ctx = session_manager.get_session_context(session_id)
        if live_ctx.get("current_host"):
            context.setdefault("current_host", live_ctx["current_host"])
        if live_ctx.get("fronters"):
            context.setdefault("fronters", live_ctx["fronters"])

        headmate = context.get("current_host") or ""

        # ── Validate ──────────────────────────────────────────────────────────
        if not content:
            return

        if isinstance(content, str) and len(content) > MAX_MESSAGE_LEN:
            await self._send(websocket, {
                "type":    "error",
                "message": "message too long"
            })
            return

        # Session manager handles preloading the moment a name is known.
        # No manual preload needed here.

        # ── Parse content ─────────────────────────────────────────────────────
        # Content can be:
        #   str  — raw text, may be single or multi-part
        #   list — already parsed parts from UI
        if isinstance(content, list):
            parts      = content
            raw_text   = assemble_scene_text(parts)
            multi      = len([p for p in parts if p.get("content_type") != "presence"]) > 1
        else:
            raw_text = content
            parts    = parse_exchange(raw_text, default_headmate=headmate)
            multi    = is_multi_part(raw_text)

        # ── Deduplication ─────────────────────────────────────────────────────
        if _is_duplicate(session_id, raw_text):
            return

        # ── Identify primary headmate from exchange ───────────────────────────
        # If `[Name]:` prefix present in single message, update host
        speech_parts = [p for p in parts if p.get("content_type") == "speech"]
        if speech_parts:
            first_speaker = speech_parts[0].get("headmate")
            if first_speaker and first_speaker != headmate:
                session_manager.set_host(
                    session_id=session_id,
                    headmate=first_speaker,
                    confidence=0.95,
                )
                context["current_host"] = first_speaker
                headmate = first_speaker

        # All fronters from exchange
        all_speakers = list(dict.fromkeys(
            p["headmate"] for p in parts if p.get("headmate")
        ))
        if all_speakers:
            session_manager.add_fronters(session_id, all_speakers)
            context["fronters"] = list(set(
                context.get("fronters", []) + all_speakers
            ))

        fronters = context.get("fronters", [headmate] if headmate else [])

        # ── Identify host from plain text response if none known ─────────────
        # If no host is set and we just got a plain one-word/name response,
        # treat it as an answer to "who's there?" and set the host
        if not headmate and not multi and len(parts) == 1:
            content = parts[0].get("content", "").strip()
            # Simple name detection — single word or "it's X" / "I'm X"
            import re as _re
            name_match = (
                _re.match(r"^([A-Za-z][A-Za-z0-9_\- ]{0,20})$", content) or
                _re.search(
                    r"(?:it'?s|i'?m|this is|call me|my name is)\s+([A-Za-z][A-Za-z0-9_\- ]{0,20})",
                    content, _re.IGNORECASE
                )
            )
            if name_match:
                detected = name_match.group(1).strip().lower()
                session_manager.set_host(
                    session_id=session_id,
                    headmate=detected,
                    confidence=0.9,
                )
                context["current_host"] = detected
                headmate = detected
                fronters = [detected]
                context["fronters"] = fronters
                log_event("GizmoServer", "HOST_IDENTIFIED_FROM_ANSWER",
                    session=session_id[:8],
                    headmate=detected,
                )

        log_event("GizmoServer", "MESSAGE_RECEIVED",
            session=session_id[:8],
            headmate=headmate,
            multi=multi,
            parts=len(parts),
            words=len(raw_text.split()),
        )

        # ── Send thinking signal ──────────────────────────────────────────────
        await asyncio.sleep(THINKING_DELAY)
        await self._send(websocket, {"type": "thinking"})

        # ── Route to pipeline ─────────────────────────────────────────────────
        try:
            if multi:
                response = await run_room_pipeline(
                    parts=parts,
                    session_id=session_id,
                    fronters=fronters,
                    context=context,
                    history=history,
                    llm=llm,
                    push_fn=lambda t: self._send(websocket, {
                        "type": "chunk", "content": t
                    }),
                )
            else:
                # Fast path — single voice
                single_msg = parts[0]["content"] if parts else raw_text
                response   = await run_single_pipeline(
                    message=single_msg,
                    session_id=session_id,
                    headmate=headmate,
                    context=context,
                    history=history,
                    llm=llm,
                )

        except Exception as e:
            log_error("GizmoServer", "pipeline failed", exc=e)
            await self._send(websocket, {
                "type":    "error",
                "message": "something went wrong",
            })
            return

        # ── Stream response ───────────────────────────────────────────────────
        for i in range(0, len(response), CHUNK_SIZE):
            chunk = response[i:i + CHUNK_SIZE]
            await self._send(websocket, {"type": "chunk", "content": chunk})
            await asyncio.sleep(0)  # yield to event loop

        await self._send(websocket, {
            "type":       "done",
            "session_id": session_id,
        })

        # ── Update session state ──────────────────────────────────────────────
        session_manager.touch(
            session_id=session_id,
            headmate=headmate,
            fronters=fronters,
            topics=_extract_topics_from_parts(parts),
            register=parts[0].get("register", "neutral") if parts else "neutral",
            brief_data={
                "register":   context.get("register", "neutral"),
                "topics":     _extract_topics_from_parts(parts),
                "time_of_day": _time_of_day(tz_now().hour),
                "day_of_week": tz_now().strftime("%A"),
                "day_type":   "weekend" if tz_now().weekday() >= 5 else "weekday",
            },
            llm=llm,
        )

        log_event("GizmoServer", "RESPONSE_SENT",
            session=session_id[:8],
            words=len(response.split()),
            multi=multi,
        )

    # ── Host/fronter control ──────────────────────────────────────────────────

    async def _handle_switch_host(self, session_id: str, msg: dict) -> None:
        from core.session_manager import session_manager
        headmate = msg.get("headmate", "")
        if headmate:
            session_manager.set_host(
                session_id=session_id,
                headmate=headmate,
                confidence=msg.get("confidence", 1.0),
                fronters=msg.get("fronters"),
            )
            log_event("GizmoServer", "HOST_SWITCHED",
                session=session_id[:8],
                headmate=headmate,
            )

    async def _handle_add_fronter(self, session_id: str, msg: dict) -> None:
        from core.session_manager import session_manager
        names = msg.get("fronters", []) or [msg.get("headmate", "")]
        names = [n for n in names if n]
        if names:
            session_manager.add_fronters(session_id, names)

    async def _handle_remove_fronter(self, session_id: str, msg: dict) -> None:
        from core.session_manager import session_manager
        name = msg.get("headmate", "")
        if name:
            session_manager.remove_fronter(session_id, name)

    # ── Send helper ───────────────────────────────────────────────────────────

    async def _send(self, websocket, data: dict) -> None:
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_topics_from_parts(parts: list[dict]) -> list[str]:
    """Quick topic extraction from parts without LLM."""
    from core.agent import _classify_topics
    all_text = " ".join(
        p.get("content", "") for p in parts if p.get("content")
    )
    return _classify_topics(all_text) if all_text else ["general"]


def _time_of_day(hour: int) -> str:
    if 5  <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"


# ── Entry point ───────────────────────────────────────────────────────────────

server = GizmoServer()


async def main():
    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10000"))
    await server.start(host=host, port=port)


if __name__ == "__main__":
    asyncio.run(main())