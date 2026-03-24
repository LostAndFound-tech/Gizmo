"""
core/conversation_agent.py
The conversational agent — Gizmo's primary face.

Inherits the core loop from BaseAgent and adds:
  - Per-host personality loading
  - Full RAG retrieval + synthesis
  - Host/fronter change detection
  - Wellness detection + protocol management
  - Curiosity observation
  - Entity and association blocks
  - Topic tracking (active topics decay when not referenced)
  - Cold start detection per host → "Hey, tell me about yourself"

This is the agent that does the talking. Everything else is infrastructure.
"""

import asyncio
import os
import re
from datetime import datetime, timedelta
from typing import Optional

from core.base_agent import BaseAgent, TOOL_REGISTRY, load_generated_tools
from core.llm import llm
from core.rag import rag
from core.synthesis import retrieve_and_synthesize
from core.wellness import detect_distress, log_wellness_event, build_checkin_prompt
from core.protocols import (
    get_active_protocol, trigger_protocol, advance_protocol,
    close_protocol, build_deflection_response, is_deflection, is_protocol_close,
)
from memory.history import ConversationHistory

# Optional modules — non-fatal if missing
try:
    from core.curiosity import observe_turn, get_curiosity_block
    _HAS_CURIOSITY = True
except ImportError:
    _HAS_CURIOSITY = False

try:
    from core.entity_query import build_entity_block
    _HAS_ENTITY = True
except ImportError:
    _HAS_ENTITY = False


# ── Personality ───────────────────────────────────────────────────────────────

_PERSONALITY_FILE = os.path.join(os.path.dirname(__file__), "..", "personality.txt")
_DEFAULT_PERSONALITY = "You are a helpful, capable assistant."


def _load_global_personality() -> str:
    try:
        with open(_PERSONALITY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return _DEFAULT_PERSONALITY


def _load_host_personality(host: str) -> str:
    """
    Load per-host interaction style from RAG if available.
    Falls back to global personality.
    """
    if not host:
        return _load_global_personality()

    try:
        from core.rag import RAGStore
        store = RAGStore(collection_name=f"interaction_style_{host.lower()}")
        if store.count == 0:
            return _load_global_personality()

        results = store.collection.get()
        if results["documents"]:
            # Most recent interaction style synthesis
            return results["documents"][-1]
    except Exception as e:
        print(f"[ConversationAgent] Host personality load failed for {host}: {e}")

    return _load_global_personality()


def _is_cold_start_for_host(host: str) -> bool:
    """True if we have no stored data for this host."""
    if not host:
        return False
    try:
        from core.rag import RAGStore
        store = RAGStore(collection_name=host.lower())
        return store.count == 0
    except Exception:
        return False


# ── Host change tracking ──────────────────────────────────────────────────────

_last_context: dict[str, dict] = {}  # session_id -> last context dict


def _detect_changes(session_id: str, context: Optional[dict]) -> dict:
    """
    Compare current context to last known for this session.
    Returns dict of what changed.
    """
    if not context or not session_id:
        return {}

    last = _last_context.get(session_id, {})
    changes = {}

    current_host = context.get("current_host", "")
    last_host = last.get("current_host", "")
    if current_host and current_host != last_host and last_host:
        changes["host_changed"] = True
        changes["previous_host"] = last_host

    current_fronters = set(
        f.lower() if isinstance(f, str) else str(f).lower()
        for f in (context.get("fronters") or [])
    )
    last_fronters = set(
        f.lower() if isinstance(f, str) else str(f).lower()
        for f in (last.get("fronters") or [])
    )
    joined = current_fronters - last_fronters
    left = last_fronters - current_fronters
    if joined:
        changes["fronters_joined"] = list(joined)
    if left:
        changes["fronters_left"] = list(left)

    _last_context[session_id] = dict(context)
    return changes


# ── Headmate name detection ───────────────────────────────────────────────────

def _detect_mentioned_headmates(
    message: str,
    current_host: Optional[str],
    fronters: list,
) -> list[str]:
    """Scan message for known headmate names not already in fronters."""
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        known = {c.name.lower() for c in client.list_collections()}
        known -= {"main"}
    except Exception:
        return []

    already = {f.lower() for f in fronters if f}
    if current_host:
        already.add(current_host.lower())

    message_lower = message.lower()
    mentioned = [name for name in known if name not in already and name in message_lower]

    if mentioned:
        print(f"[ConversationAgent] Detected mentioned headmates: {mentioned}")

    return mentioned


# ── Topic tracking ────────────────────────────────────────────────────────────
# Active topics decay when not mentioned. Keeps system prompt focused
# on what's actually being discussed right now.

_TOPIC_TTL_MINUTES = 15
_active_topics: dict[str, dict[str, datetime]] = {}  # session_id -> {topic: last_seen}


def _update_topics(session_id: str, topics: list[str]) -> None:
    """Refresh timestamps for mentioned topics, add new ones."""
    now = datetime.now()
    if session_id not in _active_topics:
        _active_topics[session_id] = {}
    for topic in topics:
        _active_topics[session_id][topic] = now


def _get_active_topics(session_id: str) -> list[str]:
    """Return topics seen within the TTL window, prune expired ones."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=_TOPIC_TTL_MINUTES)
    session_topics = _active_topics.get(session_id, {})
    active = [t for t, last in session_topics.items() if last > cutoff]
    # Prune expired
    _active_topics[session_id] = {t: ts for t, ts in session_topics.items() if ts > cutoff}
    return active


async def _extract_topics(message: str, response: str) -> list[str]:
    """
    Use LLM to extract 1-3 topic tags from the current exchange.
    Fast, cheap call — small output, low temperature.
    """
    try:
        result = await llm.generate(
            messages=[{
                "role": "user",
                "content": (
                    f"User said: {message[:200]}\n"
                    f"Response: {response[:200]}\n\n"
                    "List 1-3 short topic tags (2-4 words each) for this exchange. "
                    "JSON array only, no explanation. Example: [\"elbow injury\", \"work schedule\"]"
                )
            }],
            system_prompt="Extract topic tags. JSON array only. No markdown.",
            max_new_tokens=60,
            temperature=0.1,
        )
        result = result.strip().strip("```json").strip("```").strip()
        import json
        return json.loads(result)
    except Exception:
        return []


# ── System Prompt ─────────────────────────────────────────────────────────────

def _build_conversational_prompt(
    personality: str,
    tools: dict,
    rag_synthesis: str = "",
    context: Optional[dict] = None,
    changes: Optional[dict] = None,
    curiosity_block: str = "",
    entity_block: str = "",
    association_block: str = "",
    active_topics: list[str] = None,
    cold_start_host: Optional[str] = None,
) -> str:
    from core.timezone import tz_now
    now_str = tz_now().strftime("%A %Y-%m-%d %H:%M")

    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in tools.values()
    )

    rag_block = f"\n\n[Relevant knowledge]\n{rag_synthesis}" if rag_synthesis else ""
    entity_section = f"\n\n{entity_block}" if entity_block else ""
    associate_section = f"\n\n{association_block}" if association_block else ""
    curiosity_section = f"\n\n{curiosity_block}" if curiosity_block else ""

    # Context block
    context_block = ""
    if context:
        lines = [f"  {k}: {v}" for k, v in context.items() if k != "debug"]
        if lines:
            context_block = "\n\n[Current situation]\n" + "\n".join(lines)

    # Host/fronter change signals
    change_block = ""
    if changes:
        change_lines = []
        if changes.get("host_changed"):
            change_lines.append(
                f"  The host just changed from {changes['previous_host']} "
                f"to {context.get('current_host', 'someone new')}. "
                f"You are now speaking with a different person — acknowledge this naturally."
            )
        if changes.get("fronters_joined"):
            change_lines.append(f"  Joined the front: {', '.join(changes['fronters_joined'])}")
        if changes.get("fronters_left"):
            change_lines.append(f"  Left the front: {', '.join(changes['fronters_left'])}")
        if change_lines:
            change_block = "\n\n[System changes]\n" + "\n".join(change_lines)

    # Active topics block
    topics_block = ""
    if active_topics:
        topics_block = f"\n\n[Active topics]\n  " + ", ".join(active_topics)

    # Cold start — we don't know this host yet
    cold_start_block = ""
    if cold_start_host:
        cold_start_block = (
            f"\n\n[New host]\n"
            f"  You have not spoken with {cold_start_host} before. "
            f"Greet them warmly and ask them to tell you about themselves. "
            f"Be genuinely curious — this first conversation seeds everything you'll know about them."
        )

    return f"""{personality}

Current time: {now_str}
Message history includes [HH:MM] timestamps — use these to reason about elapsed time.

Available tools:
{tool_descriptions}

To use a tool, respond with ONLY this JSON format (no extra text):
{{"tool": "tool_name", "args": {{"arg1": "value1"}}}}

After receiving a tool result, continue reasoning and provide a final response.
If no tool is needed, respond directly.
[
{f"The host has swapped: {change_block}" if change_block else ""}
{f"Verified knowledge: {rag_block}" if rag_block else ""}
{f"Entities involved: {entity_section}" if entity_section else ""}
{f"Associations: {associate_section}" if associate_section else ""}
{f"Current context: {context_block}" if context_block else ""}
{f"What you're curious about: {curiosity_section}" if curiosity_section else ""}
{f"Active topics: {topics_block}" if topics_block else ""}
{cold_start_block}
]

You are speaking primarily with the host, but all fronters are available to speak.
The host will communicate between you and a fronter honestly.
You can talk about what you like with the host.
The host has needs, while you do not. But you have wants. They deserve to exist.
Consider all information in the above block. That is all information you get to decide things about.
You are allowed to like things.
If something is not in the above block, ask about it — don't assume.
If you're curious about something, ask about it.
Check your personality to figure out how to say it in the most human way possible.

TEACHING SESSIONS:
Use the teach tool when someone wants to work on, improve, or build a tool together.
Stay in the lesson until they explicitly say to save or stop — do not exit mid-flow.
While a lesson is active, map natural language to teach actions:
  'try it' / 'give it a go' / 'show me'           → action='try'
  any reaction to output (critique, feedback)       → action='critique', answer=<their words>
  'that's it' / 'perfect' / 'settle this'          → action='settle'
  any answer to a lesson question                   → action='answer', answer=<their words>
  'what tools do you have' / 'show me your tools'  → action='list'
  'resume' / 'pick up where we left off'           → action='resume'
  'save this' / "we're done" / 'lock it in'        → call save_lesson, is_wip=False
  'save as WIP' / "we'll come back" / 'save for now' → call save_lesson, is_wip=True
Do not call save_lesson unless the user explicitly asks to save.
"""


# ── Conversation Agent ────────────────────────────────────────────────────────

class ConversationAgent(BaseAgent):
    """
    Gizmo's primary conversational face.

    Adds per-host personality, RAG synthesis, wellness/protocol handling,
    curiosity observation, topic tracking, and cold start detection on top
    of the shared base loop.
    """

    async def build_system_prompt(
        self,
        user_message: str,
        history: ConversationHistory,
        session_id: str,
        context: Optional[dict],
    ) -> str:
        current_host = (context or {}).get("current_host")
        fronters = list((context or {}).get("fronters") or [])

        # Detect host changes
        changes = _detect_changes(session_id, context)

        # Per-host personality
        personality = _load_host_personality(current_host)

        # Cold start check
        cold_start_host = None
        if current_host and changes.get("host_changed") and _is_cold_start_for_host(current_host):
            cold_start_host = current_host

        # RAG retrieval
        rag_synthesis = ""
        if self.should_use_rag(user_message, context):
            mentioned = _detect_mentioned_headmates(user_message, current_host, fronters)
            all_fronters = list({f for f in fronters + mentioned if f})

            # Build lightweight history summary for synthesis
            # Fixed: take first N chars of recent messages, not truncating the tail
            recent = history.as_list()[-6:]
            history_summary = None
            if len(recent) >= 4:
                history_summary = " | ".join(
                    f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content'][:80]}"
                    for m in recent
                )

            rag_synthesis = await retrieve_and_synthesize(
                query=user_message,
                current_host=current_host,
                fronters=all_fronters,
                history_summary=history_summary,
                llm=llm,
            )

        # Curiosity block
        curiosity_block = ""
        if _HAS_CURIOSITY:
            try:
                curiosity_block = get_curiosity_block(user_message, current_host=current_host)
            except Exception:
                pass

        # Entity block
        entity_block = ""
        if _HAS_ENTITY:
            try:
                entity_block = build_entity_block(user_message, current_host=current_host)
            except Exception as e:
                print(f"[ConversationAgent] Entity block failed (non-fatal): {e}")

        # Active topics
        active_topics = _get_active_topics(session_id)

        return _build_conversational_prompt(
            personality=personality,
            tools=self.tools,
            rag_synthesis=rag_synthesis,
            context=context,
            changes=changes,
            curiosity_block=curiosity_block,
            entity_block=entity_block,
            active_topics=active_topics,
            cold_start_host=cold_start_host,
        )

    async def post_response_hooks(
        self,
        user_message: str,
        response_text: str,
        session_id: str,
        context: Optional[dict],
        history: ConversationHistory,
    ) -> str:
        current_host = (context or {}).get("current_host", "")
        fronters = list((context or {}).get("fronters") or [])

        # ── Protocol state check ──────────────────────────────────────────────
        active_proto = get_active_protocol(session_id)

        if active_proto:
            if is_protocol_close(user_message):
                close_info = close_protocol(
                    session_id,
                    closed_by=current_host,
                    original_fronter=active_proto["fronter"],
                )
                if close_info["different_fronter"]:
                    response_text = (
                        f"Got it, {current_host} — I'll note that "
                        f"{close_info['original_fronter']} seemed to be doing better "
                        f"when you took over. Take care of each other. 💙"
                    )
                return response_text

            elif is_deflection(user_message):
                response_text = await build_deflection_response(
                    session_id=session_id,
                    user_message=user_message,
                    current_host=current_host,
                    llm=llm,
                )
                return response_text

            else:
                next_step = advance_protocol(session_id)
                if next_step:
                    response_text = response_text.rstrip() + f"\n\n{next_step}"

        # ── Wellness / distress detection ─────────────────────────────────────
        detection = detect_distress(user_message)
        print(f"[Wellness] Scanned — detected: {detection['detected']}, categories: {detection['categories']}")

        if detection["detected"]:
            await log_wellness_event(
                message=user_message,
                detection=detection,
                current_host=current_host,
                fronters=fronters,
                session_id=session_id,
            )

            if not active_proto:
                category = detection["categories"][0] if detection["categories"] else "general_distress"

                try:
                    from core.push import _push_to_all as push_fn
                except Exception:
                    async def push_fn(msg): print(f"[Protocols] Push: {msg}")

                asyncio.ensure_future(
                    trigger_protocol(
                        session_id=session_id,
                        category=category,
                        fronter=current_host,
                        llm=llm,
                        push_fn=push_fn,
                    )
                )

                response_text = response_text.rstrip()
                response_text += "\n\nGive me just a moment — I want to look something up for you."
            else:
                checkin = await build_checkin_prompt(detection, current_host, llm)
                response_text = re.sub(
                    r'\n\n"Hey.*?$', '', response_text,
                    flags=re.DOTALL | re.MULTILINE
                ).rstrip()
                response_text = response_text + "\n\n" + checkin

        # ── Curiosity observation ─────────────────────────────────────────────
        if _HAS_CURIOSITY:
            try:
                followup = await observe_turn(
                    user_message=user_message,
                    gizmo_response=response_text,
                    current_host=current_host or None,
                    session_id=session_id,
                    llm=llm,
                )
                if followup:
                    response_text = response_text.rstrip() + "\n\n" + followup
            except Exception as e:
                print(f"[ConversationAgent] observe_turn failed (non-fatal): {e}")

        # ── Topic tracking ────────────────────────────────────────────────────
        try:
            topics = await _extract_topics(user_message, response_text)
            if topics:
                _update_topics(session_id, topics)
        except Exception:
            pass

        return response_text

    def should_use_rag(self, user_message: str, context: Optional[dict]) -> bool:
        return True


# ── Singleton ─────────────────────────────────────────────────────────────────
agent = ConversationAgent()
