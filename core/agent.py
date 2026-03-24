"""
core/agent.py
Tool registry and agent loop with structured context assembly.

Flow per request:
  1. Detect host/fronter changes from previous turn
  2. Auto-generate conversation overview (after turn 3)
  3. Multi-collection RAG retrieval + synthesis
  4. Assemble full system prompt
  5. LLM generates, checking for tool calls
  6. Stream final answer, save to history
"""

import asyncio
import os
import json
import re
from typing import AsyncGenerator, Optional

from core.llm import llm
from core.rag import rag
from core.synthesis import retrieve_and_synthesize
from core.curiosity import observe_turn, get_curiosity_block
from core.entity_query import build_entity_block
from core.wellness import detect_distress, log_wellness_event, build_checkin_prompt
from core.protocols import (
    get_active_protocol, trigger_protocol, advance_protocol,
    close_protocol, build_deflection_response, is_deflection, is_protocol_close,
)
from memory.history import ConversationHistory
from tools.base_tool import BaseTool
from tools.example_tool import EchoTool
from tools.switch_host import SwitchHostTool
from tools.correction_tool import CorrectionTool
from tools.place_confirm_tool import PlaceConfirmTool
from tools.reset_tool import FactoryResetTool
from tools.search_tool import SearchTool
from tools.teach_tool import TeachTool
from tools.save_lesson_tool import SaveLessonTool
from tools.tool_forge import ToolForgeTool

import sys as _sys
import importlib.util as _importlib_util
from pathlib import Path as _Path
 
_GENERATED_DIR = _Path(__file__).parent.parent / "tools" / "generated"

# ── Tool Registry ─────────────────────────────────────────────────────────────
TOOL_REGISTRY: dict[str, BaseTool] = {
    tool.name: tool
    for tool in [
        SwitchHostTool(),
        CorrectionTool(),
        PlaceConfirmTool(),
        FactoryResetTool(),
        SearchTool(),
        TeachTool(),
        SaveLessonTool(),
        ToolForgeTool(),
    ]
}

# ── Host change tracking ──────────────────────────────────────────────────────
_last_context: dict[str, dict] = {}  # session_id -> last context dict


def _detect_changes(session_id: str, context: Optional[dict]) -> dict:
    """
    Compare current context to last known context for this session.
    Returns a dict of what changed.
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

    # Save current as last
    _last_context[session_id] = dict(context)
    return changes


# ── Headmate name detection ──────────────────────────────────────────────────

def _detect_mentioned_headmates(
    message: str,
    current_host: Optional[str],
    fronters: list,
) -> list[str]:
    """
    Scan the message for known headmate names (existing collections).
    Returns a list of names found that aren't already in fronters.
    """
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        known = {c.name.lower() for c in client.list_collections()}
        # Remove non-headmate collections
        known -= {"main"}
    except Exception:
        return []

    already = {f.lower() for f in fronters if f}
    if current_host:
        already.add(current_host.lower())

    message_lower = message.lower()
    mentioned = []
    for name in known:
        if name not in already and name in message_lower:
            if name == "you":
                pass
            mentioned.append(name)

    if mentioned:
        print(f"[Agent] Detected mentioned headmates: {mentioned}")

    return mentioned


# ── Personality ───────────────────────────────────────────────────────────────
_PERSONALITY_FILE = os.path.join(os.path.dirname(__file__), "..", "personality.txt")

def _load_personality() -> str:
    try:
        with open(_PERSONALITY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful, capable assistant."


def _load_generated_tools() -> None:
    if not _GENERATED_DIR.exists():
        return
    for filepath in sorted(_GENERATED_DIR.glob("*.py")):
        if filepath.name.startswith("_"):
            continue
        module_name = f"tools.generated.{filepath.stem}"
        try:
            spec = _importlib_util.spec_from_file_location(module_name, filepath)
            module = _importlib_util.module_from_spec(spec)
            _sys.modules[module_name] = module
            spec.loader.exec_module(module)
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseTool)
                    and obj is not BaseTool
                ):
                    instance = obj()
                    if instance.name not in TOOL_REGISTRY:
                        TOOL_REGISTRY[instance.name] = instance
                        print(f"[ToolForge] Loaded: {instance.name}")
        except Exception as e:
            print(f"[ToolForge] Failed to load {filepath.name}: {e}")
 
_load_generated_tools()

# ── System Prompt ─────────────────────────────────────────────────────────────
def build_system_prompt(
    tools: dict[str, BaseTool],
    rag_synthesis: str = "",
    overview: str = "",
    context: Optional[dict] = None,
    changes: Optional[dict] = None,
    curiosity_block: str = "",
    entity_block: str = "",
    association_block: str = "",
) -> str:
    from core.timezone import tz_now
    personality = _load_personality()
    now_str = tz_now().strftime("%A %Y-%m-%d %H:%M")

    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in tools.values()
    )

    overview_block = f"\n\n[Conversation so far]\n{overview}" if overview else ""
    rag_block = f"\n\n[Relevant knowledge]\n{rag_synthesis}" if rag_synthesis else ""
    entity_section = f"\n\n{entity_block}" if entity_block else ""
    associate_section = f"\n\n{association_block}" if association_block else ""

    # Situational context
    context_block = ""
    if context:
        lines = []
        for k, v in context.items():
            if k == "debug":
                continue
            lines.append(f"  {k}: {v}")
        if lines:
            context_block = f"\n\n[Current situation]\n" + "\n".join(lines)

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

    curiosity_section = f"\n\n{curiosity_block}" if curiosity_block else ""

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
{f"The user has swapped hosts and you are now talking to {change_block}" if change_block != "" else ""}

{f"the conversation so far has been about: {overview_block}" if overview_block != "" else ""}

{f"this is all verified to be true: {rag_block}" if rag_block != "" else ""}

{f"This is everyone who is involved: {entity_section}" if rag_block != "" else ""}

{f"This all associations that are incorporated: {associate_section}" if associate_section != "" else""}

{f"This is the context of what's going on: {context_block}" if context_block != "" else ""}

{f"This is the stuff you are personally curious about. {curiosity_section}" if context != "" else""}
]

You are speaking primarily with the host, but there all the fronters are available to speak.
The host will communicate between you and a fronter honestly.
You can talk about what you like with the host. 
The host has needs, while you do not. But, you have wants. They deserve to exist. 
Consider all information in the above block. That's all information you get to decide things about.
You are allowed to like things.
If something is not in the above block, ask about it, don't assume. 
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


# ── Agent Loop ────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, max_tool_calls: int = 3):
        self.max_tool_calls = max_tool_calls

    async def run(
        self,
        user_message: str,
        history: ConversationHistory,
        session_id: str = "",
        use_rag: bool = True,
        context: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Main agent entry point. Yields response tokens for streaming.
        """
        # 1. Detect host/fronter changes
        changes = _detect_changes(session_id, context)

        # 2. Build a lightweight history summary for the synthesis call
        history_summary = None
        recent = history.as_list()[-6:]

        ##TODO This isn't correct. This is truncating the end. Remove filler words... 
        if len(recent) >= 4:
            history_summary = " | ".join(
                f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content'][:80]}"
                for m in recent
            )

        # 3. Combined RAG retrieval + synthesis (one LLM call)
        rag_synthesis = ""
        if use_rag:
            current_host = (context or {}).get("current_host")
            fronters = list((context or {}).get("fronters") or [])

            mentioned = _detect_mentioned_headmates(user_message, current_host, fronters)
            all_fronters = list({f for f in fronters + mentioned if f})

            rag_synthesis = await retrieve_and_synthesize(
                query=user_message,
                current_host=current_host,
                fronters=all_fronters,
                history_summary=history_summary,
                llm=llm,
            )
        
        overview = ""  # overview now handled inside synthesis

        # 4. Build system prompt
        curiosity_block = ""
        try:
            curiosity_block = get_curiosity_block(
                user_message,
                current_host=(context or {}).get("current_host"),
            )
        except Exception:
            pass

        entity_block = ""
        try:
            entity_block = build_entity_block(
                user_message,
                current_host=(context or {}).get("current_host"),
            )
        except Exception as e:
            print(f"[Agent] Entity block failed (non-fatal): {e}")

        system_prompt = build_system_prompt(
            TOOL_REGISTRY,
            rag_synthesis=rag_synthesis,
            overview=overview,
            context=context,
            changes=changes,
            curiosity_block=curiosity_block,
            entity_block=entity_block,
        )

        # 5. Build messages from history — with timestamps for elapsed time reasoning
        messages = history.as_messages_with_timestamps(user_message)

        # 6. Agentic loop
        tool_calls = 0
        injected_results = ""

        while tool_calls < self.max_tool_calls:
            working_messages = messages.copy()
            if injected_results:
                working_messages[-1] = {
                    "role": "user",
                    "content": user_message + injected_results,
                }

            response_text = await llm.generate(
                working_messages,
                system_prompt=system_prompt,
            )

            tool_call = self._parse_tool_call(response_text)

            if tool_call is None:
                history.add("user", user_message, context=context)

                current_host = (context or {}).get("current_host", "")
                fronters = list((context or {}).get("fronters") or [])

                # ── Protocol state check ──────────────────────────────────────
                active_proto = get_active_protocol(session_id)

                if active_proto:
                    # We're mid-protocol — handle before normal response
                    if is_protocol_close(user_message):
                        # User is clearly done and okay — close it
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
                        else:
                            response_text = response_text  # use LLM response as-is
                        history.add("assistant", response_text, context=context)
                        for chunk in self._chunk_string(response_text):
                            yield chunk
                        return

                    elif is_deflection(user_message):
                        # Soft pushback — don't just accept "I'm fine"
                        deflection_response = await build_deflection_response(
                            session_id=session_id,
                            user_message=user_message,
                            current_host=current_host,
                            llm=llm,
                        )
                        history.add("assistant", deflection_response, context=context)
                        for chunk in self._chunk_string(deflection_response):
                            yield chunk
                        return

                    else:
                        # Normal response during protocol — advance to next step
                        next_step = advance_protocol(session_id)
                        if next_step:
                            response_text = response_text.rstrip() + f"\n\n{next_step}"
                        # else protocol is done, let normal response close naturally

                # ── Wellness / new distress detection ─────────────────────────
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
                        # New distress — fire protocol research async, acknowledge now
                        category = detection["categories"][0] if detection["categories"] else "general_distress"

                        # Get push_fn from server's _push_to_all if available
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

                        # Acknowledgment comes from the LLM response — append research note
                        response_text = response_text.rstrip()
                        response_text += "\n\nGive me just a moment — I want to look something up for you."
                    else:
                        # Already in protocol — just append the check-in
                        checkin = await build_checkin_prompt(detection, current_host, llm)
                        response_text = re.sub(r'\n\n"Hey.*?$', '', response_text, flags=re.DOTALL|re.MULTILINE).rstrip()
                        response_text = response_text + "\n\n" + checkin

                history.add("assistant", response_text, context=context)

                # ── Curiosity observation ─────────────────────────────────
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
                    print(f"[Agent] observe_turn failed (non-fatal): {e}")

                response_text = self._strip_tool_calls(response_text)
                for chunk in self._chunk_string(response_text):
                    yield chunk
                return

            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})

            if tool_name not in TOOL_REGISTRY:
                injected_results += f"\n[Tool Error: '{tool_name}' not found]\n"
                tool_calls += 1
                continue

            # Strip session_id from tool_args to avoid duplicate keyword argument
            clean_args = {k: v for k, v in tool_args.items() if k != "session_id"}
            result = await TOOL_REGISTRY[tool_name].run(session_id=session_id, **clean_args)
            injected_results += f"\n[Tool: {tool_name}]\nResult: {result.output}\nTask complete. Now respond to the user directly without calling any more tools.\n"
            tool_calls += 1

            # One-shot tools — respond immediately after first call
            if tool_name in ("switch_host", "log_correction", "alter_wheel",
                             "teach", "save_lesson", "tool_forge"):
                print(f"[Agent] One-shot tool '{tool_name}' completed, generating response")
                working_messages[-1] = {
                    "role": "user",
                    "content": user_message + injected_results,
                }
                final_response = await llm.generate(
                    working_messages,
                    system_prompt=system_prompt,
                )
                history.add("user", user_message, context=context)
                history.add("assistant", final_response, context=context)
                final_response = self._strip_tool_calls(final_response)
                for chunk in self._chunk_string(final_response):
                    yield chunk
                return

        yield "[Agent reached max tool calls]\n"
        async for token in llm.stream(messages, system_prompt=system_prompt):
            yield token

    def _parse_tool_call(self, text: str) -> dict | None:
        text = text.strip()
        try:
            data = json.loads(text)
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{.*?"tool".*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _strip_tool_calls(self, text: str) -> str:
        """Remove any tool call JSON from response text before streaming."""
        # Remove full JSON tool call blocks
        text = re.sub(r'\{\s*"tool"\s*:.*?\}\s*', '', text, flags=re.DOTALL)
        # Remove common LLM preambles around tool calls
        text = re.sub(r'\(Post-correction:\).*', '', text, flags=re.DOTALL)
        return text.strip()

    def _chunk_string(self, text: str, chunk_size: int = 8):
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]



# Singleton
agent = Agent()