"""
core/base_agent.py
Abstract base agent — core loop, tool dispatch, registry management.

All specialized agents inherit from this. Override:
  - build_system_prompt() — controls what context the LLM sees
  - post_response_hooks() — runs after LLM response, before streaming
  - should_use_rag()      — whether to run RAG retrieval for this message

The loop itself (tool dispatch, LLM call, streaming) is shared and unchanged.
"""

import asyncio
import json
import os
import re
import sys as _sys
import importlib.util as _importlib_util
from abc import ABC, abstractmethod
from pathlib import Path as _Path
from typing import AsyncGenerator, Optional

from core.llm import llm
from memory.history import ConversationHistory
from tools.base_tool import BaseTool

# ── Generated tools directory ─────────────────────────────────────────────────
_GENERATED_DIR = _Path(__file__).parent.parent / "tools" / "generated"


# ── Tool Registry ─────────────────────────────────────────────────────────────
# Populated at import time. Specialized agents can extend with their own tools.

def build_base_registry() -> dict[str, BaseTool]:
    """Build the base tool registry shared by all agents."""
    from tools.switch_host import SwitchHostTool
    from tools.correction_tool import CorrectionTool
    from tools.place_confirm_tool import PlaceConfirmTool
    from tools.reset_tool import FactoryResetTool
    from tools.search_tool import SearchTool
    from tools.teach_tool import TeachTool
    from tools.save_lesson_tool import SaveLessonTool
    from tools.tool_forge import ToolForgeTool

    tools = [
        SwitchHostTool(),
        CorrectionTool(),
        PlaceConfirmTool(),
        FactoryResetTool(),
        SearchTool(),
        TeachTool(),
        SaveLessonTool(),
        ToolForgeTool(),
    ]
    return {t.name: t for t in tools}


def load_generated_tools(registry: dict[str, BaseTool]) -> None:
    """
    Dynamically load all tools from tools/generated/ into the given registry.
    Called at startup and after tool forge creates a new tool.
    """
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
                    if instance.name not in registry:
                        registry[instance.name] = instance
                        print(f"[ToolForge] Loaded: {instance.name}")
        except Exception as e:
            print(f"[ToolForge] Failed to load {filepath.name}: {e}")


# Shared base registry — all agents start from this
TOOL_REGISTRY: dict[str, BaseTool] = build_base_registry()
load_generated_tools(TOOL_REGISTRY)


# ── One-shot tools ────────────────────────────────────────────────────────────
# These tools respond immediately after a single call rather than looping.
ONE_SHOT_TOOLS = frozenset({
    "switch_host",
    "log_correction",
    "alter_wheel",
    "teach",
    "save_lesson",
    "tool_forge",
})


# ── Base Agent ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base agent. Handles the core agentic loop — tool dispatch,
    LLM calls, and streaming. Subclasses control context assembly and
    post-response behavior.
    """

    def __init__(self, max_tool_calls: int = 3):
        self.max_tool_calls = max_tool_calls
        # Each agent gets its own registry copy so they can add tools
        # without polluting each other
        self.tools: dict[str, BaseTool] = dict(TOOL_REGISTRY)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    async def build_system_prompt(
        self,
        user_message: str,
        history: ConversationHistory,
        session_id: str,
        context: Optional[dict],
    ) -> str:
        """
        Assemble the system prompt for this agent type.
        Called once per turn before the agentic loop.
        """
        ...

    async def post_response_hooks(
        self,
        user_message: str,
        response_text: str,
        session_id: str,
        context: Optional[dict],
        history: ConversationHistory,
    ) -> str:
        """
        Optional post-processing after LLM response, before streaming.
        Return the (possibly modified) response text.
        Override in subclasses to add wellness checks, curiosity, etc.
        Default: pass through unchanged.
        """
        return response_text

    def should_use_rag(self, user_message: str, context: Optional[dict]) -> bool:
        """
        Whether to run RAG retrieval for this message.
        Override to disable RAG in data-capture modes.
        Default: True.
        """
        
        return True

    # ── Core loop ─────────────────────────────────────────────────────────────

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
        # Allow caller to override RAG for this turn
        if not use_rag:
            self._rag_override = False
        else:
            self._rag_override = None  # defer to should_use_rag()

        # Build system prompt — delegated to subclass
        system_prompt = await self.build_system_prompt(
            user_message=user_message,
            history=history,
            session_id=session_id,
            context=context,
        )

        # Build message history
        messages = history.as_messages_with_timestamps(user_message)

        # Agentic loop
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
                # No tool call — run post-response hooks and stream
                history.add("user", user_message, context=context)

                response_text = await self.post_response_hooks(
                    user_message=user_message,
                    response_text=response_text,
                    session_id=session_id,
                    context=context,
                    history=history,
                )

                history.add("assistant", response_text, context=context)
                response_text = self._strip_tool_calls(response_text)
                for chunk in self._chunk_string(response_text):
                    yield chunk
                return

            # Tool call detected
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})

            if tool_name not in self.tools:
                injected_results += f"\n[Tool Error: '{tool_name}' not found]\n"
                tool_calls += 1
                continue

            clean_args = {k: v for k, v in tool_args.items() if k != "session_id"}
            result = await self.tools[tool_name].run(session_id=session_id, **clean_args)
            injected_results += (
                f"\n[Tool: {tool_name}]\nResult: {result.output}\n"
                f"Task complete. Now respond to the user directly without calling any more tools.\n"
            )
            tool_calls += 1

            # One-shot tools respond immediately after a single call
            if tool_name in ONE_SHOT_TOOLS:
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

        # Max tool calls reached — stream directly
        yield "[Agent reached max tool calls]\n"
        async for token in llm.stream(messages, system_prompt=system_prompt):
            yield token

    # ── Utilities ─────────────────────────────────────────────────────────────

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
        text = re.sub(r'\{\s*"tool"\s*:.*?\}\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'\(Post-correction:\).*', '', text, flags=re.DOTALL)
        return text.strip()

    def _chunk_string(self, text: str, chunk_size: int = 8):
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
