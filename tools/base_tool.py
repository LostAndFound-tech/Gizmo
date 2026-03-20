"""
tools/base_tool.py
Abstract base for all agent tools.

To add a new tool:
1. Subclass BaseTool
2. Implement `name`, `description`, and `run()`
3. Register it in core/agent.py's TOOL_REGISTRY

That's it. No other changes needed.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any


class ToolResult(BaseModel):
    success: bool
    output: str
    data: Any = None  # optional structured data for downstream use


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier. Used by the agent to call it."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Plain-language description of what the tool does and when to use it.
        The LLM reads this to decide whether to invoke the tool.
        Be specific about inputs and outputs.
        """
        ...

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Execute the tool. Kwargs are parsed from the LLM's tool call."""
        ...

    def schema(self) -> dict:
        """Returns tool schema for injection into the system prompt."""
        return {
            "name": self.name,
            "description": self.description,
        }
