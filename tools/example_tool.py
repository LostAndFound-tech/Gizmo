"""
tools/example_tool.py
A stub tool showing the pattern. Replace or extend with your proprietary integrations.
"""

from tools.base_tool import BaseTool, ToolResult


class EchoTool(BaseTool):
    """
    Trivial example tool. Echoes back whatever it receives.
    Replace this with something real (file system, home automation, etc.)
    """

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return (
            "Echoes back a message. Use when the user explicitly asks you to echo something. "
            "Input: message (str). Output: the same message."
        )

    async def run(self, message: str = "", **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"Echo: {message}")
