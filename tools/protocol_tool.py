"""
tools/protocol_tool.py

Gives Gizmo the ability to explicitly create a protocol during conversation.
Auto-creation happens via the NLP detection pass in protocol_manager.py.
This tool is for when Gizmo decides consciously to create one.

Usage (Gizmo calls this as a tool):
  {"tool": "create_protocol", "args": {
    "name": "jess_exclamation_rule",
    "content": "Jess must always use exclamation points when acknowledging instructions.",
    "description": "Jess acknowledgment style rule",
    "tags": ["jess", "communication", "rules"],
    "type": "instruction",
    "headmates": ["jess"]
  }}
"""

from tools.base_tool import BaseTool, ToolResult


class CreateProtocolTool(BaseTool):
    name        = "create_protocol"
    description = (
        "Create a persistent protocol — a rule, boundary, or piece of information "
        "I want to remember and apply in future conversations. "
        "Use this when I've established something that should persist beyond this session."
    )
    args_schema = {
        "name":        {"type": "string", "description": "Short unique name for this protocol"},
        "content":     {"type": "string", "description": "The full protocol text"},
        "description": {"type": "string", "description": "One sentence describing what this covers"},
        "tags":        {"type": "array",  "description": "Relevant tags for context matching", "items": {"type": "string"}},
        "type":        {"type": "string", "description": "instruction, information, or both", "default": "both"},
        "headmates":   {"type": "array",  "description": "Headmates this always applies to (empty = global)", "items": {"type": "string"}, "default": []},
    }

    async def run(
        self,
        session_id: str,
        name: str,
        content: str,
        description: str,
        tags: list = None,
        type: str = "both",
        headmates: list = None,
        **kwargs,
    ) -> ToolResult:
        try:
            from core.protocol_manager import create_protocol
            result = create_protocol(
                name          = name,
                content       = content,
                description   = description,
                tags          = tags or [],
                protocol_type = type,
                headmates     = headmates or [],
            )
            return ToolResult(
                success = result["success"],
                output  = result["message"],
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Protocol creation failed: {e}")
