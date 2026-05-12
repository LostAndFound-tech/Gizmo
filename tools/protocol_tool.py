"""
tools/protocol_tool.py

Gives Gizmo the ability to explicitly create a protocol during conversation.
Returns immediately with a holding response, does the work async,
then pushes the result back to the client when done.
"""

import asyncio
from tools.base_tool import BaseTool, ToolResult


class CreateProtocolTool(BaseTool):
    name        = "create_protocol"
    description = (
    "Create a persistent behavioral protocol — a rule, boundary, or standing instruction "
    "that should change how I behave in future conversations. "
    "Use this for rules, limits, and patterns worth keeping permanently. "
    "NOT for drafts, notes, working documents, or things we're building together."
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
        # Fire the actual work async — never blocks the response
        asyncio.ensure_future(
            _write_protocol_and_push(
                session_id    = session_id,
                name          = name,
                content       = content,
                description   = description,
                tags          = tags or [],
                protocol_type = type,
                headmates     = headmates or [],
            )
        )

        # Return immediately so response isn't held
        return ToolResult(
            success = True,
            output  = f"On it — writing '{name}' now.",
        )


async def _write_protocol_and_push(
    session_id: str,
    name: str,
    content: str,
    description: str,
    tags: list,
    protocol_type: str,
    headmates: list,
) -> None:
    """
    Does the actual file write async, then pushes confirmation back to the client.
    """
    try:
        from core.protocol_manager import create_protocol
        result = create_protocol(
            name          = name,
            content       = content,
            description   = description,
            tags          = tags,
            protocol_type = protocol_type,
            headmates     = headmates,
        )

        if result["success"]:
            headmate_note = ""
            if headmates:
                names = " or ".join(h.title() for h in headmates)
                headmate_note = f" I'll load it automatically whenever {names} is here."

            push_message = (
                f"Done — wrote '{name}'.{headmate_note}\n\n"
                f"Here's what I saved:\n\n{content}"
            )
        else:
            push_message = f"Couldn't write '{name}' — {result['message']}"

        from core.agent import push
        await push(push_message)

    except Exception as e:
        try:
            from core.agent import push
            await push(f"Something went wrong writing '{name}' — {e}")
        except Exception:
            print(f"[ProtocolTool] push failed after write error: {e}")