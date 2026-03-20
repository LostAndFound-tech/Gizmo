"""
tools/chattiness_tool.py
Lets Gizmo adjust his own talkativeness in response to social signals.

The agent calls this when it detects meta-signals about Gizmo's chattiness —
"shut up", "you're quiet today", "talk more", etc.

The SocialRegulator in gizmo_voice.py handles this automatically for
ambient transcripts. This tool handles it for direct conversation turns,
where the agent is responding to an explicit message.
"""

from tools.base_tool import BaseTool, ToolResult


class ChattinessTool(BaseTool):
    @property
    def name(self) -> str:
        return "set_chattiness"

    @property
    def description(self) -> str:
        return (
            "Adjust how often Gizmo speaks up unprompted. "
            "Use when someone tells Gizmo to be quieter or more talkative, "
            "or when you notice social signals about your talkativeness. "
            "Args: "
            "level (int) — 1 (nearly silent) to 5 (very talkative). "
            "reason (str) — brief note on why you're adjusting."
        )

    async def run(
        self,
        level: int = 3,
        reason: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        try:
            from voice.gizmo_voice import set_chattiness, get_chattiness
            old = get_chattiness()
            set_chattiness(level)
            new = get_chattiness()

            direction = "quieter" if new < old else "more talkative" if new > old else "the same"
            return ToolResult(
                success=True,
                output=f"Chattiness adjusted to {new}/5 ({direction}). {reason}",
                data={"old_level": old, "new_level": new},
            )
        except Exception as e:
            return ToolResult(success=False, output=f"Couldn't adjust chattiness: {e}")
