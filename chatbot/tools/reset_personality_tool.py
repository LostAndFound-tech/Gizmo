"""
tools/reset_personality_tool.py
Factory reset tool. Wipes Gizmo's personality and starts onboarding over.

This is a real consequence. Corrections are never wiped.
Gizmo earns everything back through conversation.

The tool requires explicit double confirmation before firing —
the agent must see unambiguous intent before calling this.
"""

from tools.base_tool import BaseTool, ToolResult


class ResetPersonalityTool(BaseTool):
    @property
    def name(self) -> str:
        return "reset_personality"

    @property
    def description(self) -> str:
        return (
            "Factory reset Gizmo's personality. Wipes everything learned — "
            "voice, tone, values, interests, context, observations. "
            "Corrections are NEVER wiped. Onboarding runs again on the next turn. "
            "EXPLICIT TRIGGERS ONLY: 'reset your personality', 'factory reset', "
            "'start over from scratch', 'forget everything about yourself', "
            "'wipe your personality', 'reset yourself'. "
            "Do NOT call for general frustration, minor complaints, or vague requests. "
            "This is irreversible. Confirm intent before calling. "
            "Args: confirmed_by (str) — who triggered the reset."
        )

    async def run(
        self,
        confirmed_by: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        try:
            from core.personality_growth import reset_personality
            success = await reset_personality(confirmed_by=confirmed_by or "unknown")
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Reset failed: {e}",
            )

        if not success:
            return ToolResult(
                success=False,
                output=(
                    "Reset failed — check logs. Personality collections may be partially wiped. "
                    "Corrections are intact."
                ),
            )

        # Force the current session back into onboarding.
        # Without this, the agent's onboarding gate won't re-trigger
        # mid-session even though personality_core is now empty.
        if session_id:
            from core.agent import _onboarding_sessions
            # Remove any stale onboarding state first, then mark as needing onboarding
            _onboarding_sessions.pop(session_id, None)
            _onboarding_sessions[session_id] = []   # empty list = onboarding not yet started

        from core.personality_growth import run_onboarding
        from core.llm import llm as _llm
        opening = await run_onboarding(_llm)

        return ToolResult(
            success=True,
            output=(
                f"Personality reset complete. Everything learned is gone. "
                f"Corrections are preserved. "
                f"Onboarding is now active for this session. "
                f"Respond to the user with EXACTLY this, nothing else:\n\n{opening}"
            ),
            data={
                "confirmed_by": confirmed_by,
                "session_id": session_id,
                "corrections_preserved": True,
                "opening_question": opening,
            }
        )