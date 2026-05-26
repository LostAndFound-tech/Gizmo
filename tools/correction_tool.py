"""
tools/correction_tool.py
Hard behavioral rules. Writes to store.corrections. Never wiped.
"""
from core.store import store
from core.log import log_event

class CorrectionTool:
    name        = "log_correction"
    description = "Save a hard behavioral rule Gizmo must always follow. Never overridden."

    async def execute(self, args, session_id, headmate, llm) -> str:
        rule = args.get("rule") or args.get("correction") or args.get("text", "")
        if not rule:
            return "no rule provided"
        store.add_correction(
            rule=rule,
            headmate=headmate,
            who_corrected=headmate,
            session_id=session_id,
        )
        log_event("CorrectionTool", "CORRECTION_SAVED", rule=rule[:60])
        return f"correction saved: \"{rule[:80]}\""
