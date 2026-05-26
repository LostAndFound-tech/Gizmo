"""
tools/switch_host.py
Switch the current host / update fronters. Writes to session manager.
"""
from core.log import log_event

class SwitchHostTool:
    name        = "switch_host"
    description = "Switch the current host or update who's fronting."

    async def execute(self, args, session_id, headmate, llm) -> str:
        from core.session_manager import session_manager

        new_host = (
            args.get("headmate") or
            args.get("host") or
            args.get("name", "")
        ).strip().lower()

        fronters = args.get("fronters", [])
        if isinstance(fronters, str):
            fronters = [f.strip() for f in fronters.split(",")]

        if not new_host and not fronters:
            return "no headmate specified"

        if new_host:
            changed = session_manager.set_host(
                session_id=session_id,
                headmate=new_host,
                confidence=1.0,
                fronters=fronters or None,
            )
            log_event("SwitchHostTool", "HOST_SWITCHED",
                session=session_id[:8],
                headmate=new_host,
            )
            return f"switched to {new_host.title()}"

        if fronters:
            session_manager.add_fronters(session_id, fronters)
            return f"fronters updated: {', '.join(f.title() for f in fronters)}"

        return "done"
