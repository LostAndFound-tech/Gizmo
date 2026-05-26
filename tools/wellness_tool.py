"""tools/wellness_tool.py — Log wellbeing observations."""
from core.store import store
from core.log import log_event


class WellnessTool:
    name        = "log_wellness"
    description = "Log a wellbeing observation for a headmate."

    async def execute(self, args, session_id, headmate, llm) -> str:
        about       = args.get("headmate") or headmate or ""
        category    = args.get("category", "pattern")
        observation = args.get("observation") or args.get("text", "")
        context     = args.get("context", "")
        register    = args.get("register", "neutral")

        if not observation:
            return "no observation provided"

        store.write("wellbeing", {
            "headmate":    about.lower() if about else None,
            "category":    category,
            "observation": observation,
            "context":     context,
            "register":    register,
            "session_id":  session_id,
            "source":      "gizmo",
            "confidence":  0.8,
            "tags":        f"wellbeing,{category},{about.lower() if about else 'unknown'}",
        })
        log_event("WellnessTool", "WELLNESS_LOGGED", about=about, category=category)
        return f"wellness observation logged for {about or 'unknown'}"
