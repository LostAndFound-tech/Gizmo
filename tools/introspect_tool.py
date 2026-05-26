"""tools/introspect_tool.py — Gizmo reflects on himself."""
from core.store import store


class IntrospectTool:
    name        = "introspect"
    description = "Gizmo reflects on himself — writes a thought to reflections."

    async def execute(self, args, session_id, headmate, llm) -> str:
        text    = args.get("text") or args.get("thought", "")
        topic   = args.get("topic", "")
        valence = float(args.get("valence", 0.0))

        if not text:
            return "nothing to reflect on"

        store.write("reflections", {
            "text":       text,
            "topic":      topic,
            "valence":    valence,
            "headmate":   headmate.lower() if headmate else None,
            "session_id": session_id,
            "source":     "gizmo",
            "outcome":    "pending",
            "tags":       f"reflection,{topic},{headmate.lower() if headmate else 'gizmo'}",
        })
        return "reflection stored"
