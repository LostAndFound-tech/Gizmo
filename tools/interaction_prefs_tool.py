"""tools/interaction_prefs_tool.py — Per-headmate interaction preferences."""
from core.store import store


class InteractionPrefsTool:
    name        = "set_interaction_pref"
    description = "Set an interaction preference for a headmate (tone, format, persona, boundary)."

    async def execute(self, args, session_id, headmate, llm) -> str:
        about     = args.get("headmate") or headmate or ""
        pref_type = args.get("pref_type", "explicit")
        content   = args.get("content") or args.get("text", "")

        if not content:
            return "no preference content provided"

        store.write("interaction_prefs", {
            "headmate":   about.lower() if about else None,
            "pref_type":  pref_type,
            "content":    content,
            "session_id": session_id,
            "source":     "user",
            "tags":       f"interaction_pref,{pref_type},{about.lower() if about else 'global'}",
        })
        return f"interaction preference saved for {about or 'global'}"


class ViewInteractionPrefsTool:
    name        = "view_interaction_prefs"
    description = "View interaction preferences for a headmate."

    async def execute(self, args, session_id, headmate, llm) -> str:
        about = args.get("headmate") or headmate or ""
        prefs = store.get_active("interaction_prefs",
            headmate=about.lower() if about else None,
            limit=20,
        )
        if not prefs:
            return f"no interaction preferences for {about or 'unknown'}"
        lines = [f"- [{p.get('pref_type','?')}] {p.get('content','')}" for p in prefs]
        return "\n".join(lines)
