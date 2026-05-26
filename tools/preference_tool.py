"""
tools/preference_tool.py
Soft preferences — context-weighted, Gizmo can override.
Follow-up question queue — shameless, non-intrusive.
"""
import json
from core.store import store
from core.log import log_event


class SetPreferenceTool:
    name        = "set_preference"
    description = "Save a soft preference for a headmate. Context-aware. Gizmo can override."

    async def execute(self, args, session_id, headmate, llm) -> str:
        pref      = args.get("preference") or args.get("text", "")
        about     = args.get("headmate") or headmate or ""
        valence   = args.get("valence", "positive")
        contexts  = args.get("default_context", [])
        avoid     = args.get("avoid_context", [])
        override  = args.get("gizmo_override", True)
        note      = args.get("override_note", "")

        if not pref:
            return "no preference provided"

        store.add_preference(
            headmate=about,
            preference=pref,
            valence=valence,
            default_context=contexts if isinstance(contexts, list) else [contexts],
            avoid_context=avoid if isinstance(avoid, list) else [avoid],
            gizmo_override=override,
            override_note=note,
            session_id=session_id,
        )
        log_event("SetPreferenceTool", "PREF_SAVED",
            headmate=about, pref=pref[:60])
        return f"preference saved for {about or 'unknown'}: \"{pref[:80]}\""


class ViewPreferenceTool:
    name        = "view_preferences"
    description = "View soft preferences for a headmate."

    async def execute(self, args, session_id, headmate, llm) -> str:
        about = args.get("headmate") or headmate or ""
        prefs = store.get_preferences(about) if about else []
        if not prefs:
            return f"no preferences on file for {about or 'unknown'}"
        lines = []
        for p in prefs:
            ctx = ", ".join(p.get("default_context") or []) or "any"
            lines.append(
                f"- [{p.get('valence','?')}] {p['preference']} "
                f"(context: {ctx}, "
                f"landed: {p.get('times_landed',0)}/{p.get('times_used',0)})"
            )
        return "\n".join(lines)


class QueueQuestionTool:
    name        = "queue_question"
    description = "Queue a follow-up question to ask a headmate when the moment is right."

    async def execute(self, args, session_id, headmate, llm) -> str:
        question = args.get("question") or args.get("text", "")
        about    = args.get("headmate") or headmate or ""
        gap      = args.get("gap_identified") or args.get("reason", "")
        context  = args.get("context") or args.get("timing", "")

        if not question:
            return "no question provided"

        store.queue_question(
            headmate=about,
            question=question,
            gap_identified=gap,
            context=context,
            session_id=session_id,
        )
        log_event("QueueQuestionTool", "QUESTION_QUEUED",
            headmate=about, question=question[:60])
        return f"question queued for {about or 'unknown'}: \"{question[:80]}\""
