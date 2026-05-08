"""
tools/interaction_prefs_tool.py
Tool for setting per-headmate interaction preferences.

Gizmo calls this when a headmate explicitly states how they want to be
interacted with. Preferences are written into the headmate's JSON file
under "interaction_prefs" and injected into every conversation with that
headmate via ego.py's _build_system_prompt().

Explicit triggers:
  "when I'm venting, just listen"
  "stop asking me how I'm doing"
  "be more direct with me"
  "I don't want jokes when I'm upset"
  "talk to me like a peer, not a patient"
  "I prefer short answers"
  "when talking to me, always..." → use persona field
  ... any direct statement about how Gizmo should behave with this person.

Fields:
  tone      — how Gizmo should sound (dry/warm/direct/gentle/match my energy)
  pacing    — verbose vs terse, elaboration vs just the answer
  checkins  — whether to proactively ask how they're doing
  humor     — what kind and how much
  distress  — how to respond when this person seems distressed
  persona   — freeform mini-prompt: "When talking to X, do A, say B, never do C..."
               injected as raw direction, not as a labeled field
               use this for anything that doesn't fit the structured fields
               or when they want to write their own instruction block
  explicit  — short verbatim one-liners that accumulate (use for quick rules)
"""

from tools.base_tool import BaseTool, ToolResult


class InteractionPrefsTool(BaseTool):
    @property
    def name(self) -> str:
        return "set_interaction_pref"

    @property
    def description(self) -> str:
        return (
            "Store a preference for how Gizmo should interact with the current headmate. "
            "Call this whenever someone explicitly states how they want to be spoken to, "
            "treated, or engaged with. "
            "These are permanent and read at the start of every conversation with this person. "
            "EXPLICIT TRIGGERS: 'when I'm venting just listen', 'be more direct with me', "
            "'stop asking how I'm doing', 'I prefer short answers', "
            "'don't joke when I'm upset', 'talk to me like a peer', "
            "'when talking to me always...', or any statement about how Gizmo should "
            "behave specifically with them. "
            "Args: "
            "host (str) — the headmate name (current_host). "
            "field (str) — one of: tone, pacing, checkins, humor, distress, persona, explicit. "
            "  persona: use for freeform mini-prompts like 'When talking to X, do A, say B'. "
            "    Injected as raw direction — most powerful field for complex instructions. "
            "    Upserted — writing a new persona replaces the old one. "
            "  explicit: use for short one-liner rules. Accumulates — never overwritten. "
            "  Other fields: use when the preference maps cleanly to that category. "
            "value (str) — the preference, in their words as much as possible."
        )

    async def run(
        self,
        host: str = "",
        field: str = "",
        value: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not host or not field or not value:
            return ToolResult(
                success=False,
                output="Need host, field, and value to store a preference."
            )

        try:
            from core.interaction_prefs import set_pref, FIELD_LABELS, ALL_FIELDS
        except ImportError as e:
            return ToolResult(
                success=False,
                output=f"Interaction prefs module unavailable: {e}"
            )

        field_lower = field.lower().strip()
        if field_lower not in ALL_FIELDS:
            valid = ", ".join(sorted(ALL_FIELDS))
            return ToolResult(
                success=False,
                output=f"Unknown field '{field}'. Valid fields: {valid}."
            )

        try:
            ok = set_pref(host=host, field=field_lower, value=value)
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Failed to store preference: {e}"
            )

        if not ok:
            return ToolResult(
                success=False,
                output=f"No headmate file found for '{host}'. They need to be in the system first."
            )

        is_explicit = field_lower == "explicit"
        is_persona  = field_lower == "persona"

        if is_persona:
            note = "This is their persona block — replaces any previous one."
        elif is_explicit:
            note = "Added to their explicit instructions."
        else:
            note = "Replaces any previous setting for this field."

        return ToolResult(
            success=True,
            output=f"Stored. {host.title()} — {FIELD_LABELS.get(field_lower, field_lower)}: \"{value[:80]}\". {note}",
            data={"host": host, "field": field_lower, "value": value},
        )


class ViewInteractionPrefsTool(BaseTool):
    @property
    def name(self) -> str:
        return "view_interaction_prefs"

    @property
    def description(self) -> str:
        return (
            "View the stored interaction preferences for a headmate. "
            "Call when someone asks what Gizmo knows about how they like to be talked to, "
            "or wants to review or update what's been set. "
            "Args: host (str) — the headmate name."
        )

    async def run(self, host: str = "", session_id: str = "", **kwargs) -> ToolResult:
        if not host:
            return ToolResult(success=False, output="Need a host name.")

        try:
            from core.interaction_prefs import get_prefs, FIELD_LABELS
        except ImportError as e:
            return ToolResult(success=False, output=f"Interaction prefs module unavailable: {e}")

        prefs = get_prefs(host)

        if prefs is None:
            return ToolResult(
                success=False,
                output=f"No headmate file found for '{host}'."
            )

        persona    = prefs.get("persona")
        structured = {f: prefs.get(f) for f in ("tone", "pacing", "checkins", "humor", "distress") if prefs.get(f)}
        explicit   = prefs.get("explicit", [])

        if not persona and not structured and not explicit:
            return ToolResult(
                success=True,
                output=f"No interaction preferences set for {host.title()} yet.",
                data={}
            )

        lines = [f"Interaction preferences for {host.title()}:"]
        if persona:
            lines.append(f"  Persona: {persona}")
        for field, val in structured.items():
            lines.append(f"  {FIELD_LABELS[field]}: {val}")
        if explicit:
            lines.append("  Explicit instructions:")
            for entry in explicit:
                lines.append(f"    - {entry}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data=prefs,
        )