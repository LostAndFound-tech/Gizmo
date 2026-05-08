"""
tools/interaction_prefs_tool.py
Tool for setting per-host interaction preferences.

Gizmo calls this when a headmate explicitly tells him how they want to be
interacted with. These preferences are stored verbatim and injected into
every conversation with that host — they are never inferred, never softened,
never synthesized away.

Explicit triggers:
  "when I'm venting, just listen"
  "stop asking me how I'm doing"
  "be more direct with me"
  "I don't want jokes when I'm upset"
  "talk to me like a peer, not a patient"
  "I prefer short answers"
  ... any direct statement about how Gizmo should behave with this person.

Fields:
  tone        — how Gizmo should sound (dry/warm/direct/gentle/match my energy/etc.)
  pacing      — verbose vs terse, elaboration vs just the answer
  checkins    — whether to proactively ask how they're doing
  humor       — what kind and how much
  distress    — how to respond when this person seems distressed
  explicit    — freeform verbatim statement (accumulates, never overwritten)
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
            "treated, or engaged with — tone, pacing, humor, check-ins, how to handle distress, "
            "or any direct instruction about Gizmo's behavior with them specifically. "
            "These are permanent and injected into every conversation with this person. "
            "EXPLICIT TRIGGERS: 'when I'm venting just listen', 'be more direct', "
            "'stop asking how I'm doing', 'I prefer short answers', 'don't joke when I'm upset', "
            "'talk to me like a peer', or any statement about how Gizmo should behave with them. "
            "Args: "
            "host (str) — the headmate name. "
            "field (str) — one of: tone, pacing, checkins, humor, distress, explicit. "
            "  Use 'explicit' for verbatim instructions that don't fit a specific field. "
            "  Use the specific field when the preference maps cleanly to one. "
            "value (str) — the preference, in their words as much as possible. "
            "set_by (str) — who is setting this (usually the same as host)."
        )

    async def run(
        self,
        host: str = "",
        field: str = "",
        value: str = "",
        set_by: str = "",
        **kwargs,
    ) -> ToolResult:
        if not host or not field or not value:
            return ToolResult(
                success=False,
                output="Need host, field, and value to store a preference."
            )

        try:
            from core.interaction_prefs import set_pref, FIELD_LABELS, ALL_FIELDS
        except ImportError:
            return ToolResult(
                success=False,
                output="Interaction prefs module not available."
            )

        field_lower = field.lower().strip()
        if field_lower not in ALL_FIELDS:
            valid = ", ".join(sorted(ALL_FIELDS))
            return ToolResult(
                success=False,
                output=f"Unknown field '{field}'. Valid fields: {valid}."
            )

        try:
            pref_id = set_pref(
                host=host,
                field=field_lower,
                value=value,
                set_by=set_by or host,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Failed to store preference: {e}"
            )

        label = FIELD_LABELS.get(field_lower, field_lower)
        is_explicit = field_lower == "explicit"

        summary = (
            f"Got it. Stored {host.title()}'s preference — "
            f"{label}: \"{value}\". "
            f"{'This will stack with any other explicit instructions.' if is_explicit else 'This replaces any previous setting for this field.'}"
        )

        return ToolResult(
            success=True,
            output=summary,
            data={
                "host": host,
                "field": field_lower,
                "value": value,
                "pref_id": pref_id,
            }
        )


class ViewInteractionPrefsTool(BaseTool):
    """
    Secondary tool — lets Gizmo or a headmate review what's currently set.
    Useful when someone asks 'what do you know about how I like to be talked to?'
    """
    @property
    def name(self) -> str:
        return "view_interaction_prefs"

    @property
    def description(self) -> str:
        return (
            "View the stored interaction preferences for a headmate. "
            "Call when someone asks what Gizmo knows about their preferences, "
            "or wants to review/update what's set. "
            "Args: host (str) — the headmate name."
        )

    async def run(self, host: str = "", **kwargs) -> ToolResult:
        if not host:
            return ToolResult(success=False, output="Need a host name.")

        try:
            from core.interaction_prefs import get_prefs, FIELD_LABELS
        except ImportError:
            return ToolResult(success=False, output="Interaction prefs module not available.")

        prefs = get_prefs(host)
        if not prefs:
            return ToolResult(
                success=True,
                output=f"No interaction preferences stored for {host.title()} yet.",
                data={}
            )

        lines = [f"Interaction preferences for {host.title()}:"]
        for field in ("tone", "pacing", "checkins", "humor", "distress"):
            if field in prefs:
                lines.append(f"  {FIELD_LABELS[field]}: {prefs[field]}")

        explicit = prefs.get("explicit", [])
        if explicit:
            lines.append("  Explicit instructions:")
            for entry in explicit:
                lines.append(f"    [{entry['id']}] {entry['value']}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data=prefs
        )
