"""
tools/place_confirm_tool.py
Stores a confirmed place after Gizmo asks about it.

Triggered when the user answers a place clarification question.
The agent recognizes confirmation language and calls this tool
with the place name, type, and any description extracted from the answer.

place_type options:
  "headspace"  — inner world space belonging to a headmate
  "local"      — real-world local place (store, park, school, etc.)
  "online"     — digital space (website, server, platform)
  "other"      — anything else

Examples of triggers:
  Gizmo: "What's dairy mart?"
  User:  "It's a convenience store near us"
  → place_type: "local", description: "convenience store nearby"

  Gizmo: "Is your office your headspace?"
  User:  "Yeah, it's where I go to think"
  → place_type: "headspace", owner: current_host

  Gizmo: "What's dairy mart — inner world or real?"
  User:  "Real, it's just a corner store we like"
  → place_type: "local", description: "corner store"
"""

from tools.base_tool import BaseTool, ToolResult


class PlaceConfirmTool(BaseTool):
    @property
    def name(self) -> str:
        return "store_place"

    @property
    def description(self) -> str:
        return (
            "Store a confirmed place after asking what it is. "
            "Call this when the user answers a question about an unfamiliar place. "
            "Args: "
            "place_name (str) — the name of the place as mentioned. "
            "place_type (str) — one of: 'headspace', 'local', 'online', 'other'. "
            "owner (str, optional) — for headspaces, whose inner space it is. "
            "description (str, optional) — brief description from user's answer. "
            "Do NOT call this speculatively — only after the user has confirmed what a place is."
        )

    async def run(
        self,
        place_name: str = "",
        place_type: str = "other",
        owner: str = "",
        description: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not place_name:
            return ToolResult(success=False, output="No place name provided.")

        if place_type not in ("headspace", "local", "online", "other"):
            place_type = "other"

        try:
            from core.curiosity import store_place
            store_place(
                name=place_name,
                place_type=place_type,
                owner=owner or None,
                description=description,
                session_id=session_id,
            )

            if place_type == "headspace" and owner:
                msg = f"Got it — {place_name} is {owner}'s headspace. I'll remember that."
            elif place_type == "headspace":
                msg = f"Got it — {place_name} is a headspace. I'll remember that."
            elif place_type == "local":
                desc_note = f" ({description})" if description else ""
                msg = f"Got it — {place_name}{desc_note}. I'll remember it."
            else:
                msg = f"Got it — I'll remember {place_name}."

            return ToolResult(
                success=True,
                output=msg,
                data={
                    "place_name": place_name,
                    "place_type": place_type,
                    "owner": owner,
                    "description": description,
                },
            )

        except Exception as e:
            return ToolResult(success=False, output=f"Failed to store place: {e}")
