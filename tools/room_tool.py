"""tools/room_tool.py — Room contract management."""
from core.store import store
from core.log import log_event


class SetRoomContractTool:
    name        = "set_room_contract"
    description = "Record a room contract — who consented to what dynamic when multiple people are present."

    async def execute(self, args, session_id, headmate, llm) -> str:
        speaker = args.get("speaker") or headmate or ""
        entity  = args.get("entity") or args.get("about", "")
        label   = args.get("label") or args.get("contract", "")

        if not all([speaker, label]):
            return "need speaker and contract label"

        store.write("relationships", {
            "speaker":               speaker.lower(),
            "entity":                entity.lower() if entity else "",
            "relationship_label":    label.lower(),
            "relationship_category": "room_contract",
            "confidence_type":       "stated",
            "intimate":              1,
            "headmate":              speaker.lower(),
            "session_id":            session_id,
            "source":                "gizmo",
            "tags":                  f"room_contract,{speaker.lower()}",
        })
        log_event("SetRoomContractTool", "CONTRACT_SET", speaker=speaker, label=label)
        return f"room contract recorded: {speaker} → {label}"
