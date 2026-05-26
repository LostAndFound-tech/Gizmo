"""tools/protocol_tool.py — Create behavioral protocols."""
import os
from pathlib import Path
from core.store import store
from core.log import log_event


class CreateProtocolTool:
    name        = "create_protocol"
    description = "Create or update a behavioral protocol — a rule or commitment from conversation."

    async def execute(self, args, session_id, headmate, llm) -> str:
        name    = args.get("name") or args.get("title", "")
        content = args.get("content") or args.get("text", "")
        scope   = args.get("scope", "global")

        if not content:
            return "no protocol content provided"

        proto_id = store.write("protocols", {
            "name":       name,
            "content":    content,
            "scope":      scope,
            "headmate":   headmate.lower() if headmate and scope != "global" else None,
            "session_id": session_id,
            "source":     "gizmo",
            "tags":       f"protocol,{scope},{headmate.lower() if headmate else 'global'}",
        })

        # Also write to disk for human readability
        protocols_dir = Path(os.getenv("PERSONALITY_DIR", "/data/personality")) / "protocols"
        protocols_dir.mkdir(parents=True, exist_ok=True)
        safe_name = (name or proto_id).lower().replace(" ", "_")[:40]
        file_path = protocols_dir / f"{safe_name}.txt"
        file_path.write_text(content, encoding="utf-8")

        store.write("files", {
            "path":        str(file_path),
            "description": name or content[:60],
            "file_type":   "protocol",
            "content_ref": f"protocols:{proto_id}",
            "headmate":    headmate.lower() if headmate else None,
            "source":      "gizmo",
            "tags":        "file,protocol",
        })

        log_event("CreateProtocolTool", "PROTOCOL_CREATED", name=name, scope=scope)
        return f"protocol saved: {name or 'unnamed'}"
