"""tools/entity_tool.py — Create and view entity records."""
from core.store import store
from core.log import log_event


class CreateEntityTool:
    name        = "create_entity"
    description = "Create or update an entity (headmate, external person, pet, object)."

    async def execute(self, args, session_id, headmate, llm) -> str:
        name        = args.get("name", "").strip().lower()
        entity_type = args.get("entity_type", "external")
        pronouns    = args.get("pronouns", "")
        age         = args.get("age", "")
        persona     = args.get("persona", "")
        notes       = args.get("notes", "")

        if not name:
            return "no name provided"

        store.write("entities", {
            "id":          f"entity_{name.replace(' ','_')}",
            "name":        name,
            "entity_type": entity_type,
            "pronouns":    pronouns,
            "age":         age,
            "persona":     persona,
            "notes":       notes,
            "headmate":    name if entity_type == "headmate" else None,
            "source":      "gizmo",
            "tags":        f"entity,{entity_type},{name}",
        })
        log_event("CreateEntityTool", "ENTITY_CREATED", name=name, type=entity_type)
        return f"entity created: {name} ({entity_type})"


class ViewEntityTool:
    name        = "view_entity"
    description = "View what Gizmo knows about a person or entity."

    async def execute(self, args, session_id, headmate, llm) -> str:
        name   = args.get("name") or args.get("headmate") or headmate or ""
        entity = store.get_entity(name) if name else None

        if not entity:
            return f"no entity on file for {name or 'unknown'}"

        facts = store.query("facts", headmate=name.lower(), active=1, limit=8)
        rels  = store.query("relationships", headmate=name.lower(), active=1, limit=5)

        lines = [
            f"Entity: {entity.get('name','?')} ({entity.get('entity_type','?')})",
            f"Pronouns: {entity.get('pronouns','unknown')}",
            f"Persona: {entity.get('persona','none on file')}",
        ]
        if facts:
            lines.append("Facts:")
            lines.extend(f"  - {f['fact']}" for f in facts)
        if rels:
            lines.append("Relationships:")
            lines.extend(
                f"  - {r['speaker']} → {r['relationship_label']} → {r['entity']}"
                for r in rels if r.get("relationship_label")
            )
        return "\n".join(lines)
