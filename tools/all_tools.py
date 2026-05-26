"""
tools/entity_tool.py
Create and view entity records in the store.
"""
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
        log_event("CreateEntityTool", "ENTITY_CREATED",
            name=name, type=entity_type)
        return f"entity created: {name} ({entity_type})"


class ViewEntityTool:
    name        = "view_entity"
    description = "View what Gizmo knows about a person or entity."

    async def execute(self, args, session_id, headmate, llm) -> str:
        name   = args.get("name") or args.get("headmate") or headmate or ""
        entity = store.get_entity(name) if name else None

        if not entity:
            return f"no entity on file for {name or 'unknown'}"

        facts = store.query("facts",
            headmate=name.lower(), active=1, limit=8)
        rels  = store.query("relationships",
            headmate=name.lower(), active=1, limit=5)

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


# ── Pattern tool ──────────────────────────────────────────────────────────────

class ViewPatternTool:
    name        = "view_patterns"
    description = "View active behavioral patterns for a headmate."

    async def execute(self, args, session_id, headmate, llm) -> str:
        about = args.get("headmate") or headmate or ""
        if not about:
            return "no headmate specified"

        patterns = store.get_patterns(about, min_confidence=0.2)
        if not patterns:
            return f"no patterns on file for {about}"

        lines = []
        for p in patterns:
            lines.append(
                f"[{p.get('action','?').upper()}] {p.get('pattern_type','?')} "
                f"conf={p.get('confidence',0):.2f} "
                f"pts={p.get('data_points',0)} "
                f"quality={p.get('outcome_quality_avg',0):.2f}"
            )
            if p.get("approach"):
                lines.append(f"  → {p['approach'][:80]}")
        return "\n".join(lines)


# ── Room contract tool ────────────────────────────────────────────────────────

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
            "speaker":                speaker.lower(),
            "entity":                 entity.lower() if entity else "",
            "relationship_label":     label.lower(),
            "relationship_category":  "room_contract",
            "confidence_type":        "stated",
            "intimate":               1,
            "headmate":               speaker.lower(),
            "session_id":             session_id,
            "source":                 "gizmo",
            "tags":                   f"room_contract,{speaker.lower()}",
        })
        log_event("SetRoomContractTool", "CONTRACT_SET",
            speaker=speaker, label=label)
        return f"room contract recorded: {speaker} → {label}"


# ── Wellness tool ─────────────────────────────────────────────────────────────

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
        log_event("WellnessTool", "WELLNESS_LOGGED",
            about=about, category=category)
        return f"wellness observation logged for {about or 'unknown'}"


# ── Interaction prefs tool ────────────────────────────────────────────────────

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
            "headmate":  about.lower() if about else None,
            "pref_type": pref_type,
            "content":   content,
            "session_id": session_id,
            "source":    "user",
            "tags":      f"interaction_pref,{pref_type},{about.lower() if about else 'global'}",
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
        lines = [f"- [{p.get('pref_type','?')}] {p.get('content','')}"
                 for p in prefs]
        return "\n".join(lines)


# ── Introspect tool ───────────────────────────────────────────────────────────

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
            "text":      text,
            "topic":     topic,
            "valence":   valence,
            "headmate":  headmate.lower() if headmate else None,
            "session_id": session_id,
            "source":    "gizmo",
            "outcome":   "pending",
            "tags":      f"reflection,{topic},{headmate.lower() if headmate else 'gizmo'}",
        })
        return f"reflection stored"


# ── Protocol tool ─────────────────────────────────────────────────────────────

class CreateProtocolTool:
    name        = "create_protocol"
    description = "Create or update a behavioral protocol — a rule or commitment from conversation."

    async def execute(self, args, session_id, headmate, llm) -> str:
        import os
        from pathlib import Path

        name    = args.get("name") or args.get("title", "")
        content = args.get("content") or args.get("text", "")
        scope   = args.get("scope", "global")

        if not content:
            return "no protocol content provided"

        # Write to store
        proto_id = store.write("protocols", {
            "name":      name,
            "content":   content,
            "scope":     scope,
            "headmate":  headmate.lower() if headmate and scope != "global" else None,
            "session_id": session_id,
            "source":    "gizmo",
            "tags":      f"protocol,{scope},{headmate.lower() if headmate else 'global'}",
        })

        # Also write to disk for human readability
        protocols_dir = Path(os.getenv("PERSONALITY_DIR", "/data/personality")) / "protocols"
        protocols_dir.mkdir(parents=True, exist_ok=True)
        safe_name = (name or proto_id).lower().replace(" ", "_")[:40]
        file_path = protocols_dir / f"{safe_name}.txt"
        file_path.write_text(content, encoding="utf-8")

        # Index in files table
        store.write("files", {
            "path":        str(file_path),
            "description": name or content[:60],
            "file_type":   "protocol",
            "content_ref": f"protocols:{proto_id}",
            "headmate":    headmate.lower() if headmate else None,
            "source":      "gizmo",
            "tags":        f"file,protocol",
        })

        log_event("CreateProtocolTool", "PROTOCOL_CREATED",
            name=name, scope=scope)
        return f"protocol saved: {name or 'unnamed'}"


# ── Report tool (re-export) ───────────────────────────────────────────────────

class ReportTool:
    name        = "generate_report"
    description = (
        "Generate a PDF report for one or more headmates. "
        "Understands plain language — ask for clarification if needed."
    )

    def __init__(self):
        from tools.report_tool import ReportTool as _RT
        self._inner = _RT()

    async def execute(self, args, session_id, headmate, llm) -> str:
        return await self._inner.execute(
            args=args,
            session_id=session_id,
            headmate=headmate,
            llm=llm,
        )
