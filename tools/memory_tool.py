"""
tools/memory_tool.py
Read, write, search, delete facts from store.
"""
from core.store import store
from core.log import log_event

class MemoryWriteTool:
    name        = "memory_write"
    description = "Save a fact about someone or something to memory."

    async def execute(self, args, session_id, headmate, llm) -> str:
        fact    = args.get("fact") or args.get("text", "")
        about   = args.get("about") or args.get("headmate") or headmate or ""
        context = args.get("context", "")
        if not fact:
            return "no fact provided"
        store.write("facts", {
            "fact":       fact,
            "headmate":   about.lower() if about else None,
            "fact_type":  args.get("type", "observation"),
            "context":    context,
            "session_id": session_id,
            "source":     "gizmo",
            "tags":       f"manual,{about.lower() if about else 'general'}",
        })
        log_event("MemoryWriteTool", "FACT_WRITTEN", fact=fact[:60])
        return f"remembered: \"{fact[:80]}\""


class MemoryReadTool:
    name        = "memory_read"
    description = "Read facts from memory about someone."

    async def execute(self, args, session_id, headmate, llm) -> str:
        about = args.get("about") or args.get("headmate") or headmate or ""
        limit = int(args.get("limit", 10))
        facts = store.query("facts",
            headmate=about.lower() if about else None,
            active=1,
            limit=limit,
        )
        if not facts:
            return f"nothing on file for {about or 'unknown'}"
        lines = [f"- {f['fact']}" for f in facts if f.get("fact")]
        return "\n".join(lines)


class MemorySearchTool:
    name        = "memory_search"
    description = "Search memory for relevant facts using a query."

    async def execute(self, args, session_id, headmate, llm) -> str:
        query = args.get("query") or args.get("text", "")
        about = args.get("about") or args.get("headmate") or headmate
        limit = int(args.get("limit", 8))
        if not query:
            return "no query provided"
        results = store.search(
            query=query,
            tables=["facts", "messages", "reflections"],
            headmate=about.lower() if about else None,
            limit=limit,
        )
        if not results:
            return "nothing found"
        lines = []
        for r in results:
            text = r.get("fact") or r.get("content", "")[:100]
            table = r.get("_table", "?")
            lines.append(f"[{table}] {text}")
        return "\n".join(lines)


class MemoryDeleteTool:
    name        = "memory_delete"
    description = "Soft-delete a memory fact by ID."

    async def execute(self, args, session_id, headmate, llm) -> str:
        fact_id = args.get("id", "")
        if not fact_id:
            return "no id provided"
        store.delete("facts", fact_id)
        return f"deleted {fact_id}"
