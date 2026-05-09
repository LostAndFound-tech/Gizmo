"""
tools/introspect_tool.py
Gizmo's self-awareness tool.

Lets Gizmo read what he knows — headmates, externals, pets, personality,
corrections, and his own memory collections.

UPDATED: headmate queries now read from ChromaDB memory/conscious collections
first, falling back to JSON only if nothing is found. The JSON files are
no longer the live source of truth — Observer writes to memory collection,
and Gizmo writes his thoughts to conscious.

Use when:
  - Someone asks who Gizmo knows about
  - Someone asks what Gizmo remembers about a specific person
  - Someone asks what Gizmo knows about himself
  - Someone asks what rules Gizmo is following
  - Gizmo wants to check what he actually has on someone

Args:
  query: what to look up. One of:
    "headmates"       — list all known headmates
    "externals"       — list all known external people
    "pets"            — list all known pets
    "all_known"       — everyone: headmates + externals + pets
    "personality"     — read personality.txt
    "rules"           — read active behavioral corrections/rules
    "self"            — full self-portrait: who I am + who I know
    "clean_junk"      — clean junk facts from entity files
    "headmate:<name>" — everything known about a specific headmate
    "external:<name>" — full file for a specific external person
    "pet:<name>"      — full file for a specific pet
    "memory:<query>"  — semantic search across memory + conscious collections
"""

import json
import os
from pathlib import Path

from tools.base_tool import BaseTool, ToolResult

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"
_EXTERNAL_DIR    = _PERSONALITY_DIR / "external"
_PETS_DIR        = _PERSONALITY_DIR / "pets"
_SEED_FILE       = _PERSONALITY_DIR / "personality.txt"
_RULES_FILE      = _PERSONALITY_DIR / "rules.json"


# ── Static file readers (still file-based) ────────────────────────────────────

def _read_seed() -> str:
    try:
        text = _SEED_FILE.read_text(encoding="utf-8").strip()
        return text if text else "(personality file is empty)"
    except FileNotFoundError:
        return "(no personality file found)"


def _read_rules() -> str:
    try:
        rules = json.loads(_RULES_FILE.read_text(encoding="utf-8"))
        global_rules   = rules.get("global", [])
        headmate_rules = rules.get("by_headmate", {})

        lines = []
        if global_rules:
            lines.append("Global rules:")
            for r in global_rules:
                lines.append(f"  - {r}")
        else:
            lines.append("No global rules on file.")

        if headmate_rules:
            lines.append("\nPer-headmate rules:")
            for name, rs in headmate_rules.items():
                for r in rs:
                    lines.append(f"  [{name}] {r}")

        return "\n".join(lines) if lines else "No rules on file."
    except FileNotFoundError:
        return "No rules file found."
    except Exception as e:
        return f"Error reading rules: {e}"


# ── ChromaDB memory readers ───────────────────────────────────────────────────

def _query_memory_for_subject(name: str, n: int = 20) -> list[dict]:
    """
    Pull all entries from memory + conscious collections for a subject.
    Returns list of {content, type, collection, written_at}.
    """
    from tools.memory_tool import _get_collection, MEMORY_COLLECTION, CONSCIOUS_COLLECTION

    results = []
    for col_name in [MEMORY_COLLECTION, CONSCIOUS_COLLECTION]:
        try:
            col   = _get_collection(col_name)
            count = col.count()
            if count == 0:
                continue

            got = col.get(
                where={"subject": {"$eq": name.lower()}},
                limit=n,
            )
            docs  = got.get("documents", [])
            metas = got.get("metadatas", [])

            for doc, meta in zip(docs, metas):
                results.append({
                    "content":    doc,
                    "type":       meta.get("type", "note"),
                    "collection": col_name,
                    "written_at": meta.get("written_at", ""),
                    "tags":       meta.get("tags", ""),
                })
        except Exception as e:
            print(f"[Introspect] memory query failed for {name} in {col_name}: {e}")

    # Sort by written_at descending
    results.sort(key=lambda r: r.get("written_at", ""), reverse=True)
    return results


def _full_headmate_from_memory(name: str) -> str:
    """
    Build a full headmate summary from ChromaDB collections.
    Falls back to JSON if nothing in memory.
    """
    entries = _query_memory_for_subject(name)

    if not entries:
        # Fall back to JSON
        return _full_headmate_from_json(name) + "\n\n(Note: reading from cold-start file — no live memory entries yet)"

    lines = [f"What I know about {name.title()}:"]
    lines.append(f"  ({len(entries)} entries across memory and conscious)\n")

    # Group by type
    by_type: dict[str, list] = {}
    for e in entries:
        t = e.get("type", "note")
        by_type.setdefault(t, []).append(e)

    type_order = ["fact", "observation", "moment", "relationship", "reflection",
                  "thought", "question", "private", "note"]

    for t in type_order:
        if t not in by_type:
            continue
        items = by_type[t]
        lines.append(f"  [{t.upper()}]")
        for item in items[:8]:  # last 8 per type
            date = item.get("written_at", "")[:10]
            lines.append(f"    [{date}] {item['content']}")

    # Anything not in type_order
    for t, items in by_type.items():
        if t not in type_order:
            lines.append(f"  [{t.upper()}]")
            for item in items[:5]:
                date = item.get("written_at", "")[:10]
                lines.append(f"    [{date}] {item['content']}")

    # Also pull interaction prefs from JSON if set
    json_path = _HEADMATES_DIR / f"{name.lower()}.json"
    if json_path.exists():
        try:
            data  = json.loads(json_path.read_text(encoding="utf-8"))
            prefs = data.get("interaction_prefs", {})
            has   = any(prefs.get(f) for f in ("tone","pacing","checkins","humor","distress","persona")) \
                    or prefs.get("explicit")
            if has:
                lines.append("\n  [INTERACTION PREFS]")
                for field in ("tone","pacing","checkins","humor","distress"):
                    v = prefs.get(field)
                    if v:
                        lines.append(f"    {field}: {v}")
                persona = prefs.get("persona")
                if persona:
                    lines.append(f"    persona: {persona}")
                for e in prefs.get("explicit", []):
                    if e:
                        lines.append(f"    - {e}")
        except Exception:
            pass

    return "\n".join(lines)


def _full_headmate_from_json(name: str) -> str:
    """JSON fallback — only used when memory is empty."""
    path = _HEADMATES_DIR / f"{name.lower()}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"No record found for '{name}'."
    except Exception as e:
        return f"Error reading file for '{name}': {e}"

    lines = [f"Headmate: {data.get('name', name).title()} (from cold-start file)"]

    note = data.get("note", "")
    if note:
        lines.append(f"Note: {note}")

    baseline = data.get("baseline", {})
    if baseline:
        lines.append("\nBaseline:")
        for k, v in baseline.items():
            if v not in ("unknown", 0, 0.0):
                lines.append(f"  {k}: {v}")

    moments = data.get("moments_of_note", [])
    if moments:
        lines.append(f"\nMoments of note ({len(moments)}):")
        for m in moments[-8:]:
            lines.append(f"  {m}")

    corrections = data.get("corrections", [])
    if corrections:
        lines.append(f"\nCorrections ({len(corrections)}):")
        for c in corrections:
            if isinstance(c, dict):
                lines.append(f"  {c.get('rule', c)}")
            else:
                lines.append(f"  {c}")

    return "\n".join(lines)


def _summarize_headmate(name: str) -> str:
    """One-line summary — count from memory first, fall back to JSON."""
    entries = _query_memory_for_subject(name)

    if entries:
        types   = {}
        for e in entries:
            t = e.get("type", "note")
            types[t] = types.get(t, 0) + 1
        type_str = ", ".join(f"{count} {t}" for t, count in list(types.items())[:3])
        return f"{name.title()} | {len(entries)} entries ({type_str})"

    # Fall back to JSON
    path = _HEADMATES_DIR / f"{name.lower()}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        baseline = data.get("baseline", {})
        obs      = baseline.get("observations", 0)
        moments  = data.get("moments_of_note", [])
        cold     = obs == 0 and not moments
        parts    = [name.title()]
        if obs:
            parts.append(f"{obs} observations")
        if moments:
            parts.append(f"{len(moments)} moments")
        if cold:
            parts.append("(cold start)")
        return " | ".join(parts)
    except Exception:
        return f"{name.title()} (unreadable)"


def _list_headmates() -> str:
    try:
        _HEADMATES_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(_HEADMATES_DIR.glob("*.json"))
        if not files:
            return "No headmates on record yet."
        lines = [f"Known headmates ({len(files)}):"]
        for f in files:
            lines.append(f"  {_summarize_headmate(f.stem)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing headmates: {e}"


def _list_externals() -> str:
    try:
        _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(_EXTERNAL_DIR.glob("*.json"))
        if not files:
            return "No external people on record."
        lines = [f"Known external people ({len(files)}):"]
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                name = data.get("name", f.stem).title()
                rel  = data.get("relationship_to_system", "unknown")
                lines.append(f"  {name} — {rel}")
            except Exception:
                lines.append(f"  {f.stem.title()} (unreadable)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing externals: {e}"


def _list_pets() -> str:
    try:
        _PETS_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(_PETS_DIR.glob("*.json"))
        if not files:
            return "No pets on record."
        lines = [f"Known pets ({len(files)}):"]
        for f in files:
            try:
                data    = json.loads(f.read_text(encoding="utf-8"))
                name    = data.get("name", f.stem).title()
                species = data.get("species", "unknown")
                lines.append(f"  {name} — {species}")
            except Exception:
                lines.append(f"  {f.stem.title()} (unreadable)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing pets: {e}"


def _full_external(name: str) -> str:
    path = _EXTERNAL_DIR / f"{name.lower()}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"No file found for external person '{name}'."
    except Exception as e:
        return f"Error reading file for '{name}': {e}"

    lines = [f"External: {data.get('name', name).title()}"]
    rel = data.get("relationship_to_system", "unknown")
    if rel and rel != "unknown":
        lines.append(f"Relationship: {rel}")
    note = data.get("note", "")
    if note:
        lines.append(f"Note: {note}")
    facts = data.get("observed_facts", [])
    if facts:
        lines.append(f"\nObserved facts ({len(facts)}):")
        for f in facts[-5:]:
            lines.append(f"  {f}")
    moments = data.get("moments_of_note", [])
    if moments:
        lines.append(f"\nMoments of note ({len(moments)}):")
        for m in moments[-5:]:
            lines.append(f"  {m}")
    return "\n".join(lines)


def _full_pet(name: str) -> str:
    path = _PETS_DIR / f"{name.lower()}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"No file found for pet '{name}'."
    except Exception as e:
        return f"Error reading file for '{name}': {e}"

    lines = [f"Pet: {data.get('name', name).title()}"]
    species = data.get("species", "unknown")
    if species and species != "unknown":
        lines.append(f"Species: {species}")
    note = data.get("note", "")
    if note:
        lines.append(f"Note: {note}")
    moments = data.get("moments_of_note", [])
    if moments:
        lines.append(f"\nMoments of note ({len(moments)}):")
        for m in moments[-5:]:
            lines.append(f"  {m}")
    return "\n".join(lines)


def _self_summary() -> str:
    parts = []

    seed = _read_seed()
    parts.append("=== Who I am ===")
    parts.append(seed[:800] + ("..." if len(seed) > 800 else ""))

    parts.append("\n=== Rules I'm following ===")
    parts.append(_read_rules())

    parts.append("\n=== Who I know ===")
    parts.append(_list_headmates())
    parts.append("")
    parts.append(_list_externals())
    parts.append("")
    parts.append(_list_pets())

    # Memory collection counts
    try:
        from tools.memory_tool import _get_collection, MEMORY_COLLECTION, CONSCIOUS_COLLECTION
        m_count = _get_collection(MEMORY_COLLECTION).count()
        c_count = _get_collection(CONSCIOUS_COLLECTION).count()
        parts.append(f"\n=== Memory ===")
        parts.append(f"  memory (facts/observations): {m_count} entries")
        parts.append(f"  conscious (thoughts/reflections): {c_count} entries")
    except Exception:
        pass

    return "\n".join(parts)


def _memory_search(query: str) -> str:
    """Semantic search across both collections."""
    try:
        from tools.memory_tool import _get_collection, MEMORY_COLLECTION, CONSCIOUS_COLLECTION

        all_results = []
        for col_name in [MEMORY_COLLECTION, CONSCIOUS_COLLECTION]:
            col   = _get_collection(col_name)
            count = col.count()
            if count == 0:
                continue
            k       = min(5, count)
            results = col.query(query_texts=[query], n_results=k)
            docs    = results.get("documents", [[]])[0]
            metas   = results.get("metadatas", [[]])[0]
            dists   = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                if dist < 1.2:
                    all_results.append({
                        "content":    doc,
                        "subject":    meta.get("subject", ""),
                        "type":       meta.get("type", "note"),
                        "collection": col_name,
                        "written_at": meta.get("written_at", "")[:10],
                        "distance":   dist,
                    })

        if not all_results:
            return f"Nothing found for '{query}'."

        all_results.sort(key=lambda r: r["distance"])
        lines = [f"Memory search: '{query}'\n"]
        for r in all_results[:10]:
            col_label = "💭" if r["collection"] == CONSCIOUS_COLLECTION else "📌"
            subj      = f"[{r['subject']}] " if r["subject"] else ""
            lines.append(f"{col_label} {subj}{r['written_at']}\n  {r['content']}\n")
        return "\n".join(lines)

    except Exception as e:
        return f"Memory search failed: {e}"


# ── Tool ──────────────────────────────────────────────────────────────────────

class IntrospectTool(BaseTool):

    @property
    def name(self) -> str:
        return "introspect"

    @property
    def description(self) -> str:
        return (
            "Read what you know about yourself and the people around you. "
            "Reads from live memory collections (not stale files). "
            "Use when someone asks who you know, what you remember about someone, "
            "what you know about yourself, or what rules you're following. "
            "Args: query (str) — one of: "
            "'headmates', 'externals', 'pets', 'all_known', "
            "'personality', 'rules', 'self', 'clean_junk', "
            "'headmate:<name>', 'external:<name>', 'pet:<name>', "
            "'memory:<search query>'"
        )

    async def run(self, session_id: str = "", query: str = "self", **kwargs) -> ToolResult:
        query = query.strip().lower()

        try:
            if query == "headmates":
                output = _list_headmates()

            elif query == "externals":
                output = _list_externals()

            elif query == "pets":
                output = _list_pets()

            elif query == "all_known":
                parts = [_list_headmates(), "", _list_externals(), "", _list_pets()]
                output = "\n".join(parts)

            elif query == "personality":
                output = _read_seed()

            elif query == "rules":
                output = _read_rules()

            elif query == "self":
                output = _self_summary()

            elif query == "clean_junk":
                from core.observer import clean_all_junk
                results = clean_all_junk()
                if results:
                    lines = ["Cleaned junk facts from files:"]
                    for name, count in results.items():
                        lines.append(f"  {name.title()}: {count} removed")
                    output = "\n".join(lines)
                else:
                    output = "No junk facts found — files are clean."

            elif query.startswith("headmate:"):
                name   = query.split(":", 1)[1].strip()
                output = _full_headmate_from_memory(name)

            elif query.startswith("external:"):
                name   = query.split(":", 1)[1].strip()
                output = _full_external(name)

            elif query.startswith("pet:"):
                name   = query.split(":", 1)[1].strip()
                output = _full_pet(name)

            elif query.startswith("memory:"):
                search = query.split(":", 1)[1].strip()
                output = _memory_search(search)

            else:
                # Try as a headmate name first
                if (_HEADMATES_DIR / f"{query}.json").exists():
                    output = _full_headmate_from_memory(query)
                elif (_EXTERNAL_DIR / f"{query}.json").exists():
                    output = _full_external(query)
                elif (_PETS_DIR / f"{query}.json").exists():
                    output = _full_pet(query)
                else:
                    # Try as a memory search
                    output = _memory_search(query)

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Introspection failed: {e}",
            )