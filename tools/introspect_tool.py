"""
tools/introspect_tool.py
Gizmo's self-awareness tool.

Lets Gizmo read his own files — headmates, externals, pets, personality,
corrections, and basic code structure. Read-only, always.

This is not RAG retrieval. This is direct file inspection — structured,
authoritative, complete. Gizmo looks at what he actually knows, not what
he can find in embeddings.

Use when:
  - Someone asks who Gizmo knows about ("do you have a file on Oren?")
  - Someone asks what Gizmo remembers about himself
  - Someone asks what rules Gizmo is following
  - Someone asks if Gizmo has files / what files he has
  - Gizmo is uncertain whether he knows someone and wants to check

Args:
  query: what to look up. One of:
    "headmates"   — list all known headmates with brief summaries
    "externals"   — list all known external people
    "pets"        — list all known pets
    "all_known"   — everyone: headmates + externals + pets
    "personality" — read personality.txt (who Gizmo is)
    "rules"       — read active behavioral corrections/rules
    "headmate:<name>" — full file for a specific headmate
    "external:<name>" — full file for a specific external person
    "pet:<name>"      — full file for a specific pet
    "self"        — summary of everything: who Gizmo is + who he knows

Returns a plain-text summary Gizmo can speak naturally from.
"""

from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
import json

import os

_PERSONALITY_DIR = Path(os.getenv("PERSONALITY_DIR", "/data/personality"))
_HEADMATES_DIR   = _PERSONALITY_DIR / "headmates"
_EXTERNAL_DIR    = _PERSONALITY_DIR / "external"
_PETS_DIR        = _PERSONALITY_DIR / "pets"
_SEED_FILE       = _PERSONALITY_DIR / "personality.txt"
_RULES_FILE      = _PERSONALITY_DIR / "rules.json"


def _read_seed() -> str:
    try:
        text = _SEED_FILE.read_text(encoding="utf-8").strip()
        return text if text else "(personality file is empty)"
    except FileNotFoundError:
        return "(no personality file found)"


def _read_rules() -> str:
    try:
        rules = json.loads(_RULES_FILE.read_text(encoding="utf-8"))
        global_rules = rules.get("global", [])
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


def _summarize_headmate(data: dict) -> str:
    """One-line summary of a headmate file."""
    name = data.get("name", "unknown").title()
    baseline = data.get("baseline", {})
    obs = baseline.get("observations", 0)
    register = baseline.get("register", "unknown")
    patterns = data.get("observed_patterns", [])
    moments = data.get("moments_of_note", [])
    corrections = data.get("corrections", [])

    parts = [f"{name}"]
    if obs:
        parts.append(f"{obs} observation{'s' if obs != 1 else ''}")
    if register != "unknown":
        parts.append(f"baseline register: {register}")
    if patterns:
        parts.append(f"{len(patterns)} pattern{'s' if len(patterns) != 1 else ''} noted")
    if moments:
        parts.append(f"{len(moments)} moment{'s' if len(moments) != 1 else ''} of note")
    if corrections:
        parts.append(f"{len(corrections)} correction{'s' if len(corrections) != 1 else ''}")

    cold = obs == 0
    if cold:
        parts.append("(cold start — still learning)")

    return " | ".join(parts)


def _full_headmate(name: str) -> str:
    path = _HEADMATES_DIR / f"{name.lower()}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"No file found for headmate '{name}'."
    except Exception as e:
        return f"Error reading file for '{name}': {e}"

    lines = [f"Headmate: {data.get('name', name).title()}"]

    note = data.get("note", "")
    if note:
        lines.append(f"Note: {note}")

    baseline = data.get("baseline", {})
    if baseline:
        lines.append("\nBaseline:")
        for k, v in baseline.items():
            if v not in ("unknown", 0, 0.0):
                lines.append(f"  {k}: {v}")

    patterns = data.get("observed_patterns", [])
    if patterns:
        lines.append(f"\nObserved patterns ({len(patterns)}):")
        for p in patterns[-5:]:  # last 5
            if isinstance(p, dict):
                lines.append(f"  [{p.get('timestamp', '')}] {p.get('pattern', p)}")
            else:
                lines.append(f"  {p}")

    moments = data.get("moments_of_note", [])
    if moments:
        lines.append(f"\nMoments of note ({len(moments)}):")
        for m in moments[-5:]:
            lines.append(f"  {m}")

    corrections = data.get("corrections", [])
    if corrections:
        lines.append(f"\nCorrections ({len(corrections)}):")
        for c in corrections:
            if isinstance(c, dict):
                lines.append(f"  [{c.get('timestamp', '')}] {c.get('rule', c)}")
            else:
                lines.append(f"  {c}")

    return "\n".join(lines)


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


def _list_headmates() -> str:
    try:
        _HEADMATES_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(_HEADMATES_DIR.glob("*.json"))
        if not files:
            return "No headmate files on record yet."
        lines = [f"Known headmates ({len(files)}):"]
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                lines.append(f"  {_summarize_headmate(data)}")
            except Exception:
                lines.append(f"  {f.stem.title()} (file unreadable)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing headmates: {e}"


def _list_externals() -> str:
    try:
        _EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(_EXTERNAL_DIR.glob("*.json"))
        if not files:
            return "No external person files on record."
        lines = [f"Known external people ({len(files)}):"]
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                name = data.get("name", f.stem).title()
                rel  = data.get("relationship_to_system", "unknown")
                lines.append(f"  {name} — {rel}")
            except Exception:
                lines.append(f"  {f.stem.title()} (file unreadable)")
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
                data = json.loads(f.read_text(encoding="utf-8"))
                name    = data.get("name", f.stem).title()
                species = data.get("species", "unknown")
                lines.append(f"  {name} — {species}")
            except Exception:
                lines.append(f"  {f.stem.title()} (file unreadable)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing pets: {e}"


def _self_summary() -> str:
    """Full self-portrait: who I am + who I know."""
    parts = []

    # Personality
    seed = _read_seed()
    parts.append("=== Who I am ===")
    parts.append(seed[:800] + ("..." if len(seed) > 800 else ""))

    # Rules
    parts.append("\n=== Rules I'm following ===")
    parts.append(_read_rules())

    # Everyone I know
    parts.append("\n=== Who I know ===")
    parts.append(_list_headmates())
    parts.append("")
    parts.append(_list_externals())
    parts.append("")
    parts.append(_list_pets())

    return "\n".join(parts)


class IntrospectTool(BaseTool):

    @property
    def name(self) -> str:
        return "introspect"

    @property
    def description(self) -> str:
        return (
            "Read your own files and knowledge. Use when someone asks who you know, "
            "whether you have a file on someone, what you remember about yourself, "
            "or what rules you're following. "
            "Args: query (str) — one of: "
            "'headmates', 'externals', 'pets', 'all_known', "
            "'personality', 'rules', 'self', "
            "'headmate:<name>', 'external:<name>', 'pet:<name>'"
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

            elif query.startswith("headmate:"):
                name = query.split(":", 1)[1].strip()
                output = _full_headmate(name)

            elif query.startswith("external:"):
                name = query.split(":", 1)[1].strip()
                output = _full_external(name)

            elif query.startswith("pet:"):
                name = query.split(":", 1)[1].strip()
                output = _full_pet(name)

            else:
                # Try to guess — maybe they passed a name directly
                # Check all three directories
                for dir_, reader in [
                    (_HEADMATES_DIR, _full_headmate),
                    (_EXTERNAL_DIR,  _full_external),
                    (_PETS_DIR,      _full_pet),
                ]:
                    if (dir_ / f"{query}.json").exists():
                        output = reader(query)
                        break
                else:
                    output = (
                        f"Unknown query '{query}'. "
                        f"Valid options: headmates, externals, pets, all_known, "
                        f"personality, rules, self, headmate:<name>, external:<name>, pet:<name>"
                    )

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Introspection failed: {e}",
            )