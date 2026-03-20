"""
tools/personality_tool.py
Tool for Gizmo to interact with the personality signal system.

Two functions:
  1. resolve_contradiction — called after human decides how to handle a conflict
  2. query_personality — retrieve stored signals for a speaker (for synthesis)

The agent uses resolve_contradiction when it presents a contradiction
notice to the user and they respond with their preference.

query_personality is called by synthesis.py when building context
for a specific headmate — surfaces their known traits, interests, etc.
"""

from tools.base_tool import BaseTool, ToolResult


class PersonalityResolveTool(BaseTool):
    @property
    def name(self) -> str:
        return "resolve_personality_contradiction"

    @property
    def description(self) -> str:
        return (
            "Resolve a personality signal contradiction — called after presenting "
            "a contradiction to the user and receiving their decision. "
            "Args: "
            "speaker (str) — whose signal this is. "
            "subject (str) — what the signal is about. "
            "keep (str) — 'new', 'old', or 'both'. "
            "new_signal_id (str, optional) — ID of the pending new signal."
        )

    async def run(
        self,
        speaker: str = "",
        subject: str = "",
        keep: str = "both",
        new_signal_id: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not speaker or not subject:
            return ToolResult(success=False, output="Need speaker and subject.")

        if keep not in ("new", "old", "both"):
            return ToolResult(
                success=False,
                output=f"keep must be 'new', 'old', or 'both' — got '{keep}'"
            )

        try:
            from voice.personality import resolve_contradiction
            resolve_contradiction(
                speaker=speaker,
                subject=subject,
                keep=keep,
                new_signal_id=new_signal_id or None,
            )

            messages = {
                "new": f"Got it — updated {speaker.capitalize()}'s record with the new signal about {subject}.",
                "old": f"Okay — keeping the existing record for {speaker.capitalize()} about {subject}.",
                "both": f"Noted — keeping both signals for {speaker.capitalize()} about {subject}, treating it as an evolution over time.",
            }

            return ToolResult(
                success=True,
                output=messages[keep],
                data={"speaker": speaker, "subject": subject, "resolution": keep},
            )

        except Exception as e:
            return ToolResult(success=False, output=f"Resolution failed: {e}")


class PersonalityQueryTool(BaseTool):
    @property
    def name(self) -> str:
        return "query_personality"

    @property
    def description(self) -> str:
        return (
            "Retrieve stored personality signals for a headmate. "
            "Use when asked about someone's preferences, interests, traits, or values. "
            "Args: "
            "speaker (str) — whose signals to retrieve. "
            "signal_type (str, optional) — filter by type: "
            "'preference', 'interest', 'value', 'emotional', 'belief', or 'all'. "
            "subject (str, optional) — narrow to signals about a specific topic."
        )

    async def run(
        self,
        speaker: str = "",
        signal_type: str = "all",
        subject: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not speaker:
            return ToolResult(success=False, output="Need a speaker name.")

        try:
            signals = _fetch_signals(speaker, signal_type, subject)

            if not signals:
                return ToolResult(
                    success=True,
                    output=f"No personality signals stored for {speaker.capitalize()} yet.",
                )

            lines = []
            by_type: dict[str, list] = {}
            for s in signals:
                t = s.get("signal_type", "other")
                by_type.setdefault(t, []).append(s)

            for stype, items in sorted(by_type.items()):
                lines.append(f"\n{stype.upper()}S:")
                for item in items:
                    sentiment = item.get("sentiment", "")
                    date = item.get("date", "")
                    stmt = item.get("statement", "")
                    lines.append(f"  • {stmt} [{sentiment}, {date}]")

            summary = f"Personality signals for {speaker.capitalize()}:" + "\n".join(lines)

            return ToolResult(
                success=True,
                output=summary,
                data={"speaker": speaker, "signals": signals},
            )

        except Exception as e:
            return ToolResult(success=False, output=f"Query failed: {e}")


def _fetch_signals(speaker: str, signal_type: str = "all", subject: str = "") -> list[dict]:
    """Fetch active personality signals for a speaker from their collection."""
    from core.rag import RAGStore, CHROMA_PERSIST_DIR
    import chromadb

    collection_name = speaker.lower().strip()

    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            return []

        store = RAGStore(collection_name=collection_name)
        if store.count == 0:
            return []

        # Build where clause
        conditions = [
            {"type": {"$eq": "personality_signal"}},
            {"status": {"$eq": "active"}},
            {"speaker": {"$eq": speaker.lower()}},
        ]
        if signal_type != "all":
            conditions.append({"signal_type": {"$eq": signal_type}})

        where = {"$and": conditions}

        if subject:
            # Semantic search narrowed to this subject
            results = store.retrieve(
                query=f"{speaker} {subject} personality",
                n_results=10,
                where=where,
            )
            return [r["metadata"] for r in results]
        else:
            # Get all active signals
            results = store.collection.get(where=where)
            return results.get("metadatas", [])

    except Exception as e:
        print(f"[PersonalityTool] Fetch failed: {e}")
        return []


def get_personality_context(speaker: str) -> str:
    """
    Build a compact personality context block for injection into synthesis.
    Called by synthesis.py when current_host matches speaker.
    Returns empty string if no signals stored.
    """
    signals = _fetch_signals(speaker, signal_type="all")
    if not signals:
        return ""

    by_type: dict[str, list[str]] = {}
    for s in signals:
        t = s.get("signal_type", "other")
        stmt = s.get("statement", "").strip()
        if stmt:
            by_type.setdefault(t, []).append(stmt)

    if not by_type:
        return ""

    lines = [f"Known about {speaker.capitalize()}:"]
    type_labels = {
        "preference": "Preferences",
        "interest": "Interests",
        "value": "Values",
        "emotional": "Emotional patterns",
        "belief": "Beliefs/worldview",
    }
    for t, stmts in sorted(by_type.items()):
        label = type_labels.get(t, t.capitalize())
        lines.append(f"  {label}: {'; '.join(stmts[:5])}")  # cap at 5 per type

    return "\n".join(lines)
