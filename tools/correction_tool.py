"""
tools/correction_tool.py
Permanent correction system. When Gizmo does something wrong,
this tool logs the correction to personality_corrections via
personality_growth.ingest_correction() — where it lives permanently,
survives rewrites, and is always injected into the system prompt.

The old personality.txt append and standalone corrections RAG collection
are no longer used. Corrections now live in personality_corrections,
which is one of the five personality collections and treated as sacred
by the weekly rewrite process.
"""

from datetime import datetime
from tools.base_tool import BaseTool, ToolResult


class CorrectionTool(BaseTool):
    @property
    def name(self) -> str:
        return "log_correction"

    @property
    def description(self) -> str:
        return (
            "Log a permanent behavioral correction. "
            "EXPLICIT TRIGGERS: 'don't do that', 'stop doing that', 'that's wrong', "
            "'never do that again', 'stop making things up', 'that's not right', "
            "'I've told you this before', 'you keep doing this', "
            "or any clear statement that Gizmo did something wrong and should stop. "
            "Args: "
            "what_was_wrong (str) — clear description of what Gizmo did incorrectly. "
            "rule (str) — the permanent rule to follow going forward. "
            "who_corrected (str) — who gave the correction. "
            "session_id (str) — current session id."
        )

    async def run(
        self,
        what_was_wrong: str = "",
        rule: str = "",
        who_corrected: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if not what_was_wrong or not rule:
            return ToolResult(
                success=False,
                output="Need both what was wrong and the rule to apply going forward."
            )

        who = who_corrected or "the system"

        try:
            from core.personality_growth import ingest_correction
            await ingest_correction(
                what_was_wrong=what_was_wrong,
                rule=rule,
                who_corrected=who,
                session_id=session_id,
            )
            personality_updated = True
        except Exception as e:
            print(f"[Correction] personality_growth.ingest_correction() failed: {e}")
            personality_updated = False

        # Also log to main RAG for conversational memory retrieval
        await _log_to_main_rag(
            what_was_wrong=what_was_wrong,
            rule=rule,
            who_corrected=who,
            session_id=session_id,
        )

        summary = (
            f"Correction logged. "
            f"What was wrong: {what_was_wrong} "
            f"Rule going forward: {rule} "
            f"{'Personality updated.' if personality_updated else 'Note: personality store could not be updated — check logs.'}"
        )

        return ToolResult(
            success=True,
            output=summary,
            data={
                "what_was_wrong": what_was_wrong,
                "rule": rule,
                "who_corrected": who,
                "personality_updated": personality_updated,
            }
        )


async def _log_to_main_rag(
    what_was_wrong: str,
    rule: str,
    who_corrected: str,
    session_id: str,
) -> None:
    """
    Also ingest into main RAG so corrections surface in conversational context.
    The personality_corrections collection is the authoritative store —
    this is just for retrieval completeness.
    """
    try:
        from core.rag import RAGStore

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = (
            f"Correction logged on {timestamp} by {who_corrected}. "
            f"What Gizmo did wrong: {what_was_wrong} "
            f"Rule to follow going forward: {rule}"
        )

        metadata = {
            "source": "correction",
            "type": "correction",
            "date": timestamp,
            "session_id": session_id,
            "who_corrected": who_corrected,
            "rule": rule,
            "collection": "main",
        }

        store = RAGStore(collection_name="main")
        store.ingest_texts([entry], metadatas=[metadata])
        print(f"[Correction] Logged to main RAG — {rule[:60]}")

    except Exception as e:
        print(f"[Correction] main RAG log failed: {e}")