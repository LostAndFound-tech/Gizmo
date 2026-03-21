"""
tools/reset_tool.py
Scorched earth factory reset. NOTHING survives. No exceptions.

"sudo reset yourself motherfucker" — full and complete wipe:
  - Entire ChromaDB persist directory deleted with shutil.rmtree (not just collections)
  - In-memory conversation history and session state
  - Overview cache
  - Agent last-context tracking

After reset:
  - is_cold_start() returns True
  - Onboarding fires on next conversation turn
  - Gizmo remembers nothing

NOTE: This tool is also intercepted directly at the server level in server.py
before it reaches the agent, so the LLM never sees the passphrase.
The tool exists so the reset logic lives in one place and can be called cleanly.

EXACT PASSPHRASE REQUIRED: "sudo reset yourself motherfucker"
"""

import os
import shutil
from datetime import datetime
from tools.base_tool import BaseTool, ToolResult

RESET_PASSPHRASE = "sudo reset yourself motherfucker"


class FactoryResetTool(BaseTool):
    @property
    def name(self) -> str:
        return "factory_reset"

    @property
    def description(self) -> str:
        return (
            "Scorched earth reset. Wipes EVERYTHING — entire ChromaDB directory, "
            "all history, all session state, all context. Nothing is preserved. "
            "Onboarding re-runs automatically on the next turn. "
            "ONLY triggers on the EXACT passphrase: 'sudo reset yourself motherfucker'. "
            "No other phrasing activates this tool. "
            "Args: passphrase (str) — must match exactly, character for character."
        )

    async def run(
        self,
        passphrase: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:
        if passphrase.strip() != RESET_PASSPHRASE:
            return ToolResult(
                success=False,
                output=(
                    "Reset rejected. Wrong passphrase. "
                    "Required exact string: 'sudo reset yourself motherfucker'"
                ),
            )

        results = []
        errors = []

        # 1. Nuke the entire ChromaDB persist directory from disk
        r = _nuke_chroma_dir()
        if r["success"]:
            results.append(f"ChromaDB: deleted {r['path']} from disk")
        else:
            errors.append(f"ChromaDB nuke failed: {r['error']}")

        # 2. Wipe entity store (SQLite)
        r = _wipe_entity_store()
        if r["success"]:
            results.append("Entity store: wiped")
        else:
            errors.append(f"Entity store wipe failed: {r['error']}")

        # 3. Wipe in-memory conversation history
        r = _wipe_history()
        if r["success"]:
            results.append(f"History: cleared {r['sessions_cleared']} sessions")
        else:
            errors.append(f"History wipe failed: {r['error']}")

        # 3. Wipe overview cache
        r = _wipe_overviews()
        if r["success"]:
            results.append("Overview cache: cleared")
        else:
            errors.append(f"Overview wipe failed: {r['error']}")

        # 4. Wipe agent last-context tracking
        r = _wipe_agent_context()
        if r["success"]:
            results.append("Agent context tracking: cleared")
        else:
            errors.append(f"Agent context wipe failed: {r['error']}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_lines = [f"[Scorched Earth Reset @ {timestamp}]"] + results
        if errors:
            summary_lines += ["ERRORS:"] + errors
        summary_lines.append("Everything is gone. Onboarding fires on next turn.")

        return ToolResult(
            success=len(errors) == 0,
            output="\n".join(summary_lines),
            data={"errors": errors, "completed": results},
        )


# ── Wipe helpers ──────────────────────────────────────────────────────────────

def _wipe_entity_store() -> dict:
    try:
        from core.entity_store import wipe_all
        wipe_all()
        return {"success": True}
    except Exception as e:
        print(f"[Reset] Entity store wipe error: {e}")
        return {"success": False, "error": str(e)}


def _nuke_chroma_dir() -> dict:
    """
    Delete the entire ChromaDB persist directory from disk.
    This is the only reliable wipe — delete_collection() leaves files on disk
    and ChromaDB will reload them from the persist dir on next connection.
    """
    try:
        from core.rag import CHROMA_PERSIST_DIR

        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print(f"[Reset] ChromaDB: nuked {CHROMA_PERSIST_DIR}")
        else:
            print(f"[Reset] ChromaDB: directory didn't exist, nothing to delete")

        # Recreate empty directory so RAGStore doesn't error on next init
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        return {"success": True, "path": CHROMA_PERSIST_DIR}
    except Exception as e:
        print(f"[Reset] ChromaDB nuke error: {e}")
        return {"success": False, "error": str(e)}


def _wipe_history() -> dict:
    try:
        from memory.history import get_all_sessions
        sessions = get_all_sessions()
        count = len(sessions)
        for history in sessions.values():
            history.clear()
        sessions.clear()
        print(f"[Reset] History: cleared {count} sessions")
        return {"success": True, "sessions_cleared": count}
    except Exception as e:
        print(f"[Reset] History wipe error: {e}")
        return {"success": False, "error": str(e), "sessions_cleared": 0}


def _wipe_overviews() -> dict:
    try:
        from memory import overview as ov
        ov._overviews.clear()
        ov._overview_turn.clear()
        print("[Reset] Overview cache: cleared")
        return {"success": True}
    except Exception as e:
        print(f"[Reset] Overview wipe error: {e}")
        return {"success": False, "error": str(e)}


def _wipe_agent_context() -> dict:
    try:
        from core import agent as ag
        ag._last_context.clear()
        print("[Reset] Agent context tracking: cleared")
        return {"success": True}
    except Exception as e:
        print(f"[Reset] Agent context wipe error: {e}")
        return {"success": False, "error": str(e)}
