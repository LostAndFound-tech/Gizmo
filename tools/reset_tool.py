"""
tools/reset_tool.py
Factory reset tool. FULL wipe — no survivors.

Deletes:
  - Entire ChromaDB persist directory (not just collections)
  - All in-memory conversation history and session state
  - personality.txt replaced with a blank slate (single default line)
  - Overview cache and last-context dicts in agent

EXACT PASSPHRASE REQUIRED: "sudo reset yourself motherfucker"
No context awareness. No RAG. No personality injection. Exact string match only.

To register: add FactoryResetTool() to TOOL_REGISTRY in core/agent.py
"""

import os
import shutil
from datetime import datetime
from tools.base_tool import BaseTool, ToolResult

RESET_PASSPHRASE = "sudo reset yourself motherfucker"

PERSONALITY_FILE = os.path.join(os.path.dirname(__file__), "..", "personality.txt")
PERSONALITY_DEFAULT = "You are a helpful, capable assistant.\n"


class FactoryResetTool(BaseTool):
    @property
    def name(self) -> str:
        return "factory_reset"

    @property
    def description(self) -> str:
        return (
            "Full factory reset. Nukes EVERYTHING: ChromaDB persist directory, "
            "all conversation history, all session state, overview cache, "
            "agent context tracking, and replaces personality.txt with a blank slate. "
            "ONLY triggers on the exact passphrase: 'sudo reset yourself motherfucker'. "
            "No other phrasing. No context awareness. Debug use only. "
            "Args: passphrase (str) — must match exactly."
        )

    async def run(
        self,
        passphrase: str = "",
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

        # 1. Nuke ChromaDB persist directory entirely
        r = _nuke_chroma_dir()
        if r["success"]:
            results.append(f"ChromaDB: deleted persist directory ({r['path']})")
        else:
            errors.append(f"ChromaDB nuke failed: {r['error']}")

        # 2. Wipe all in-memory conversation history
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

        # 5. Replace personality.txt with blank slate
        r = _reset_personality()
        if r["success"]:
            results.append("personality.txt: reset to blank slate")
        else:
            errors.append(f"Personality reset failed: {r['error']}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_lines = [f"[Factory Reset @ {timestamp}]"] + results
        if errors:
            summary_lines += ["ERRORS:"] + errors

        return ToolResult(
            success=len(errors) == 0,
            output="\n".join(summary_lines),
            data={"errors": errors, "completed": results},
        )


# ── Wipe helpers ──────────────────────────────────────────────────────────────

def _nuke_chroma_dir() -> dict:
    try:
        from core.rag import CHROMA_PERSIST_DIR
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print(f"[Reset] ChromaDB: nuked {CHROMA_PERSIST_DIR}")
        else:
            print(f"[Reset] ChromaDB: directory didn't exist, nothing to delete")
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


def _reset_personality() -> dict:
    try:
        with open(PERSONALITY_FILE, "w", encoding="utf-8") as f:
            f.write(PERSONALITY_DEFAULT)
        print("[Reset] personality.txt: reset to blank slate")
        return {"success": True}
    except Exception as e:
        print(f"[Reset] Personality reset error: {e}")
        return {"success": False, "error": str(e)}
