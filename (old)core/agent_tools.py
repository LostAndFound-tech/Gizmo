"""
core/agent_tools.py
Tool registry.

All tools write to store. No JSON files. No ChromaDB.
New tools: report_tool, wellness_tool, pattern_tool.
Rewritten tools: correction, memory, switch_host, interaction_prefs.

Registry is a flat dict: {tool_name: tool_instance}
Agent reads TOOL_REGISTRY and TOOL_DESCRIPTIONS for prompt injection.

Tool interface (all tools implement):
  tool.name:        str
  tool.description: str
  tool.execute(args, session_id, headmate, llm) -> str
"""

from core.log import log, log_error


def _load(import_path: str, class_name: str, *args):
    """Safe tool loader. Returns None and logs on failure."""
    try:
        module = __import__(import_path, fromlist=[class_name])
        cls    = getattr(module, class_name)
        return cls(*args)
    except Exception as e:
        log_error("ToolRegistry", f"failed to load {class_name}: {e}", exc=None)
        return None


# ── Core tools ────────────────────────────────────────────────────────────────

_correction      = _load("tools.correction_tool",      "CorrectionTool")
_switch_host     = _load("tools.switch_host",          "SwitchHostTool")
_interaction     = _load("tools.interaction_prefs_tool","InteractionPrefsTool")
_view_prefs      = _load("tools.interaction_prefs_tool","ViewInteractionPrefsTool")
_memory_write    = _load("tools.memory_tool",          "MemoryWriteTool")
_memory_read     = _load("tools.memory_tool",          "MemoryReadTool")
_memory_search   = _load("tools.memory_tool",          "MemorySearchTool")
_memory_delete   = _load("tools.memory_tool",          "MemoryDeleteTool")
_read_file       = _load("tools.file_tool",            "ReadFileTool")
_write_file      = _load("tools.file_tool",            "WriteFileTool")
_append_file     = _load("tools.file_tool",            "AppendFileTool")
_list_files      = _load("tools.file_tool",            "ListFilesTool")
_delete_file     = _load("tools.file_tool",            "DeleteFileTool")
_protocol        = _load("tools.protocol_tool",        "CreateProtocolTool")
_introspect      = _load("tools.introspect_tool",      "IntrospectTool")
_report          = _load("tools.report_tool",          "ReportTool")
_wellness        = _load("tools.wellness_tool",        "WellnessTool")
_preference_set  = _load("tools.preference_tool",     "SetPreferenceTool")
_preference_view = _load("tools.preference_tool",     "ViewPreferenceTool")
_question_queue  = _load("tools.preference_tool",     "QueueQuestionTool")
_entity_create   = _load("tools.entity_tool",         "CreateEntityTool")
_entity_view     = _load("tools.entity_tool",         "ViewEntityTool")
_pattern_view    = _load("tools.pattern_tool",        "ViewPatternTool")
_room_contract   = _load("tools.room_tool",           "SetRoomContractTool")

# ── Build registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict = {}

for _tool in [
    _correction,
    _switch_host,
    _interaction,
    _view_prefs,
    _memory_write,
    _memory_read,
    _memory_search,
    _memory_delete,
    _read_file,
    _write_file,
    _append_file,
    _list_files,
    _delete_file,
    _protocol,
    _introspect,
    _report,
    _wellness,
    _preference_set,
    _preference_view,
    _question_queue,
    _entity_create,
    _entity_view,
    _pattern_view,
    _room_contract,
]:
    if _tool is not None:
        TOOL_REGISTRY[_tool.name] = _tool

log("ToolRegistry", f"loaded {len(TOOL_REGISTRY)} tools: {list(TOOL_REGISTRY.keys())}")


# ── Tool descriptions for prompt injection ────────────────────────────────────

def get_tool_descriptions() -> str:
    """Compact tool list for system prompt."""
    if not TOOL_REGISTRY:
        return "(no tools available)"
    return "\n".join(
        f"- {name}: {tool.description}"
        for name, tool in TOOL_REGISTRY.items()
    )


# ── Tool dispatcher ───────────────────────────────────────────────────────────

async def dispatch_tool(
    tool_name:  str,
    args:       dict,
    session_id: str,
    headmate:   str,
    llm,
) -> str:
    """
    Execute a tool by name. Returns tool output string.
    Logs and returns error string on failure — never raises.
    """
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        return f"[tool '{tool_name}' not found]"

    try:
        result = await tool.execute(
            args=args,
            session_id=session_id,
            headmate=headmate,
            llm=llm,
        )
        log("ToolRegistry", f"tool '{tool_name}' executed → {str(result)[:60]}")
        return str(result) if result is not None else "[done]"

    except Exception as e:
        log_error("ToolRegistry", f"tool '{tool_name}' failed: {e}", exc=e)
        return f"[tool error: {e}]"
