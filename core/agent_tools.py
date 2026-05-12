"""
core/agent_tools.py
Tool registry. Separate from the membrane so tools can be
registered, hot-swapped, and eventually managed by the tool forge
without touching agent.py.

v1: manual registration of the tools that survived the rebuild.
Future: tool forge registers tools here dynamically.
"""

from core.log import log
from tools.introspect_tool import IntrospectTool
from tools.protocol_tool import CreateProtocolTool
from tools.active_file_tool import SetActiveFileTool, CloseActiveFileTool

try:
    from tools.switch_host import SwitchHostTool
    _switch = SwitchHostTool()
except Exception as e:
    log("Agent", f"WARNING: switch_host tool failed to load: {e}")
    _switch = None

try:
    from tools.correction_tool import CorrectionTool
    _correction = CorrectionTool()
except Exception as e:
    log("Agent", f"WARNING: correction_tool failed to load: {e}")
    _correction = None

try:
    from tools.interaction_prefs_tool import InteractionPrefsTool, ViewInteractionPrefsTool
    _set_prefs  = InteractionPrefsTool()
    _view_prefs = ViewInteractionPrefsTool()
except Exception as e:
    log("Agent", f"WARNING: interaction_prefs_tool failed to load: {e}")
    _set_prefs  = None
    _view_prefs = None

try:
    from tools.memory_tool import (
        MemoryWriteTool,
        MemoryReadTool,
        MemoryListTool,
        MemoryDeleteTool,
    )
    _memory_write  = MemoryWriteTool()
    _memory_read   = MemoryReadTool()
    _memory_list   = MemoryListTool()
    _memory_delete = MemoryDeleteTool()
except Exception as e:
    log("Agent", f"WARNING: memory_tool failed to load: {e}")
    _memory_write  = None
    _memory_read   = None
    _memory_list   = None
    _memory_delete = None

try:
    from tools.file_tool import (
        ReadFileTool,
        WriteFileTool,
        AppendFileTool,
        ListFilesTool,
        DeleteFileTool,
        ConfirmWriteTool,
        CancelWriteTool,
    )
    _read_file     = ReadFileTool()
    _write_file    = WriteFileTool()
    _append_file   = AppendFileTool()
    _list_files    = ListFilesTool()
    _delete_file   = DeleteFileTool()
    _confirm_write = ConfirmWriteTool()
    _cancel_write  = CancelWriteTool()
except Exception as e:
    log("Agent", f"WARNING: file_tool failed to load: {e}")
    _read_file     = None
    _write_file    = None
    _append_file   = None
    _list_files    = None
    _delete_file   = None
    _confirm_write = None
    _cancel_write  = None

# Build registry — skip any tools that failed to load
TOOL_REGISTRY = {
    tool.name: tool for tool in [
        IntrospectTool(),
        CreateProtocolTool(),
        SetActiveFileTool(),
        CloseActiveFileTool()

    ]
}

for tool in [
    _switch,
    _correction,
    _memory_write,
    _memory_read,
    _memory_list,
    _memory_delete,
    _set_prefs,
    _view_prefs,
    _read_file,
    _write_file,
    _append_file,
    _list_files,
    _delete_file,
    _confirm_write,
    _cancel_write,
]:
    if tool is not None:
        TOOL_REGISTRY[tool.name] = tool

log("Agent", f"tool registry loaded: {list(TOOL_REGISTRY.keys())}")