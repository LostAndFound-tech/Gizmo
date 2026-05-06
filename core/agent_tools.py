"""
core/agent_tools.py
Tool registry. Separate from the membrane so tools can be
registered, hot-swapped, and eventually managed by the tool forge
without touching agent.py.

v1: manual registration of the tools that survived the rebuild.
Future: tool forge registers tools here dynamically.
"""

from core.log import log

# Import surviving tools
# (others deleted — switch_host and correction_tool are temporary
#  until Archivist and Id absorb them)
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

# Build registry — skip any tools that failed to load
TOOL_REGISTRY: dict = {}

for tool in [_switch, _correction]:
    if tool is not None:
        TOOL_REGISTRY[tool.name] = tool

log("Agent", f"tool registry loaded: {list(TOOL_REGISTRY.keys())}")
