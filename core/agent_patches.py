"""
AGENT.PY PATCHES
Add these in the order shown.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. IMPORTS — add after existing tool imports
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from tools.teach_tool import TeachTool
from tools.save_lesson_tool import SaveLessonTool
from tools.tool_forge import ToolForgeTool

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. TOOL_REGISTRY — add the three new tools
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

TOOL_REGISTRY: dict[str, BaseTool] = {
    tool.name: tool
    for tool in [
        EchoTool(),
        SwitchHostTool(),
        CorrectionTool(),
        TeachTool(),
        SaveLessonTool(),
        ToolForgeTool(),
        # generated tools auto-loaded below
    ]
}

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. AUTO-DISCOVERY — paste after TOOL_REGISTRY definition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys as _sys
import importlib.util as _importlib_util
from pathlib import Path as _Path

_GENERATED_DIR = _Path(__file__).parent.parent / "tools" / "generated"

def _load_generated_tools() -> None:
    if not _GENERATED_DIR.exists():
        return
    for filepath in sorted(_GENERATED_DIR.glob("*.py")):
        if filepath.name.startswith("_"):
            continue
        module_name = f"tools.generated.{filepath.stem}"
        try:
            spec = _importlib_util.spec_from_file_location(module_name, filepath)
            module = _importlib_util.module_from_spec(spec)
            _sys.modules[module_name] = module
            spec.loader.exec_module(module)
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseTool)
                    and obj is not BaseTool
                ):
                    instance = obj()
                    if instance.name not in TOOL_REGISTRY:
                        TOOL_REGISTRY[instance.name] = instance
                        print(f"[ToolForge] Loaded: {instance.name}")
        except Exception as e:
            print(f"[ToolForge] Failed to load {filepath.name}: {e}")

_load_generated_tools()

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. ONE-SHOT TOOLS — update the if-check in the agent loop
   Find the existing: if tool_name in ("switch_host", "log_correction", ...)
   Replace with this expanded version.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ONE_SHOT_TOOLS = {
    "switch_host",
    "log_correction",
    "alter_wheel",
    "teach",        # every teach action streams its result directly
    "save_lesson",  # commit is a one-shot, always respond immediately after
    "tool_forge",   # forge is also one-shot
}

# Replace the existing one-shot check in the agent loop with:
if tool_name in ONE_SHOT_TOOLS:
    print(f"[Agent] One-shot tool '{tool_name}' completed, generating response")
    working_messages[-1] = {
        "role": "user",
        "content": user_message + injected_results,
    }
    final_response = await llm.generate(
        working_messages,
        system_prompt=system_prompt,
    )
    history.add("user", user_message, context=context)
    history.add("assistant", final_response, context=context)
    final_response = self._strip_tool_calls(final_response)
    for chunk in self._chunk_string(final_response):
        yield chunk
    return

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. SYSTEM PROMPT ADDITIONS
   Add to the bottom of the system prompt string (near the
   "Use the switch_host tool whenever..." lines):

   "Use the teach tool when someone wants to work on, improve, or
   build a tool together. Teaching is iterative — stay in the lesson
   until they say save.
   When in an active lesson, map natural language to teach actions:
     'try it' / 'give it a go' → action='try'
     'too stiff' / any critique of output → action='critique', answer=<their words>
     'that's it' / 'perfect' / 'settle this' → action='settle'
     any answer to a lesson question → action='answer', answer=<their words>
     'save this' / 'we're done' → call save_lesson
     'save as WIP' / 'we'll come back' → call save_lesson with is_wip=True
   Do not exit a lesson mid-flow without saving. Do not call save_lesson
   unless the user explicitly asks to save."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
