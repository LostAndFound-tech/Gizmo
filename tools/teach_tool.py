"""
tools/teach_tool.py
The Teaching Session — collaborative, iterative tool editing from inside the bot.

Gizmo opens any registered tool (core or generated), reads it back in plain
English, then works through it with you round by round:
  - One question at a time
  - Shows what changed after each answer
  - Tries the tool live mid-lesson (sandbox — nothing saves until you say so)
  - Takes your critique, interprets it intelligently into spec changes, tries again
  - Marks things settled when you're happy
  - Saves only when you say so

Trigger phrases:
  "let's work on [tool]"              → open existing tool for editing
  "let's teach you to [thing]"        → start a new tool from scratch
  "what tools do you have?"           → list all registered tools
  "resume the lesson"                 → pick up a WIP lesson
  "try it" / "give it a go"           → run the current draft live
  "too stiff" / "needs more detail"   → critique → spec update → retry
  "that's it" / "perfect"             → settle current behavior, move on
  "save this" / "save it"             → commit and write the file
  "save as WIP" / "we'll come back"   → save checkpoint, close for now

The lesson lives across turns in lesson_state.py.
Only one active lesson per session at a time.
The sandbox runs the behavior spec against the live LLM but never touches
the saved file until save_lesson is called.
"""

import re
import ast
import sys
import inspect
import importlib.util
from pathlib import Path
from typing import Optional

from tools.base_tool import BaseTool, ToolResult

# Directories
_TOOLS_DIR = Path(__file__).parent
_GENERATED_DIR = _TOOLS_DIR / "generated"

# Fields we teach through, in order
_TEACH_FIELDS = [
    "trigger_phrases",   # When should this tool be called?
    "description",       # What does it do, in plain English?
    "behavior",          # How does it behave / what does it return?
    "live_run",          # Actually try it — react and refine
]

_FIELD_QUESTIONS = {
    "trigger_phrases": (
        "What should make me reach for this tool? "
        "Give me phrases, situations, or vibes — "
        "the more specific, the less I'll misfire."
    ),
    "description": (
        "How would you describe what this tool does in one or two sentences? "
        "I'll use this to decide whether to invoke it."
    ),
    "behavior": (
        "When I actually run this — what should happen? "
        "What do I generate, fetch, or return? "
        "Be as specific as you want."
    ),
    "live_run": (
        "Alright — let me try it. I'll run the current behavior spec right now "
        "and you tell me how I did. React however feels natural: "
        "'too stiff', 'needs more detail', 'that's exactly it' — "
        "I'll interpret your reaction and adjust."
    ),
}


# ── Live sandbox executor ─────────────────────────────────────────────────────

async def _run_sandbox(behavior: str, prompt: str = "", **kwargs) -> str:
    """
    Execute the current behavior spec against the live LLM without touching
    any saved file. This is the lesson's sandbox.

    behavior: the working behavior spec string
    prompt:   optional user-supplied prompt to feed in (e.g. "tell me a story about a fox")
    """
    from core.llm import llm as _llm

    system = (
        "You are Gizmo, running in tool sandbox mode. "
        "Execute the following behavior spec exactly as described. "
        "Do not explain what you're doing — just do it.\n\n"
        f"BEHAVIOR SPEC:\n{behavior}"
    )

    user_content = prompt.strip() if prompt.strip() else "Run the behavior spec now."

    try:
        result = await _llm.generate(
            [{"role": "user", "content": user_content}],
            system_prompt=system,
        )
        return result
    except Exception as e:
        return f"[Sandbox error: {e}]"


# ── Smart critique interpreter ─────────────────────────────────────────────────

async def _interpret_critique(
    behavior: str,
    last_output: str,
    critique: str,
) -> dict:
    """
    Given:
      - the current behavior spec
      - the output Gizmo just produced
      - the user's critique (can be vague: "too stiff", "needs more punch")

    Returns:
      {
        "interpretation": "what the critique means in concrete terms",
        "changes": ["specific change 1", "specific change 2", ...],
        "new_behavior": "the full updated behavior spec"
      }
    """
    from core.llm import llm as _llm

    prompt = f"""You are refining a behavior spec for an AI tool based on user feedback.

CURRENT BEHAVIOR SPEC:
{behavior}

OUTPUT THAT WAS PRODUCED:
{last_output}

USER'S CRITIQUE:
{critique}

Your job:
1. Interpret what the critique means concretely — even if it's vague ("too stiff" = overly formal language, no personality, doesn't sound like Gizmo)
2. Identify the specific changes needed to the behavior spec
3. Rewrite the full behavior spec incorporating those changes

Respond ONLY with valid JSON in this exact format (no markdown fences):
{{
  "interpretation": "what the critique means in plain terms",
  "changes": ["change 1", "change 2"],
  "new_behavior": "the complete rewritten behavior spec"
}}"""

    try:
        raw = await _llm.generate([{"role": "user", "content": prompt}])
        # Strip any accidental fences
        raw = re.sub(r"```json\s*|```", "", raw).strip()
        import json
        result = json.loads(raw)
        return result
    except Exception as e:
        # Fallback: treat critique as direct spec addition
        return {
            "interpretation": critique,
            "changes": [f"Incorporate: {critique}"],
            "new_behavior": behavior + f"\n\nAdditional note: {critique}",
        }


def _locate_tool_file(tool_name: str) -> Optional[Path]:
    """Find the .py file for a registered tool by name."""
    # Check generated first
    gen_path = _GENERATED_DIR / f"{tool_name}.py"
    if gen_path.exists():
        return gen_path
    # Check tools dir
    for path in _TOOLS_DIR.glob("*.py"):
        if path.stem in ("__init__", "base_tool", "lesson_state", "teach_tool", "save_lesson_tool"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseTool)
                    and obj is not BaseTool
                ):
                    instance = obj()
                    if instance.name == tool_name:
                        return path
        except Exception:
            continue
    return None


def _read_tool_source(path: Path) -> dict:
    """
    Parse a tool file and extract its name, description, trigger phrases,
    behavior notes, and run() source.
    """
    source = path.read_text(encoding="utf-8")
    result = {
        "raw_source": source,
        "description": "",
        "trigger_phrases": [],
        "behavior": "",
        "run_source": "",
    }

    # Extract description string
    desc_match = re.search(
        r'def description.*?return\s*\(\s*(.*?)\s*\)',
        source,
        re.DOTALL,
    )
    if desc_match:
        raw = desc_match.group(1).replace('"\n            "', " ").replace('"', "").strip()
        result["description"] = raw

    # Extract trigger phrases from description
    triggers = re.findall(r"'([^']+)'", result["description"])
    result["trigger_phrases"] = triggers[:12]  # cap at 12

    # Extract run() source
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == "run":
                lines = source.splitlines()
                run_lines = lines[node.lineno - 1 : node.end_lineno]
                result["run_source"] = "\n".join(run_lines)
                # Infer behavior from run body
                body_text = "\n".join(run_lines[1:])
                if "llm" in body_text.lower() or "generate" in body_text.lower():
                    prompt_match = re.search(r'prompt\s*=\s*["\'](.+?)["\']', body_text, re.DOTALL)
                    result["behavior"] = prompt_match.group(1)[:300] if prompt_match else "Calls the LLM to generate a response."
                else:
                    result["behavior"] = "Returns a direct result without calling the LLM."
    except Exception:
        pass

    return result


def _all_registered_tools() -> list[dict]:
    """Return all tools currently in TOOL_REGISTRY with their metadata."""
    try:
        from core.agent import TOOL_REGISTRY
        tools = []
        for name, tool in TOOL_REGISTRY.items():
            path = _locate_tool_file(name)
            tools.append({
                "name": name,
                "description_snippet": tool.description[:80] + "...",
                "file": str(path) if path else "(built-in / unknown)",
                "editable": path is not None,
            })
        return tools
    except Exception as e:
        return [{"error": str(e)}]


def _render_lesson_state(lesson: dict) -> str:
    """Plain-English summary of where the lesson is right now."""
    settled = lesson.get("settled", [])
    rounds = len(lesson.get("rounds", []))
    lines = [
        f"📖 Working on: **{lesson['tool_name']}**",
        f"   Rounds so far: {rounds}",
        f"   Settled: {', '.join(settled) if settled else 'nothing yet'}",
        "",
        f"   Current description:",
        f"   \"{lesson['description'][:120]}{'...' if len(lesson['description']) > 120 else ''}\"",
        "",
        f"   Current behavior:",
        f"   \"{lesson['behavior'][:120]}{'...' if len(lesson['behavior']) > 120 else ''}\"",
    ]
    return "\n".join(lines)


def _next_question(lesson: dict) -> tuple[str, str]:
    """Return (field, question) for the next unsettled teaching field."""
    settled = lesson.get("settled", [])
    for field in _TEACH_FIELDS:
        if field not in settled:
            return field, _FIELD_QUESTIONS[field]
    return "done", "Everything looks settled. Want to save this, or keep refining?"


class TeachTool(BaseTool):

    @property
    def name(self) -> str:
        return "teach"

    @property
    def description(self) -> str:
        return (
            "Opens a collaborative teaching session to build or refine any registered tool. "
            "TRIGGERS: 'let's work on [tool]', 'teach you to [thing]', "
            "'let's improve [tool]', 'open [tool] for editing', "
            "'what tools do you have', 'show me your tools', "
            "'resume the lesson', 'pick up where we left off'. "
            "Works on ALL tools — core and generated. "
            "The lesson runs round by round: one question at a time, "
            "shows changes, tries things, asks how it went. "
            "Nothing saves until you say 'save this' or 'save as WIP'. "
            "Args: "
            "action (str) — 'open', 'list', 'resume', 'answer', 'settle', 'status'. "
            "tool_name (str, optional) — which tool to open. "
            "answer (str, optional) — your answer to the current question. "
            "session_id (str) — current session."
        )

    async def run(
        self,
        action: str = "status",
        tool_name: str = "",
        answer: str = "",
        session_id: str = "",
        **kwargs,
    ) -> ToolResult:

        from tools.lesson_state import (
            start_lesson, get_lesson, update_lesson,
            add_round, settle_field, list_open_lessons,
        )

        # ── LIST ──────────────────────────────────────────────────────────────
        if action == "list":
            tools = _all_registered_tools()
            if not tools or "error" in tools[0]:
                return ToolResult(success=False, output="Couldn't read the tool registry.")
            lines = ["Here are all my registered tools:\n"]
            for t in tools:
                editable = "✏️" if t.get("editable") else "⚠️ (no file found)"
                lines.append(f"  {editable} **{t['name']}** — {t['description_snippet']}")
            lines.append(
                "\nSay 'let's work on [name]' to open one, "
                "or 'let's teach you to [thing]' to build something new."
            )
            return ToolResult(success=True, output="\n".join(lines))

        # ── RESUME ────────────────────────────────────────────────────────────
        if action == "resume":
            lesson = get_lesson(session_id)
            if lesson:
                field, question = _next_question(lesson)
                summary = _render_lesson_state(lesson)
                return ToolResult(
                    success=True,
                    output=(
                        f"Picking up where we left off.\n\n{summary}\n\n"
                        f"Next up — {field}:\n{question}"
                    ),
                    data={"lesson": lesson, "next_field": field},
                )
            open_lessons = list_open_lessons()
            if open_lessons:
                names = ", ".join(l["tool_name"] for l in open_lessons)
                return ToolResult(
                    success=True,
                    output=f"I have open WIP lessons for: {names}. Which one should we pick up?",
                    data={"open_lessons": open_lessons},
                )
            return ToolResult(success=True, output="No open lessons right now. Want to start one?")

        # ── STATUS ────────────────────────────────────────────────────────────
        if action == "status":
            lesson = get_lesson(session_id)
            if not lesson:
                return ToolResult(
                    success=True,
                    output=(
                        "No active lesson right now. "
                        "Say 'what tools do you have' to browse, "
                        "or 'let's work on [tool]' to start."
                    ),
                )
            summary = _render_lesson_state(lesson)
            field, question = _next_question(lesson)
            return ToolResult(
                success=True,
                output=f"{summary}\n\nWhere we are: {field}\n{question}",
                data={"lesson": lesson},
            )

        # ── OPEN ──────────────────────────────────────────────────────────────
        if action == "open":
            if not tool_name:
                return ToolResult(
                    success=False,
                    output="Which tool do you want to work on? Tell me the name.",
                )

            safe_name = re.sub(r"[^\w]", "_", tool_name.strip().lower())

            # Check if already in a lesson
            existing = get_lesson(session_id)
            if existing and existing["status"] == "active":
                return ToolResult(
                    success=False,
                    output=(
                        f"We're already mid-lesson on '{existing['tool_name']}'. "
                        f"Say 'save as WIP' to set it aside, or 'resume' to keep going."
                    ),
                )

            # Try to find the tool
            path = _locate_tool_file(safe_name)

            if path:
                # Existing tool — read it
                parsed = _read_tool_source(path)
                lesson = start_lesson(
                    session_id=session_id,
                    tool_name=safe_name,
                    source_file=str(path),
                    description=parsed["description"],
                    behavior=parsed["behavior"],
                    trigger_phrases=parsed["trigger_phrases"],
                    run_source=parsed["run_source"],
                    started_by=kwargs.get("current_host", ""),
                )
                summary = _render_lesson_state(lesson)
                field, question = _next_question(lesson)
                return ToolResult(
                    success=True,
                    output=(
                        f"Opened **{safe_name}**. Here's where it stands:\n\n"
                        f"{summary}\n\n"
                        f"Let's start with — {field}:\n{question}"
                    ),
                    data={"lesson": lesson, "next_field": field},
                )
            else:
                # New tool — start from scratch
                lesson = start_lesson(
                    session_id=session_id,
                    tool_name=safe_name,
                    source_file=None,
                    description="",
                    behavior="",
                    trigger_phrases=[],
                    run_source="",
                    started_by=kwargs.get("current_host", ""),
                )
                return ToolResult(
                    success=True,
                    output=(
                        f"No tool called '{safe_name}' exists yet — let's build it.\n\n"
                        f"First question:\n{_FIELD_QUESTIONS['trigger_phrases']}"
                    ),
                    data={"lesson": lesson, "next_field": "trigger_phrases"},
                )

        # ── TRY ───────────────────────────────────────────────────────────────
        if action == "try":
            lesson = get_lesson(session_id)
            if not lesson:
                return ToolResult(
                    success=False,
                    output="No active lesson to try. Start one with 'let's work on [tool]'.",
                )
            if not lesson.get("behavior"):
                return ToolResult(
                    success=False,
                    output=(
                        "No behavior spec yet — let's define what the tool should do first, "
                        "then I can try it."
                    ),
                )

            # Use supplied prompt or generate a natural one from the tool name
            run_prompt = answer.strip() if answer.strip() else ""
            if not run_prompt:
                tool_name = lesson["tool_name"].replace("_", " ")
                run_prompt = f"Please {tool_name} for me."

            output = await _run_sandbox(lesson["behavior"], prompt=run_prompt)

            # Store last output in lesson for critique to reference
            lesson["last_sandbox_output"] = output
            lesson["last_sandbox_prompt"] = run_prompt
            update_lesson(session_id, lesson)

            return ToolResult(
                success=True,
                output=(
                    f"[Sandbox run — not saved yet]\n\n"
                    f"{output}\n\n"
                    f"---\n"
                    f"How was that? Tell me what you think — "
                    f"'too stiff', 'needs more detail', 'the ending fell flat', anything. "
                    f"I'll interpret it and adjust the spec. "
                    f"Or say 'that's it' if it's there."
                ),
                data={
                    "sandbox_output": output,
                    "sandbox_prompt": run_prompt,
                    "behavior_used": lesson["behavior"],
                },
            )

        # ── CRITIQUE ──────────────────────────────────────────────────────────
        if action == "critique":
            lesson = get_lesson(session_id)
            if not lesson:
                return ToolResult(
                    success=False,
                    output="No active lesson to critique.",
                )
            if not answer.strip():
                return ToolResult(success=False, output="What's your critique?")

            last_output = lesson.get("last_sandbox_output", "")
            if not last_output:
                return ToolResult(
                    success=False,
                    output=(
                        "I haven't run the tool yet this lesson — "
                        "say 'try it' first so I have something to react to."
                    ),
                )

            # Interpret the critique intelligently
            interpreted = await _interpret_critique(
                behavior=lesson["behavior"],
                last_output=last_output,
                critique=answer.strip(),
            )

            old_behavior = lesson["behavior"]
            new_behavior = interpreted.get("new_behavior", old_behavior)
            interpretation = interpreted.get("interpretation", answer)
            changes = interpreted.get("changes", [])

            # Update the lesson spec
            lesson["behavior"] = new_behavior
            lesson.setdefault("critique_history", []).append({
                "critique": answer.strip(),
                "interpretation": interpretation,
                "changes": changes,
                "old_behavior": old_behavior,
                "new_behavior": new_behavior,
            })
            update_lesson(session_id, lesson)
            add_round(session_id, f"critique: {answer.strip()}", answer, {"behavior": new_behavior})

            change_lines = "\n".join(f"  • {c}" for c in changes) if changes else "  (no discrete changes identified)"

            # Auto-run with the new spec so they see the result immediately
            last_prompt = lesson.get("last_sandbox_prompt", "")
            new_output = await _run_sandbox(new_behavior, prompt=last_prompt)
            lesson["last_sandbox_output"] = new_output
            update_lesson(session_id, lesson)

            return ToolResult(
                success=True,
                output=(
                    f"Got it. I heard: \"{interpretation}\"\n\n"
                    f"Changes made to behavior spec:\n{change_lines}\n\n"
                    f"---\n"
                    f"[New sandbox run]\n\n"
                    f"{new_output}\n\n"
                    f"---\n"
                    f"Better? Worse? Keep reacting — or say 'that's it' to settle this "
                    f"and move on, or 'save this' when you're done."
                ),
                data={
                    "interpretation": interpretation,
                    "changes": changes,
                    "new_behavior": new_behavior,
                    "new_output": new_output,
                },
            )

        # ── ANSWER ────────────────────────────────────────────────────────────
        if action == "answer":
            lesson = get_lesson(session_id)
            if not lesson:
                return ToolResult(
                    success=False,
                    output="No active lesson. Say 'let's work on [tool]' to start one.",
                )
            if not answer.strip():
                return ToolResult(success=False, output="What's your answer?")

            field, _ = _next_question(lesson)
            changed = {}

            if field == "trigger_phrases":
                phrases = [p.strip().strip("'\"") for p in re.split(r"[,\n]+", answer) if p.strip()]
                lesson["trigger_phrases"] = phrases
                changed["trigger_phrases"] = phrases
                if lesson["description"]:
                    trigger_str = ", ".join(f"'{p}'" for p in phrases[:6])
                    lesson["description"] = re.sub(
                        r"TRIGGERS?:.*?(?=\.|Args:|$)",
                        f"TRIGGERS: {trigger_str}. ",
                        lesson["description"],
                        flags=re.DOTALL,
                    )
                    if "TRIGGER" not in lesson["description"]:
                        trigger_str = ", ".join(f"'{p}'" for p in phrases[:6])
                        lesson["description"] = f"TRIGGERS: {trigger_str}. " + lesson["description"]

            elif field == "description":
                lesson["description"] = answer.strip()
                changed["description"] = answer.strip()

            elif field == "behavior":
                lesson["behavior"] = answer.strip()
                changed["behavior"] = answer.strip()

            elif field == "live_run":
                # They're reacting to the live run field prompt —
                # treat their answer as a critique if we have prior output,
                # or as a prompt to run if we don't
                if lesson.get("last_sandbox_output"):
                    # Treat as critique
                    interpreted = await _interpret_critique(
                        behavior=lesson["behavior"],
                        last_output=lesson["last_sandbox_output"],
                        critique=answer.strip(),
                    )
                    lesson["behavior"] = interpreted.get("new_behavior", lesson["behavior"])
                    changed["behavior"] = lesson["behavior"]
                    changed["via_critique"] = interpreted.get("interpretation", "")
                else:
                    # No prior output — run it first
                    output = await _run_sandbox(lesson["behavior"], prompt=answer.strip())
                    lesson["last_sandbox_output"] = output
                    lesson["last_sandbox_prompt"] = answer.strip()
                    changed["sandbox_run"] = True
                    update_lesson(session_id, lesson)
                    add_round(session_id, "live_run", answer, changed)
                    return ToolResult(
                        success=True,
                        output=(
                            f"[Sandbox run]\n\n{output}\n\n---\n"
                            f"How was that? React freely."
                        ),
                        data={"sandbox_output": output},
                    )

            update_lesson(session_id, lesson)
            add_round(session_id, _FIELD_QUESTIONS.get(field, field), answer, changed)

            change_lines = []
            for k, v in changed.items():
                if isinstance(v, list):
                    change_lines.append(f"  • {k}: {', '.join(str(x) for x in v)}")
                elif k == "via_critique":
                    change_lines.append(f"  • interpreted as: {v}")
                else:
                    snippet = str(v)[:100] + ("..." if len(str(v)) > 100 else "")
                    change_lines.append(f"  • {k}: \"{snippet}\"")

            change_summary = "Updated:\n" + "\n".join(change_lines) if change_lines else ""
            next_field, next_question = _next_question(lesson)

            if next_field == "live_run" and lesson.get("behavior"):
                # Auto-run as we enter the live_run phase
                output = await _run_sandbox(lesson["behavior"])
                lesson["last_sandbox_output"] = output
                lesson["last_sandbox_prompt"] = ""
                update_lesson(session_id, lesson)
                return ToolResult(
                    success=True,
                    output=(
                        f"{change_summary}\n\n"
                        f"Alright — let me try it right now.\n\n"
                        f"[Sandbox run]\n\n{output}\n\n---\n"
                        f"How was that? React however feels natural — "
                        f"I'll interpret your feedback and adjust."
                    ),
                    data={"lesson": lesson, "next_field": "live_run", "sandbox_output": output},
                )

            if next_field == "done":
                return ToolResult(
                    success=True,
                    output=(
                        f"{change_summary}\n\n"
                        f"That covers everything. Here's the full picture:\n\n"
                        f"{_render_lesson_state(lesson)}\n\n"
                        f"Say 'save this' when you're ready, or keep refining anything."
                    ),
                    data={"lesson": lesson, "next_field": "done"},
                )

            return ToolResult(
                success=True,
                output=(
                    f"{change_summary}\n\n"
                    f"Good. Next — {next_field}:\n{next_question}"
                ),
                data={"lesson": lesson, "next_field": next_field},
            )

        # ── SETTLE ────────────────────────────────────────────────────────────
        if action == "settle":
            lesson = get_lesson(session_id)
            if not lesson:
                return ToolResult(success=False, output="No active lesson.")
            field, _ = _next_question(lesson)
            settle_field(session_id, field)
            next_field, next_question = _next_question(lesson)
            return ToolResult(
                success=True,
                output=f"Marked '{field}' as settled. Next:\n{next_question}",
                data={"settled_field": field, "next_field": next_field},
            )

        return ToolResult(
            success=False,
            output=(
                f"Unknown action '{action}'. "
                f"Try: 'list', 'open', 'answer', 'resume', 'settle', 'status'."
            ),
        )
