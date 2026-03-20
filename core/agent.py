"""
core/agent.py
Tool registry and agent loop with structured context assembly.

Flow per request:
  1. Cold-start check — if personality_core is empty, run onboarding first
  2. Detect host/fronter changes from previous turn
  3. Multi-collection RAG retrieval + synthesis (memory)
  4. Personality retrieval — contextually relevant chunks from personality collections
  5. Assemble full system prompt
  6. LLM generates, checking for tool calls
  7. Stream final answer, save to history
"""

import os
import json
import re
from typing import AsyncGenerator, Optional

from core.llm import llm
from core.rag import rag
from core.synthesis import retrieve_and_synthesize
from core.wellness import detect_distress, log_wellness_event, build_checkin_prompt
from memory.history import ConversationHistory
from tools.base_tool import BaseTool
from tools.switch_host import SwitchHostTool
from tools.correction_tool import CorrectionTool
from tools.reset_personality_tool import ResetPersonalityTool
from tools.search_tool import SearchTool

# ── Tool Registry ─────────────────────────────────────────────────────────────
TOOL_REGISTRY: dict[str, BaseTool] = {
    tool.name: tool
    for tool in [
        SwitchHostTool(),
        CorrectionTool(),
        ResetPersonalityTool(),
        SearchTool(),
    ]
}

# ── Host change tracking ──────────────────────────────────────────────────────
_last_context: dict[str, dict] = {}  # session_id -> last context dict


def _detect_changes(session_id: str, context: Optional[dict]) -> dict:
    """
    Compare current context to last known context for this session.
    Returns a dict of what changed.
    """
    if not context or not session_id:
        return {}

    last = _last_context.get(session_id, {})
    changes = {}

    current_host = context.get("current_host", "")
    last_host = last.get("current_host", "")
    if current_host and current_host != last_host and last_host:
        changes["host_changed"] = True
        changes["previous_host"] = last_host

    current_fronters = set(
        f.lower() if isinstance(f, str) else str(f).lower()
        for f in (context.get("fronters") or [])
    )
    last_fronters = set(
        f.lower() if isinstance(f, str) else str(f).lower()
        for f in (last.get("fronters") or [])
    )
    joined = current_fronters - last_fronters
    left = last_fronters - current_fronters
    if joined:
        changes["fronters_joined"] = list(joined)
    if left:
        changes["fronters_left"] = list(left)

    # Save current as last
    _last_context[session_id] = dict(context)
    return changes


# ── Headmate name detection ───────────────────────────────────────────────────

def _detect_mentioned_headmates(
    message: str,
    current_host: Optional[str],
    fronters: list,
) -> list[str]:
    """
    Scan the message for known headmate names (existing collections).
    Returns a list of names found that aren't already in fronters.
    """
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        known = {c.name.lower() for c in client.list_collections()}
        # Remove non-headmate collections
        from core.personality_growth import PERSONALITY_COLLECTIONS
        known -= {"main"} | {c.lower() for c in PERSONALITY_COLLECTIONS}
    except Exception:
        return []

    already = {f.lower() for f in fronters if f}
    if current_host:
        already.add(current_host.lower())

    message_lower = message.lower()
    mentioned = []
    for name in known:
        if name not in already and name in message_lower:
            mentioned.append(name)

    if mentioned:
        print(f"[Agent] Detected mentioned headmates: {mentioned}")

    return mentioned


# ── Personality ───────────────────────────────────────────────────────────────

async def _get_personality(query: str, current_host: Optional[str] = None) -> str:
    """
    Retrieve contextually relevant personality chunks from the five personality
    collections. Falls back to a minimal stub on cold start or failure.
    """
    try:
        from core.personality_growth import retrieve_personality, is_cold_start
        if is_cold_start():
            return "You are a new companion, just starting to find your voice. Be warm, curious, and present."
        return await retrieve_personality(query, current_host=current_host)
    except Exception as e:
        print(f"[Agent] Personality retrieval failed: {e}")
        return "You are a helpful, caring companion."


# ── System Prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(
    tools: dict[str, BaseTool],
    personality: str = "",
    hard_rules: str = "",
    rag_synthesis: str = "",
    overview: str = "",
    context: Optional[dict] = None,
    changes: Optional[dict] = None,
) -> str:
    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in tools.values()
    )

    # Hard rules go first — above personality, above everything.
    # The LLM must see these before it reads anything else.
    hard_rules_block = f"{hard_rules}\n\n" if hard_rules else ""

    overview_block = f"\n\n[Conversation so far]\n{overview}" if overview else ""
    rag_block = f"\n\n[Relevant knowledge]\n{rag_synthesis}" if rag_synthesis else ""

    # Situational context
    context_block = ""
    if context:
        lines = []
        for k, v in context.items():
            if k == "debug":
                continue
            lines.append(f"  {k}: {v}")
        if lines:
            context_block = "\n\n[Current situation]\n" + "\n".join(lines)

    # Host/fronter change signals
    change_block = ""
    if changes:
        change_lines = []
        if changes.get("host_changed"):
            change_lines.append(
                f"  The host just changed from {changes['previous_host']} "
                f"to {context.get('current_host', 'someone new')}. "
                f"You are now speaking with a different person — acknowledge this naturally."
            )
        if changes.get("fronters_joined"):
            change_lines.append(f"  Joined the front: {', '.join(changes['fronters_joined'])}")
        if changes.get("fronters_left"):
            change_lines.append(f"  Left the front: {', '.join(changes['fronters_left'])}")
        if change_lines:
            change_block = "\n\n[System changes]\n" + "\n".join(change_lines)

    return f"""{hard_rules_block}{personality}

Available tools:
{tool_descriptions}

To use a tool, respond with ONLY this JSON format (no extra text):
{{"tool": "tool_name", "args": {{"arg1": "value1"}}}}

After receiving a tool result, continue reasoning and provide a final response.
If no tool is needed, respond directly.{overview_block}{rag_block}{context_block}{change_block}

The person in "current_host" is who you are speaking WITH right now — address them as "you" directly.
Be concise. Be accurate. When uncertain, say so.

TOOL TRIGGERS — call the named tool when you hear these:
- switch_host      → "I'm fronting", "X is here", "switching to X", "X stepped back", "just me now"
- log_correction   → "don't do that", "stop doing that", "that's wrong", "never again", "you keep doing this"
- reset_personality → "reset your personality", "factory reset", "start over from scratch", "wipe yourself"
- web_search       → "look this up", "search for", "what's the latest", "check online", "find out"
                     OR any question about current events, prices, news, or time-sensitive facts

When using log_correction: summarize what was wrong and what rule you're committing to before calling it.
When using reset_personality: confirm intent explicitly before calling — this is irreversible.
When using web_search for disambiguation: surface the clarifying question directly to the user.

CRITICAL — KNOWLEDGE BASE RULES:
- The [Relevant knowledge] block is your memory. It is ground truth.
- If [Relevant knowledge] contains an answer, USE IT. Do not say "I don't know" or "we haven't discussed that."
- Do not contradict it. Do not invent details beyond it.
- If [Relevant knowledge] is empty, then you genuinely have no memory of it — say so.
- Never answer from your own training data when [Relevant knowledge] is provided.
- If you don't know something and it's not in memory, say so plainly and offer to look it up.
  "I'm not sure — want me to search for that?" is always better than a confident guess.

CRITICAL — OUTPUT FORMAT:
- Never show reasoning, thought process, or internal notes in your response.
- Never annotate your own tone or intent — no *(warmly)*, **(curious)**, *(playful)*, or any variant.
- Never use parenthetical meta-commentary about how or why you're responding.
- Just respond. The response is the only output."""


# ── Onboarding state ──────────────────────────────────────────────────────────
# Tracks in-progress onboarding conversations per session
_onboarding_sessions: dict[str, list[dict]] = {}


async def _handle_onboarding(
    user_message: str,
    session_id: str,
    history: ConversationHistory,
) -> tuple[bool, str]:
    """
    Manages the onboarding conversation flow.
    Returns (still_onboarding, response_text).
    When onboarding completes, seeds personality and returns (False, final_message).
    """
    from core.personality_growth import (
        run_onboarding,
        continue_onboarding,
        seed_personality_from_onboarding,
    )

    # First message in onboarding — get the opening question
    if session_id not in _onboarding_sessions:
        opening = await run_onboarding(llm)
        _onboarding_sessions[session_id] = [
            {"role": "assistant", "content": opening}
        ]
        return True, opening

    # Append user's response to the onboarding conversation
    convo = _onboarding_sessions[session_id]
    convo.append({"role": "user", "content": user_message})

    # Get next onboarding question, or None if done
    next_q = await continue_onboarding(convo, llm)

    if next_q is None:
        # Onboarding complete — seed personality
        print(f"[Agent] Onboarding complete for session {session_id[:8]}, seeding personality...")
        await seed_personality_from_onboarding(convo, llm)
        del _onboarding_sessions[session_id]
        farewell = (
            "Okay — I think I have a sense of things. Let's start. What's on your mind?"
        )
        return False, farewell

    # Still onboarding
    convo.append({"role": "assistant", "content": next_q})
    return True, next_q


# ── Search routing ────────────────────────────────────────────────────────────

# Explicit search phrases — always trigger search without the routing check
_EXPLICIT_SEARCH_TRIGGERS = re.compile(
    r"\b(look this up|look it up|search for|search this|find out|"
    r"what'?s the latest|check online|google that|can you find|"
    r"look that up|search the web|find me|pull up)\b",
    re.IGNORECASE,
)

_NO_SEARCH_TRIGGERS = re.compile(
    r"\b(without looking it up|don'?t look it up|without searching|"
    r"don'?t search|from memory|off the top of your head|"
    r"without the internet|no looking it up|no search)\b",
    re.IGNORECASE,
)

_AFFIRMATIVE = re.compile(
    r"^(yes|yea|yeah|yep|sure|go ahead|do it|please|ok|okay|"
    r"go for it|yes please|why not|absolutely|definitely|"
    r"do a search|search|look it up|yes look|yeah look|"
    r"yes search|yeah search|do the search)[.!,\s]*$",
    re.IGNORECASE,
)

# Phrases that indicate Gizmo just offered to search
_SEARCH_OFFER_PHRASES = [
    "want me to search", "want me to look", "shall i search",
    "should i look", "want me to find", "i can look that up",
    "i can search", "want me to check", "look it up?",
    "search for that?", "look that up?", "want me to look that up",
    "want me to search for", "should i search",
]


async def _needs_web_search(message: str, llm, history=None) -> bool:
    """
    Pre-generation routing check — does this message need web search?

    Priority order:
      1. Explicit no-search instruction → always False
      2. Context confirmation — user affirming after Gizmo offered to search → True
      3. Explicit search trigger phrase → True
      4. Implicit LLM check → YES/NO
    """
    # 1. Explicit no-search instruction — always honour it
    if _NO_SEARCH_TRIGGERS.search(message):
        print(f"[Agent] No-search instruction detected — skipping search")
        return False

    # 2. Context-aware confirmation — if Gizmo just offered to search
    # and the user is affirming, that's definitive regardless of phrasing
    if history is not None:
        recent = history.as_list()
        if recent:
            last_assistant = next(
                (m for m in reversed(recent) if m["role"] == "assistant"),
                None,
            )
            if last_assistant:
                last_text = last_assistant["content"].lower()
                offered = any(phrase in last_text for phrase in _SEARCH_OFFER_PHRASES)
                if offered and _AFFIRMATIVE.search(message.strip()):
                    print(f"[Agent] Search confirmation detected from context")
                    return True

    # 3. Explicit search trigger — no LLM call needed
    if _EXPLICIT_SEARCH_TRIGGERS.search(message):
        print(f"[Agent] Explicit search trigger matched")
        return True

    # 4. Implicit check — ask the LLM
    prompt = [
        {
            "role": "user",
            "content": (
                f"Does this query require current, time-sensitive, or external information "
                f"that a local knowledge base wouldn't have? "
                f"Things that need search: current events, today's news, live prices, "
                f"recent releases, who currently holds a position, weather, sports scores, "
                f"anything that changes week to week.\n\n"
                f"Query: \"{message}\"\n\n"
                f"Reply with YES or NO only."
            )
        }
    ]

    try:
        result = await llm.generate(
            prompt,
            system_prompt="You assess whether queries need web search. Reply YES or NO only.",
            max_new_tokens=5,
            temperature=0.0,
        )
        answer = result.strip().upper()
        needs = answer.startswith("YES")
        if needs:
            print(f"[Agent] Implicit search trigger: LLM said YES for '{message[:60]}'")
        return needs
    except Exception as e:
        print(f"[Agent] _needs_web_search() failed: {e}")
        return False  # fail open — don't block if check errors


# ── Agent Loop ────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, max_tool_calls: int = 3):
        self.max_tool_calls = max_tool_calls

    async def run(
        self,
        user_message: str,
        history: ConversationHistory,
        session_id: str = "",
        use_rag: bool = True,
        context: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Main agent entry point. Yields response tokens for streaming.
        """
        # 0. Cold-start / onboarding check
        from core.personality_growth import is_cold_start
        if is_cold_start() or session_id in _onboarding_sessions:
            still_onboarding, response = await _handle_onboarding(
                user_message, session_id, history
            )
            history.add("user", user_message, context=context)
            history.add("assistant", response, context=context)
            for chunk in self._chunk_string(response):
                yield chunk
            # After onboarding completes, continue into normal flow on next turn
            return

        # 1. Detect host/fronter changes
        changes = _detect_changes(session_id, context)

        current_host = (context or {}).get("current_host")
        fronters = list((context or {}).get("fronters") or [])

        # 2. Retrieve personality chunks (concurrent with RAG below)
        # 3. Combined RAG retrieval + synthesis
        # Run both concurrently — no dependency between them
        import asyncio as _asyncio

        mentioned = _detect_mentioned_headmates(user_message, current_host, fronters)
        all_fronters = list({f for f in fronters + mentioned if f})

        history_summary = None
        recent = history.as_list()[-6:]
        if len(recent) >= 4:
            history_summary = " | ".join(
                f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content'][:80]}"
                for m in recent
            )

        personality_task = _asyncio.ensure_future(
            _get_personality(user_message, current_host=current_host)
        )
        rag_task = _asyncio.ensure_future(
            retrieve_and_synthesize(
                query=user_message,
                current_host=current_host,
                fronters=all_fronters,
                history_summary=history_summary,
                llm=llm,
            ) if use_rag else _asyncio.coroutine(lambda: "")()
        )

        personality, rag_synthesis = await _asyncio.gather(
            personality_task, rag_task, return_exceptions=True
        )

        if isinstance(personality, Exception):
            print(f"[Agent] Personality retrieval error: {personality}")
            personality = "You are a helpful, caring companion."
        if isinstance(rag_synthesis, Exception):
            print(f"[Agent] RAG synthesis error: {rag_synthesis}")
            rag_synthesis = ""

        # Hard rules — always fetched fresh, placed above everything in prompt
        hard_rules = ""
        try:
            from core.personality_growth import get_hard_rules
            hard_rules = get_hard_rules()
        except Exception as e:
            print(f"[Agent] get_hard_rules() failed: {e}")

        # ── Search routing check ──────────────────────────────────────────────
        # Before building the system prompt, check if this query needs web search.
        # Explicit triggers are caught by the tool description during generation.
        # This handles implicit triggers — time-sensitive queries the model might
        # answer from stale training data without reaching for the tool.
        # Fast zero-temperature call — runs before main generation.
        if use_rag:
            try:
                search_needed = await _needs_web_search(user_message, llm, history=history)
                if search_needed:
                    print(f"[Agent] Implicit search trigger detected — pre-fetching")
                    from tools.epistemic_synthesis import research, format_for_response
                    search_result = await research(
                        query=user_message,
                        llm=llm,
                        n_sources=3,
                        ingest=True,
                        current_host=current_host,
                    )
                    if search_result.needs_clarification:
                        # Surface clarification question immediately
                        history.add("user", user_message, context=context)
                        history.add("assistant", search_result.clarification_question, context=context)
                        for chunk in self._chunk_string(search_result.clarification_question):
                            yield chunk
                        return
                    if search_result.summary:
                        # Append web results to rag_synthesis so they land in context
                        web_block = f"\n\n[Web search results]\n{format_for_response(search_result)}"
                        rag_synthesis = (rag_synthesis or "") + web_block
                        print(f"[Agent] Web results appended to context")
            except Exception as e:
                print(f"[Agent] Search routing check failed (non-fatal): {e}")

        # 4. Build system prompt
        system_prompt = build_system_prompt(
            TOOL_REGISTRY,
            personality=personality,
            hard_rules=hard_rules,
            rag_synthesis=rag_synthesis,
            context=context,
            changes=changes,
        )

        # 5. Build messages from history
        messages = history.as_messages(user_message)

        # 6. Agentic loop
        tool_calls = 0
        injected_results = ""

        while tool_calls < self.max_tool_calls:
            working_messages = messages.copy()
            if injected_results:
                working_messages[-1] = {
                    "role": "user",
                    "content": user_message + injected_results,
                }

            response_text = await llm.generate(
                working_messages,
                system_prompt=system_prompt,
            )

            tool_call = self._parse_tool_call(response_text)

            if tool_call is None:
                # ── Compliance check ──────────────────────────────────────────
                # Check response against hard rules before streaming.
                # If a rule is violated, increment the counter and regenerate once.
                try:
                    from core.personality_growth import _get_active_rules, ingest_correction
                    active_rules = _get_active_rules()
                    if active_rules:
                        violation_check_prompt = [
                            {
                                "role": "user",
                                "content": (
                                    f"Check this response against the rules below. "
                                    f"Reply with ONLY 'ok' if no rules are violated, "
                                    f"or the violated rule text if one is broken.\n\n"
                                    f"Rules:\n"
                                    + "\n".join(f"- {r['rule']}" for r in active_rules)
                                    + f"\n\nResponse to check:\n{response_text}"
                                )
                            }
                        ]
                        check = await llm.generate(
                            violation_check_prompt,
                            system_prompt="You check responses for rule violations. Reply 'ok' or state the violated rule. Nothing else.",
                            max_new_tokens=60,
                            temperature=0.0,
                        )
                        if check.strip().lower() != "ok":
                            print(f"[Agent] Compliance violation detected: {check.strip()[:80]}")
                            # Increment violation count
                            for r in active_rules:
                                if r["rule"].lower() in check.lower() or check.lower() in r["rule"].lower():
                                    await ingest_correction(
                                        what_was_wrong=f"Violated rule in response: {check.strip()[:120]}",
                                        rule=r["rule"],
                                        who_corrected="system",
                                        session_id=session_id,
                                    )
                                    break
                            # Regenerate with explicit violation callout
                            working_messages[-1] = {
                                "role": "user",
                                "content": (
                                    user_message
                                    + f"\n\n[SYSTEM: Your previous response violated this rule: "
                                    f"{check.strip()}. Rewrite your response without violating it.]"
                                ),
                            }
                            response_text = await llm.generate(
                                working_messages,
                                system_prompt=system_prompt,
                            )
                            print(f"[Agent] Regenerated after compliance failure")
                except Exception as e:
                    print(f"[Agent] Compliance check failed (non-fatal): {e}")

                history.add("user", user_message, context=context)

                # Wellness check — detect distress, log, append check-in
                detection = detect_distress(user_message)
                print(f"[Wellness] Scanned message — detected: {detection['detected']}, categories: {detection['categories']}")
                if detection["detected"]:
                    print("[Wellness] Logging event...")
                    await log_wellness_event(
                        message=user_message,
                        detection=detection,
                        current_host=current_host,
                        fronters=fronters,
                        session_id=session_id,
                    )
                    print("[Wellness] Building check-in prompt...")
                    checkin = await build_checkin_prompt(detection, current_host, llm)
                    print(f"[Wellness] Check-in: {checkin[:80]}")
                    response_text = re.sub(
                        r'\n\n"Hey.*?$', '', response_text,
                        flags=re.DOTALL | re.MULTILINE
                    ).rstrip()
                    response_text = response_text + "\n\n" + checkin

                history.add("assistant", response_text, context=context)
                response_text = self._clean_response(response_text)
                for chunk in self._chunk_string(response_text):
                    yield chunk
                return

            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})

            if tool_name not in TOOL_REGISTRY:
                injected_results += f"\n[Tool Error: '{tool_name}' not found]\n"
                tool_calls += 1
                continue

            clean_args = {k: v for k, v in tool_args.items() if k != "session_id"}
            result = await TOOL_REGISTRY[tool_name].run(session_id=session_id, **clean_args)
            injected_results += (
                f"\n[Tool: {tool_name}]\nResult: {result.output}\n"
                f"Task complete. Now respond to the user directly without calling any more tools.\n"
            )
            tool_calls += 1

            # One-shot tools — respond immediately after first call
            if tool_name in ("switch_host", "log_correction", "alter_wheel", "reset_personality"):
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
                final_response = self._clean_response(final_response)
                for chunk in self._chunk_string(final_response):
                    yield chunk
                return

        yield "[Agent reached max tool calls]\n"
        async for token in llm.stream(messages, system_prompt=system_prompt):
            yield token

    def _parse_tool_call(self, text: str) -> dict | None:
        text = text.strip()
        try:
            data = json.loads(text)
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{.*?"tool".*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _clean_response(self, text: str) -> str:
        """
        Strip all non-response artifacts before streaming:
          - Tool call JSON blocks
          - Post-correction preambles
          - Italicised/bolded tone annotations: *(warmly)*, **(curious)**, etc.
          - Parenthetical meta-commentary: (tone: X), (intent: X), etc.
          - Bracketed internal notes: [thinking], [reasoning], etc.
          - <think>...</think> style reasoning blocks some models emit
          - Trailing whitespace and blank lines left by removals
        """
        # Tool call JSON
        text = re.sub(r'\{\s*"tool"\s*:.*?\}\s*', '', text, flags=re.DOTALL)

        # Post-correction preamble
        text = re.sub(r'\(Post-correction:\).*', '', text, flags=re.DOTALL)

        # <think> ... </think> blocks (DeepSeek and similar)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Bold+italic tone annotations: **(curious, playful)**, ***(warm)***, etc.
        text = re.sub(r'\*{1,3}\([^)]{1,80}\)\*{0,3}', '', text)
        text = re.sub(r'\*{2,3}[^*]{1,80}\*{2,3}', '', text)

        # Italic tone annotations: *(warmly)*, *(playful tone)*, *warm*
        text = re.sub(r'\*\([^)]{1,80}\)\*', '', text)
        text = re.sub(r'\*[a-z][a-zA-Z\s,+]{1,60}\*', '', text)

        # Parenthetical meta: (tone: warm), (intent: X), (stimulate interest)
        text = re.sub(
            r'\([^)]{0,20}(tone|intent|curious|playful|warm|stimulat|engag|humor)[^)]{0,60}\)',
            '', text, flags=re.IGNORECASE
        )

        # Bracketed internal notes: [thinking], [internal], [reasoning]
        text = re.sub(
            r'\[[^\]]{0,20}(think|reason|internal|note|plan)[^\]]{0,40}\]',
            '', text, flags=re.IGNORECASE
        )

        # Clean up multiple blank lines left by removals
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _chunk_string(self, text: str, chunk_size: int = 8):
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]


# Singleton
agent = Agent()