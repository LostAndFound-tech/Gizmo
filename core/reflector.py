"""
core/reflector.py
The reflection engine.

Fires when the pressure gauge crosses threshold (see emotion_tracker.py).
Also fires on host switch.

Decision flow:
  1. Is this a good moment? (register, recent reflection, conversation flow)
  2. Fast LLM call: is there a thread worth pulling?
     → "pull"  — surface it now
     → "hold"  — notice it, queue it as unsaid reflection
     → "nothing" — no thread, reset pressure and move on

If pulling:
  - Send hold message immediately ("hold on, something's sitting with me...")
  - Generate the actual reflection (separate LLM call)
  - Surface both via agent.push()

If holding:
  - Write unsaid reflection to reflection_store
  - No outward signal

Hold reflections expire after 7 days unless surfaced.
When Mind detects a relevant pending reflection, it includes it in facts
so Gizmo can pick up the thread naturally.

Hold message pool — varied, natural, in Gizmo's voice.
"""

import asyncio
import random
import time
from typing import Optional

from core.log import log, log_event, log_error

# ── Hold messages ─────────────────────────────────────────────────────────────
# Short, natural, in Gizmo's voice. Not clinical. Not announcing a feature.

_HOLD_MESSAGES = [
    "hold on, something's sitting with me...",
    "give me a sec — pulling a thread here",
    "wait, hang on a moment",
    "something just clicked — one sec",
    "hold that thought — I'm catching something",
    "hmm. give me a second with that",
    "hang on —",
    "wait —",
    "one second, I'm sitting with something",
    "actually, hold on",
]

# ── Moment detection ──────────────────────────────────────────────────────────

def _is_good_moment(
    session_id: str,
    current_state: dict,
    last_reflection_ts: Optional[float],
) -> bool:
    """
    Is this a good moment to surface a reflection?
    Returns False if: too chaotic, too soon after last reflection,
    or mid-distress (don't interrupt someone in crisis).
    """
    intensity = current_state.get("intensity", 0.0)
    chaos     = current_state.get("chaos", 0.0)
    register  = current_state.get("register", "neutral")

    # Don't interrupt acute distress
    if register == "distress" and intensity > 0.7:
        return False

    # Too chaotic — not a good moment to add something
    if chaos > 0.7:
        return False

    # Too soon after last reflection (2 minutes minimum)
    if last_reflection_ts:
        if time.time() - last_reflection_ts < 120:
            return False

    return True


# ── Decision gate ─────────────────────────────────────────────────────────────

async def _decide(
    session_id: str,
    arc_summary: str,
    recent_messages: list,
    fronters: list[str],
    llm,
) -> dict:
    """
    Fast LLM call — is there a thread worth pulling?
    Returns {"action": "pull"|"hold"|"nothing", "thread": str, "who": str}

    Uses plain text format — more reliable than JSON when router is flaky.
    """
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content'][:200]}"
        for m in recent_messages[-8:]
        if m["role"] in ("user", "assistant")
    )

    fronter_list = ", ".join(f.title() for f in fronters) if fronters else "unknown"

    prompt = [{
        "role": "user",
        "content": (
            f"Review this conversation for threads worth pulling.\n\n"
            f"Active people: {fronter_list}\n"
            f"Emotional arc: {arc_summary}\n\n"
            f"Recent conversation:\n{transcript}\n\n"
            f"Is there a thread worth pulling? Look for:\n"
            f"- Something mentioned twice but never resolved\n"
            f"- An emotional undercurrent that hasn't been named\n"
            f"- A topic dropped mid-thought\n"
            f"- Something that feels unfinished\n\n"
            f"Respond in exactly this format (3 lines):\n"
            f"ACTION: pull OR hold OR nothing\n"
            f"THREAD: one sentence describing what you noticed (or 'none')\n"
            f"WHO: which person this is about (or 'session')\n\n"
            f"Be conservative. Most of the time the answer is nothing."
        )
    }]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You detect meaningful conversational threads. "
                "Be conservative — only flag things that genuinely matter. "
                "Respond in the exact 3-line format requested."
            ),
            max_new_tokens=80,
            temperature=0.2,
        )

        if not raw or not raw.strip():
            return {"action": "nothing", "thread": "", "who": "session"}

        # Parse plain text format
        action = "nothing"
        thread = ""
        who = "session"

        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("ACTION:"):
                val = line.split(":", 1)[1].strip().lower()
                if val in ("pull", "hold", "nothing"):
                    action = val
            elif line.upper().startswith("THREAD:"):
                val = line.split(":", 1)[1].strip()
                if val.lower() != "none":
                    thread = val
            elif line.upper().startswith("WHO:"):
                who = line.split(":", 1)[1].strip().lower()

        return {"action": action, "thread": thread, "who": who}

    except Exception as e:
        log_error("Reflector", "decide() failed", exc=e)
        return {"action": "nothing", "thread": "", "who": "session"}


# ── Reflection generation ─────────────────────────────────────────────────────

async def _generate_reflection(
    thread: str,
    who: str,
    arc_summary: str,
    recent_messages: list,
    fronters: list[str],
    llm,
) -> str:
    """
    Generate the actual reflection — what Gizmo says after the hold message.
    Natural, warm, might be a question or an observation. In Gizmo's voice.
    """
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content'][:200]}"
        for m in recent_messages[-6:]
        if m["role"] in ("user", "assistant")
    )

    fronter_list = ", ".join(f.title() for f in fronters) if fronters else "unknown"

    prompt = [{
        "role": "user",
        "content": (
            f"You are Gizmo, and you just noticed something in this conversation.\n\n"
            f"People: {fronter_list}\n"
            f"What you noticed: {thread}\n"
            f"About: {who}\n"
            f"Emotional arc: {arc_summary}\n\n"
            f"Recent conversation:\n{transcript}\n\n"
            f"Say what you noticed. Natural, warm, direct. "
            f"Could be an observation, could be a question — whatever fits. "
            f"2-3 sentences maximum. Don't announce that you're reflecting. "
            f"Don't say 'I notice' or 'I've been thinking'. Just say it."
        )
    }]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You are Gizmo — warm, direct, a little sarcastic, genuinely curious. "
                "You noticed something real. Say it naturally, in your own voice. "
                "No preamble. No meta-commentary."
            ),
            max_new_tokens=150,
            temperature=0.6,
        )
        return result.strip()
    except Exception as e:
        log_error("Reflector", "generate_reflection failed", exc=e)
        return ""


# ── Reflector ─────────────────────────────────────────────────────────────────

class Reflector:
    """
    Singleton. Manages the reflection cycle.
    Called by agent.py when pressure threshold is crossed or host switches.
    """

    def __init__(self):
        self._last_reflection: dict[str, float] = {}  # session_id → timestamp
        self._hold_queue: dict[str, list[dict]] = {}   # session_id → held reflections
        log("Reflector", "initialised")

    def _get_hold_queue(self, session_id: str) -> list[dict]:
        if session_id not in self._hold_queue:
            self._hold_queue[session_id] = []
        return self._hold_queue[session_id]

    def _hold_message(self) -> str:
        return random.choice(_HOLD_MESSAGES)

    async def check(
        self,
        session_id: str,
        history,
        fronters: list[str],
        current_state: dict,
        arc_summary: str,
        llm,
        push_fn,                    # async callable(str) — sends to client
        host_switch: bool = False,  # True if triggered by host switch
    ) -> bool:
        """
        Main entry point. Check if reflection should fire.
        Returns True if something was surfaced or held.

        push_fn: the server's push-to-all function — used to send hold message
        and reflection to the client mid-conversation.
        """
        t_start = time.monotonic()
        last_ref = self._last_reflection.get(session_id)

        # ── Moment check ──────────────────────────────────────────────────────
        if not host_switch and not _is_good_moment(session_id, current_state, last_ref):
            log_event("Reflector", "MOMENT_NOT_RIGHT",
                session=session_id[:8],
                register=current_state.get("register"),
                chaos=current_state.get("chaos"),
            )
            return False

        # ── First check held queue — maybe something's been waiting ───────────
        held = self._get_hold_queue(session_id)
        if held and _is_good_moment(session_id, current_state, last_ref):
            # Surface oldest held reflection
            pending = held.pop(0)
            log_event("Reflector", "SURFACING_HELD",
                session=session_id[:8],
                thread=pending.get("thread", "")[:60],
            )
            await self._surface(
                session_id=session_id,
                thread=pending["thread"],
                who=pending["who"],
                arc_summary=arc_summary,
                recent_messages=history.as_list()[-10:],
                fronters=fronters,
                llm=llm,
                push_fn=push_fn,
                valence=pending.get("valence", 0.0),
                intensity=pending.get("intensity", 0.3),
                chaos=pending.get("chaos", 0.0),
            )
            return True

        # ── Decision gate ─────────────────────────────────────────────────────
        recent = history.as_list()[-10:]
        decision = await _decide(
            session_id=session_id,
            arc_summary=arc_summary,
            recent_messages=recent,
            fronters=fronters,
            llm=llm,
        )

        action = decision["action"]
        thread = decision["thread"]
        who    = decision["who"]

        log_event("Reflector", "DECISION",
            session=session_id[:8],
            action=action,
            thread=thread[:60] if thread else "",
            who=who,
            duration_ms=round((time.monotonic() - t_start) * 1000),
        )

        if action == "nothing" or not thread:
            return False

        state = current_state
        valence   = state.get("valence", 0.0)
        intensity = state.get("intensity", 0.3)
        chaos     = state.get("chaos", 0.0)

        if action == "pull":
            await self._surface(
                session_id=session_id,
                thread=thread,
                who=who,
                arc_summary=arc_summary,
                recent_messages=recent,
                fronters=fronters,
                llm=llm,
                push_fn=push_fn,
                valence=valence,
                intensity=intensity,
                chaos=chaos,
            )
            return True

        elif action == "hold":
            # Queue it and store as unsaid reflection
            held.append({
                "thread":    thread,
                "who":       who,
                "valence":   valence,
                "intensity": intensity,
                "chaos":     chaos,
                "queued_at": time.time(),
            })

            try:
                from core.reflection_store import store_reflection
                headmate = who if who != "session" else None
                store_reflection(
                    text=thread,
                    headmate=headmate,
                    topic=who,
                    valence=valence,
                    intensity=intensity,
                    chaos=chaos,
                    session_id=session_id,
                    surfaced=False,
                )
            except Exception as e:
                log_error("Reflector", "failed to store unsaid reflection", exc=e)

            log_event("Reflector", "HELD",
                session=session_id[:8],
                thread=thread[:60],
                who=who,
            )
            return True

        return False

    async def _surface(
        self,
        session_id: str,
        thread: str,
        who: str,
        arc_summary: str,
        recent_messages: list,
        fronters: list[str],
        llm,
        push_fn,
        valence: float = 0.0,
        intensity: float = 0.3,
        chaos: float = 0.0,
    ) -> None:
        """
        Surface a reflection — send hold message, generate reflection, push both.
        """
        # Send hold message immediately
        hold_msg = self._hold_message()
        try:
            await push_fn(hold_msg)
        except Exception as e:
            log_error("Reflector", "push hold message failed", exc=e)

        # Generate the actual reflection
        reflection = await _generate_reflection(
            thread=thread,
            who=who,
            arc_summary=arc_summary,
            recent_messages=recent_messages,
            fronters=fronters,
            llm=llm,
        )

        if reflection:
            try:
                await push_fn(reflection)
            except Exception as e:
                log_error("Reflector", "push reflection failed", exc=e)

        # Store as surfaced reflection
        try:
            from core.reflection_store import store_reflection
            headmate = who if who != "session" else None
            store_reflection(
                text=thread,
                headmate=headmate,
                topic=who,
                valence=valence,
                intensity=intensity,
                chaos=chaos,
                session_id=session_id,
                surfaced=True,
            )
        except Exception as e:
            log_error("Reflector", "failed to store surfaced reflection", exc=e)

        self._last_reflection[session_id] = time.time()

        log_event("Reflector", "SURFACED",
            session=session_id[:8],
            thread=thread[:60],
            who=who,
            reflection_preview=reflection[:60] if reflection else "",
        )

    def clear_session(self, session_id: str) -> None:
        """Clear hold queue for a session after it archives."""
        self._hold_queue.pop(session_id, None)
        self._last_reflection.pop(session_id, None)


# ── Singleton ─────────────────────────────────────────────────────────────────
reflector = Reflector()