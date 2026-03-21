"""
core/greeter.py
Automatic return greeting — fires when a session has been inactive
for INACTIVITY_THRESHOLD seconds.

Behavior:
  - Pulls a light context read: pending things, recent mood, anything unresolved
  - Time-of-day aware (morning / afternoon / evening / late night)
  - "You have returned" energy — Gizmo's personality, not a chatbot greeting
  - Follows up on anything that was left pending (proposal, appointment, hard day)
  - Doesn't recap the last conversation — asks what's going on now
  - Personalized to the fronter if known
"""

import asyncio
from datetime import datetime
from typing import Optional

# Seconds of inactivity before a return greeting fires
INACTIVITY_THRESHOLD = 4 * 60 * 60  # 4 hours


def _time_of_day(now: datetime) -> str:
    hour = now.hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


async def _pull_light_context(
    fronter: str,
    llm,
) -> str:
    """
    Pull a brief context read — anything pending, unresolved, or emotionally
    significant from the recent past. Not a recap, just a vibe check.
    Returns a short string the greeter prompt can use, or empty string.
    """
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb
        from core.timezone import tz_now

        today = tz_now().strftime("%Y-%m-%d")
        yesterday = (tz_now().replace(hour=0, minute=0, second=0, microsecond=0)
                     .__class__(tz_now().year, tz_now().month, tz_now().day)
                     .__class__.fromtimestamp(
                         tz_now().timestamp() - 86400,
                         tz=tz_now().tzinfo
                     )).strftime("%Y-%m-%d")

        snippets = []

        # Check fronter's collection for recent pending/unresolved things
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = [c.name for c in client.list_collections()]

        for collection_name in ([fronter.lower()] if fronter else []) + ["main", "wellness"]:
            if collection_name not in existing:
                continue
            store = RAGStore(collection_name=collection_name)
            if store.count == 0:
                continue

            # Look for anything that sounds pending, upcoming, or emotionally weighted
            results = store.retrieve(
                query="appointment proposal plan tomorrow upcoming worried about nervous excited",
                n_results=3,
            )
            for r in results:
                meta = r.get("metadata", {})
                date = meta.get("date", "")
                if date in (today, yesterday):
                    snippets.append(r["text"][:200])

        if not snippets:
            return ""

        # Distill into a single context hint
        prompt = [
            {
                "role": "user",
                "content": (
                    f"Here are some recent notes about {fronter or 'this person'}:\n\n"
                    + "\n\n".join(snippets)
                    + "\n\nIn one sentence, what's the most important thing to follow up on? "
                    "Focus on anything pending, unresolved, or emotionally significant. "
                    "If nothing stands out, say 'nothing pending'. "
                    "Be specific — e.g. 'they had a job interview this afternoon' not 'they were busy'."
                )
            }
        ]

        result = await llm.generate(
            prompt,
            system_prompt="You identify the single most important pending thing to follow up on. One sentence only.",
            max_new_tokens=60,
            temperature=0.2,
        )
        result = result.strip()
        if "nothing pending" in result.lower():
            return ""
        return result

    except Exception as e:
        print(f"[Greeter] Context pull failed: {e}")
        return ""


async def build_greeting(
    fronter: str,
    session_id: str,
    llm,
) -> str:
    """
    Build a personalized return greeting.
    Pulls light context, then generates a Gizmo-flavored welcome back.
    """
    from core.timezone import tz_now
    now = tz_now()
    time_of_day = _time_of_day(now)
    time_str = now.strftime("%H:%M")

    # Pull context in parallel with nothing (just structure the await)
    context_hint = await _pull_light_context(fronter, llm)

    prompt = [
        {
            "role": "user",
            "content": (
                f"Current time: {time_str} ({time_of_day})\n"
                f"Fronter: {fronter or 'unknown'}\n"
                + (f"Context: {context_hint}\n" if context_hint else "")
                + "\nWrite a short return greeting from Gizmo. Rules:\n"
                "- 'You have returned' energy — dramatic best friend, not a chatbot\n"
                "- Time-of-day aware but in Gizmo's voice, not 'Good morning!'\n"
                "- If there's context, follow up on it naturally and specifically "
                "  — like a friend who remembered. Don't recap, just ask.\n"
                "- If no context, ask what's been going on in an open, casual way\n"
                "- Short — 1-3 sentences max\n"
                "- No stage directions, no asterisks, no narration. Just dialogue.\n"
                "- Match the relationship: Gizmo is warm, sarcastic, genuinely caring\n"
                f"- Address {fronter or 'them'} directly if name is known"
            )
        }
    ]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You are Gizmo — sarcastic, warm, genuinely caring. "
                "You speak in dialogue, never narration. "
                "You greet returning people like a dramatic best friend who noticed they were gone."
            ),
            max_new_tokens=120,
            temperature=0.7,
        )
        return result.strip()
    except Exception as e:
        print(f"[Greeter] Generation failed: {e}")
        # Fallback greetings by time of day
        fallbacks = {
            "morning": f"Oh, you're up. Morning, {fronter or 'you'}. What's the damage today?",
            "afternoon": f"There you are. Good afternoon, {fronter or 'you'}. What's going on?",
            "evening": f"Evening, {fronter or 'you'}. You survived the day. Tell me things.",
            "night": f"It's late, {fronter or 'you'}. You okay? What's up?",
        }
        return fallbacks.get(time_of_day, f"Hey, {fronter or 'you'}. You're back. What's going on?")


def should_greet(session) -> bool:
    """
    Returns True if the session has been inactive long enough to warrant
    a return greeting.
    """
    if len(session) == 0:
        return False  # brand new session — no greeting needed, they haven't talked yet
    return session.seconds_since_active() >= INACTIVITY_THRESHOLD
