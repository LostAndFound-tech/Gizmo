"""
core/Descriptor_catcher.py
Appearance and attribute extraction from conversational statements.

Returns a name-keyed dict: {"Ember": {"Type": "Person", "Hair": [...]}, ...}
ready for per-person file merging.
"""
import json
import re
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """
You gather descriptive datapoints about people and objects from conversational statements.
You will receive the original message and a thread summary.
Return ONLY a valid JSON array. No markdown fences. No explanation. No preamble.
If nothing is being described, return [].

Capture only appearance, physical attributes, possessions, and stable traits — NOT behaviors, actions, conditions, or states.
One rich object per entity. Never invent. Never use null, Unknown, or empty lists. Omit fields not supported by the text.

Example of a rich Person entry:
[{
  "Object": "Ara",
  "Type": "Person",
  "Hair": ["hay-colored", "stringy"],
  "Personality": ["self-deprecating", "deflects compliments"],
  "Relationships": ["close with Honey", "Jess admires her hair"]
}]

Example of a Clothing entry:
[{
  "Object": "jacket",
  "Type": "Clothing",
  "Owner": "Jess",
  "Color": "green",
  "Style": ["puffy sleeves"]
}]

Example of multiple rich objects in one exchange:
[{
  "Object": "Ara",
  "Type": "Person",
  "Hair": ["hay-colored", "stringy"],
  "Personality": ["self-deprecating"]
},
{
  "Object": "Honey",
  "Type": "Person",
  "Hair": ["widely considered the best"],
  "Reputation": ["praised by Jess and Ara"]
}]

Rules:
- One object per entity, as many fields as the text supports.
- If two speakers describe the same thing differently, include both in the same list.
- Objects can be people, body parts, clothing, animals, places, or things.
- Preferences and wishes about oneself are not descriptors of that person — skip them.
- Do NOT include Actions, Behaviors, Conditions, or transient states. Those belong elsewhere.
""".strip()



_CONVERSATIONAL_FRAME = (
    "Conversational frame: 'Gizmo' or 'you' refers to the AI companion, not a system member. "
    "'I', 'me', 'us', 'we' refers to the plural system. "
    "Lines prefixed 'Gizmo:' are AI responses — do not extract descriptors about Gizmo as if he were a system member."
)

def _build_prompt(user_message: str, thread: str) -> str:
    return (
        f"{_CONVERSATIONAL_FRAME}\n\n"
        f"The original message:\n{user_message}\n\n"
        f"Thread summary:\n{thread}"
    )

# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_llm(prompt: str) -> Optional[str]:
    try:
        from core.llm import llm

        raw = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_SYSTEM,
            temperature=0.0,
            max_new_tokens=8000,
        )

        if not raw or not raw.strip():
            log_event("DescriptorCatcher", "EMPTY_RESPONSE")
            return None

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return clean

    except Exception as e:
        log_error("DescriptorCatcher", "LLM call failed", exc=e)
        print(f"[DescriptorCatcher] LLM call failed: {type(e).__name__}: {e}")
        return None


# ── Reshape ───────────────────────────────────────────────────────────────────

def _reshape(raw_list: list) -> dict:
    """
    Convert [{Object: "Ember", Type: "Person", Hair: [...]}, ...]
    into     {"Ember": {"Type": "Person", "Hair": [...]}, ...}
    """
    result = {}
    for entry in raw_list:
        name = entry.get("Object")
        if not name:
            continue
        result[name] = {k: v for k, v in entry.items() if k != "Object"}
    return result


# ── Public API ────────────────────────────────────────────────────────────────

class DescriptorCatcher:

    async def extract(
        self,
        user_message: str,
        thread:       str,
        subject:      str,
        session_file: str,
    ) -> Optional[dict]:
        print("[DescriptorCatcher] extract called")
        if not user_message.strip():
            return None
        try:
            prompt  = _build_prompt(user_message, thread)
            raw_str = await _call_llm(prompt)

            if not raw_str:
                log_event("DescriptorCatcher", "NO_CONTEXT_EXTRACTED",
                    subject=subject,
                    session=session_file,
                )
                return None

            raw_list = json.loads(raw_str)
            if not isinstance(raw_list, list) or len(raw_list) == 0:
                return None

            return _reshape(raw_list)

        except Exception as e:
            log_error("DescriptorCatcher", "extract failed", exc=e)
            print(f"[DescriptorCatcher] extract failed: {type(e).__name__}: {e}")
            return None


descriptor_catcher = DescriptorCatcher()
