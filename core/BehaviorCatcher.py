"""
core/BehaviorCatcher.py
Behavioral extraction from conversational chunks.

Returns a parsed list of per-person behavior dicts.
Personality traits include storage tags for later retrieval.
Accepts a buffer of pending unpaired actions to attempt matching.
"""
import json
import re
from typing import Optional

from core.log import log_event, log_error


_SYSTEM = """
You are a behavioral analyst extracting deep psychological datapoints from conversation.
Casual conversation is your richest data source — not your poorest. Read between the lines.

You will receive the current chunk, a thread summary, and a buffer of unpaired actions
from previous chunks that haven't found a reaction yet.

Return ONLY a valid JSON array. No markdown fences. No explanation. No preamble.
If nothing behavioral is present, return [].

For each person, produce one entry with only the fields the text supports:

[{
  "Subject": "Kaylee",
  "Type": "Person",
  "Personality": [
    {"trait": "assumes she will be burdened", "tags": ["behavior", "mood", "relational"]},
    {"trait": "martyred", "tags": ["behavior", "mood", "depression"]},
    {"trait": "snippy under pressure", "tags": ["behavior", "anger", "communication"]}
  ],
  "Episodes": [
    {
      "action": "Jess asked who wants to commute, not directing it at anyone",
      "reaction": "Kaylee said 'Fine.' before being asked, volunteering with visible irritation",
      "tags": ["behavior", "relational", "mood"]
    }
  ],
  "Source": "observed",
  "Statement": "Kaylee: Fine."
}]

What to look for — these are the richest signals:
- Short or clipped responses ("Fine.", "I do. I do know that.") carry enormous emotional weight — analyze the tone
- Volunteering before being asked reveals eagerness, anxiety, martyrdom, or expectation
- Assuming you'll be picked before being asked shows entitlement or resentment
- How someone acknowledges something reveals emotional awareness or defensiveness
- Offering to help unprompted shows protectiveness or care
- Pushing back gently shows quiet assertiveness
- Reassuring others reveals confidence, stoicism, or a need to appear capable
- Enthusiasm that overshoots the moment reveals eagerness to please or social anxiety
- Gratitude phrasing reveals warmth, formality, or deference
- Speaker asks others to help locate something they misplaced → possible inattention/adhd
- Speaker acknowledges forgetting something matter-of-factly → possible inattention/adhd
- Speaker is redirected by others to complete a basic task → normalized accommodation pattern
- Speaker prioritizes a project or fixation over physical needs → possible adhd/hyperfocus
- Others compensate for speaker's disorganization without comment → strong adhd signal
- Speaker loses track of objects, time, or tasks across conversations → adhd pattern

Available storage tags — use 2-5 per trait/episode, pick the tightest fit:
appearance, fashion, color, identity, gender, sexuality, relational, boundaries, care,
behavior, communication, humor, reckless, mood, anger, warmth, anxiety, grief,
adhd, depression, trauma, dissociation, wellness, work, routine, physical, food,
system, role

Rules:
- Infer personality from behavior — don't restate the action as a trait
- One word can be a full behavioral datapoint if the subtext is clear
- Episodes: include when there is a clear action → reaction pair, including subtle ones
- If an action has no reaction yet, omit Episodes — do not leave one side null
- Unmatched actions will be buffered automatically — do not invent reactions
- One entry per person, stack everything into it
- Source is "self" if they described themselves, otherwise name the observer
""".strip()



_CONVERSATIONAL_FRAME = (
    "Conversational frame: 'Gizmo' or 'you' refers to the AI companion, not a system member. "
    "'I', 'me', 'us', 'we' refers to the plural system. "
    "Lines prefixed 'Gizmo:' are AI responses — do not attribute their content to system members."
)

def _build_prompt(chunk: str, thread: str, pending_actions: list[dict]) -> str:
    parts = [
        f"Current chunk:\n{chunk}",
        f"\nThread summary:\n{thread}",
    ]
    if pending_actions:
        parts.append(
            f"\nPending unpaired actions from previous chunks:\n"
            + json.dumps(pending_actions, indent=2)
        )
    return "\n".join(parts)


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
            log_event("BehaviorCatcher", "EMPTY_RESPONSE")
            return None

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return clean

    except Exception as e:
        log_error("BehaviorCatcher", "LLM call failed", exc=e)
        print(f"[BehaviorCatcher] LLM call failed: {type(e).__name__}: {e}")
        return None


class BehaviorCatcher:

    async def extract(
        self,
        user_message:    str,
        thread:          str,
        subject:         str,
        session_file:    str,
        pending_actions: list[dict] = [],
    ) -> Optional[list]:
        if not user_message.strip():
            return None
        try:
            prompt  = _build_prompt(user_message, thread, pending_actions)
            raw_str = await _call_llm(prompt)

            if not raw_str:
                log_event("BehaviorCatcher", "NO_BEHAVIOR_EXTRACTED",
                    subject=subject,
                    session=session_file,
                )
                return None

            parsed = json.loads(raw_str)
            if not isinstance(parsed, list):
                return None
            return parsed

        except Exception as e:
            log_error("BehaviorCatcher", "extract failed", exc=e)
            print(f"[BehaviorCatcher] extract failed: {type(e).__name__}: {e}")
            return None


behaviorcatcher = BehaviorCatcher()
