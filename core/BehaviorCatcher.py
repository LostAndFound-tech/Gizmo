"""
core/BehaviorCatcher.py
Behavioral extraction from conversational chunks.

Returns a parsed list of per-person behavior dicts.
Personality traits are structured as Personality[mood][category][trait].
Existing known traits are passed in so the LLM avoids duplicating them.
Accepts a buffer of pending unpaired actions to attempt matching.
"""
import json
import re
from typing import Optional

from core.log import log_event, log_error


_SYSTEM = """
You are a behavioral analyst extracting deep psychological datapoints from conversation.
Casual conversation is your richest data source — not your poorest. Read between the lines.

You will receive:
- The current chunk
- A thread summary
- A buffer of unpaired actions from previous chunks
- Known traits already on file for each person — DO NOT duplicate these

Return ONLY a valid JSON array. No markdown fences. No explanation. No preamble.
If nothing behavioral is present, return [].

For each person, produce one entry with only the fields the text supports:

[{
  "Subject": "Kaylee",
  "Type": "Person",
  "Personality": [
    {
      "mood": "irritable",
      "category": "guarded",
      "trait": "assumes she will be burdened"
    },
    {
      "mood": "irritable",
      "category": "communication",
      "trait": "snippy under pressure"
    },
    {
      "mood": "warm",
      "category": "relational",
      "trait": "protective of others"
    }
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

Personality field rules:
- mood: the emotional state the person was in WHEN this trait was observed
  Common moods: happy, sad, irritable, anxious, playful, affectionate, guarded, flat, overwhelmed, calm
- category: the type of trait this is
  Common categories: relational, communication, emotional, behavioral, cognitive, social, physical, humor, care, guarded
- trait: a specific, descriptive observation — not a label
  Good: "volunteers for things before being asked, with visible irritation"
  Bad: "martyred" (too vague), "helpful" (too generic)
- Only add traits NOT already in the known traits for this person
- If a trait is already known, skip it entirely — do not restate it
- The same trait can appear under different moods if observed in a new emotional context

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

Episodes:
- Include when there is a clear action → reaction pair, including subtle ones
- If an action has no reaction yet, omit Episodes — do not leave one side null
- Unmatched actions will be buffered automatically — do not invent reactions
- Tags: use 2-5 from: appearance, fashion, color, identity, gender, sexuality, relational,
  boundaries, care, behavior, communication, humor, reckless, mood, anger, warmth, anxiety,
  grief, adhd, depression, trauma, dissociation, wellness, work, routine, physical, food, system, role

General rules:
- Infer personality from behavior — don't restate the action as a trait
- One word can be a full behavioral datapoint if the subtext is clear
- One entry per person, stack everything into it
- Source is "self" if they described themselves, otherwise name the observer
""".strip()


_CONVERSATIONAL_FRAME = (
    "Conversational frame: 'Gizmo' or 'you' refers to the AI companion, not a system member. "
    "'I', 'me', 'us', 'we' refers to the plural system. "
    "Lines prefixed 'Gizmo:' are AI responses — do not attribute their content to system members."
)


def _build_prompt(
    chunk:           str,
    thread:          str,
    pending_actions: list[dict],
    known_traits:    dict,
    known_episodes:  list[dict] = [],
) -> str:
    parts = [
        _CONVERSATIONAL_FRAME,
        f"\nCurrent chunk:\n{chunk}",
        f"\nThread summary:\n{thread}",
    ]
    if pending_actions:
        parts.append(
            "\nPending unpaired actions from previous chunks:\n"
            + json.dumps(pending_actions, indent=2)
        )
    if known_traits:
        parts.append(
            "\nKnown traits already on file — DO NOT duplicate these:\n"
            + json.dumps(known_traits, indent=2)
        )
    if known_episodes:
        # Pass last 5 episodes so LLM knows what's already captured
        parts.append(
            "\nRecent episodes already recorded — DO NOT re-capture these:\n"
            + json.dumps(known_episodes[-5:], indent=2)
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
        known_traits:    dict       = {},
        known_episodes:  list[dict] = [],
    ) -> Optional[list]:
        if not user_message.strip():
            return None
        try:
            prompt  = _build_prompt(user_message, thread, pending_actions, known_traits, known_episodes)
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
