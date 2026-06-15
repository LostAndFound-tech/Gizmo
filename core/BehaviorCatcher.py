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

You will receive a structured exchange: what Gizmo said or did, and how the subject responded.
Gizmo's lines are context only — extract behavior about the subject, not Gizmo.
You will also receive a thread summary and a buffer of pending unpaired actions.

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
      "cause": "Jess asked who wants to commute, not directing it at anyone",
      "action": "Kaylee said 'Fine.' before being asked",
      "reaction": "volunteered with visible irritation before her name was called",
      "trait": "assumes she will be chosen and resents it before it happens",
      "tags": ["behavior", "relational", "mood"]
    }
  ],
  "Source": "observed",
  "Statement": "Kaylee: Fine."
}]

Episode fields:
- cause: what Gizmo said or did, or what was happening that prompted the subject's response
- action: what the subject did or said
- reaction: the immediate consequence or follow-through — their own or others'
- trait: the distillation — what this episode says about who this person is, stated as a stable characteristic
- tags: 2-5 storage tags

The trait is the longitudinal signal. It should be earned by the episode — not restated from it.
Bad trait: "volunteered before being asked" (that's just the action again)
Good trait: "assumes she will be burdened and resents it preemptively" (that's what it means)

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
- Laughter at something dark or absurd versus laughter at warmth — context changes everything
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
- Extract behavior about the subject only — Gizmo's lines are cause/context, not subject matter
- Infer personality from behavior — don't restate the action as a trait
- One word can be a full behavioral datapoint if the subtext is clear
- Episodes require cause + action + reaction + trait — omit if you can't fill all four honestly
- If an action has no reaction yet, omit Episodes — do not leave one side null
- Unmatched actions will be buffered automatically — do not invent reactions
- One entry per person, stack everything into it
- Source is "self" if they described themselves, otherwise name the observer
""".strip()



def _build_prompt(exchanges: list[dict], thread: str, pending_actions: list[dict]) -> str:
    """
    exchanges: list of {"gizmo": str, "subject": str, "subject_name": str}
    Each entry is one back-and-forth. Gizmo's line is cause context.
    Subject's line is what gets analyzed.
    """
    exchange_lines = []
    for ex in exchanges:
        subject_name = ex.get("subject_name", "Subject")
        if ex.get("gizmo"):
            exchange_lines.append(f"Gizmo: {ex['gizmo']}")
        if ex.get("subject"):
            exchange_lines.append(f"{subject_name}: {ex['subject']}")

    parts = [
        f"Structured exchange:\n" + "\n".join(exchange_lines),
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
        exchanges:       list[dict],
        thread:          str,
        subject:         str,
        session_file:    str,
        pending_actions: list[dict] = [],
    ) -> Optional[list]:
        if not exchanges:
            return None
        try:
            prompt  = _build_prompt(exchanges, thread, pending_actions)
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
