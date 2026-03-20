"""
ambient/conflict_detector.py
Intent-aware personality conflict detection.

Runs AFTER tagging, ONLY when an utterance contains a reconsiderable intent.
Does NOT inject personality into every response — signals stay silent until
situationally relevant.

Flow:
  1. detect_intent() — quick heuristic: does this utterance contain a
     decision/choice/commitment that could be reconsidered?
  2. If yes → extract_intent_details() — LLM pass to understand:
       - what they're about to do/choose
       - what entities/subjects are involved
       - how reversible the decision is (ordering now vs. considering)
  3. check_conflicts() — query speaker's stored personality signals
     for anything that conflicts with the intent subjects
  4. assess_awareness() — LLM pass: does the speaker seem aware of
     the conflict, or did they likely not notice?
  5. If conflict + unaware (or partially aware):
       → determine urgency based on reversibility
       → surface via appropriate channel

Surfacing modes (in priority order):
  - "aside"     : gentle note after main response, non-urgent
  - "interrupt" : queued into directed_queue immediately, before decision finalizes
  - "flag"      : noted in ambient log only, no active surfacing

The detector never surfaces conflicts for things the speaker clearly
already knows — "I know I hate mushrooms but I'm trying it anyway" is
not a conflict worth flagging.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional

# ── Intent heuristics ─────────────────────────────────────────────────────────

_INTENT_PATTERN = re.compile(
    r"\b("
    # Ordering / food
    r"i('ll| will| want to| think i('ll| will))? (get|order|have|try|go with|grab)|"
    r"(sounds|looks|seems) (good|interesting|nice|tasty|fun)|"
    r"i('m| am) going to (get|order|have|try)|"
    # Buying
    r"(i('ll| will| want to| think i('ll| will))?) (buy|get|pick up|grab|purchase|order)|"
    r"thinking (of|about) (buying|getting|picking up)|"
    # Planning / committing
    r"(i('m| am)) (going|planning|thinking of)|"
    r"(we('re| are)) (going|planning|thinking of)|"
    r"(let('s| us)) |i('ll| will) (go|do|try|sign up|join|book|reserve)|"
    # Interest / considering
    r"(i('m| am)) (interested in|curious about|considering|thinking about)|"
    r"(might|maybe|could) (try|go|get|do|check out)|"
    r"(what (do you think|about)|how (is|was|are))"
    r")\b",
    re.IGNORECASE,
)


def detect_intent(transcript: str) -> bool:
    """
    Quick heuristic: does this utterance contain a reconsiderable intent?
    Cheap pre-filter before LLM call.
    """
    return bool(_INTENT_PATTERN.search(transcript))


# ── Intent detail extraction ──────────────────────────────────────────────────

@dataclass
class IntentDetails:
    has_intent: bool
    intent_type: str          # "ordering" | "buying" | "planning" | "considering"
    subjects: list[str]       # entities involved (e.g. ["shiitake skewers", "mushrooms"])
    reversibility: str        # "low" | "medium" | "high"
                              # low = ordering right now, high = vague future plan
    description: str          # natural language summary of the intent


async def extract_intent_details(
    transcript: str,
    speaker: str,
    llm,
) -> Optional[IntentDetails]:
    """
    LLM pass to understand what the speaker is intending to do
    and what subjects/entities are involved.
    """
    prompt = [
        {
            "role": "user",
            "content": (
                f"Speaker: {speaker}\n"
                f"Transcript: \"{transcript}\"\n\n"
                f"Does this transcript contain a decision, choice, or commitment "
                f"that could potentially be reconsidered?\n\n"
                f"If yes, extract the details. Respond with ONLY valid JSON:\n"
                f'{{\n'
                f'  "has_intent": true,\n'
                f'  "intent_type": "ordering|buying|planning|considering",\n'
                f'  "subjects": ["specific thing they want", "broader category if relevant"],\n'
                f'  "reversibility": "low|medium|high",\n'
                f'  "description": "one sentence: what they are about to do"\n'
                f'}}\n\n'
                f"reversibility guide:\n"
                f"  low    = happening right now (ordering food, about to buy)\n"
                f"  medium = soon but not immediate (planning for today/tomorrow)\n"
                f"  high   = vague future consideration\n\n"
                f"subjects should include BOTH the specific item AND its category "
                f"if relevant — e.g. 'shiitake skewers' AND 'mushrooms'.\n\n"
                f'If no intent found: {{"has_intent": false}}'
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You identify decision intent in natural speech. "
                "Be specific about subjects — include both specific items and their "
                "broader categories when relevant. Respond with valid JSON only."
            ),
            max_new_tokens=200,
            temperature=0.1,
        )
        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        if not parsed.get("has_intent"):
            return None

        return IntentDetails(
            has_intent=True,
            intent_type=parsed.get("intent_type", "considering"),
            subjects=[s.lower().strip() for s in parsed.get("subjects", []) if s.strip()],
            reversibility=parsed.get("reversibility", "high"),
            description=parsed.get("description", "").strip(),
        )

    except Exception as e:
        print(f"[ConflictDetector] Intent extraction failed: {e}")
        return None


# ── Conflict checking ─────────────────────────────────────────────────────────

@dataclass
class ConflictMatch:
    signal_statement: str     # stored signal text
    signal_type: str          # preference | interest | value | etc.
    sentiment: str            # positive | negative
    subject: str              # what the stored signal is about
    intent_subject: str       # what in the intent matched it
    date: str                 # when the signal was recorded


def check_conflicts(
    speaker: str,
    intent: IntentDetails,
) -> list[ConflictMatch]:
    """
    Query stored personality signals for conflicts with the intent subjects.
    Only returns NEGATIVE signals (things they dislike/avoid) that match
    something they're about to do — the meaningful conflict case.
    """
    if not intent.subjects:
        return []

    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb

        collection_name = speaker.lower().strip()
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            return []

        store = RAGStore(collection_name=collection_name)
        if store.count == 0:
            return []

        conflicts = []

        for subject in intent.subjects:
            # Search for signals about this subject
            results = store.retrieve(
                query=f"{speaker} {subject} preference dislike",
                n_results=5,
                where={
                    "$and": [
                        {"type": {"$eq": "personality_signal"}},
                        {"status": {"$eq": "active"}},
                        {"speaker": {"$eq": speaker.lower()}},
                        # Only flag negative signals — positive ones aren't conflicts
                        {"sentiment": {"$eq": "negative"}},
                    ]
                },
            )

            for r in results:
                meta = r.get("metadata", {})
                stored_subject = meta.get("subject", "").lower()

                # Check for meaningful subject overlap
                if _subjects_conflict(subject, stored_subject):
                    conflicts.append(ConflictMatch(
                        signal_statement=meta.get("statement", ""),
                        signal_type=meta.get("signal_type", "preference"),
                        sentiment=meta.get("sentiment", "negative"),
                        subject=stored_subject,
                        intent_subject=subject,
                        date=meta.get("date", ""),
                    ))

        # Deduplicate by subject
        seen = set()
        unique = []
        for c in conflicts:
            if c.subject not in seen:
                seen.add(c.subject)
                unique.append(c)

        return unique

    except Exception as e:
        print(f"[ConflictDetector] Conflict check failed: {e}")
        return []


def _subjects_conflict(intent_subject: str, stored_subject: str) -> bool:
    """
    Check if an intent subject and a stored signal subject overlap meaningfully.
    Handles: exact match, containment, shared meaningful words.
    """
    a = intent_subject.lower().strip()
    b = stored_subject.lower().strip()

    if a == b:
        return True
    if a in b or b in a:
        return True

    # Word overlap (ignore short/common words)
    stopwords = {"a", "an", "the", "and", "or", "of", "to", "in", "for", "with"}
    words_a = {w for w in a.split() if len(w) > 3 and w not in stopwords}
    words_b = {w for w in b.split() if len(w) > 3 and w not in stopwords}
    return bool(words_a & words_b)


# ── Awareness assessment ──────────────────────────────────────────────────────

async def assess_awareness(
    transcript: str,
    conflict: ConflictMatch,
    llm,
) -> str:
    """
    Assess whether the speaker seems aware of the conflict.
    Returns: "aware" | "probably_aware" | "probably_unaware" | "unaware"

    "aware" = they explicitly acknowledged it ("I know I hate mushrooms but...")
    "probably_aware" = they mentioned the subject confidently, likely know
    "probably_unaware" = item name doesn't obviously signal the conflict
    "unaware" = no indication they know
    """
    prompt = [
        {
            "role": "user",
            "content": (
                f"Transcript: \"{transcript}\"\n\n"
                f"Stored signal: {conflict.signal_statement}\n"
                f"They are about to: {conflict.intent_subject}\n\n"
                f"Does the speaker seem aware that '{conflict.intent_subject}' "
                f"conflicts with their stored preference?\n\n"
                f"Consider:\n"
                f"- Did they explicitly acknowledge the conflict?\n"
                f"- Is the conflict obvious from the item name? "
                f"  ('mushroom soup' is obvious, 'shiitake skewers' less so)\n"
                f"- Did they express hesitation or awareness?\n\n"
                f"Respond with ONLY one of: aware | probably_aware | probably_unaware | unaware"
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt="Assess speaker awareness in one word. No explanation.",
            max_new_tokens=10,
            temperature=0.1,
        )
        result = raw.strip().lower()
        if result in ("aware", "probably_aware", "probably_unaware", "unaware"):
            return result
        return "probably_unaware"
    except Exception:
        return "probably_unaware"


# ── Surfacing ─────────────────────────────────────────────────────────────────

@dataclass
class ConflictAlert:
    speaker: str
    conflict: ConflictMatch
    intent: IntentDetails
    awareness: str
    mode: str              # "interrupt" | "aside" | "flag"
    message: str           # what Gizmo should say


def _build_alert_message(
    speaker: str,
    conflict: ConflictMatch,
    intent: IntentDetails,
    awareness: str,
) -> str:
    """
    Build a minimal, non-patronizing alert. Just the flag — no explanation,
    no restatement of their preference, no "just so you know" padding.
    They can work out the implication themselves.
    """
    subject = conflict.intent_subject

    if awareness in ("probably_unaware", "unaware"):
        return f"You know {subject}, right?"
    else:
        return f"You know that's {subject}, right?"


def _determine_mode(intent: IntentDetails, awareness: str) -> str:
    """
    Decide how urgently to surface the conflict.

    interrupt : decision is imminent (low reversibility) and they seem unaware
    aside     : decision is soon but not immediate, or they're probably aware
    flag      : high reversibility (vague plan) or they already know
    """
    if awareness == "aware":
        return "flag"  # they already know, don't bother them

    if intent.reversibility == "low" and awareness in ("unaware", "probably_unaware"):
        return "interrupt"
    elif intent.reversibility == "medium" or awareness == "probably_unaware":
        return "aside"
    else:
        return "flag"


async def run_conflict_detection(
    transcript: str,
    speaker: str,
    llm,
    directed_queue: Optional[asyncio.Queue] = None,
    context: Optional[dict] = None,
) -> Optional[ConflictAlert]:
    """
    Full conflict detection pipeline for one utterance.
    Returns a ConflictAlert if a surfaceable conflict is found, else None.

    Steps:
      1. Heuristic intent check
      2. LLM intent extraction
      3. Signal conflict check
      4. Awareness assessment
      5. Surface via appropriate mode
    """
    if not speaker or not detect_intent(transcript):
        return None

    # Extract intent details
    intent = await extract_intent_details(transcript, speaker=speaker, llm=llm)
    if not intent:
        return None

    print(f"[ConflictDetector] Intent detected for '{speaker}': {intent.description} "
          f"(reversibility={intent.reversibility})")

    # Check for conflicts with stored signals
    conflicts = check_conflicts(speaker, intent)
    if not conflicts:
        return None

    print(f"[ConflictDetector] {len(conflicts)} conflict(s) found for '{speaker}'")

    # Take the strongest conflict (first after dedup)
    conflict = conflicts[0]

    # Assess awareness
    awareness = await assess_awareness(transcript, conflict, llm=llm)
    print(f"[ConflictDetector] Awareness: {awareness}")

    # Determine surfacing mode
    mode = _determine_mode(intent, awareness)
    if mode == "flag":
        print(f"[ConflictDetector] Flagging only — speaker is aware or decision is reversible")
        return None  # silent log, no active surface

    # Build alert
    message = _build_alert_message(speaker, conflict, intent, awareness)
    alert = ConflictAlert(
        speaker=speaker,
        conflict=conflict,
        intent=intent,
        awareness=awareness,
        mode=mode,
        message=message,
    )

    print(f"[ConflictDetector] Surfacing as '{mode}': {message[:80]}")

    # Surface it
    if directed_queue:
        await directed_queue.put({
            "transcript": f"[CONFLICT ALERT] {message}",
            "context": context or {},
            "type": "conflict_alert",
            "mode": mode,
            "speaker": speaker,
        })

    return alert
