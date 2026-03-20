"""
ambient/tagger.py
LLM-based topic extraction and summarization for ambient transcripts.

For each meaningful transcript, produces:
  - topics: list of 1-4 emergent topic tags (snake_case, short)
  - summary: 1-2 sentence factual summary of what was said
  - directed_at_gizmo: bool — was this speech directed at Gizmo?
  - reminder: parsed reminder dict if reminder intent detected, else None

Topics are fully emergent — the LLM invents appropriate tags based on content.
They're stored as a comma-separated string in ChromaDB metadata.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional

from ambient.reminders import detect_reminder_intent, parse_reminder


@dataclass
class TagResult:
    topics: list[str]
    summary: str
    directed_at_gizmo: bool
    raw_transcript: str
    reminder: Optional[dict] = None   # set if reminder intent was detected


# Names/phrases that suggest speech is directed at Gizmo
GIZMO_TRIGGERS = re.compile(
    r"\b(gizmo|hey gizmo|giz)\b",
    re.IGNORECASE,
)

_COMMON_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and",
    "or", "but", "for", "with", "that", "this", "i", "you", "we", "they",
    "was", "were", "be", "been", "have", "has", "do", "did", "will", "would",
    "could", "should", "just", "like", "so", "about", "up", "out", "if",
    "what", "when", "where", "who", "how", "my", "your", "our", "their",
    "its", "me", "him", "her", "us", "them", "also", "then", "than",
}


def _heuristic_topics(text: str) -> list[str]:
    words = re.findall(r"\b[a-z]{5,}\b", text.lower())
    filtered = [w for w in words if w not in _COMMON_STOPWORDS]
    from collections import Counter
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(3)]


async def tag(
    transcript: str,
    llm,
    context: Optional[dict] = None,
) -> TagResult:
    """
    Extract topics, summary, and reminder (if any) from a transcript.
    Reminder parsing runs concurrently with topic tagging — no added latency.
    """
    directed = bool(GIZMO_TRIGGERS.search(transcript))
    has_reminder = detect_reminder_intent(transcript)

    fronter_hint = ""
    if context:
        host = context.get("current_host", "")
        if host:
            fronter_hint = f"The person speaking is likely {host}. "

    topic_prompt = [
        {
            "role": "user",
            "content": (
                f"{fronter_hint}"
                f"Analyze this speech transcript and respond with ONLY valid JSON — "
                f"no markdown, no explanation, no backticks:\n\n"
                f'"{transcript}"\n\n'
                f"Respond with exactly this JSON structure:\n"
                f'{{"topics": ["tag1", "tag2"], "summary": "One or two sentence summary."}}\n\n'
                f"Rules for topics:\n"
                f"- 1 to 4 tags maximum\n"
                f"- snake_case, lowercase, 1-3 words each\n"
                f"- Be specific (e.g. 'movie_recommendation' not 'media')\n"
                f"- Reflect what was actually discussed, not meta-commentary\n\n"
                f"Rules for summary:\n"
                f"- Past tense, factual, 1-2 sentences\n"
                f"- Include who said what if attributable\n"
                f"- No filler phrases"
            )
        }
    ]

    # Run topic tagging + optional reminder parsing concurrently
    tasks = [llm.generate(
        topic_prompt,
        system_prompt=(
            "You extract structured metadata from speech transcripts. "
            "Always respond with valid JSON only. No markdown. No explanation."
        ),
        max_new_tokens=150,
        temperature=0.2,
    )]
    if has_reminder:
        tasks.append(parse_reminder(transcript, llm=llm))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process topic result
    topics = []
    summary = ""
    topic_raw = results[0]

    if isinstance(topic_raw, Exception):
        print(f"[Tagger] Topic LLM failed: {topic_raw}")
        topics = _heuristic_topics(transcript)
        summary = transcript[:200]
    else:
        try:
            clean = topic_raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(clean)
            topics = [
                t.lower().replace(" ", "_").strip()
                for t in parsed.get("topics", [])
                if isinstance(t, str) and t.strip()
            ][:4]
            summary = parsed.get("summary", "").strip()
            if not topics:
                topics = _heuristic_topics(transcript)
            if not summary:
                summary = transcript[:200]
        except Exception as e:
            print(f"[Tagger] Topic parse failed ({e}), using heuristic fallback")
            topics = _heuristic_topics(transcript)
            summary = transcript[:200]

    # Process reminder result
    reminder = None
    if has_reminder:
        reminder_result = results[1] if len(results) > 1 else None
        if isinstance(reminder_result, Exception):
            print(f"[Tagger] Reminder parse failed: {reminder_result}")
        else:
            reminder = reminder_result
            if reminder:
                print(f"[Tagger] Reminder: '{reminder['message']}' due {reminder['due_iso']}")
                if "reminder" not in topics:
                    topics.append("reminder")

    print(f"[Tagger] Topics: {topics} | Directed: {directed} | Reminder: {reminder is not None}")

    return TagResult(
        topics=topics,
        summary=summary,
        directed_at_gizmo=directed,
        raw_transcript=transcript,
        reminder=reminder,
    )