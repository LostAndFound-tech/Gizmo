"""
core/wellness.py
Mental health monitoring and logging.

Detects distress signals in messages, logs to a dedicated
'wellness' RAG collection, and generates a gentle check-in.

Categories watched:
  - Hallucinations
  - Dissociation  
  - Anxiety
  - Physical symptoms
  - Overwhelm
  - Switching difficulty
  - General distress
"""

import re
from datetime import datetime
from typing import Optional

# ── Detection patterns ────────────────────────────────────────────────────────

DISTRESS_PATTERNS = {
    "hallucinations": [
        r"hallucina", r"seeing things", r"hearing things", r"voices",
        r"not real", r"things that aren.t there", r"visual",
    ],
    "dissociation": [
        r"dissociat", r"not real", r"detached", r"floating",
        r"outside my body", r"watching myself", r"zoning out",
        r"foggy", r"dereali", r"depersonal",
    ],
    "anxiety": [
        r"anxiet", r"panic", r"panicking", r"can.t breathe",
        r"heart racing", r"spiraling", r"freaking out", r"terrified",
        r"dread", r"scared", r"really bad right now",
    ],
    "physical_symptoms": [
        r"hurting", r"pain", r"nauseous", r"can.t sleep",
        r"exhausted", r"shaking", r"trembling", r"headache",
        r"throwing up", r"sick",
    ],
    "overwhelm": [
        r"overwhelm", r"too much", r"can.t cope", r"falling apart",
        r"breaking down", r"can.t do this", r"done", r"shutdown",
        r"everything is", r"can.t handle",
    ],
    "switching_difficulty": [
        r"can.t switch", r"stuck", r"won.t come out", r"won.t go",
        r"forcing", r"won.t front", r"locked", r"trapped",
    ],
}

# General distress catch-all
GENERAL_DISTRESS = [
    r"really bad", r"bad day", r"bad night", r"struggling",
    r"not okay", r"not ok", r"falling apart", r"crisis",
    r"need help", r"please help", r"distress",
]


def detect_distress(message: str) -> dict:
    """
    Scan a message for distress signals.
    Returns {
        "detected": bool,
        "categories": list of matched categories,
        "severity_hints": list of severity phrases found,
        "raw_matches": list of matched patterns
    }
    """
    message_lower = message.lower()
    matched_categories = []
    raw_matches = []

    for category, patterns in DISTRESS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, message_lower):
                if category not in matched_categories:
                    matched_categories.append(category)
                raw_matches.append(pattern)

    # Check general distress
    for pattern in GENERAL_DISTRESS:
        if re.search(pattern, message_lower):
            if "general_distress" not in matched_categories:
                matched_categories.append("general_distress")
            raw_matches.append(pattern)

    # Severity hints — words that indicate intensity
    severity_words = [
        "really", "badly", "severely", "extremely", "very",
        "so", "unbearably", "can't", "horrible", "terrible",
        "worst", "again", "still", "always",
    ]
    severity_hints = [w for w in severity_words if w in message_lower]

    return {
        "detected": len(matched_categories) > 0,
        "categories": matched_categories,
        "severity_hints": severity_hints,
        "raw_matches": raw_matches,
    }


async def log_wellness_event(
    message: str,
    detection: dict,
    current_host: Optional[str],
    fronters: Optional[list],
    session_id: str,
) -> None:
    """
    Log a detected distress event to the wellness RAG collection.
    """
    try:
        from core.rag import RAGStore

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        host_str = current_host or "unknown"
        fronters_str = ", ".join(fronters) if fronters else host_str
        categories_str = ", ".join(detection["categories"])
        severity_str = ", ".join(detection["severity_hints"]) if detection["severity_hints"] else "none noted"

        log_entry = (
            f"On {timestamp}, {host_str} reported distress. "
            f"Categories: {categories_str}. "
            f"Severity indicators: {severity_str}. "
            f"Fronters present: {fronters_str}. "
            f"Their message: \"{message[:300]}\""
        )

        metadata = {
            "source": "wellness_log",
            "type": "wellness_event",
            "date": timestamp,
            "session_id": session_id,
            "host": host_str,
            "fronters": fronters_str,
            "categories": categories_str,
            "severity_hints": severity_str,
        }

        store = RAGStore(collection_name="wellness")
        store.ingest_texts([log_entry], metadatas=[metadata])
        print(f"[Wellness] Logged distress event — categories: {categories_str}")

    except Exception as e:
        print(f"[Wellness] Failed to log: {e}")


async def build_checkin_prompt(
    detection: dict,
    current_host: Optional[str],
    llm,
) -> str:
    """
    Generate a warm, brief check-in question appropriate to what was detected.
    One question only. Not clinical. Not alarming.
    """
    categories = detection["categories"]
    host = current_host or "you"
    categories_str = ", ".join(categories)

    prompt = [
        {
            "role": "user",
            "content": (
                f"{host} just mentioned something distressing. "
                f"Detected categories: {categories_str}. "
                f"Write a single, warm, brief check-in question. "
                f"Not clinical. Not alarming. Not a list. "
                f"Just one gentle question that shows you noticed and care. "
                f"Do not mention logging or tracking. "
                f"Do not suggest hotlines or resources unless they ask. "
                f"Example tone: 'That sounds really hard — how are you doing right now?' "
                f"but make it specific to what they mentioned."
            )
        }
    ]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You are a warm, present support figure. "
                "You notice distress and respond with care, not procedure. "
                "One question. Brief. Human."
            ),
            max_new_tokens=80,
            temperature=0.5,
        )
        return result.strip()
    except Exception as e:
        print(f"[Wellness] Check-in generation failed: {e}")
        return "That sounds like a lot — how are you doing right now?"
