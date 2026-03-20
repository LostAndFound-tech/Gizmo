"""
ambient/personality.py
Real-time personality signal extraction from ambient transcripts.

Runs in parallel with the tagger for every meaningful utterance.
Extracts structured signals about the speaker, stores them in their
personal ChromaDB collection, and detects contradictions with existing
stored signals — surfacing those for human review rather than silently
overwriting.

Signal types tracked:
  - preference     : likes/dislikes (food, media, aesthetics, anything)
  - interest       : topics they engage with enthusiastically or return to
  - value          : what they care about, what matters to them
  - emotional      : recurring mood/emotional patterns, not momentary states
  - belief         : how they see the world, opinions, worldview

ChromaDB metadata schema per signal:
  {
    "type":         "personality_signal",
    "signal_type":  "preference" | "interest" | "value" | "emotional" | "belief",
    "subject":      "mushrooms",           # what the signal is about
    "sentiment":    "positive" | "negative" | "neutral" | "complex",
    "speaker":      "alice",
    "confidence":   "0.85",                # LLM confidence in extraction
    "date":         "2026-03-18",
    "source":       "ambient_transcript",
    "raw_snippet":  "I've always hated...", # short supporting quote
    "status":       "active" | "contradicted" | "pending_review",
  }

Contradiction flow:
  - New signal extracted for speaker X about subject Y
  - Query X's collection for existing signals about Y
  - If conflict found → put contradiction notice into directed_queue
  - New signal stored with status="pending_review" until resolved
  - Human resolves via chat → signal updated to "active" or discarded
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Heuristic pre-filter — does this transcript likely contain a personality signal?
_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"love|hate|like|dislike|enjoy|can't stand|obsessed|passionate|"
    r"always|never|usually|tend to|prefer|favourite|favorite|"
    r"care about|matter|important|believe|think that|feel like|"
    r"interested in|into|really into|not into|don't care|"
    r"makes me|drives me|reminds me|used to|anymore|still"
    r")\b",
    re.IGNORECASE,
)

PERSONALITY_COLLECTION_SUFFIX = ""  # signals go into the speaker's own collection


@dataclass
class PersonalitySignal:
    signal_type: str       # preference | interest | value | emotional | belief
    subject: str           # what it's about
    sentiment: str         # positive | negative | neutral | complex
    statement: str         # natural language summary of the signal
    confidence: float      # 0.0-1.0
    raw_snippet: str       # supporting text from transcript
    speaker: str


@dataclass
class ExtractionResult:
    signals: list[PersonalitySignal]
    contradictions: list[dict]   # [{new_signal, existing_signal, speaker}]
    speaker: str


def detect_signal_intent(transcript: str) -> bool:
    """Quick heuristic — does this transcript likely contain a personality signal?"""
    return bool(_SIGNAL_PATTERN.search(transcript))


async def extract_signals(
    transcript: str,
    speaker: str,
    llm,
) -> list[PersonalitySignal]:
    """
    Ask the LLM to extract personality signals from a transcript.
    Returns a list of PersonalitySignal objects (may be empty).
    """
    if not speaker:
        return []

    prompt = [
        {
            "role": "user",
            "content": (
                f"The speaker is: {speaker}\n\n"
                f"Transcript:\n\"{transcript}\"\n\n"
                f"Extract any personality signals from this transcript. "
                f"A signal is something that reveals a lasting trait, preference, interest, "
                f"value, emotional pattern, or belief about the speaker.\n\n"
                f"DO NOT extract:\n"
                f"- Momentary emotions (being happy right now)\n"
                f"- Factual statements about the world\n"
                f"- Things said about other people\n"
                f"- Anything uncertain or hypothetical\n\n"
                f"DO extract:\n"
                f"- 'I hate mushrooms' → preference, negative\n"
                f"- 'I keep coming back to philosophy' → interest, positive\n"
                f"- 'Honesty matters more than comfort to me' → value, positive\n"
                f"- 'I always shut down when overwhelmed' → emotional, neutral\n"
                f"- 'I think most people are fundamentally good' → belief, positive\n\n"
                f"Respond with ONLY valid JSON, no markdown:\n"
                f'{{"signals": [\n'
                f'  {{\n'
                f'    "signal_type": "preference|interest|value|emotional|belief",\n'
                f'    "subject": "short label for what this is about",\n'
                f'    "sentiment": "positive|negative|neutral|complex",\n'
                f'    "statement": "one clear sentence describing the signal",\n'
                f'    "confidence": 0.0-1.0,\n'
                f'    "raw_snippet": "shortest supporting quote from transcript"\n'
                f'  }}\n'
                f']}}\n\n'
                f'If no signals found, respond with {{"signals": []}}'
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You extract personality signals from natural speech. "
                "Be conservative — only extract clear, lasting signals. "
                "Respond with valid JSON only. No markdown."
            ),
            max_new_tokens=400,
            temperature=0.1,
        )

        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        signals = []
        for s in parsed.get("signals", []):
            sig_type = s.get("signal_type", "").lower()
            if sig_type not in ("preference", "interest", "value", "emotional", "belief"):
                continue
            confidence = float(s.get("confidence", 0.5))
            if confidence < 0.6:
                continue  # skip low-confidence extractions

            signals.append(PersonalitySignal(
                signal_type=sig_type,
                subject=s.get("subject", "").strip()[:60],
                sentiment=s.get("sentiment", "neutral").lower(),
                statement=s.get("statement", "").strip(),
                confidence=confidence,
                raw_snippet=s.get("raw_snippet", "")[:200],
                speaker=speaker,
            ))

        return signals

    except Exception as e:
        print(f"[Personality] Extraction failed: {e}")
        return []


def _find_existing_signals(speaker: str, subject: str) -> list[dict]:
    """
    Query the speaker's RAG collection for existing signals about the same subject.
    Returns list of matching metadata dicts.
    """
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb

        collection_name = speaker.lower().strip()
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name not in existing_collections:
            return []

        store = RAGStore(collection_name=collection_name)
        if store.count == 0:
            return []

        # Search semantically for signals about the same subject
        results = store.retrieve(
            query=f"personality signal about {subject}",
            n_results=5,
            where={"type": {"$eq": "personality_signal"}},
        )

        # Filter to signals actually about this subject (fuzzy match)
        subject_lower = subject.lower()
        matching = []
        for r in results:
            meta = r.get("metadata", {})
            stored_subject = meta.get("subject", "").lower()
            # Check if subjects overlap meaningfully
            if (subject_lower in stored_subject or
                stored_subject in subject_lower or
                _subject_overlap(subject_lower, stored_subject)):
                if meta.get("status", "active") != "pending_review":
                    matching.append(meta)

        return matching

    except Exception as e:
        print(f"[Personality] Existing signal lookup failed: {e}")
        return []


def _subject_overlap(a: str, b: str) -> bool:
    """Check if two subject strings share meaningful words."""
    stopwords = {"a", "an", "the", "and", "or", "of", "to", "in", "for"}
    words_a = {w for w in a.split() if w not in stopwords and len(w) > 3}
    words_b = {w for w in b.split() if w not in stopwords and len(w) > 3}
    return bool(words_a & words_b)


def _is_contradiction(new_signal: PersonalitySignal, existing_meta: dict) -> bool:
    """
    Check if a new signal contradicts an existing stored signal.
    Contradiction = same subject, opposite sentiment.
    """
    new_sentiment = new_signal.sentiment
    old_sentiment = existing_meta.get("sentiment", "")

    opposites = {
        ("positive", "negative"),
        ("negative", "positive"),
    }
    return (new_sentiment, old_sentiment) in opposites or (old_sentiment, new_sentiment) in opposites


def store_signal(signal: PersonalitySignal, status: str = "active") -> str:
    """
    Persist a personality signal to the speaker's RAG collection.
    Returns the signal ID.
    """
    try:
        from core.rag import RAGStore

        collection_name = signal.speaker.lower().strip()
        signal_id = f"psig_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        doc_text = (
            f"{signal.speaker.capitalize()} — {signal.signal_type}: {signal.statement}"
        )

        metadata = {
            "type": "personality_signal",
            "signal_type": signal.signal_type,
            "subject": signal.subject,
            "sentiment": signal.sentiment,
            "speaker": signal.speaker,
            "confidence": str(round(signal.confidence, 3)),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "source": "ambient_transcript",
            "raw_snippet": signal.raw_snippet[:200],
            "status": status,
            "statement": signal.statement,
        }

        store = RAGStore(collection_name=collection_name)
        store.ingest_texts([doc_text], metadatas=[metadata], ids=[signal_id])
        print(f"[Personality] Stored signal for '{signal.speaker}': {signal.signal_type} / {signal.subject} ({status})")
        return signal_id

    except Exception as e:
        print(f"[Personality] Store failed: {e}")
        return ""


def resolve_contradiction(
    speaker: str,
    subject: str,
    keep: str,  # "new" | "old" | "both"
    new_signal_id: Optional[str] = None,
) -> None:
    """
    Called when the human resolves a flagged contradiction.
    keep="new"  → mark old signal as superseded, activate new one
    keep="old"  → discard new signal (mark pending_review → discarded)
    keep="both" → activate both with a note that preference evolved
    """
    try:
        from core.rag import RAGStore, CHROMA_PERSIST_DIR
        import chromadb

        collection_name = speaker.lower().strip()
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            return

        store = RAGStore(collection_name=collection_name)
        now_iso = datetime.now().isoformat(timespec="seconds")

        # Find existing active signals about this subject
        results = store.collection.get(
            where={
                "$and": [
                    {"type": {"$eq": "personality_signal"}},
                    {"subject": {"$eq": subject}},
                    {"speaker": {"$eq": speaker.lower()}},
                ]
            }
        )

        if not results["ids"]:
            return

        for sig_id, meta in zip(results["ids"], results["metadatas"]):
            updated_meta = dict(meta)

            if keep == "new":
                if sig_id == new_signal_id:
                    updated_meta["status"] = "active"
                else:
                    updated_meta["status"] = "superseded"
                    updated_meta["superseded_at"] = now_iso
            elif keep == "old":
                if sig_id == new_signal_id:
                    updated_meta["status"] = "discarded"
            elif keep == "both":
                updated_meta["status"] = "active"
                updated_meta["note"] = "preference evolved over time"

            store.collection.update(ids=[sig_id], metadatas=[updated_meta])

        print(f"[Personality] Contradiction resolved for '{speaker}' / '{subject}': keep={keep}")

    except Exception as e:
        print(f"[Personality] Resolve failed: {e}")


async def process_transcript(
    transcript: str,
    speaker: str,
    llm,
    directed_queue: Optional[asyncio.Queue] = None,
    context: Optional[dict] = None,
) -> ExtractionResult:
    """
    Main entry point. Extract signals, check for contradictions,
    store active signals, queue contradiction notices for human review.

    Called in parallel with tagger.tag() — adds no blocking latency
    to the main pipeline.
    """
    if not speaker or not detect_signal_intent(transcript):
        return ExtractionResult(signals=[], contradictions=[], speaker=speaker)

    signals = await extract_signals(transcript, speaker=speaker, llm=llm)

    if not signals:
        return ExtractionResult(signals=[], contradictions=[], speaker=speaker)

    contradictions = []

    for signal in signals:
        existing = _find_existing_signals(speaker, signal.subject)

        conflicting = [e for e in existing if _is_contradiction(signal, e)]

        if conflicting:
            # Store as pending_review — don't blindly overwrite
            new_id = store_signal(signal, status="pending_review")

            for old in conflicting:
                contradiction = {
                    "speaker": speaker,
                    "subject": signal.subject,
                    "new_signal": signal,
                    "new_signal_id": new_id,
                    "existing_statement": old.get("statement", ""),
                    "existing_sentiment": old.get("sentiment", ""),
                    "existing_date": old.get("date", ""),
                }
                contradictions.append(contradiction)
                print(
                    f"[Personality] Contradiction for '{speaker}' / '{signal.subject}': "
                    f"was '{old.get('sentiment')}', now '{signal.sentiment}'"
                )

            # Surface to directed_queue for human resolution
            if directed_queue and conflicting:
                old = conflicting[0]
                notice = (
                    f"[PERSONALITY CONTRADICTION] "
                    f"{speaker.capitalize()} said something that conflicts with a stored signal. "
                    f"Previously: \"{old.get('statement', '')}\" ({old.get('date', '')}). "
                    f"Now: \"{signal.statement}\". "
                    f"Should I update the record, keep the old one, or note that both are true?"
                )
                await directed_queue.put({
                    "transcript": notice,
                    "context": context or {},
                    "type": "personality_contradiction",
                    "contradiction": contradiction,
                })

        else:
            # No conflict — store directly as active
            store_signal(signal, status="active")

    return ExtractionResult(
        signals=signals,
        contradictions=contradictions,
        speaker=speaker,
    )
