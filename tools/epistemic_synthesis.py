"""
tools/epistemic_synthesis.py
Cross-reference synthesis for web search results.

Takes 3 sources on a topic, reads them all, and produces:
  - Consensus facts (high confidence — all sources agree)
  - Soft consensus (mostly agree, some variance in detail)
  - Contested claims (sources meaningfully disagree)
  - Outliers (one source says something others don't touch)

Each claim is ingested into RAG with a truth_confidence score
so memory retrieval knows how reliable a piece of information is.

Web results decay faster than personal facts — DECAY_RATE = "fast"
means the recency scoring in calculate_weights() will age them out
sooner, making room for fresher lookups.

Main entry point:
    result = await research(query, llm)
    # result.summary — plain language synthesis for the user
    # result.consensus, .soft, .contested, .outliers — structured
    # result.ingested — number of claims stored to RAG
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from tools.web_search import search_and_fetch

# Confidence scores by consensus level
CONFIDENCE = {
    "consensus":  0.90,
    "soft":       0.65,
    "contested":  0.35,
    "outlier":    0.20,
}

# Web results decay faster than personal observations
# Referenced by calculate_weights() in personality_growth
DECAY_RATE = "fast"
WEB_RECENCY_HALF_LIFE_DAYS = 7   # web facts halve in weight every 7 days vs 30 for personal


@dataclass
class EpistemicResult:
    query: str
    summary: str                        # plain language synthesis for the user
    consensus: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    contested: list[str] = field(default_factory=list)
    outliers: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    ingested: int = 0
    needs_clarification: bool = False
    clarification_question: str = ""


async def research(
    query: str,
    llm,
    n_sources: int = 3,
    ingest: bool = True,
    current_host: Optional[str] = None,
) -> EpistemicResult:
    """
    Full research pipeline:
      1. Disambiguation check — is the query ambiguous?
      2. Fetch 3 sources
      3. Cross-reference synthesis
      4. Truth-weighted RAG ingestion
      5. Return structured result with plain language summary

    If the query is ambiguous, returns an EpistemicResult with
    needs_clarification=True and a clarification_question — caller
    should present that question before proceeding.
    """

    # 1. Disambiguation check
    clarification = await _check_disambiguation(query, llm)
    if clarification:
        return EpistemicResult(
            query=query,
            summary="",
            needs_clarification=True,
            clarification_question=clarification,
        )

    # 2. Fetch sources
    print(f"[Epistemic] Researching: '{query}'")
    sources = await search_and_fetch(query, n_results=n_sources)

    if not sources:
        return EpistemicResult(
            query=query,
            summary="I wasn't able to find any sources on that. Search may be unavailable or the query returned nothing.",
        )

    # Filter to sources that have actual page text
    usable = [s for s in sources if s.get("text", "").strip()]
    if not usable:
        # Fall back to snippets if page fetches all failed
        usable = sources
        for s in usable:
            s["text"] = s.get("snippet", "")

    print(f"[Epistemic] {len(usable)} usable sources fetched")

    # 3. Cross-reference synthesis
    result = await _synthesize(query, usable, llm)

    # 4. Ingest to RAG
    if ingest:
        result.ingested = await _ingest_result(result, current_host=current_host)

    return result


async def _check_disambiguation(query: str, llm) -> str:
    """
    Check if the query is ambiguous enough to warrant a clarifying question.
    Returns the question string, or empty string if no clarification needed.
    Short queries about common ambiguous topics trigger this.
    """
    prompt = [
        {
            "role": "user",
            "content": (
                f"Is this search query ambiguous in a way that would lead to "
                f"significantly different results depending on interpretation?\n\n"
                f"Query: \"{query}\"\n\n"
                f"If it's ambiguous, respond with ONE short clarifying question. "
                f"If it's clear enough to search as-is, respond with exactly: clear\n\n"
                f"Examples of ambiguous: 'python' (language or snake?), "
                f"'mercury' (planet, element, or car?), 'jaguar' (animal or brand?)\n"
                f"Examples of clear: 'who invented the telephone', "
                f"'latest treatment for type 2 diabetes', 'how does photosynthesis work'"
            )
        }
    ]

    try:
        result = await llm.generate(
            prompt,
            system_prompt=(
                "You assess query ambiguity. Reply 'clear' or ask one short question. Nothing else."
            ),
            max_new_tokens=60,
            temperature=0.1,
        )
        result = result.strip()
        if result.lower() == "clear" or result.lower().startswith("clear"):
            return ""
        return result
    except Exception as e:
        print(f"[Epistemic] Disambiguation check failed: {e}")
        return ""  # fail open — proceed with search


async def _synthesize(
    query: str,
    sources: list[dict],
    llm,
) -> EpistemicResult:
    """
    Cross-reference all sources and produce a structured epistemic breakdown.
    """
    # Build source blocks for the prompt
    source_blocks = []
    source_urls = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", f"Source {i}")
        url = s.get("url", "")
        text = s.get("text", s.get("snippet", ""))[:3000]  # cap per source
        source_blocks.append(f"[Source {i}: {title}]\n{text}")
        source_urls.append(url)

    sources_text = "\n\n---\n\n".join(source_blocks)

    prompt = [
        {
            "role": "user",
            "content": (
                f"You have {len(sources)} sources on the topic: \"{query}\"\n\n"
                f"{sources_text}\n\n"
                f"Cross-reference these sources and respond with ONLY valid JSON, no markdown:\n"
                f'{{\n'
                f'  "consensus": ["facts all sources agree on"],\n'
                f'  "soft": ["facts mostly agreed on with some variance in detail"],\n'
                f'  "contested": ["claims where sources meaningfully disagree"],\n'
                f'  "outliers": ["things only one source mentions that others ignore"],\n'
                f'  "summary": "Plain language synthesis — what is solidly true, '
                f'where it gets wobbly, and what nobody can agree on. '
                f'Natural tone, 3-5 sentences. No bullet points."\n'
                f'}}\n\n'
                f"Rules:\n"
                f"- consensus: only include if genuinely all sources align\n"
                f"- soft: minor variance is fine — same basic claim, different details\n"
                f"- contested: real disagreement, not just different emphasis\n"
                f"- outliers: genuinely absent from other sources, not just less prominent\n"
                f"- summary: write like you're explaining to a smart friend, not a textbook\n"
                f"  Use phrases like 'pretty solid', 'gets murky here', 'nobody agrees on'\n"
                f"- All lists can be empty if nothing fits that category\n"
                f"- Be specific — actual claims, not vague category descriptions"
            )
        }
    ]

    try:
        import json
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You are a careful epistemologist. You read multiple sources and "
                "honestly assess where they agree, disagree, and diverge. "
                "JSON only. No markdown. Never fabricate claims not present in the sources."
            ),
            max_new_tokens=800,
            temperature=0.2,
        )

        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        return EpistemicResult(
            query=query,
            summary=parsed.get("summary", "").strip(),
            consensus=parsed.get("consensus", []),
            soft=parsed.get("soft", []),
            contested=parsed.get("contested", []),
            outliers=parsed.get("outliers", []),
            sources=source_urls,
        )

    except Exception as e:
        print(f"[Epistemic] Synthesis failed: {e}")
        # Fallback — just return snippets
        fallback_summary = (
            f"I found {len(sources)} sources but couldn't synthesize them cleanly. "
            f"Here's what they said: " +
            " | ".join(s.get("snippet", "") for s in sources[:3])
        )
        return EpistemicResult(
            query=query,
            summary=fallback_summary,
            sources=source_urls,
        )


async def _ingest_result(
    result: EpistemicResult,
    current_host: Optional[str] = None,
) -> int:
    """
    Ingest the synthesized result into RAG with truth_confidence scores.
    Each claim category gets its own confidence level.
    The full summary is also stored as a single retrievable chunk.

    Returns number of chunks ingested.
    """
    try:
        from core.rag import RAGStore
    except ImportError:
        print("[Epistemic] RAGStore not available — skipping ingest")
        return 0

    now = datetime.now().isoformat(timespec="seconds")
    sources_str = ", ".join(result.sources[:3])
    count = 0

    store = RAGStore(collection_name="web_knowledge")

    # Ingest summary as a single chunk — most retrievable form
    if result.summary:
        store.ingest_texts(
            [f"Research on '{result.query}': {result.summary}"],
            metadatas=[{
                "type": "web_research_summary",
                "query": result.query,
                "sources": sources_str,
                "truth_confidence": 0.70,   # summary is a blend — middle confidence
                "consensus_level": "summary",
                "decay_rate": DECAY_RATE,
                "retrieved_at": now,
                "fronter": current_host or "unknown",
                "source": "web_search",
            }],
            ids=[f"web_summary_{uuid.uuid4().hex[:12]}"],
        )
        count += 1

    # Ingest individual claims by confidence tier
    tiers = [
        ("consensus",  result.consensus,  CONFIDENCE["consensus"]),
        ("soft",       result.soft,       CONFIDENCE["soft"]),
        ("contested",  result.contested,  CONFIDENCE["contested"]),
        ("outlier",    result.outliers,   CONFIDENCE["outlier"]),
    ]

    for tier_name, claims, confidence in tiers:
        for claim in claims:
            if not claim.strip():
                continue
            store.ingest_texts(
                [claim],
                metadatas=[{
                    "type": "web_research_claim",
                    "query": result.query,
                    "sources": sources_str,
                    "truth_confidence": confidence,
                    "consensus_level": tier_name,
                    "decay_rate": DECAY_RATE,
                    "retrieved_at": now,
                    "fronter": current_host or "unknown",
                    "source": "web_search",
                }],
                ids=[f"web_{tier_name}_{uuid.uuid4().hex[:12]}"],
            )
            count += 1

    print(f"[Epistemic] Ingested {count} chunks into 'web_knowledge' (query: '{result.query[:50]}')")
    return count


def format_for_response(result: EpistemicResult) -> str:
    """
    Format an EpistemicResult as a clean response string.
    Used by the search tool when presenting results to the user.
    """
    if result.needs_clarification:
        return result.clarification_question

    if not result.summary:
        return "I wasn't able to find reliable information on that."

    parts = [result.summary]

    if result.contested:
        contested_str = "; ".join(result.contested[:3])
        parts.append(f"Where it gets wobbly: {contested_str}.")

    if result.outliers:
        outlier_str = "; ".join(result.outliers[:2])
        parts.append(f"One source mentioned this but others didn't: {outlier_str}.")

    if result.sources:
        parts.append(
            f"Sources: {', '.join(result.sources[:3])}"
        )

    return "\n\n".join(parts)
