"""
core/graph_extractor.py

Async LLM pass that extracts typed knowledge edges from exchanges.

After the tagger runs and produces topics/mood/cause/effect,
this pass reads the exchange and extracts structured relationships
to write into the knowledge graph.

Input: exchange + tagger output
Output: typed edges written to knowledge_graph

Example extraction from:
  "I get really scared when things change suddenly"
  (speaker: princess, topics: anxiety, identity)

Produces:
  princess ──[feels]──▶ scared  (context: sudden_change, strength: 0.7)
  princess ──[triggered_by]──▶ sudden change  (strength: 0.7)
  princess ──[struggles_with]──▶ transitions  (strength: 0.6, inferred)

This runs after tag_exchange in archivist.receive_outgoing,
fired via asyncio.ensure_future — never blocks.
"""

import json
import time
from typing import Optional

from core.log import log, log_event, log_error


# ── Noise gate ────────────────────────────────────────────────────────────────

_MIN_WORDS = 8


def _worth_extracting(user_message: str, gizmo_response: str) -> bool:
    combined = len(user_message.split()) + len(gizmo_response.split())
    return combined >= _MIN_WORDS


# ── Main extraction coroutine ─────────────────────────────────────────────────

async def extract_edges(
    session_id: str,
    user_message: str,
    gizmo_response: str,
    host: Optional[str],
    fronters: list,
    topics: list,
    mood: str,
    cause: Optional[str],
    effect: Optional[str],
    llm,
) -> None:
    """
    Extract typed knowledge edges from an exchange.
    Writes directly to knowledge_graph.
    Called async after tagger — never blocks.
    """
    try:
        if not _worth_extracting(user_message, gizmo_response):
            return

        if not host and not fronters:
            return

        host_str     = (host or fronters[0] if fronters else "unknown").lower()
        fronter_str  = ", ".join(f.title() for f in fronters) if fronters else host_str.title()
        topic_str    = ", ".join(topics) if topics else "general"

        # Load known entities so the LLM knows who's who
        try:
            from core.knowledge_graph import get_known_entities, PREDICATES
            entities     = get_known_entities()
            entity_names = [e["id"] for e in entities] if entities else [host_str, "gizmo"]
            entity_str   = ", ".join(entity_names)
            pred_str     = ", ".join(sorted(PREDICATES))
        except Exception:
            entity_str = f"{host_str}, gizmo"
            pred_str   = "feels, fears, wants, enjoys, dislikes, struggles_with, trusts, loves, protects, tends_to, triggered_by, calmed_by, curious_about, is, values, pattern, taught"

        prompt = [{
            "role": "user",
            "content": (
                f"Extract knowledge edges from this exchange.\n\n"
                f"Speaker: {fronter_str}\n"
                f"Topics: {topic_str}\n"
                f"Mood: {mood or 'unknown'}\n"
                f"Cause: {cause or 'unknown'}\n"
                f"Effect: {effect or 'unknown'}\n\n"
                f"Exchange:\n"
                f"{fronter_str}: {user_message.strip()}\n"
                f"Gizmo: {gizmo_response.strip()}\n\n"
                f"Known entities: {entity_str}\n"
                f"Available predicates: {pred_str}\n\n"
                f"Extract relationship edges. Each edge:\n"
                f"  subject: who this is about (use entity name)\n"
                f"  predicate: relationship type from the list above\n"
                f"  object: what it points to (concept, emotion, entity, behavior)\n"
                f"  object_type: 'entity', 'emotion', 'concept', or 'behavior'\n"
                f"  strength: 0.1-1.0 (how strongly supported by this exchange)\n"
                f"  confidence: 0.1-1.0 (told=0.9, observed=0.6, inferred=0.4)\n"
                f"  source: 'told', 'observed', or 'inferred'\n"
                f"  context: one short phrase describing when/why this applies\n\n"
                f"Rules:\n"
                f"- Only extract what's genuinely supported\n"
                f"- Prefer specific objects over vague ones\n"
                f"- Include both explicit statements AND reasonable inferences\n"
                f"- Gizmo can be a subject too (what Gizmo felt or did)\n"
                f"- 3-8 edges per exchange maximum\n"
                f"- If nothing meaningful, return empty array\n\n"
                f"Respond with ONLY valid JSON array, no markdown:\n"
                f'[{{"subject":"jess","predicate":"feels","object":"safe","object_type":"emotion","strength":0.8,"confidence":0.9,"source":"told","context":"when with gizmo"}}]'
            )
        }]

        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You extract typed knowledge edges from conversation exchanges. "
                "Be specific and grounded. JSON array only. No preamble."
            ),
            max_new_tokens=500,
            temperature=0.2,
        )

        if not raw or not raw.strip():
            return

        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()

        bracket = raw.find("[")
        if bracket > 0:
            raw = raw[bracket:]

        edges = json.loads(raw)

        if not isinstance(edges, list) or not edges:
            return

        from core.knowledge_graph import add_edge, ensure_entity

        # Ensure all fronters are registered as entities
        for name in ([host] + fronters) if host else fronters:
            if name:
                ensure_entity(name.lower(), "headmate")
        ensure_entity("gizmo", "ai")

        count = 0
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            subject   = edge.get("subject", "").strip().lower()
            predicate = edge.get("predicate", "").strip().lower()
            object_   = edge.get("object", "").strip().lower()
            if not subject or not predicate or not object_:
                continue

            add_edge(
                subject     = subject,
                predicate   = predicate,
                object_     = object_,
                object_type = edge.get("object_type", "concept"),
                strength    = float(edge.get("strength", 0.5)),
                confidence  = float(edge.get("confidence", 0.5)),
                source      = edge.get("source", "observed"),
                session_id  = session_id,
                context     = {"note": edge.get("context", ""), "topics": topics},
                tags        = topics,
            )
            count += 1

        log_event("GraphExtractor", "EXTRACTED",
            session  = session_id[:8],
            host     = host_str,
            edges    = count,
            topics   = topics,
        )

    except Exception as e:
        log_error("GraphExtractor", "extract_edges failed", exc=e)
