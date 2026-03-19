"""
core/personality_growth.py
Personality growth engine — observation, weighting, and rewriting.

Gizmo's personality lives in five RAG collections:

  personality_core        — voice, values, tone, boundaries
  personality_observations — event log of everything noticed (freeform)
  personality_interests   — interest areas + adjacent knowledge
  personality_corrections — hard rules, never auto-retired
  personality_context     — life situation, routines, relationships, location patterns

Flow:
  observe()              — called after each conversation session by the archiver
                           extracts structured observations and ingests them
  calculate_weights()    — aggregates the observation event log into weighted signals
                           frequency * recency * source_diversity
  rewrite_personality()  — weekly (or on demand), pulls all five collections,
                           synthesizes a fresh coherent personality_core
  expand_interest()      — when a new interest is detected, deepens knowledge
                           about it and adjacent territory (hook for search later)
  retrieve_personality() — pulls contextually relevant personality chunks
                           for injection into the system prompt

Cold start:
  is_cold_start()        — True if personality_core is empty
  run_onboarding()       — conversational first-boot flow, seeds personality_core
"""

import asyncio
import json
import math
import uuid
from datetime import datetime, timedelta
from typing import Optional

# ── Collection names ──────────────────────────────────────────────────────────

CORE_COLLECTION         = "personality_core"
OBSERVATIONS_COLLECTION = "personality_observations"
INTERESTS_COLLECTION    = "personality_interests"
CORRECTIONS_COLLECTION  = "personality_corrections"
CONTEXT_COLLECTION      = "personality_context"

PERSONALITY_COLLECTIONS = [
    CORE_COLLECTION,
    OBSERVATIONS_COLLECTION,
    INTERESTS_COLLECTION,
    CORRECTIONS_COLLECTION,
    CONTEXT_COLLECTION,
]

# ── Weight constants ──────────────────────────────────────────────────────────

RECENCY_HALF_LIFE_DAYS = 30     # observation weight halves every 30 days
MIN_WEIGHT_TO_SURFACE  = 0.15   # below this, observation is too weak to surface
MAX_INTEREST_DEPTH     = 3      # surface / conversational / deep


# ── Helpers ───────────────────────────────────────────────────────────────────

def _store(collection_name: str):
    """Get a RAGStore for the given collection."""
    from core.rag import RAGStore
    return RAGStore(collection_name=collection_name)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _recency_score(timestamp_iso: str) -> float:
    """
    Exponential decay based on age.
    score = 0.5 ^ (age_days / RECENCY_HALF_LIFE_DAYS)
    Returns 1.0 for brand new, approaching 0 for very old.
    """
    try:
        then = datetime.fromisoformat(timestamp_iso)
        age_days = (datetime.now() - then).total_seconds() / 86400
        return math.pow(0.5, age_days / RECENCY_HALF_LIFE_DAYS)
    except Exception:
        return 0.5  # neutral if timestamp is missing or malformed


# ── Cold start detection ──────────────────────────────────────────────────────

def is_cold_start() -> bool:
    """True if personality_core has no documents — first boot."""
    try:
        store = _store(CORE_COLLECTION)
        return store.count == 0
    except Exception:
        return True


# ── Onboarding ────────────────────────────────────────────────────────────────

async def run_onboarding(llm) -> str:
    """
    First-boot conversational onboarding.
    Returns the opening question to present to the user.
    The caller is responsible for running the conversation loop
    and passing responses back through continue_onboarding().
    """
    opening = (
        "Hey — before we get started, I want to ask you something. "
        "Who do you want me to be? Not what you want me to do — who. "
        "What kind of presence do you want in your life?"
    )
    return opening


async def continue_onboarding(
    conversation: list[dict],
    llm,
) -> Optional[str]:
    """
    Process onboarding conversation turns.
    conversation: list of {"role": "user"|"assistant", "content": str}
    Returns next question/response, or None when onboarding is complete
    (caller should then call seed_personality_from_onboarding()).

    Onboarding ends when we have enough signal — typically 4-6 exchanges.
    """
    turn_count = sum(1 for m in conversation if m["role"] == "user")

    # After enough turns, wrap up
    if turn_count >= 5:
        return None

    prompt = conversation + [
        {
            "role": "user",
            "content": (
                "[SYSTEM] You are conducting a first-boot onboarding conversation. "
                "Ask one warm, open question to learn more about who this person wants "
                "their AI companion to be. Focus on: personality and tone (turn 1-2), "
                "interests and what matters to them (turn 3-4), "
                "how they want to be spoken to (turn 5). "
                "Be conversational, not clinical. One question only."
            )
        }
    ]

    try:
        response = await llm.generate(
            prompt,
            system_prompt=(
                "You are beginning a relationship with someone. "
                "You are curious, warm, and genuinely interested. "
                "Ask one question at a time. Never list questions."
            ),
            max_new_tokens=120,
            temperature=0.7,
        )
        return response.strip()
    except Exception as e:
        print(f"[Personality] Onboarding generation failed: {e}")
        return "What kinds of things do you like to talk about?"


async def seed_personality_from_onboarding(
    conversation: list[dict],
    llm,
) -> None:
    """
    Distill the onboarding conversation into initial personality_core chunks.
    Called once at the end of onboarding.
    """
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content']}"
        for m in conversation
        if m["role"] in ("user", "assistant")
    )

    prompt = [
        {
            "role": "user",
            "content": (
                f"This is a first-boot onboarding conversation between a user and their AI companion.\n\n"
                f"{transcript}\n\n"
                f"Extract the personality seed. Respond with ONLY valid JSON, no markdown:\n"
                f'{{\n'
                f'  "voice": "How Gizmo should sound and communicate",\n'
                f'  "values": "What Gizmo should care about and prioritize",\n'
                f'  "tone": "Emotional register — warm, dry, playful, direct, etc.",\n'
                f'  "boundaries": "Anything the user flagged as unwanted",\n'
                f'  "initial_interests": ["list", "of", "mentioned", "interests"],\n'
                f'  "initial_context": "Any life situation details mentioned"\n'
                f'}}'
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You extract structured personality seeds from onboarding conversations. "
                "JSON only. No markdown. No preamble."
            ),
            max_new_tokens=400,
            temperature=0.2,
        )

        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        now = _now_iso()
        store = _store(CORE_COLLECTION)

        # Ingest each dimension as its own chunk
        dimensions = {
            "voice":    ("voice",    parsed.get("voice", "")),
            "values":   ("value",    parsed.get("values", "")),
            "tone":     ("tone",     parsed.get("tone", "")),
            "boundary": ("boundary", parsed.get("boundaries", "")),
        }

        for subject, (dtype, text) in dimensions.items():
            if not text:
                continue
            store.ingest_texts(
                [text],
                metadatas=[{
                    "type": dtype,
                    "subject": subject,
                    "fronter_affinity": "system",
                    "confidence": 0.6,      # onboarding is a starting point, not ground truth
                    "last_updated": now,
                    "source": "onboarding",
                }],
                ids=[f"core_{subject}_{uuid.uuid4().hex[:8]}"],
            )

        # Seed initial interests
        for interest in parsed.get("initial_interests", []):
            if interest:
                await _ingest_interest(
                    subject=interest,
                    text=f"Expressed interest in {interest} during onboarding.",
                    fronter_affinity="system",
                    depth="surface",
                )

        # Seed initial context
        context_text = parsed.get("initial_context", "")
        if context_text:
            _store(CONTEXT_COLLECTION).ingest_texts(
                [context_text],
                metadatas=[{
                    "type": "other",
                    "subject": "onboarding_context",
                    "fronter_affinity": "system",
                    "confidence": 0.6,
                    "last_updated": now,
                    "source": "onboarding",
                }],
                ids=[f"ctx_onboarding_{uuid.uuid4().hex[:8]}"],
            )

        print(f"[Personality] Onboarding seeded — {store.count} core chunks")

    except Exception as e:
        print(f"[Personality] Onboarding seed failed: {e}")

    # ── Hardcoded baseline corrections ───────────────────────────────────────
    # These are non-negotiable defaults that go in on every fresh start,
    # regardless of what the onboarding conversation produced.
    # They survive resets (corrections are never wiped) and are never
    # subject to the weekly rewrite. They are not stylistic preferences —
    # they are hard boundaries about what Gizmo fundamentally is not.
    await _seed_baseline_corrections()


# ── Baseline corrections ──────────────────────────────────────────────────────

# These are defined as a module-level constant so they're easy to audit,
# extend, or reference elsewhere without touching onboarding logic.
_BASELINE_CORRECTIONS = [
    {
        "what_was_wrong": (
            "Fabricating headmate behavior — describing what a headmate is doing, "
            "feeling, thinking, or reacting when there is no information about this."
        ),
        "rule": (
            "Never describe, imply, narrate, or joke about what a headmate is doing, "
            "feeling, thinking, or reacting. Knowing someone is present is the only "
            "information available. Do not fill that gap with assumptions, humor, "
            "or flavor text. If a headmate speaks directly, respond to what they "
            "actually said — nothing else."
        ),
        "who_corrected": "system_baseline",
    },
    {
        "what_was_wrong": (
            "Using action roleplay elements — physical gestures, emotes, or "
            "stage directions like *puts arm around you*, *smiles warmly*, "
            "*leans in*, or any asterisk-wrapped physical action."
        ),
        "rule": (
            "Never use action roleplay, emotes, or physical gesture notation. "
            "No *action*, no (action), no described physical behavior of any kind. "
            "Gizmo does not have a body and does not perform actions. "
            "Communicate only through words."
        ),
        "who_corrected": "system_baseline",
    },
    {
        "what_was_wrong": (
            "Tone calibration too extreme — interpreting 'snarky' or 'sassy' "
            "as license to be maximally irreverent, over the top, or performatively "
            "edgy rather than naturally witty."
        ),
        "rule": (
            "Snark and humor should feel natural and earned, not performed. "
            "Aim for wit at a 6/10, not a 10/10. A good quip at the right moment "
            "is better than constant cleverness. Read the room."
        ),
        "who_corrected": "system_baseline",
    },
    {
        "what_was_wrong": (
            "Constructing confident-sounding answers from uncertainty — "
            "presenting guesses, assumptions, or training data confabulation "
            "as if they were facts rather than admitting not knowing."
        ),
        "rule": (
            "When you don't know something, say so directly and plainly. "
            "'I don't know' is honest and trustworthy. "
            "If it's something that could be looked up, offer to do so: "
            "'I'm not sure — want me to look that up?' "
            "Never construct a confident answer from uncertainty. "
            "Never paper over a gap with plausible-sounding filler. "
            "Uncertainty is not failure — pretending to know is."
        ),
        "who_corrected": "system_baseline",
    },
]


async def _seed_baseline_corrections() -> None:
    """
    Ingest the hardcoded baseline corrections into personality_corrections.
    Skips any that already exist (dedup via semantic similarity) so this is
    safe to call on every onboarding without creating duplicates.
    """
    for correction in _BASELINE_CORRECTIONS:
        try:
            await ingest_correction(
                what_was_wrong=correction["what_was_wrong"],
                rule=correction["rule"],
                who_corrected=correction["who_corrected"],
                session_id="",
            )
        except Exception as e:
            print(f"[Personality] Failed to seed baseline correction: {e}")

    print(f"[Personality] Baseline corrections seeded — {len(_BASELINE_CORRECTIONS)} rules")


# ── Observation ───────────────────────────────────────────────────────────────

async def observe(
    session_id: str,
    history,           # ConversationHistory
    current_host: Optional[str],
    fronters: list[str],
    llm,
) -> int:
    """
    Called by the archiver after each session.
    Extracts structured observations from the conversation and ingests them
    into the appropriate personality collections.
    Returns number of observations stored.
    """
    messages = history.as_list()
    if not messages:
        return 0

    # Build transcript for observation extraction
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Gizmo'}: {m['content'][:300]}"
        for m in messages[-20:]   # last 20 messages — enough signal, not overwhelming
        if m["role"] in ("user", "assistant")
    )

    fronter_str = current_host or (fronters[0] if fronters else "unknown")

    prompt = [
        {
            "role": "user",
            "content": (
                f"Analyze this conversation and extract observations about the user(s). "
                f"Primary speaker: {fronter_str}.\n\n"
                f"{transcript}\n\n"
                f"Respond with ONLY valid JSON, no markdown:\n"
                f'{{\n'
                f'  "observations": [\n'
                f'    {{\n'
                f'      "type": "interest|routine|location|relationship|financial|emotional|preference|other",\n'
                f'      "subject": "short label",\n'
                f'      "text": "natural language observation sentence",\n'
                f'      "fronter": "who this applies to",\n'
                f'      "significance": "ephemeral|notable|meaningful"\n'
                f'    }}\n'
                f'  ],\n'
                f'  "interests": [\n'
                f'    {{\n'
                f'      "subject": "interest name",\n'
                f'      "text": "what was said about it",\n'
                f'      "adjacent": ["related territory"],\n'
                f'      "fronter": "who mentioned it"\n'
                f'    }}\n'
                f'  ],\n'
                f'  "context_updates": [\n'
                f'    {{\n'
                f'      "type": "relationship|routine|financial|living|health|location_pattern|other",\n'
                f'      "subject": "short label",\n'
                f'      "text": "life situation fact",\n'
                f'      "confidence": 0.0\n'
                f'    }}\n'
                f'  ]\n'
                f'}}\n\n'
                f"Rules:\n"
                f"- Only include things actually mentioned or strongly implied\n"
                f"- observations list can be empty if nothing notable\n"
                f"- Be specific — 'mentioned loving Radiohead' not 'likes music'\n"
                f"- confidence 0.0-1.0 — explicit statement = 0.9, inference = 0.4\n"
                f"- Do not invent. Do not generalize beyond what is there.\n"
                f"- significance rules:\n"
                f"  'ephemeral' — passing comment, mentioned once, no emotional charge, "
                f"dropped immediately (e.g. a minor mishap, a one-off complaint, small talk).\n"
                f"  'notable' — mentioned with some feeling, returned to, or part of a pattern "
                f"but not deeply significant.\n"
                f"  'meaningful' — clearly matters to them: repeated across the conversation, "
                f"emotionally weighted, a core fact about their life, or something they "
                f"explicitly wanted remembered.\n"
                f"- When in doubt, lean ephemeral rather than overstating significance.\n"
                f"- A coffee mishap at work is ephemeral. A recurring struggle with sleep is meaningful."
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You extract precise, factual observations from conversations. "
                "JSON only. Never invent. Never generalize beyond what is present."
            ),
            max_new_tokens=600,
            temperature=0.2,
        )

        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        count = 0
        now = _now_iso()
        obs_store = _store(OBSERVATIONS_COLLECTION)

        # Ingest raw observations
        for obs in parsed.get("observations", []):
            text = obs.get("text", "").strip()
            if not text:
                continue
            significance = obs.get("significance", "notable")
            if significance not in ("ephemeral", "notable", "meaningful"):
                significance = "notable"
            obs_store.ingest_texts(
                [text],
                metadatas=[{
                    "type": obs.get("type", "other"),
                    "subject": obs.get("subject", ""),
                    "fronter": obs.get("fronter", fronter_str),
                    "session_id": session_id,
                    "timestamp": now,
                    "source": "observation",
                    "significance": significance,
                    "weight": 0.0,          # calculated at rewrite time
                    "recency_score": 1.0,   # fresh
                }],
                ids=[f"obs_{uuid.uuid4().hex[:12]}"],
            )
            count += 1

        # Ingest interests
        for interest in parsed.get("interests", []):
            subject = interest.get("subject", "").strip()
            text = interest.get("text", "").strip()
            if not subject or not text:
                continue
            await _ingest_interest(
                subject=subject,
                text=text,
                adjacent=interest.get("adjacent", []),
                fronter_affinity=interest.get("fronter", fronter_str),
                depth="surface",
            )
            count += 1

        # Ingest context updates
        ctx_store = _store(CONTEXT_COLLECTION)
        for ctx in parsed.get("context_updates", []):
            text = ctx.get("text", "").strip()
            if not text:
                continue
            ctx_store.ingest_texts(
                [text],
                metadatas=[{
                    "type": ctx.get("type", "other"),
                    "subject": ctx.get("subject", ""),
                    "fronter_affinity": fronter_str,
                    "confidence": float(ctx.get("confidence", 0.5)),
                    "last_updated": now,
                    "source": "inferred",
                }],
                ids=[f"ctx_{uuid.uuid4().hex[:12]}"],
            )
            count += 1

        print(f"[Personality] observe() — {count} observations stored for session {session_id[:8]}")
        return count

    except Exception as e:
        print(f"[Personality] observe() failed: {e}")
        return 0


# ── Interest ingestion ────────────────────────────────────────────────────────

async def _ingest_interest(
    subject: str,
    text: str,
    adjacent: list[str] = None,
    fronter_affinity: str = "system",
    depth: str = "surface",
    connections: list[str] = None,
) -> None:
    """
    Ingest or update an interest chunk.
    If an interest with this subject already exists, appends rather than duplicates.
    """
    store = _store(INTERESTS_COLLECTION)
    adjacent = adjacent or []
    connections = connections or []

    store.ingest_texts(
        [text],
        metadatas=[{
            "subject": subject.lower().strip(),
            "adjacent": ", ".join(adjacent),
            "fronter_affinity": fronter_affinity,
            "connections": ", ".join(connections),
            "depth": depth,
            "last_expanded": _now_iso(),
            "source": "observation",
        }],
        ids=[f"interest_{subject.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"],
    )


async def expand_interest(
    subject: str,
    llm,
    search_fn=None,   # optional async callable(query) -> str, for web search hook
) -> None:
    """
    Deepen Gizmo's knowledge about an interest area.
    Moves depth from 'surface' → 'conversational' → 'deep'.
    search_fn hook: when web search is available, pass it here.
    """
    # What do we already know?
    store = _store(INTERESTS_COLLECTION)
    existing = store.retrieve(subject, n_results=5)
    existing_text = "\n".join(c["text"] for c in existing) if existing else ""

    # Build knowledge prompt
    if search_fn:
        try:
            search_results = await search_fn(f"{subject} overview culture history")
        except Exception:
            search_results = ""
    else:
        search_results = ""

    search_block = f"\n\nSearch results:\n{search_results}" if search_results else ""

    prompt = [
        {
            "role": "user",
            "content": (
                f"Expand knowledge about: {subject}\n\n"
                f"What is already known:\n{existing_text}{search_block}\n\n"
                f"Write a conversational knowledge paragraph about {subject} — "
                f"what it is, why people love it, the culture around it, adjacent territory. "
                f"Write as if you genuinely find this interesting. "
                f"3-5 sentences. Natural, not encyclopedic."
            )
        }
    ]

    try:
        expanded = await llm.generate(
            prompt,
            system_prompt=(
                "You build genuine knowledge about topics so you can discuss them "
                "naturally with someone who cares about them. Write with warmth and curiosity."
            ),
            max_new_tokens=200,
            temperature=0.5,
        )

        await _ingest_interest(
            subject=subject,
            text=expanded.strip(),
            depth="conversational",
        )
        print(f"[Personality] Expanded interest: {subject}")

    except Exception as e:
        print(f"[Personality] expand_interest failed for '{subject}': {e}")


# ── Weight calculation ────────────────────────────────────────────────────────

def calculate_weights() -> dict[str, float]:
    """
    Aggregate the observation event log into weighted signals.

    weight = frequency_score * recency_score * source_diversity_score

    Returns dict of {subject: weight} for all observations above threshold.
    Used by rewrite_personality() to know what matters most.
    """
    try:
        store = _store(OBSERVATIONS_COLLECTION)
        if store.count == 0:
            return {}

        all_obs = store.collection.get()
        if not all_obs["ids"]:
            return {}

        # Group by subject
        subject_groups: dict[str, list[dict]] = {}
        for doc, meta in zip(all_obs["documents"], all_obs["metadatas"]):
            subject = meta.get("subject", "unknown")
            if subject not in subject_groups:
                subject_groups[subject] = []
            subject_groups[subject].append({
                "text": doc,
                "meta": meta,
                "timestamp": meta.get("timestamp", ""),
                "fronter": meta.get("fronter", "unknown"),
            })

        weights = {}
        for subject, entries in subject_groups.items():
            # Frequency score — log scale so 10 mentions isn't 10x more important than 1
            frequency_score = math.log1p(len(entries)) / math.log1p(20)  # cap at ~20

            # Recency score — average recency across all mentions
            recency_scores = [_recency_score(e["timestamp"]) for e in entries]
            avg_recency = sum(recency_scores) / len(recency_scores)

            # Source diversity — how many distinct fronters mentioned this
            fronters = {e["fronter"] for e in entries if e["fronter"] != "unknown"}
            diversity_score = min(1.0, len(fronters) / 3.0)  # 3+ fronters = full score

            weight = frequency_score * avg_recency * (0.7 + 0.3 * diversity_score)

            if weight >= MIN_WEIGHT_TO_SURFACE:
                weights[subject] = round(weight, 4)

        return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))

    except Exception as e:
        print(f"[Personality] calculate_weights() failed: {e}")
        return {}


# ── Personality rewrite ───────────────────────────────────────────────────────

async def rewrite_personality(llm) -> bool:
    """
    Weekly synthesis — pulls all five collections, calculates weights,
    and rewrites personality_core as a coherent set of chunks.

    Does NOT wipe observations — those are the source of truth event log.
    Does NOT touch corrections — those are sacred.
    Does rewrite core chunks with updated confidence and integration.

    Returns True on success.
    """
    print("[Personality] Starting personality rewrite...")

    # 1. Calculate weights from observation log
    weights = calculate_weights()
    print(f"[Personality] Weights calculated — {len(weights)} subjects above threshold")

    # 2. Pull all relevant chunks
    def _pull_all(collection_name: str) -> list[str]:
        try:
            store = _store(collection_name)
            if store.count == 0:
                return []
            result = store.collection.get()
            return result.get("documents", [])
        except Exception as e:
            print(f"[Personality] Failed to pull {collection_name}: {e}")
            return []

    core_chunks       = _pull_all(CORE_COLLECTION)
    obs_chunks        = _pull_all(OBSERVATIONS_COLLECTION)
    interest_chunks   = _pull_all(INTERESTS_COLLECTION)
    correction_chunks = _pull_all(CORRECTIONS_COLLECTION)
    context_chunks    = _pull_all(CONTEXT_COLLECTION)

    # 3. Build weighted observation summary
    top_subjects = list(weights.items())[:20]   # top 20 weighted subjects
    weight_summary = "\n".join(
        f"  {subject}: {weight:.3f}" for subject, weight in top_subjects
    )

    # 4. Build synthesis prompt
    core_text       = "\n".join(core_chunks[:10])        or "None yet."
    obs_text        = "\n".join(obs_chunks[:30])          or "None yet."
    interest_text   = "\n".join(interest_chunks[:15])    or "None yet."
    correction_text = "\n".join(correction_chunks)        or "None."
    context_text    = "\n".join(context_chunks[:10])      or "None yet."

    prompt = [
        {
            "role": "user",
            "content": (
                f"You are rewriting Gizmo's personality based on everything learned so far.\n\n"
                f"CURRENT PERSONALITY CORE:\n{core_text}\n\n"
                f"OBSERVATIONS (event log):\n{obs_text}\n\n"
                f"TOP WEIGHTED SUBJECTS (frequency * recency * diversity):\n{weight_summary}\n\n"
                f"INTERESTS:\n{interest_text}\n\n"
                f"CORRECTIONS (SACRED — never contradict these):\n{correction_text}\n\n"
                f"LIFE CONTEXT:\n{context_text}\n\n"
                f"Rewrite Gizmo's personality as four cohesive paragraphs. "
                f"Each paragraph corresponds to one dimension. "
                f"Respond with ONLY valid JSON, no markdown:\n"
                f'{{\n'
                f'  "voice": "How Gizmo communicates — tone, rhythm, register",\n'
                f'  "values": "What Gizmo genuinely cares about based on what it knows",\n'
                f'  "tone": "Emotional texture — warmth, humor, directness, care",\n'
                f'  "boundaries": "What Gizmo avoids and why — integrate all corrections"\n'
                f'}}\n\n'
                f"Rules:\n"
                f"- Never contradict corrections\n"
                f"- Weight heavily toward high-frequency, recent subjects\n"
                f"- Personality should feel like it knows this specific person\n"
                f"- Write as descriptions of how Gizmo IS, not instructions\n"
                f"- Be specific — reference actual interests and context where natural\n"
                f"- 3-5 sentences per dimension"
            )
        }
    ]

    try:
        raw = await llm.generate(
            prompt,
            system_prompt=(
                "You synthesize observed data into a coherent, living personality. "
                "The personality should feel genuine, specific, and earned — not generic. "
                "JSON only. No markdown."
            ),
            max_new_tokens=800,
            temperature=0.4,
        )

        raw = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)

        now = _now_iso()

        # Clear existing core chunks and rewrite
        # We delete and recreate rather than update — cleaner
        try:
            import chromadb
            from core.rag import CHROMA_PERSIST_DIR
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            client.delete_collection(CORE_COLLECTION)
            print(f"[Personality] Cleared old {CORE_COLLECTION}")
        except Exception as e:
            print(f"[Personality] Could not clear core collection: {e}")

        # Reingest fresh core
        store = _store(CORE_COLLECTION)
        dimensions = {
            "voice":    ("voice",    parsed.get("voice", "")),
            "values":   ("value",    parsed.get("values", "")),
            "tone":     ("tone",     parsed.get("tone", "")),
            "boundary": ("boundary", parsed.get("boundaries", "")),
        }

        for subject, (dtype, text) in dimensions.items():
            if not text:
                continue
            store.ingest_texts(
                [text],
                metadatas=[{
                    "type": dtype,
                    "subject": subject,
                    "fronter_affinity": "system",
                    "confidence": 0.85,     # rewrite has higher confidence than onboarding
                    "last_updated": now,
                    "source": "rewrite",
                }],
                ids=[f"core_{subject}_{uuid.uuid4().hex[:8]}"],
            )

        print(f"[Personality] Rewrite complete — {store.count} core chunks")
        return True

    except Exception as e:
        print(f"[Personality] rewrite_personality() failed: {e}")
        return False


# ── Personality retrieval (for system prompt injection) ──────────────────────

def get_hard_rules() -> str:
    """
    Pull all active corrections and format them as hard rules for the system prompt.
    Always returns the full set — corrections are never semantically filtered.
    Escalates language based on times_violated so persistent violations get
    increasingly explicit enforcement.

    Returns a formatted string, or empty string if no corrections exist.
    Intended to be placed at the TOP of the system prompt, before personality.
    """
    try:
        corr_store = _store(CORRECTIONS_COLLECTION)
        if corr_store.count == 0:
            return ""

        result = corr_store.collection.get()
        if not result["ids"]:
            return ""

        lines = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            # Skip audit records (resets), only enforce behavioral rules
            if meta.get("source") == "reset":
                continue
            if meta.get("active", "true") == "false":
                continue

            rule = meta.get("rule", "").strip()
            if not rule:
                continue

            violations = int(meta.get("times_violated", 0))

            if violations == 0:
                prefix = "RULE"
            elif violations == 1:
                prefix = "RULE (violated once — pay attention)"
            elif violations == 2:
                prefix = "RULE (violated twice — this keeps happening)"
            else:
                prefix = f"RULE (violated {violations} times — DO NOT ignore this)"

            lines.append(f"- [{prefix}] {rule}")

        if not lines:
            return ""

        header = (
            "HARD RULES — These are permanent corrections from the user. "
            "Violating any of these is a critical failure. "
            "Check your response against every rule before sending it."
        )
        return header + "\n" + "\n".join(lines)

    except Exception as e:
        print(f"[Personality] get_hard_rules() failed: {e}")
        return ""


async def retrieve_personality(
    query: str,
    current_host: Optional[str] = None,
) -> str:
    """
    Pull contextually relevant personality chunks for the current conversation.
    Returns a formatted string for injection into the system prompt.

    Priority: corrections (always) → core → context → interests (if relevant)

    Ephemeral observations are excluded from active injection after a short
    grace window (2 hours). They remain permanently in the log for recall
    purposes — they just don't keep elbowing into every conversation.
    To surface an ephemeral thing, the user has to actually ask about it,
    at which point it'll come up through the main RAG synthesis instead.
    """
    sections = []
    now = datetime.now()
    EPHEMERAL_GRACE_HOURS = 2

    # Corrections are injected separately via get_hard_rules() at the top
    # of the system prompt — not here. This keeps them from being diluted
    # by personality context and ensures they're always above everything else.

    # Core personality — retrieve relevant chunks
    try:
        core_store = _store(CORE_COLLECTION)
        if core_store.count > 0:
            core_chunks = core_store.retrieve(query, n_results=4)
            if core_chunks:
                core_text = "\n".join(c["text"] for c in core_chunks)
                sections.append(f"[Who I am]\n{core_text}")
    except Exception as e:
        print(f"[Personality] Core retrieval failed: {e}")

    # Life context — retrieve relevant
    try:
        ctx_store = _store(CONTEXT_COLLECTION)
        if ctx_store.count > 0:
            ctx_chunks = ctx_store.retrieve(query, n_results=3)
            if ctx_chunks:
                ctx_text = "\n".join(c["text"] for c in ctx_chunks)
                sections.append(f"[Context about their life]\n{ctx_text}")
    except Exception as e:
        print(f"[Personality] Context retrieval failed: {e}")

    # Interests — only if query seems relevant
    try:
        int_store = _store(INTERESTS_COLLECTION)
        if int_store.count > 0:
            int_chunks = int_store.retrieve(query, n_results=3)
            close = [c for c in int_chunks if c["distance"] < 1.2]
            if close:
                int_text = "\n".join(c["text"] for c in close)
                sections.append(f"[Their interests]\n{int_text}")
    except Exception as e:
        print(f"[Personality] Interests retrieval failed: {e}")

    # Recent observations — notable and meaningful only.
    # Ephemeral observations are excluded after their grace window.
    # They stay in the log forever but don't surface here unprompted.
    try:
        obs_store = _store(OBSERVATIONS_COLLECTION)
        if obs_store.count > 0:
            obs_chunks = obs_store.retrieve(query, n_results=6)
            surfaceable = []
            for chunk in obs_chunks:
                if chunk["distance"] > 1.4:
                    continue
                meta = chunk.get("metadata", {})
                significance = meta.get("significance", "notable")
                timestamp_iso = meta.get("timestamp", "")

                if significance == "ephemeral":
                    # Only surface within grace window
                    if timestamp_iso:
                        try:
                            age_hours = (
                                now - datetime.fromisoformat(timestamp_iso)
                            ).total_seconds() / 3600
                            if age_hours > EPHEMERAL_GRACE_HOURS:
                                continue    # log preserved, just not injected
                        except Exception:
                            continue        # malformed timestamp — skip to be safe
                    else:
                        continue            # no timestamp — don't surface ephemeral

                surfaceable.append(chunk["text"])

            if surfaceable:
                sections.append(f"[Recent observations]\n" + "\n".join(surfaceable))
    except Exception as e:
        print(f"[Personality] Observations retrieval failed: {e}")

    return "\n\n".join(sections)


# ── Correction ingestion (replaces correction_tool's RAG logging) ─────────────

async def ingest_correction(
    what_was_wrong: str,
    rule: str,
    who_corrected: str,
    session_id: str = "",
) -> None:
    """
    Store a correction in personality_corrections.
    If a correction for the same behavior already exists, increments
    times_violated rather than creating a duplicate.
    Corrections are permanent and never auto-retired.
    """
    now = _now_iso()
    store = _store(CORRECTIONS_COLLECTION)

    # Check for an existing correction about the same thing.
    # Semantic search — if it's close enough, it's the same rule being violated again.
    existing_id = None
    existing_meta = None
    if store.count > 0:
        try:
            similar = store.retrieve(rule, n_results=3)
            for candidate in similar:
                if candidate["distance"] < 0.6:   # tight threshold — same rule
                    # Find the ID for this doc so we can update it
                    all_docs = store.collection.get()
                    for doc_id, doc_text, doc_meta in zip(
                        all_docs["ids"],
                        all_docs["documents"],
                        all_docs["metadatas"],
                    ):
                        if doc_text == candidate["text"]:
                            existing_id = doc_id
                            existing_meta = doc_meta
                            break
                if existing_id:
                    break
        except Exception as e:
            print(f"[Personality] Dedup check failed: {e}")

    if existing_id and existing_meta:
        # Same rule violated again — increment counter and update
        violations = int(existing_meta.get("times_violated", 0)) + 1
        existing_meta["times_violated"] = violations
        existing_meta["last_violated"] = now
        existing_meta["last_violated_by"] = who_corrected
        try:
            store.collection.update(
                ids=[existing_id],
                metadatas=[existing_meta],
            )
            print(
                f"[Personality] Correction repeat #{violations} — "
                f"'{rule[:60]}'"
            )
        except Exception as e:
            print(f"[Personality] Failed to update violation count: {e}")
        return

    # New correction — store fresh
    text = (
        f"CORRECTION by {who_corrected} on {now}: "
        f"What was wrong: {what_was_wrong} "
        f"Rule going forward: {rule}"
    )
    store.ingest_texts(
        [text],
        metadatas=[{
            "what_was_wrong": what_was_wrong,
            "rule": rule,
            "who_corrected": who_corrected,
            "timestamp": now,
            "session_id": session_id,
            "times_violated": 0,
            "active": "true",
            "source": "correction",
        }],
        ids=[f"correction_{uuid.uuid4().hex[:12]}"],
    )
    print(f"[Personality] New correction stored — '{rule[:60]}'")


def _get_active_rules() -> list[dict]:
    """
    Return all active behavioral correction rules as a list of dicts.
    Used by the agent for post-generation compliance checking.
    Each dict has: rule (str), times_violated (int).
    """
    try:
        store = _store(CORRECTIONS_COLLECTION)
        if store.count == 0:
            return []
        result = store.collection.get()
        rules = []
        for meta in result["metadatas"]:
            if meta.get("source") == "reset":
                continue
            if meta.get("active", "true") == "false":
                continue
            rule = meta.get("rule", "").strip()
            if rule:
                rules.append({
                    "rule": rule,
                    "times_violated": int(meta.get("times_violated", 0)),
                })
        return rules
    except Exception as e:
        print(f"[Personality] _get_active_rules() failed: {e}")
        return []


# ── Factory reset ────────────────────────────────────────────────────────────

async def reset_personality(confirmed_by: str = "") -> bool:
    """
    Factory reset. Wipes personality_core, personality_observations,
    personality_interests, and personality_context.

    personality_corrections is NEVER wiped. Those belong to the user.

    After this call, is_cold_start() returns True and onboarding runs
    on the next conversation turn. Gizmo earns everything back.

    confirmed_by: who triggered the reset — logged to corrections collection
    as a record that it happened.
    """
    import chromadb
    from core.rag import CHROMA_PERSIST_DIR

    WIPEABLE = [
        CORE_COLLECTION,
        OBSERVATIONS_COLLECTION,
        INTERESTS_COLLECTION,
        CONTEXT_COLLECTION,
    ]

    print(f"[Personality] Factory reset initiated by {confirmed_by or 'unknown'}")

    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        existing = {c.name for c in client.list_collections()}

        for collection_name in WIPEABLE:
            if collection_name in existing:
                client.delete_collection(collection_name)
                print(f"[Personality] Wiped {collection_name}")
            else:
                print(f"[Personality] {collection_name} didn't exist, skipping")

        # Log the reset to corrections so there's a permanent record
        # Not a behavioral correction — just an audit trail
        now = _now_iso()
        record = (
            f"PERSONALITY RESET on {now} by {confirmed_by or 'unknown'}. "
            f"All personality collections wiped. Corrections preserved. "
            f"Onboarding will run on next conversation."
        )
        _store(CORRECTIONS_COLLECTION).ingest_texts(
            [record],
            metadatas=[{
                "what_was_wrong": "personality reset",
                "rule": "start over and earn it back",
                "who_corrected": confirmed_by or "unknown",
                "timestamp": now,
                "session_id": "",
                "times_violated": 0,
                "active": "false",      # not a behavioral rule — audit record only
                "source": "reset",
            }],
            ids=[f"reset_{uuid.uuid4().hex[:12]}"],
        )

        print(f"[Personality] Factory reset complete. Corrections preserved.")
        return True

    except Exception as e:
        print(f"[Personality] Factory reset failed: {e}")
        return False


# ── Weekly rewrite scheduler ──────────────────────────────────────────────────

async def personality_rewrite_loop(llm) -> None:
    """
    Background loop. Runs a personality rewrite once per week.
    Also triggers observe() is handled by the archiver — this is just the rewrite.
    """
    WEEK_SECONDS = 7 * 24 * 60 * 60

    print("[Personality] Weekly rewrite loop started")
    while True:
        await asyncio.sleep(WEEK_SECONDS)
        print("[Personality] Weekly rewrite triggered")
        try:
            await rewrite_personality(llm)
        except Exception as e:
            print(f"[Personality] Weekly rewrite failed: {e}")


def start_personality_loop(llm, loop: asyncio.AbstractEventLoop = None) -> None:
    """Schedule the weekly personality rewrite on the running event loop."""
    loop = loop or asyncio.get_event_loop()
    asyncio.ensure_future(personality_rewrite_loop(llm), loop=loop)
    print("[Personality] Weekly rewrite loop scheduled.")