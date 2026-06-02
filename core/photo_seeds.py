"""
core/photo_seeds.py

Photo seed system. Detects when something visually notable happens in
public and might have been photographed. Builds a full scene description
from telemetry body data, then passes to media for caption generation
and platform posting.

A photo seed is born in the world reactor when:
  - Someone is notably undressed in public
  - An intimate or sexual act is visible
  - Something visually unusual is happening
  - Bystander presence is detected

Each subject in the scene gets a full body description built from their
PersonObject — clothing state, modifiers, manner, recent descriptors.
The caption generator receives this full description and produces something
that makes sense for what's actually in the frame.

Same photo can be posted multiple times with different captions by
different people — each feeding different culture threads.

Nudity and intimacy multiply platform traction significantly:
  Flick:  likes ×3-5, saves ×6, extends reach to adjacent circles
  Nexus:  multiple circle cross-posts, long comment threads
  Pulse:  half-life extends 6hr→24hr, repost rate ×4

Usage:
    from core.photo_seeds import photo_seed_engine

    # Called from world reactor after each observation
    await photo_seed_engine.check(
        session_id, loc, telem, observation_text, llm
    )

    # Called from media hourly pass
    await photo_seed_engine.process_queue(llm)
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.log import log_event, log_error
from core.timezone import tz_now


# ── Paths ─────────────────────────────────────────────────────────────────────

def _queue_path() -> Path:
    return Path("/data/personality/social/photo_queue.json")


# ── Visual register detection ──────────────────────────────────────────────────

# Tags that suggest something visually notable happened
_NOTABLE_TAGS = {
    "naked", "nude", "undressed", "intimate", "sexual", "crawling",
    "kneeling", "leashed", "on_all_fours", "bound", "exposed",
    "strip", "undress", "collar", "public_display",
}

_INTIMACY_TAGS = {
    "intimate", "sexual", "aroused", "subby", "dominant",
    "touching", "kissing", "grinding", "explicit",
}

# Visual register of a scene
_VISUAL_REGISTERS = {
    "naked_public":   {"traction_mult": 4.0, "save_mult": 6.0},
    "intimate":       {"traction_mult": 3.5, "save_mult": 5.0},
    "leashed":        {"traction_mult": 3.0, "save_mult": 4.0},
    "unusual":        {"traction_mult": 1.5, "save_mult": 2.0},
    "ordinary":       {"traction_mult": 0.3, "save_mult": 0.5},
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SubjectDescription:
    """Full description of one person in a photo, built from their PersonObject."""
    name:          str  = ""
    person_type:   str  = "headmate"

    # Physical state at capture time
    clothing:      str  = "unknown"
    clothing_detail: str = ""
    modifiers:     list = field(default_factory=list)
    body_desc:     str  = ""      # generated from PersonObject data
    manner:        str  = ""      # how they were moving/being
    descriptors:   list = field(default_factory=list)  # from recent actions
    position:      str  = ""      # standing, crawling, kneeling, etc.
    expression:    str  = ""      # emotional register visible in photo

    def to_prompt_str(self) -> str:
        lines = [f"{self.name}:"]
        if self.clothing == "naked":
            lines.append("  naked")
        elif self.clothing:
            lines.append(f"  wearing: {self.clothing_detail or self.clothing}")
        if self.modifiers:
            lines.append(f"  modifiers: {', '.join(self.modifiers)}")
        if self.position:
            lines.append(f"  position: {self.position}")
        if self.manner:
            lines.append(f"  manner: {self.manner}")
        if self.descriptors:
            lines.append(f"  descriptors: {', '.join(self.descriptors[:4])}")
        if self.expression:
            lines.append(f"  expression/register: {self.expression}")
        if self.body_desc:
            lines.append(f"  appearance: {self.body_desc}")
        return "\n".join(lines)


@dataclass
class PhotoSeed:
    """A potential photo of a notable scene."""
    seed_id:        str   = field(default_factory=lambda: f"photo_{uuid.uuid4().hex[:8]}")
    captured_at:    float = field(default_factory=time.time)

    location:       str   = ""
    scene_context:  str   = ""    # what was happening (from now_block)
    visual_register: str  = "ordinary"
    subjects:       list  = field(default_factory=list)   # list of SubjectDescription

    # Bystander info
    bystanders_present: bool  = False
    bystander_count:    int   = 0
    phones_out:         bool  = False

    # Processing state
    processed:      bool  = False
    posts_created:  list  = field(default_factory=list)   # post_ids

    def to_dict(self) -> dict:
        return {
            "seed_id":           self.seed_id,
            "captured_at":       self.captured_at,
            "location":          self.location,
            "scene_context":     self.scene_context,
            "visual_register":   self.visual_register,
            "subjects":          [
                {
                    "name":           s.name,
                    "person_type":    s.person_type,
                    "clothing":       s.clothing,
                    "clothing_detail": s.clothing_detail,
                    "modifiers":      s.modifiers,
                    "body_desc":      s.body_desc,
                    "manner":         s.manner,
                    "descriptors":    s.descriptors,
                    "position":       s.position,
                    "expression":     s.expression,
                }
                for s in self.subjects
            ],
            "bystanders_present": self.bystanders_present,
            "bystander_count":    self.bystander_count,
            "phones_out":         self.phones_out,
            "processed":          self.processed,
            "posts_created":      self.posts_created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PhotoSeed":
        seed = cls(
            seed_id          = d.get("seed_id", f"photo_{uuid.uuid4().hex[:8]}"),
            captured_at      = d.get("captured_at", time.time()),
            location         = d.get("location", ""),
            scene_context    = d.get("scene_context", ""),
            visual_register  = d.get("visual_register", "ordinary"),
            bystanders_present = d.get("bystanders_present", False),
            bystander_count  = d.get("bystander_count", 0),
            phones_out       = d.get("phones_out", False),
            processed        = d.get("processed", False),
            posts_created    = d.get("posts_created", []),
        )
        seed.subjects = [
            SubjectDescription(**{
                k: v for k, v in s.items()
                if k in SubjectDescription.__dataclass_fields__
            })
            for s in d.get("subjects", [])
        ]
        return seed


# ── Photo seed engine ─────────────────────────────────────────────────────────

class PhotoSeedEngine:

    # Probability of someone having their phone out by visual register
    _PHONE_PROB = {
        "naked_public": 0.80,
        "intimate":     0.60,
        "leashed":      0.70,
        "unusual":      0.40,
        "ordinary":     0.05,
    }

    # Probability of posting given phone out
    _POST_PROB = {
        "naked_public": 0.75,
        "intimate":     0.65,
        "leashed":      0.70,
        "unusual":      0.35,
        "ordinary":     0.10,
    }

    # Number of independent posters (same photo, different captions)
    _POSTER_COUNT = {
        "naked_public": (1, 4),
        "intimate":     (1, 3),
        "leashed":      (1, 3),
        "unusual":      (1, 2),
        "ordinary":     (0, 1),
    }

    def __init__(self):
        self._queue: list = []
        self._load()

    def _load(self) -> None:
        try:
            p = _queue_path()
            if p.exists():
                data = json.loads(p.read_text())
                self._queue = [
                    PhotoSeed.from_dict(s)
                    for s in data.get("seeds", [])
                    if not s.get("processed")
                ]
        except Exception as e:
            log_error("PhotoSeedEngine", "queue load failed", exc=e)

    def _save(self) -> None:
        try:
            p = _queue_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(
                {"seeds": [s.to_dict() for s in self._queue[-50:]]},
                indent=2,
            ))
        except Exception as e:
            log_error("PhotoSeedEngine", "queue save failed", exc=e)

    # ── Detection ─────────────────────────────────────────────────────────────

    async def check(
        self,
        session_id:       str,
        loc,              # SessionLocation
        telem,            # SessionTelemetry
        observation_text: str,
        llm,
    ) -> Optional[PhotoSeed]:
        """
        Called from world reactor after each observation.
        Checks if a photo seed should be planted.
        Returns a PhotoSeed if planted, None otherwise.
        """
        if not telem or not loc:
            return None

        # Must be in public with bystanders possible
        location = loc.gizmo_location or loc.user_location
        if not location:
            return None

        # Determine visual register from person states
        visual_register = self._assess_visual_register(telem)

        # Check if phones are plausibly out
        phone_prob = self._PHONE_PROB.get(visual_register, 0.05)

        # Bystanders in observation text increases probability
        bystander_signals = [
            "watching", "staring", "noticed", "looked", "clocked",
            "slowed", "stopped", "crowd", "people", "someone",
        ]
        bystander_count = sum(
            1 for signal in bystander_signals
            if signal in observation_text.lower()
        )
        if bystander_count > 0:
            phone_prob = min(1.0, phone_prob + bystander_count * 0.1)

        if random.random() > phone_prob:
            return None

        # Build subject descriptions from person objects
        subjects = await self._build_subject_descriptions(telem, llm)
        if not subjects:
            return None

        seed = PhotoSeed(
            location         = location,
            scene_context    = telem.now_block(),
            visual_register  = visual_register,
            subjects         = subjects,
            bystanders_present = bystander_count > 0,
            bystander_count  = bystander_count,
            phones_out       = True,
        )

        self._queue.append(seed)
        self._save()

        log_event("PhotoSeedEngine", "SEED_PLANTED",
            location        = location,
            visual_register = visual_register,
            subjects        = [s.name for s in subjects],
        )

        return seed

    def _assess_visual_register(self, telem) -> str:
        """Determine what register this scene is in visually."""
        if not telem:
            return "ordinary"

        all_flags  = set()
        all_tags   = set()
        all_mods   = set()

        for person in telem.persons.values():
            if person.person_type == "gizmo":
                continue
            # Flags
            all_flags.update(person.active_flags.keys())
            # Modifiers
            all_mods.update(person.modifiers)
            # Recent action tags
            for action in person.actions[-5:]:
                all_tags.update(action.tags)
                all_tags.update(action.descriptors)

        all_signals = all_flags | all_tags | all_mods

        # Check registers in priority order
        if "naked" in all_flags or "naked" in all_signals:
            if all_signals & _INTIMACY_TAGS:
                return "intimate"
            return "naked_public"

        if all_signals & _INTIMACY_TAGS:
            return "intimate"

        if "on_leash" in all_mods or "leashed" in all_signals:
            return "leashed"

        if all_signals & _NOTABLE_TAGS:
            return "unusual"

        return "ordinary"

    async def _build_subject_descriptions(
        self,
        telem,
        llm,
    ) -> list:
        """
        Build SubjectDescription for each non-Gizmo person in the scene.
        Uses PersonObject data + one LLM call per person for body_desc.
        """
        subjects = []

        for key, person in telem.persons.items():
            if person.person_type == "gizmo":
                continue
            if person.clothing == "unknown" and not person.active_flags:
                continue

            # Build description from telemetry data
            recent_actions = person.actions[-3:]
            manner = ""
            descriptors = []
            position = ""

            for action in recent_actions:
                if action.manner and not manner:
                    manner = action.manner
                descriptors.extend(action.descriptors[:2])
                # Infer position from action tags/subcategory
                if "crawl" in action.subcategory or "on_all_fours" in action.tags:
                    position = "on all fours"
                elif "kneel" in action.subcategory or "kneeling" in action.tags:
                    position = "kneeling"
                elif "stand" in action.subcategory:
                    position = "standing"

            # One LLM call per person — what do they look like in this photo?
            body_desc = await self._generate_body_description(
                person   = person,
                manner   = manner,
                position = position,
                llm      = llm,
            )

            subject = SubjectDescription(
                name            = person.name,
                person_type     = person.person_type,
                clothing        = person.clothing,
                clothing_detail = person.clothing_detail,
                modifiers       = person.modifiers[:],
                body_desc       = body_desc,
                manner          = manner,
                descriptors     = list(set(descriptors))[:5],
                position        = position,
                expression      = person.current_register,
            )
            subjects.append(subject)

        return subjects

    async def _generate_body_description(
        self,
        person,
        manner:   str,
        position: str,
        llm,
    ) -> str:
        """
        One LLM call: given what we know about this person right now,
        what would they look like in a photo?
        """
        # Build input from person object
        state_parts = []
        if person.clothing == "naked":
            state_parts.append("naked")
        elif person.clothing_detail:
            state_parts.append(f"wearing {person.clothing_detail}")
        if person.modifiers:
            state_parts.append(f"modifiers: {', '.join(person.modifiers[:4])}")
        if position:
            state_parts.append(f"position: {position}")
        if manner:
            state_parts.append(f"manner: {manner}")
        if person.emotional_note:
            state_parts.append(f"emotional state: {person.emotional_note}")

        if not state_parts:
            return ""

        state_str = "\n".join(state_parts)

        prompt = f"""Describe what {person.name} looks like in a photograph right now.

Current state:
{state_str}

Write 1-2 sentences. Visual only — what a camera captures.
Specific. Present tense. No interpretation, no judgment.
What you'd actually see in the photo."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You write precise visual descriptions for photographs. "
                    "1-2 sentences. What the camera sees. No judgment."
                ),
                max_new_tokens=80,
                temperature=0.6,
            )
            return raw.strip() if raw else ""
        except Exception as e:
            log_error("PhotoSeedEngine", f"body desc failed: {person.name}", exc=e)
            return ""

    # ── Processing — called from media hourly pass ────────────────────────────

    async def process_queue(self, llm) -> list:
        """
        Process unprocessed photo seeds — generate captions and create posts.
        Called from media engine hourly pass.
        Returns list of created Post objects.
        """
        from core.media import media_engine

        unprocessed = [s for s in self._queue if not s.processed]
        if not unprocessed:
            return []

        created_posts = []

        for seed in unprocessed:
            posts = await self._generate_posts_for_seed(seed, llm, media_engine)
            created_posts.extend(posts)
            seed.processed     = True
            seed.posts_created = [p.post_id for p in posts]

        self._save()
        return created_posts

    async def _generate_posts_for_seed(
        self,
        seed:         PhotoSeed,
        llm,
        media_engine,
    ) -> list:
        """
        Generate 1-N posts for a photo seed.
        Each post is the same photo, different poster, different caption.
        """
        # Decide how many independent posters
        count_range = self._POSTER_COUNT.get(seed.visual_register, (0, 1))
        post_prob   = self._POST_PROB.get(seed.visual_register, 0.1)

        if random.random() > post_prob:
            return []

        n_posters = random.randint(*count_range)
        if n_posters == 0:
            return []

        # Choose platforms weighted by register
        if seed.visual_register in ("naked_public", "intimate", "leashed"):
            platform_weights = {"flick": 0.55, "pulse": 0.30, "nexus": 0.15}
        else:
            platform_weights = {"flick": 0.35, "pulse": 0.40, "nexus": 0.25}

        posts = []

        for i in range(n_posters):
            platform = random.choices(
                list(platform_weights.keys()),
                weights=list(platform_weights.values()),
            )[0]

            post = await self._generate_single_post(
                seed     = seed,
                platform = platform,
                poster_n = i,
                llm      = llm,
            )
            if post:
                # Apply traction multipliers
                post = self._apply_traction(post, seed.visual_register)

                # Plant culture seeds from photo
                asyncio.create_task(
                    self._plant_culture_seeds_from_photo(seed, post)
                )

                posts.append(post)

                # Add to media engine
                media_engine._posts[platform].append(post)

                log_event("PhotoSeedEngine", "POST_CREATED",
                    platform        = platform,
                    visual_register = seed.visual_register,
                    poster          = post.author,
                    likes           = post.likes,
                    preview         = post.content[:60],
                )

        if posts:
            media_engine._save()

        return posts

    async def _generate_single_post(
        self,
        seed:     PhotoSeed,
        platform: str,
        poster_n: int,
        llm,
    ) -> Optional["Post"]:
        """Generate one post — one person's take on the same photo."""
        from core.media import Post, PLATFORMS

        platform_info = PLATFORMS[platform]

        # Build subject descriptions for prompt
        subject_str = "\n\n".join(
            s.to_prompt_str() for s in seed.subjects
        )

        # Different poster archetypes based on position in sequence
        poster_archetypes = [
            "someone who found it striking — their reaction could go any direction",
            "someone who found it funny or absurd",
            "someone who found it beautiful or freeing",
            "someone who found it offensive or inappropriate",
            "a regular on this platform who posts a lot",
        ]
        archetype = poster_archetypes[poster_n % len(poster_archetypes)]

        now_str = tz_now().strftime("%H:%M")

        prompt = f"""Someone just photographed a scene in public and is posting it.

Platform: {platform_info['name']} ({platform_info['description']})
Platform personality: {platform_info['personality'][:150]}
Time: {now_str}
Location: {seed.location}

What's in the photo:
{subject_str}

Scene context:
{seed.scene_context[:200]}

This poster is: {archetype}

Generate their post:
{{
  "author": "username style for {platform_info['name']} — specific, memorable",
  "content": "what they wrote — platform-appropriate. Pulse is short. Nexus can be longer.",
  "caption": "caption for the photo if Flick. Else same as content.",
  "circle": "which circle/tag/hashtag this lands in",
  "initial_likes": 12,
  "initial_dislikes": 0,
  "initial_saves": 3,
  "initial_reposts": 1,
  "comment_summary": "2-sentence description of early comment energy",
  "trajectory": "quiet|growing",
  "tone": "warm|hostile|funny|awed|concerned|neutral"
}}

The caption should emerge from actually seeing this photo — specific details,
not generic. If it's hostile, it should be specifically hostile about what's
actually in the frame. If it's warm, same.

Different posters see the same thing differently. Make this one feel distinct.
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You generate authentic social media posts about a specific photo. "
                    "JSON only. The caption must reflect what's actually in the image."
                ),
                max_new_tokens=300,
                temperature=0.88,
            )
            if not raw:
                return None

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return None

            d = json.loads(raw[start:end])

            # Build image description from subjects
            image_desc = " | ".join(
                f"{s.name}: {s.body_desc or s.clothing}"
                for s in seed.subjects
            )

            post = Post(
                platform        = platform,
                author          = d.get("author", "anonymous"),
                content         = d.get("content", ""),
                image_desc      = image_desc,
                caption         = d.get("caption", d.get("content", ""))[:200],
                circle          = d.get("circle", ""),
                likes           = int(d.get("initial_likes", 8)),
                dislikes        = int(d.get("initial_dislikes", 0)),
                saves           = int(d.get("initial_saves", 2)),
                reposts         = int(d.get("initial_reposts", 0)),
                comment_count   = 0,
                comment_summary = d.get("comment_summary", "No comments yet."),
                trajectory      = d.get("trajectory", "quiet"),
                subjects        = [s.name for s in seed.subjects],
                reach           = int(d.get("initial_likes", 8)) * 4,
            )
            return post

        except Exception as e:
            log_error("PhotoSeedEngine", f"post generation failed: {e}", exc=e)
            return None

    def _apply_traction(self, post, visual_register: str) -> "Post":
        """Apply traction multipliers based on content type."""
        reg = _VISUAL_REGISTERS.get(visual_register, _VISUAL_REGISTERS["ordinary"])
        mult      = reg["traction_mult"]
        save_mult = reg["save_mult"]

        post.likes   = int(post.likes * mult * random.uniform(0.8, 1.4))
        post.saves   = int(post.saves * save_mult * random.uniform(0.7, 1.5))
        post.reach   = int(post.reach * mult * random.uniform(0.9, 1.3))
        post.reposts = int(post.reposts * mult * random.uniform(0.5, 2.0))

        # High traction → growing trajectory
        if post.likes > 100 or post.reposts > 20:
            post.trajectory = "growing"
        if post.likes > 500 or post.reposts > 100:
            post.trajectory = "viral"

        return post

    async def _plant_culture_seeds_from_photo(
        self,
        seed: PhotoSeed,
        post,
    ) -> None:
        """A viral or growing photo plants culture seeds."""
        if post.trajectory not in ("growing", "viral"):
            return
        try:
            from core.culture import culture_engine
            # Each subject in the photo gets a seed planted
            for subject in seed.subjects:
                culture_engine.plant_seed(
                    location = seed.location,
                    witness  = f"saw photo on {post.platform} (@{post.author})",
                    what     = f"photo of {subject.name} — {subject.body_desc[:80]}",
                    affect   = "varied — photo is circulating",
                    subject  = subject.name.lower(),
                )
        except Exception as e:
            log_error("PhotoSeedEngine", f"culture seed planting failed: {e}", exc=e)


# ── Singleton ─────────────────────────────────────────────────────────────────

photo_seed_engine = PhotoSeedEngine()
