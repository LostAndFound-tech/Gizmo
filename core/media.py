"""
core/media.py

Information reality layer. Sits above culture.py.

Controls how information spreads through the town's three platforms:
  Nexus  — Reddit-like. Circles, threads, long-form, upvotes/downvotes.
  Flick  — Instagram-like. Photos, captions, likes, saves, visual spread.
  Pulse  — Twitter-like. Short, fast, volatile. Things break here first.

Each platform has distinct social physics and personality.
Gizmo has accounts on all three. He can post, check, reply.

Posts are lazy — summary always present, comments generated on open.
Real-world effects close the loop: viral posts spawn culture seeds.

Files:
  /data/personality/social/nexus_posts.json
  /data/personality/social/flick_posts.json
  /data/personality/social/pulse_posts.json
  /data/personality/social/gizmo_accounts.json
  /data/personality/social/media_log.md

Passes:
  Hourly  — spread existing posts, check for new organic posts
  Daily   — paper generation, radio/TV content
  Per-session — Gizmo checks phone/paper/TV if he chooses

Usage:
    from core.media import media_engine

    await media_engine.start(llm)

    # Hourly pass (after culture pass)
    await media_engine.hourly_pass(culture_threads, llm)

    # Gizmo checks his phone
    feed = await media_engine.check_phone(session_id, platform, llm)

    # Gizmo posts something
    post = await media_engine.post(platform, content, llm)

    # Open a post's comments
    comments = await media_engine.open_comments(post_id, llm)

    # Get today's paper
    paper = media_engine.get_paper()
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

def _social_dir() -> Path:
    return Path("/data/personality/social")

def _posts_path(platform: str) -> Path:
    return _social_dir() / f"{platform}_posts.json"

def _accounts_path() -> Path:
    return _social_dir() / "gizmo_accounts.json"

def _media_log_path() -> Path:
    return _social_dir() / "media_log.md"

def _paper_path() -> Path:
    return _social_dir() / "paper.json"


# ── Platform definitions ──────────────────────────────────────────────────────

PLATFORMS = {
    "nexus": {
        "name":        "Nexus",
        "type":        "forum",
        "description": "Reddit-like. Circles, threads, upvotes/downvotes. Slow burn, deep roots.",
        "spread_speed": "slow",
        "half_life_hours": 72,
        "personality": (
            "Nexus rewards substance over spectacle. A striking post gains "
            "traction over days, not hours. Comments are substantive — sometimes "
            "thoughtful, sometimes completely unhinged. The town's circles range "
            "from local news to very specific interests. Moderators have opinions."
        ),
    },
    "flick": {
        "name":        "Flick",
        "type":        "photo",
        "description": "Instagram-like. Photos, short captions, likes, saves. Visual-first.",
        "spread_speed": "medium",
        "half_life_hours": 48,
        "personality": (
            "Flick lives and dies by the image. A blurry photo goes nowhere. "
            "Something striking can jump circles fast. Comments are short — "
            "warm or vicious, rarely in between. Save count matters more than likes. "
            "Aesthetic coherence builds followings slowly."
        ),
    },
    "pulse": {
        "name":        "Pulse",
        "type":        "microblog",
        "description": "Twitter-like. Short, fast, volatile. Things break here first.",
        "spread_speed": "fast",
        "half_life_hours": 6,
        "personality": (
            "Pulse is the town's nervous system. Things break here before the paper "
            "knows. Six-hour half-life on most posts — but a repost can take something "
            "from 12 to 12,000 in an hour. Then it's gone. Hot takes. Pile-ons. "
            "Also where the most honest reactions live, unfiltered."
        ),
    },
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Comment:
    comment_id:  str   = field(default_factory=lambda: f"c_{uuid.uuid4().hex[:6]}")
    author:      str   = ""       # pawn name or "anonymous" or gizmo account
    text:        str   = ""
    likes:       int   = 0
    tone:        str   = "neutral"
    replies:     list  = field(default_factory=list)   # list of Comment
    posted_at:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "comment_id": self.comment_id,
            "author":     self.author,
            "text":       self.text,
            "likes":      self.likes,
            "tone":       self.tone,
            "replies":    [r.to_dict() for r in self.replies],
            "posted_at":  self.posted_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Comment":
        c = cls(
            comment_id = d.get("comment_id", f"c_{uuid.uuid4().hex[:6]}"),
            author     = d.get("author", "anonymous"),
            text       = d.get("text", ""),
            likes      = d.get("likes", 0),
            tone       = d.get("tone", "neutral"),
            posted_at  = d.get("posted_at", time.time()),
        )
        c.replies = [Comment.from_dict(r) for r in d.get("replies", [])]
        return c


@dataclass
class Post:
    post_id:      str   = field(default_factory=lambda: f"post_{uuid.uuid4().hex[:8]}")
    platform:     str   = ""
    author:       str   = ""       # "gizmo" or pawn name or "anonymous"
    content:      str   = ""       # text content
    image_desc:   str   = ""       # for Flick — what the image shows
    caption:      str   = ""
    circle:       str   = ""       # Nexus circle or Pulse hashtag
    posted_at:    float = field(default_factory=time.time)

    # Metrics
    likes:        int   = 0
    dislikes:     int   = 0
    comment_count: int  = 0
    saves:        int   = 0
    reposts:      int   = 0
    reach:        int   = 0

    # Comments
    comment_summary: str  = ""     # always present
    comments:        list = field(default_factory=list)   # lazy generated
    comments_loaded: bool = False

    # Trajectory
    trajectory:   str   = "quiet"  # quiet|growing|stable|viral|fading|dead
    picked_up_by: list  = field(default_factory=list)

    # Effects
    seeds_spawned: list = field(default_factory=list)
    subjects:      list = field(default_factory=list)   # who/what it's about

    def metrics_str(self) -> str:
        parts = [f"{self.likes:,} likes"]
        if self.dislikes:
            parts.append(f"{self.dislikes:,} dislikes")
        parts.append(f"{self.comment_count:,} comments")
        if self.saves:
            parts.append(f"{self.saves:,} saves")
        if self.reposts:
            parts.append(f"{self.reposts:,} reposts")
        return " · ".join(parts)

    def to_summary(self) -> str:
        lines = [f"[{PLATFORMS[self.platform]['name']}] @{self.author}"]
        if self.image_desc:
            lines.append(f"  📷 {self.image_desc}")
        if self.content:
            lines.append(f"  {self.content[:120]}")
        if self.caption and self.caption != self.content:
            lines.append(f"  \"{self.caption[:80]}\"")
        lines.append(f"  {self.metrics_str()}")
        lines.append(f"  {self.comment_summary}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "post_id":       self.post_id,
            "platform":      self.platform,
            "author":        self.author,
            "content":       self.content,
            "image_desc":    self.image_desc,
            "caption":       self.caption,
            "circle":        self.circle,
            "posted_at":     self.posted_at,
            "likes":         self.likes,
            "dislikes":      self.dislikes,
            "comment_count": self.comment_count,
            "saves":         self.saves,
            "reposts":       self.reposts,
            "reach":         self.reach,
            "comment_summary": self.comment_summary,
            "comments":      [c.to_dict() for c in self.comments],
            "comments_loaded": self.comments_loaded,
            "trajectory":    self.trajectory,
            "picked_up_by":  self.picked_up_by,
            "seeds_spawned": self.seeds_spawned,
            "subjects":      self.subjects,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Post":
        p = cls(
            post_id      = d.get("post_id", f"post_{uuid.uuid4().hex[:8]}"),
            platform     = d.get("platform", "nexus"),
            author       = d.get("author", "anonymous"),
            content      = d.get("content", ""),
            image_desc   = d.get("image_desc", ""),
            caption      = d.get("caption", ""),
            circle       = d.get("circle", ""),
            posted_at    = d.get("posted_at", time.time()),
            likes        = d.get("likes", 0),
            dislikes     = d.get("dislikes", 0),
            comment_count = d.get("comment_count", 0),
            saves        = d.get("saves", 0),
            reposts      = d.get("reposts", 0),
            reach        = d.get("reach", 0),
            comment_summary = d.get("comment_summary", ""),
            comments_loaded = d.get("comments_loaded", False),
            trajectory   = d.get("trajectory", "quiet"),
            picked_up_by = d.get("picked_up_by", []),
            seeds_spawned = d.get("seeds_spawned", []),
            subjects     = d.get("subjects", []),
        )
        p.comments = [Comment.from_dict(c) for c in d.get("comments", [])]
        return p


@dataclass
class PaperIssue:
    """One day's paper."""
    date:         str   = ""
    headline:     str   = ""
    stories:      list  = field(default_factory=list)   # [{title, summary, page}]
    letters:      list  = field(default_factory=list)   # [{author, text}]
    editor_note:  str   = ""
    generated_at: float = field(default_factory=time.time)

    def to_summary(self) -> str:
        lines = [f"[The {_get_paper_name()} — {self.date}]"]
        if self.headline:
            lines.append(f"  HEADLINE: {self.headline}")
        for story in self.stories[:3]:
            lines.append(f"  • {story.get('title', '')} (p.{story.get('page', '?')})")
        if self.letters:
            lines.append(f"  Letters: {len(self.letters)} printed")
        return "\n".join(lines)


def _get_paper_name() -> str:
    """Load paper name from world rules or use default."""
    try:
        rp = Path("/data/personality/inner_world_rules.json")
        if rp.exists():
            rules = json.loads(rp.read_text())
            return rules.get("paper_name", "The Courier")
    except Exception:
        pass
    return "The Courier"


# ── Media engine ──────────────────────────────────────────────────────────────

class MediaEngine:

    def __init__(self):
        self._posts:       dict  = {p: [] for p in PLATFORMS}  # platform → [Post]
        self._paper:       Optional[PaperIssue] = None
        self._paper_date:  str   = ""
        self._accounts:    dict  = {}   # gizmo's account data per platform
        self._llm                = None
        self._running:     bool  = False

    # ── Start ─────────────────────────────────────────────────────────────────

    async def start(self, llm) -> None:
        self._llm = llm
        _social_dir().mkdir(parents=True, exist_ok=True)
        self._load()
        self._running = True
        log_event("MediaEngine", "STARTED",
            posts={p: len(posts) for p, posts in self._posts.items()})

    def _load(self) -> None:
        for platform in PLATFORMS:
            try:
                path = _posts_path(platform)
                if path.exists():
                    data  = json.loads(path.read_text())
                    self._posts[platform] = [
                        Post.from_dict(p)
                        for p in data.get("posts", [])
                        if p.get("trajectory") != "dead"
                    ]
            except Exception as e:
                log_error("MediaEngine", f"posts load failed: {platform}", exc=e)

        try:
            ap = _accounts_path()
            if ap.exists():
                self._accounts = json.loads(ap.read_text())
        except Exception as e:
            log_error("MediaEngine", "accounts load failed", exc=e)

        try:
            pp = _paper_path()
            if pp.exists():
                data = json.loads(pp.read_text())
                self._paper = PaperIssue(**{
                    k: v for k, v in data.items()
                    if k in PaperIssue.__dataclass_fields__
                })
                self._paper_date = data.get("date", "")
        except Exception:
            pass

    def _save(self, platform: str = None) -> None:
        platforms = [platform] if platform else list(PLATFORMS.keys())
        for p in platforms:
            try:
                path = _posts_path(p)
                path.write_text(json.dumps(
                    {"posts": [post.to_dict() for post in self._posts[p]]},
                    indent=2,
                ))
            except Exception as e:
                log_error("MediaEngine", f"posts save failed: {p}", exc=e)

    def _save_accounts(self) -> None:
        try:
            _accounts_path().write_text(json.dumps(self._accounts, indent=2))
        except Exception as e:
            log_error("MediaEngine", "accounts save failed", exc=e)

    # ── Hourly pass ───────────────────────────────────────────────────────────

    async def hourly_pass(
        self,
        culture_threads: list,
        llm,
    ) -> None:
        """
        Takes active culture threads.
        1. Spread existing posts (update metrics/trajectory)
        2. Generate organic posts from town activity
        3. Daily paper if day has changed
        """
        # Spread existing posts
        for platform in PLATFORMS:
            for post in self._posts[platform]:
                self._spread_post(post)

        # Kill dead posts
        for platform in PLATFORMS:
            self._posts[platform] = [
                p for p in self._posts[platform]
                if p.trajectory != "dead"
            ]

        # Generate organic posts from active threads
        if culture_threads:
            await self._generate_organic_posts(culture_threads, llm)

        # Daily paper
        today = tz_now().strftime("%Y-%m-%d")
        if today != self._paper_date:
            await self._generate_paper(culture_threads, llm)
            self._paper_date = today

        self._save()

        log_event("MediaEngine", "HOURLY_PASS",
            posts={p: len(posts) for p, posts in self._posts.items()})

    def _spread_post(self, post: Post) -> None:
        """Update post metrics based on platform physics and trajectory."""
        platform_info = PLATFORMS.get(post.platform, {})
        half_life     = platform_info.get("half_life_hours", 24) * 3600
        age           = time.time() - post.posted_at
        decay         = max(0.0, 1.0 - (age / half_life))

        if post.trajectory == "dead":
            return

        # Organic growth with decay
        if post.trajectory in ("growing", "viral"):
            growth = random.uniform(1.02, 1.15) * decay
            post.likes        = int(post.likes * growth)
            post.comment_count = int(post.comment_count * random.uniform(1.01, 1.05))
            post.reach        = int(post.reach * growth)

        # Decay
        if decay < 0.1 and post.trajectory not in ("viral",):
            post.trajectory = "dead"
        elif decay < 0.3 and post.trajectory == "growing":
            post.trajectory = "fading"
        elif post.trajectory == "fading" and decay < 0.05:
            post.trajectory = "dead"

    async def _generate_organic_posts(
        self,
        threads: list,
        llm,
    ) -> None:
        """Town members post about active culture threads."""
        # Only generate a few per hour
        if random.random() > 0.4:
            return

        # Pick a thread to post about
        active = [t for t in threads if not t.resolved and t.momentum > 0.2]
        if not active:
            return

        thread = random.choice(active)
        platform = random.choice(list(PLATFORMS.keys()))
        platform_info = PLATFORMS[platform]

        prompt = f"""Someone in town is posting about a cultural thread on {platform_info['name']}.

Platform: {platform_info['description']}
Platform personality: {platform_info['personality']}

Thread: {thread.name} — {thread.description}
Thread tone: {thread.tone}, momentum: {thread.momentum:.1f}

Generate ONE organic post from a town member:
{{
  "author": "pawn description — e.g. 'regulars_at_the_diner' or 'concerned_neighbor_42'",
  "content": "what they posted — platform-appropriate length and tone",
  "image_desc": "if Flick, what the image shows. Else empty.",
  "caption": "caption if different from content. Else empty.",
  "circle": "which circle/tag/hashtag",
  "initial_likes": 5,
  "initial_dislikes": 0,
  "initial_comments": 3,
  "comment_summary": "2-sentence summary of comment tone and content",
  "trajectory": "quiet|growing",
  "subjects": ["who or what it's about"]
}}

Match the platform's physics. Pulse is short. Nexus is longer.
The post should feel like someone from this specific town posted it.
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You generate authentic social media posts from town members. "
                    "JSON only. Platform-appropriate."
                ),
                max_new_tokens=300,
                temperature=0.85,
            )
            if not raw:
                return

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return

            d = json.loads(raw[start:end])

            post = Post(
                platform        = platform,
                author          = d.get("author", "anonymous"),
                content         = d.get("content", ""),
                image_desc      = d.get("image_desc", ""),
                caption         = d.get("caption", ""),
                circle          = d.get("circle", ""),
                likes           = int(d.get("initial_likes", 5)),
                dislikes        = int(d.get("initial_dislikes", 0)),
                comment_count   = int(d.get("initial_comments", 0)),
                comment_summary = d.get("comment_summary", ""),
                trajectory      = d.get("trajectory", "quiet"),
                subjects        = d.get("subjects", []),
                reach           = int(d.get("initial_likes", 5)) * 3,
            )
            self._posts[platform].append(post)

            log_event("MediaEngine", "ORGANIC_POST",
                platform = platform,
                author   = post.author,
                preview  = post.content[:60],
            )

        except Exception as e:
            log_error("MediaEngine", f"organic post failed: {e}", exc=e)

    # ── Paper ─────────────────────────────────────────────────────────────────

    async def _generate_paper(
        self,
        threads: list,
        llm,
    ) -> None:
        """Generate today's paper."""
        today = tz_now().strftime("%A, %B %d")

        thread_text = "\n".join(
            f"- {t.name}: {t.description} (momentum: {t.momentum:.1f})"
            for t in threads if not t.resolved
        ) or "(nothing notable)"

        # Recent posts that might have caught the editor's eye
        notable_posts = []
        for platform, posts in self._posts.items():
            for post in posts:
                if post.likes > 50 or post.trajectory in ("growing", "viral"):
                    notable_posts.append(post.to_summary())

        posts_context = "\n".join(notable_posts[:4]) or "(nothing notable online)"

        paper_name = _get_paper_name()

        prompt = f"""Generate today's issue of {paper_name}, the local paper.

Date: {today}

Active cultural threads:
{thread_text}

Notable social media activity:
{posts_context}

The editor is skeptical but fair. The young reporter is enthusiastic and slightly reckless.
The paper covers what's genuinely interesting to this specific town.
Not everything makes the paper. Some things the editor buries.

Return ONE JSON object:
{{
  "headline": "today's main headline — or empty if quiet day",
  "stories": [
    {{
      "title": "story title",
      "summary": "2-sentence summary",
      "page": "A1|A2|B1|B3|etc"
    }}
  ],
  "letters": [
    {{
      "author": "name or 'Name Withheld'",
      "text": "1-3 sentences"
    }}
  ],
  "editor_note": "optional editor's column note — personal, opinionated"
}}

0-3 stories. 0-2 letters. Quiet days are fine — the paper runs recipes and sports scores.
Only cover cultural threads if they've genuinely reached a level the paper would notice.
JSON only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "You generate a local newspaper. "
                    "Realistic, specific, editor-voiced. JSON only."
                ),
                max_new_tokens=400,
                temperature=0.7,
            )
            if not raw:
                return

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return

            d = json.loads(raw[start:end])

            self._paper = PaperIssue(
                date        = today,
                headline    = d.get("headline", ""),
                stories     = d.get("stories", []),
                letters     = d.get("letters", []),
                editor_note = d.get("editor_note", ""),
            )

            try:
                _paper_path().write_text(json.dumps({
                    **d, "date": today,
                    "generated_at": time.time(),
                }, indent=2))
            except Exception:
                pass

            log_event("MediaEngine", "PAPER_GENERATED",
                date     = today,
                headline = d.get("headline", "(quiet day)")[:80],
                stories  = len(d.get("stories", [])),
            )

        except Exception as e:
            log_error("MediaEngine", f"paper generation failed: {e}", exc=e)

    # ── Gizmo actions ─────────────────────────────────────────────────────────

    async def check_phone(
        self,
        session_id:  str,
        platform:    str,
        headmate:    Optional[str],
        llm,
    ) -> str:
        """
        Gizmo checks his phone — returns a feed summary for that platform.
        Marks awareness in culture engine.
        """
        try:
            from core.culture import culture_engine
            awareness = culture_engine.get_awareness(session_id)
            awareness.mark_checked_phone()
        except Exception:
            pass

        posts = self._posts.get(platform, [])
        if not posts:
            return f"[{PLATFORMS[platform]['name']}] Nothing new."

        # Sort by relevance — trajectory, recency
        relevant = sorted(
            [p for p in posts if p.trajectory != "dead"],
            key=lambda p: (
                {"viral": 4, "growing": 3, "stable": 2, "quiet": 1, "fading": 0}.get(
                    p.trajectory, 0
                ),
                p.posted_at,
            ),
            reverse=True,
        )[:5]

        lines = [f"[{PLATFORMS[platform]['name']} feed]"]
        for post in relevant:
            lines.append(post.to_summary())
            lines.append("")

        return "\n".join(lines)

    async def post(
        self,
        platform:   str,
        content:    str,
        image_desc: str = "",
        caption:    str = "",
        circle:     str = "",
        llm         = None,
    ) -> Post:
        """Gizmo posts something."""
        account = self._accounts.get(platform, {})
        handle  = account.get("handle", "gizmo")

        post = Post(
            platform    = platform,
            author      = handle,
            content     = content,
            image_desc  = image_desc,
            caption     = caption or content[:100],
            circle      = circle,
            likes       = 0,
            trajectory  = "quiet",
            reach       = account.get("followers", 10),
        )

        # Generate initial comment summary if LLM available
        if llm and content:
            summary = await self._generate_initial_summary(post, llm)
            post.comment_summary = summary

        self._posts[platform].append(post)
        self._save(platform)

        log_event("MediaEngine", "GIZMO_POSTED",
            platform = platform,
            preview  = content[:60],
        )

        return post

    async def _generate_initial_summary(self, post: Post, llm) -> str:
        """Generate initial comment summary for a new post."""
        platform_info = PLATFORMS.get(post.platform, {})

        prompt = f"""A post just went up on {platform_info.get('name', post.platform)}.

Content: {post.content[:200]}
{f"Image: {post.image_desc}" if post.image_desc else ""}

Given the platform personality ({platform_info.get('personality', '')[:100]})
and a small initial audience, what's the early comment vibe?

One sentence. E.g. "Early comments are warm, a few asking questions." JSON only:
{{"summary": "..."}}"""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt="One sentence comment summary. JSON only.",
                max_new_tokens=60,
                temperature=0.7,
            )
            if raw:
                start = raw.find("{")
                end   = raw.rfind("}") + 1
                if start != -1 and end > 0:
                    d = json.loads(raw[start:end])
                    return d.get("summary", "")
        except Exception:
            pass
        return "No comments yet."

    async def open_comments(
        self,
        post_id: str,
        llm,
    ) -> Optional[str]:
        """
        Open a post's comments. Lazy-generates if not yet loaded.
        Returns formatted comment thread.
        """
        post = self._find_post(post_id)
        if not post:
            return None

        if not post.comments_loaded:
            await self._generate_comments(post, llm)

        lines = [f"[Comments — {post.metrics_str()}]"]
        lines.append(f"Summary: {post.comment_summary}")
        lines.append("")

        for comment in post.comments[:12]:
            lines.append(f"@{comment.author}: {comment.text}")
            if comment.likes:
                lines.append(f"  ↑{comment.likes}")
            for reply in comment.replies[:2]:
                lines.append(f"  └ @{reply.author}: {reply.text}")

        return "\n".join(lines)

    async def _generate_comments(self, post: Post, llm) -> None:
        """Generate comments for a post."""
        platform_info = PLATFORMS.get(post.platform, {})

        # Load culture context
        thread_context = ""
        try:
            from core.culture import culture_engine
            thread_context = culture_engine.active_threads_block()
        except Exception:
            pass

        # Load town context
        town_context = ""
        try:
            wp = Path("/data/personality/inner_world.md")
            if wp.exists():
                town_context = wp.read_text()[:300]
        except Exception:
            pass

        ratio = post.likes / max(1, post.likes + post.dislikes)

        prompt = f"""Generate comments for this {platform_info.get('name', '')} post.

Post by @{post.author}:
{post.content[:300]}
{f"Image: {post.image_desc}" if post.image_desc else ""}

Metrics: {post.metrics_str()}
Sentiment ratio: {ratio:.0%} positive

Town context: {town_context[:200]}
{thread_context[:300]}

Platform personality: {platform_info.get('personality', '')[:150]}

Generate 8-12 comments representing the range of reactions.
Include: enthusiastic, curious, hostile, confused, funny, deep, shallow — 
whatever fits this specific post and town.

Return JSON array:
[
  {{
    "author": "pawn description or username style",
    "text": "comment text — platform-appropriate length",
    "likes": 5,
    "tone": "warm|curious|hostile|funny|confused|supportive",
    "replies": [
      {{
        "author": "username",
        "text": "reply text",
        "likes": 2,
        "tone": "tone"
      }}
    ]
  }}
]

Make them feel like real people with real opinions.
Some should clearly come from specific culture thread perspectives.
JSON array only."""

        try:
            raw = await llm.generate(
                [{"role": "user", "content": prompt}],
                system_prompt=(
                    "Generate realistic social media comments. "
                    "JSON array only. Distinct voices."
                ),
                max_new_tokens=600,
                temperature=0.85,
            )
            if not raw:
                return

            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return

            comments_data = json.loads(raw[start:end])
            post.comments = [Comment.from_dict(c) for c in comments_data]
            post.comments_loaded = True
            post.comment_count = len(post.comments)

            # Update summary from generated comments
            tones = [c.tone for c in post.comments]
            hostile_count = tones.count("hostile")
            warm_count    = sum(1 for t in tones if t in ("warm", "supportive"))

            if hostile_count > warm_count:
                post.comment_summary = f"Mostly hostile — {hostile_count} critical comments."
            elif warm_count > hostile_count * 2:
                post.comment_summary = f"Mostly positive — {warm_count} supportive comments."
            else:
                post.comment_summary = f"Mixed — {len(post.comments)} comments, divided."

            # Save updated post
            platform = post.platform
            self._save(platform)

            log_event("MediaEngine", "COMMENTS_GENERATED",
                post_id  = post.post_id,
                platform = platform,
                count    = len(post.comments),
            )

        except Exception as e:
            log_error("MediaEngine", f"comment generation failed: {e}", exc=e)

    def _find_post(self, post_id: str) -> Optional[Post]:
        for platform_posts in self._posts.values():
            for post in platform_posts:
                if post.post_id == post_id:
                    return post
        return None

    # ── Paper access ──────────────────────────────────────────────────────────

    def get_paper(self) -> Optional[PaperIssue]:
        return self._paper

    def get_paper_summary(self) -> str:
        if not self._paper:
            return "[The paper] Nothing today."
        return self._paper.to_summary()

    # ── Feed for context injection ────────────────────────────────────────────

    def notable_posts_block(self) -> str:
        """Posts worth Gizmo knowing about — if he's checked his phone."""
        lines = []
        for platform, posts in self._posts.items():
            notable = [
                p for p in posts
                if p.trajectory in ("viral", "growing")
                and p.trajectory != "dead"
            ]
            for post in notable[:2]:
                lines.append(post.to_summary())
        if not lines:
            return ""
        return "[Notable online]\n" + "\n\n".join(lines)

    def posts_about(self, subject: str) -> list:
        """Find all posts about a specific subject."""
        result = []
        for platform_posts in self._posts.values():
            for post in platform_posts:
                if subject.lower() in [s.lower() for s in post.subjects]:
                    result.append(post)
        return result

    # ── Session close ─────────────────────────────────────────────────────────

    def close_session(self, session_id: str) -> None:
        pass   # media is global, not session-scoped

    def stop(self) -> None:
        self._running = False


# ── Singleton ─────────────────────────────────────────────────────────────────

media_engine = MediaEngine()
