"""
tools/web_search.py
Brave Search API integration — search and full-page fetch.

Provides:
  search(query, n=5)      — returns top N results with title, url, snippet
  fetch_page(url)         — fetches and extracts clean text from a URL
  search_and_fetch(query) — convenience: search + fetch top 3 full pages

Rate limit: Brave free tier is 1 req/sec. All calls respect this.
Requires: BRAVE_API_KEY in .env, httpx

Page fetching strips HTML to clean readable text for LLM consumption.
Truncates to MAX_PAGE_CHARS to avoid blowing context windows.
"""

import asyncio
import os
import re
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

MAX_PAGE_CHARS = 8000       # truncate fetched pages to this length
MAX_RESULTS = 10            # hard cap on search results returned
RATE_LIMIT_DELAY = 1.1      # seconds between Brave API calls (free tier = 1/sec)

_last_search_time: float = 0.0


async def _rate_limit() -> None:
    """Enforce Brave free tier rate limit — 1 request per second."""
    global _last_search_time
    now = time.monotonic()
    elapsed = now - _last_search_time
    if elapsed < RATE_LIMIT_DELAY:
        await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_search_time = time.monotonic()


def _clean_html(html: str) -> str:
    """
    Strip HTML tags and decode common entities to plain text.
    Not a full HTML parser — good enough for LLM consumption.
    """
    # Remove script and style blocks entirely
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    html = re.sub(r'<[^>]+>', ' ', html)
    # Decode common entities
    entities = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#39;': "'", '&nbsp;': ' ',
        '&mdash;': '—', '&ndash;': '–', '&hellip;': '…',
    }
    for entity, char in entities.items():
        html = html.replace(entity, char)
    # Collapse whitespace
    html = re.sub(r'[ \t]+', ' ', html)
    html = re.sub(r'\n{3,}', '\n\n', html)
    return html.strip()


async def search(
    query: str,
    n: int = 5,
) -> list[dict]:
    """
    Search Brave and return top N results.
    Each result: {title, url, snippet, age (if available)}

    Returns empty list on failure — caller handles gracefully.
    """
    if not BRAVE_API_KEY:
        print("[WebSearch] BRAVE_API_KEY not set — search unavailable")
        return []

    try:
        import httpx
    except ImportError:
        print("[WebSearch] httpx required: pip install httpx")
        return []

    n = min(n, MAX_RESULTS)
    await _rate_limit()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                BRAVE_SEARCH_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": BRAVE_API_KEY,
                },
                params={
                    "q": query,
                    "count": n,
                    "search_lang": "en",
                    "safesearch": "moderate",
                    "text_decorations": False,
                    "spellcheck": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:n]:
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("description", ""),
                "age":     item.get("age", ""),
            })

        print(f"[WebSearch] '{query}' → {len(results)} results")
        return results

    except Exception as e:
        print(f"[WebSearch] Search failed: {e}")
        return []


async def fetch_page(url: str) -> str:
    """
    Fetch a URL and return clean extracted text.
    Truncated to MAX_PAGE_CHARS. Returns empty string on failure.
    """
    try:
        import httpx
    except ImportError:
        print("[WebSearch] httpx required: pip install httpx")
        return ""

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Gizmo/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "html" in content_type:
                text = _clean_html(resp.text)
            elif "text" in content_type:
                text = resp.text
            else:
                print(f"[WebSearch] Skipping non-text content type: {content_type}")
                return ""

        text = text[:MAX_PAGE_CHARS]
        print(f"[WebSearch] Fetched {url[:60]}... ({len(text)} chars)")
        return text

    except Exception as e:
        print(f"[WebSearch] Fetch failed for {url[:60]}: {e}")
        return ""


async def search_and_fetch(
    query: str,
    n_results: int = 3,
    max_concurrent_fetches: int = 3,
) -> list[dict]:
    """
    Search + fetch full page content for top N results.
    Fetches pages concurrently (respects rate limit on search only).

    Returns list of:
    {
        title:   str,
        url:     str,
        snippet: str,
        text:    str,   # full page text, empty if fetch failed
        age:     str,
    }
    """
    results = await search(query, n=n_results)
    if not results:
        return []

    # Fetch pages concurrently
    fetch_tasks = [fetch_page(r["url"]) for r in results[:max_concurrent_fetches]]
    page_texts = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    for i, result in enumerate(results[:max_concurrent_fetches]):
        text = page_texts[i]
        result["text"] = text if isinstance(text, str) else ""

    # Any results beyond max_concurrent_fetches get empty text
    for result in results[max_concurrent_fetches:]:
        result["text"] = ""

    return results
