"""Optional NewsAPI adapter (newsapi.org). Disabled unless NEWSAPI_KEY is set.

Free Developer plan (terms read 2026-10-01): 100 requests/day, articles delayed 24 h,
search up to one month back, development/testing use only. The engine never requires it.
"""

from __future__ import annotations

import os
from datetime import datetime

import requests
from dotenv import load_dotenv

from riskpulse.common.schemas import Document, Source

URL = "https://newsapi.org/v2/everything"


def api_key() -> str | None:
    load_dotenv()
    return os.environ.get("NEWSAPI_KEY") or None


def fetch(query: str, page_size: int = 50) -> list[Document]:
    """Fetch recent English articles for a query; returns [] when no key is configured."""
    key = api_key()
    if not key:
        return []
    r = requests.get(
        URL,
        params={"q": query, "language": "en", "pageSize": page_size, "sortBy": "publishedAt"},
        headers={"X-Api-Key": key},
        timeout=30,
    )
    r.raise_for_status()
    docs = []
    for a in r.json().get("articles", []):
        ts = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
        text = " ".join(x for x in [a.get("title"), a.get("description")] if x)
        docs.append(
            Document.build(
                Source.NEWSAPI,
                text=text,
                published_at=ts,
                title=a.get("title"),
                url=a.get("url"),
                author=a.get("author"),
                meta={"outlet": (a.get("source") or {}).get("name")},
            )
        )
    return docs
