"""GDELT DOC 2.0 API adapter for `live` mode (recent articles, no key).

Rate limit verified 2026-10-01: at most one request per 5 s; bursts trigger HTTP 429 and a
cool-down. We space requests (config ``min_seconds_between_requests``) and back off on 429.
ArtList mode returns title, URL, seendate, domain, language and source country only.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import requests

from riskpulse.common.config import load_config
from riskpulse.common.logging import get_logger
from riskpulse.common.schemas import Document, Source

log = get_logger()


class GdeltDocClient:
    """Polite GDELT DOC API client with spacing and 429 backoff."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.cfg = load_config("app")["live"]["gdelt"]
        self.session = session or requests.Session()
        self._last = 0.0
        self._lock = threading.Lock()
        self.failures = 0  # queries given up after retries (throttling or connection errors)

    def _wait_turn(self) -> None:
        with self._lock:
            gap = float(self.cfg["min_seconds_between_requests"]) - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()

    def search(self, query: str, timespan: str = "1h", maxrecords: int | None = None) -> list[dict]:
        """Raw ArtList articles for a query (``sourcelang:english`` appended)."""
        params = {
            "query": f"{query} sourcelang:english",
            "mode": "ArtList",
            "format": "json",
            "timespan": timespan,
            "maxrecords": str(maxrecords or self.cfg["maxrecords"]),
        }
        for attempt in range(int(self.cfg["max_retries"])):
            self._wait_turn()
            try:
                r = self.session.get(self.cfg["base_url"], params=params, timeout=60)
            except requests.RequestException as exc:
                log.warning(f"GDELT DOC request failed: {exc}")
                continue
            if r.status_code == 429:
                time.sleep(float(self.cfg["backoff_seconds_on_429"]) * (attempt + 1))
                continue
            r.raise_for_status()
            try:
                return r.json().get("articles", [])
            except ValueError:
                return []  # GDELT returns an empty body when nothing matches
        log.error(f"GDELT DOC gave up after retries for query {query!r}")
        self.failures += 1
        return []


def to_documents(articles: list[dict]) -> list[Document]:
    """Convert ArtList articles to normalised documents (title is the text)."""
    docs = []
    for a in articles:
        try:
            ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except (KeyError, ValueError):
            continue
        docs.append(
            Document.build(
                Source.GDELT,
                text=a.get("title", ""),
                published_at=ts,
                title=a.get("title"),
                url=a.get("url"),
                meta={"domain": a.get("domain"), "country": a.get("sourcecountry")},
            )
        )
    return docs
