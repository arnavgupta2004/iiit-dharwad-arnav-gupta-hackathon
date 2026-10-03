"""`live` mode (spec §5.1): poll current GDELT news and push new headlines through the live engine.

Sources (config ``live.sources``):
- ``gdelt_gkg``: the newest 15-minute GKG file, filtered by the same rules as the replay feed
  (``gdelt_gkg.filter_records`` + ``relink``, no org-only items, market items need an economic
  theme). Not rate-limited.
- ``gdelt_doc``: GDELT DOC 2.0 API queries (one request per 5 s; often throttled with HTTP 429).

News only: there is no free live social source. Everything this produces is labelled
"live, unvalidated" and written under ``data/live/`` (gitignored); no evaluation reads it.

The engine assumes stream time moves forward, so an item older than the newest one already
processed is counted as late and dropped. Exact duplicates (same URL or same canonical headline)
are dropped as in the batch feed.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from riskpulse.common.config import data_path, load_config
from riskpulse.common.logging import get_logger
from riskpulse.common.schemas import Document, Signal, Source
from riskpulse.engine.aggregate import ScoredMention
from riskpulse.engine.entities import MKT, EntityLinker
from riskpulse.ingestion.gdelt_doc import GdeltDocClient, to_documents
from riskpulse.ingestion.gdelt_gkg import filter_records, parse_gkg, relink
from riskpulse.ingestion.normalize import dedupe_key

log = get_logger()
LABEL = "live, unvalidated"


def gkg_documents(raw_zip: bytes, linker: EntityLinker, app: dict | None = None) -> list[Document]:
    """Documents from one GKG file, with the replay feed's gates (replay._news_frame)."""
    app = app or load_config("app")
    prefixes = app["gdelt_gkg"]["mkt_theme_prefixes"]
    econ = tuple(app["replay_feed"]["mkt_econ_theme_prefixes"])
    docs = []
    for raw in filter_records(parse_gkg(raw_zip), linker, prefixes):
        r = relink(raw, linker, prefixes)
        if r is None or (not r["tickers"] and not r["is_mkt"]):  # org-only items are dropped
            continue
        if r["is_mkt"] and not any(t.startswith(econ) for t in r["themes"]):
            continue
        docs.append(
            Document.build(
                Source.GDELT,
                text=r["title"],
                published_at=datetime.fromisoformat(r["published_at"]),
                title=r["title"],
                url=r["url"],
                meta={
                    "domain": r["domain"],
                    "regions": r["regions"],
                    "countries": r["countries"],
                    "themes": r["themes"],
                    "tone": r["tone"],
                    "link_basis": "title" if r["tickers"] else "mkt",
                    "tickers": r["tickers"] or [MKT],
                    "live_source": "gdelt_gkg",
                },
            )
        )
    return docs


class GkgLatest:
    """The newest available GKG 15-minute file; each file is read once.

    ``lastupdate.txt`` can name a GKG file before it is downloadable (HTTP 404 while GDELT lags),
    so we step back 15 minutes at a time, up to ``max_lookback_files``, to the newest file that
    exists and has not been read. Files newer than the one read are skipped for good: their items
    would be older than the engine's stream time.
    """

    def __init__(self, cfg: dict, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.session = session or requests.Session()
        self.linker = EntityLinker()
        self.last_ts: datetime | None = None
        self.lag_files = 0  # how far behind lastupdate.txt the last file read was

    def __call__(self) -> list[Document]:
        r = self.session.get(self.cfg["lastupdate_url"], timeout=60)
        r.raise_for_status()
        url = next(
            (ln.split()[-1] for ln in r.text.splitlines() if ln.endswith(".gkg.csv.zip")), None
        )
        if url is None:
            return []
        stamp = url.rsplit("/", 1)[-1].split(".")[0]
        newest = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        for k in range(int(self.cfg.get("max_lookback_files", 8)) + 1):
            ts = newest - timedelta(minutes=15 * k)
            if self.last_ts is not None and ts <= self.last_ts:
                return []
            z = self.session.get(url.replace(stamp, f"{ts:%Y%m%d%H%M%S}"), timeout=120)
            if z.status_code == 404:
                continue
            z.raise_for_status()
            self.last_ts, self.lag_files = ts, k
            return gkg_documents(z.content, self.linker)
        return []


class DocQueries:
    """GDELT DOC API queries; a throttled query returns nothing and the poll carries on."""

    def __init__(self, cfg: dict, client: GdeltDocClient | None = None) -> None:
        self.cfg, self.client = cfg, client or GdeltDocClient()

    def __call__(self) -> list[Document]:
        docs: list[Document] = []
        queries, before = self.cfg["gdelt"]["queries"], getattr(self.client, "failures", 0)
        for q in queries:
            docs += to_documents(self.client.search(q, timespan=self.cfg["timespan"]))
        if queries and getattr(self.client, "failures", 0) - before == len(queries):
            raise RuntimeError("every GDELT DOC query failed (throttled or disconnected)")
        for d in docs:
            d.meta["live_source"] = "gdelt_doc"
        return docs


class LivePoller:
    """Poll the live sources every N minutes; score new items; publish signals."""

    def __init__(
        self,
        process: Callable[[list[Document]], tuple[list[ScoredMention], list[Signal]]],
        publish: Callable[[list[Signal]], None],
        sources: dict[str, Callable[[], list[Document]]] | None = None,
        cfg: dict | None = None,
        out_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg or load_config("app")["live"]
        self.process, self.publish = process, publish
        if sources is None:
            make = {
                "gdelt_gkg": lambda: GkgLatest(self.cfg["gkg"]),
                "gdelt_doc": lambda: DocQueries(self.cfg),
            }
            sources = {name: make[name]() for name in self.cfg["sources"]}
        self.sources = sources
        self.out_dir = out_dir or data_path("live")
        self.seen: set[str] = set()
        self._source_of: dict[str, str] = {}
        self.watermark: datetime | None = None
        self.recent: deque[dict] = deque(maxlen=int(self.cfg.get("keep_recent", 500)))
        self.stats = {
            "label": LABEL,
            "sources": list(sources),
            "polls": 0,
            "fetched": 0,
            "fetched_by_source": dict.fromkeys(sources, 0),
            "source_errors": dict.fromkeys(sources, 0),
            "duplicates": 0,
            "late_dropped": 0,
            "processed": 0,
            "mentions": 0,
            "signals": 0,
            "errors": 0,
            "last_poll": None,
        }

    def _new_docs(self, docs: list[Document]) -> list[Document]:
        out = []
        for d in sorted(docs, key=lambda x: x.published_at):
            keys = {f"url:{d.url}", f"text:{dedupe_key(d.title or d.text)}"}
            if not (d.title or d.text).strip() or keys & self.seen:
                self.stats["duplicates"] += 1
                continue
            self.seen |= keys
            if self.watermark is not None and d.published_at < self.watermark:
                self.stats["late_dropped"] += 1
                continue
            out.append(d)
        return out

    def _record(self, mentions: list[ScoredMention], signals: list[Signal]) -> None:
        now = datetime.now(UTC).isoformat()
        rows = [
            {
                "label": LABEL,
                "polled_at": now,
                "published_at": m.published_at.isoformat(),
                "outlet": m.outlet,
                "live_source": self._source_of.get(m.doc_id),
                "title": m.title,
                "url": m.url,
                "ticker": m.ticker,
                "sentiment": round(m.sentiment, 4),
                "event_class": m.event_class,
                "event_confidence": round(m.event_confidence, 4),
                "impact_score": m.impact_score,
                "event_id": m.event_id,
            }
            for m in mentions
        ]
        self.recent.extend(rows)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        day = datetime.now(UTC).strftime("%Y%m%d")
        with open(self.out_dir / f"mentions_{day}.jsonl", "a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        with open(self.out_dir / f"signals_{day}.jsonl", "a", encoding="utf-8") as fh:
            fh.writelines(
                json.dumps({"label": LABEL, **json.loads(s.model_dump_json())}) + "\n"
                for s in signals
            )

    def poll_once(self) -> dict:
        """One poll over every source; a failing source is logged and skipped. Returns counts."""
        docs: list[Document] = []
        for name, fetch in self.sources.items():
            try:
                got = fetch()
            except Exception as exc:  # one source down must not stop the others
                self.stats["source_errors"][name] += 1
                log.warning(f"live source {name} failed: {exc}")
                continue
            self.stats["fetched_by_source"][name] += len(got)
            self._source_of.update({d.doc_id: name for d in got})
            docs += got
        new = self._new_docs(docs)
        mentions, signals = self.process(new) if new else ([], [])
        if new:
            self.watermark = max(d.published_at for d in new)
            self.publish(signals)
            self._record(mentions, signals)
        self._source_of.clear()
        polled = {
            "fetched": len(docs),
            "new": len(new),
            "mentions": len(mentions),
            "signals": len(signals),
        }
        self.stats["polls"] += 1
        self.stats["fetched"] += len(docs)
        self.stats["processed"] += len(new)
        self.stats["mentions"] += len(mentions)
        self.stats["signals"] += len(signals)
        self.stats["last_poll"] = datetime.now(UTC).isoformat()
        log.info(f"live poll {self.stats['polls']}: {polled}")
        return polled

    def run(self, stop: threading.Event) -> None:
        """Poll until ``stop`` is set; a failed poll is logged and retried next cycle."""
        while not stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # keep polling: one bad response must not end live mode
                self.stats["errors"] += 1
                log.exception(f"live poll failed: {exc}")
            stop.wait(float(self.cfg["poll_minutes"]) * 60)
