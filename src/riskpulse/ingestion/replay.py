"""Replay feed: build a timestamped, deduplicated news + social stream and replay it.

Build (offline):  GDELT GKG rows (news, re-linked with current rules) + Kaggle tweets
(social) -> normalised Documents -> exact/near dedupe per source -> time-sorted
``data/replay/feed.jsonl.gz`` + ``reports/feed_stats.json``.

Replay (online): :func:`iter_feed` yields documents in time order between two instants;
:func:`replay` paces them so one market day takes ``seconds_per_market_day`` seconds.
"""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from collections.abc import Iterator
from datetime import datetime

import pandas as pd

from riskpulse.common.config import load_config, repo_root
from riskpulse.common.logging import get_logger
from riskpulse.common.schemas import Document, Source
from riskpulse.engine.entities import MKT, EntityLinker
from riskpulse.ingestion.dedupe import dedupe_frame
from riskpulse.ingestion.gdelt_gkg import read_gkg_rows, relink
from riskpulse.ingestion.kaggle_tweets import load_tweets

log = get_logger()


def _news_frame(linker: EntityLinker, stats: dict) -> pd.DataFrame:
    app = load_config("app")
    prefixes = app["gdelt_gkg"]["mkt_theme_prefixes"]
    fcfg = app["replay_feed"]
    econ = tuple(fcfg["mkt_econ_theme_prefixes"])
    c = Counter()
    rows = []
    for raw in read_gkg_rows():
        c["stored"] += 1
        r = relink(raw, linker, prefixes)
        if r is None:
            c["dropped_by_current_rules"] += 1
            continue
        if not r["tickers"] and not r["is_mkt"]:
            if not fcfg["include_org_only"]:
                c["dropped_org_only"] += 1
                continue
        if r["is_mkt"] and not any(t.startswith(econ) for t in r["themes"]):
            c["dropped_mkt_non_econ"] += 1
            continue
        tickers = r["tickers"] or r["org_only_tickers"] or [MKT]
        rows.append(
            {
                "published_at": r["published_at"],
                "source": Source.GDELT.value,
                "text": r["title"],
                "title": r["title"],
                "url": r["url"],
                "tickers": tickers,
                "primary_ticker": tickers[0],
                "meta": {
                    "domain": r["domain"],
                    "regions": r["regions"],
                    "countries": r["countries"],
                    "themes": r["themes"],
                    "tone": r["tone"],
                    "link_basis": "title" if r["tickers"] else ("mkt" if r["is_mkt"] else "orgs"),
                },
            }
        )
    stats["news_filter"] = dict(c)
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def _social_frame(linker: EntityLinker, stats: dict) -> pd.DataFrame:
    tw, tstats = load_tweets(linker)
    stats["social_filter"] = tstats
    return pd.DataFrame(
        {
            "published_at": tw["published_at"],
            "source": Source.KAGGLE_TWEETS.value,
            "text": tw["text"],
            "title": None,
            "url": None,
            "tickers": tw["tickers"],
            "primary_ticker": tw["label_ticker"],
            "meta": [
                {"relevance": float(r), "n_cashtags": int(n)}
                for r, n in zip(tw["relevance"], tw["n_cashtags"], strict=True)
            ],
        }
    )


def build_feed() -> dict:
    """Build the replay feed and its stats file. Returns the stats dict."""
    app = load_config("app")
    fcfg = app["replay_feed"]
    linker = EntityLinker()
    stats: dict = {}
    frames = []
    for name, fn in (("news", _news_frame), ("social", _social_frame)):
        df = fn(linker, stats)
        df = df[(df["published_at"] >= fcfg["start"]) & (df["published_at"] < fcfg["end"])]
        df, dstats = dedupe_frame(df, text_col="text", group_col="primary_ticker")
        stats[f"{name}_dedupe"] = dstats
        frames.append(df)
    feed = pd.concat(frames, ignore_index=True).sort_values("published_at", kind="stable")

    path = repo_root() / app["paths"]["replay_feed"]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in feed.itertuples(index=False):
            doc = Document.build(
                r.source,
                text=r.text,
                published_at=r.published_at.to_pydatetime(),
                title=r.title,
                url=r.url,
                meta={**r.meta, "tickers": list(r.tickers)},
            )
            fh.write(doc.model_dump_json() + "\n")
            n += 1
    by_ticker = Counter(t for ts in feed["tickers"] for t in ts)
    stats.update(
        {
            "feed_path": app["paths"]["replay_feed"],
            "n_docs": n,
            "by_source": feed["source"].value_counts().to_dict(),
            "by_ticker": dict(sorted(by_ticker.items())),
            "by_month": feed["published_at"]
            .dt.strftime("%Y-%m")
            .value_counts()
            .sort_index()
            .to_dict(),
            "window": [fcfg["start"], fcfg["end"]],
        }
    )
    spath = repo_root() / app["paths"]["feed_stats"]
    spath.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(json.dumps(stats, indent=2, default=str))
    log.info(f"Replay feed: {n} docs -> {path}")
    return stats


def iter_feed(
    start: datetime | str | None = None, end: datetime | str | None = None
) -> Iterator[Document]:
    """Yield feed documents in time order within [start, end)."""
    path = repo_root() / load_config("app")["paths"]["replay_feed"]
    lo = pd.Timestamp(start, tz="UTC") if start is not None else None
    hi = pd.Timestamp(end, tz="UTC") if end is not None else None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            doc = Document.model_validate_json(line)
            ts = pd.Timestamp(doc.published_at)
            if lo is not None and ts < lo:
                continue
            if hi is not None and ts >= hi:
                break
            yield doc


def replay(
    start: datetime | str,
    end: datetime | str,
    seconds_per_market_day: float | None = None,
    sleep=time.sleep,
) -> Iterator[Document]:
    """Yield documents paced in simulated time: one calendar day = ``seconds_per_market_day``."""
    spd = seconds_per_market_day or load_config("app")["replay"]["seconds_per_market_day"]
    t0 = pd.Timestamp(start, tz="UTC")
    wall0 = time.monotonic()
    for doc in iter_feed(start, end):
        sim_elapsed_days = (pd.Timestamp(doc.published_at) - t0).total_seconds() / 86400
        due = wall0 + sim_elapsed_days * spd
        delay = due - time.monotonic()
        if delay > 0:
            sleep(delay)
        yield doc
