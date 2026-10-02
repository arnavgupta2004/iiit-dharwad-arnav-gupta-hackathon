"""Batch mode: run the engine over the whole replay feed and persist outputs.

Stage 1 (models) is cached in ``data/processed/cache/stage1_*`` keyed by the feed file's hash,
so re-runs (e.g. re-binning impact) only repeat stage 2. Impact bins are fitted on the burn-in
period (configs/impact.yaml) from a first stage-2 pass, then stage 2 runs again with them.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from riskpulse.common.config import data_path, load_config, repo_root
from riskpulse.common.logging import get_logger
from riskpulse.common.schemas import Document
from riskpulse.engine.events import EventPrediction
from riskpulse.engine.impact import ImpactBins
from riskpulse.engine.pipeline import (
    DocScore,
    NLPScorer,
    SignalEngine,
    apply_social_link_policy,
    stamp_versions,
)
from riskpulse.ingestion.replay import iter_feed
from riskpulse.store.signal_store import JsonlSignalWriter, mentions_frame, write_duckdb

log = get_logger()


def _feed_hash() -> str:
    path = repo_root() / load_config("app")["paths"]["replay_feed"]
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def run_stage1(batch_size: int = 128, limit: int | None = None) -> list[DocScore]:
    """Score every feed document with the models (cached)."""
    key = _feed_hash() + (f"_n{limit}" if limit else "")
    meta_path = data_path("processed", "cache", f"stage1_{key}.parquet")
    emb_path = data_path("processed", "cache", f"stage1_{key}_emb.npy")
    docs = list(iter_feed())
    if limit:
        docs = docs[:limit]
    if meta_path.exists() and emb_path.exists():
        meta = pd.read_parquet(meta_path)
        emb = np.load(emb_path).astype(np.float32)
        log.info(f"Stage 1: loaded cache for {len(meta)} docs")
        return [
            DocScore(
                doc=d,
                links=json.loads(r.links),
                regions=json.loads(r.regions),
                doc_sentiment=float(r.doc_sentiment),
                entity_sentiment=json.loads(r.entity_sentiment),
                event=EventPrediction(
                    r.event_class, float(r.event_conf), json.loads(r.event_probs)
                ),
                embedding=e,
            )
            for d, r, e in zip(docs, meta.itertuples(index=False), emb, strict=True)
        ]
    scorer = NLPScorer()
    out: list[DocScore] = []
    t0 = time.perf_counter()
    for i in range(0, len(docs), batch_size):
        out.extend(scorer.score(docs[i : i + batch_size]))
        if (i // batch_size) % 50 == 0:
            rate = len(out) / (time.perf_counter() - t0)
            log.info(f"Stage 1: {len(out)}/{len(docs)} docs ({rate:.0f} docs/s)")
    meta = pd.DataFrame(
        {
            "doc_id": [s.doc.doc_id for s in out],
            "links": [json.dumps(s.links) for s in out],
            "regions": [json.dumps(s.regions) for s in out],
            "doc_sentiment": [s.doc_sentiment for s in out],
            "entity_sentiment": [json.dumps(s.entity_sentiment) for s in out],
            "event_class": [s.event.event_class for s in out],
            "event_conf": [s.event.confidence for s in out],
            "event_probs": [
                json.dumps({k: round(v, 4) for k, v in s.event.probs.items()}) for s in out
            ],
        }
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_path, index=False)
    np.save(emb_path, np.vstack([s.embedding for s in out]).astype(np.float16))
    elapsed = time.perf_counter() - t0
    log.info(f"Stage 1 done: {len(out)} docs in {elapsed:.0f}s")
    return out


def fit_bins(scored: list[DocScore]) -> ImpactBins:
    """Fit impact quantile edges on the burn-in period's raw impacts."""
    cfg = load_config("impact")["binning"]
    lo, hi = (pd.Timestamp(x, tz="UTC") for x in cfg["burn_in"])
    burn = [s for s in scored if lo <= pd.Timestamp(s.doc.published_at) < hi]
    mentions, _ = SignalEngine(bins=ImpactBins(None)).process(burn)
    groups = np.array(["market" if m.ticker == "MKT" else "company" for m in mentions])
    bins = ImpactBins.fit(
        np.array([m.impact_raw for m in mentions]),
        int(cfg["n_bins"]),
        cfg.get("edge_quantiles"),
        groups if cfg.get("by_population") else None,
    )
    bins.save()
    log.info(f"Impact bins fitted on {len(mentions)} burn-in mentions: {bins.edges}")
    return bins


def run_batch(limit: int | None = None) -> dict:
    """Full batch run: stage 1, bin fitting, stage 2, persistence. Returns summary stats."""
    scored = run_stage1(limit=limit)
    scored, policy = apply_social_link_policy(scored)
    log.info(f"Original-ticker rule for tweets: {policy}")
    bins = fit_bins(scored)
    engine = SignalEngine(bins=bins)
    mentions, signals = engine.process(scored)
    versions = {
        "sentiment": f"finbert@{load_config('app')['sentiment']['model']}",
        "event": "embed-lr@" + load_config("app")["events"]["embedding_model"],
        "impact": "v1-burnin-quantile",
    }
    signals = stamp_versions(signals, versions)
    JsonlSignalWriter(truncate=True).write(signals)
    write_duckdb(mentions, signals)
    mdf = mentions_frame(mentions)
    mdf.to_parquet(data_path("processed", "mentions.parquet"), index=False)
    summary = {
        "n_docs": len(scored),
        "n_mentions": len(mentions),
        "n_signals": len(signals),
        "n_entity_signals": sum(s.signal_type == "entity" for s in signals),
        "n_event_signals": sum(s.signal_type == "event" for s in signals),
        "impact_bins": bins.edges,
        "tweet_original_ticker_rule": policy,
    }
    log.info(f"Batch done: {summary}")
    return summary


def score_documents(docs: list[Document], scorer: NLPScorer) -> list[DocScore]:
    """Convenience for /analyze and inject: stage 1 on a few documents."""
    return scorer.score(docs)
