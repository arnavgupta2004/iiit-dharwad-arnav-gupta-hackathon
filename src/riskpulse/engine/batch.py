"""Batch mode: run the engine over the whole replay feed and persist outputs.

Stage 1 is cached in parts under ``data/processed/cache/``: links, regions and embeddings
(``stage1base_*``, keyed by the feed and the linking/embedding setup) and sentiment
(``stage1sent_*``, keyed by the feed and the sentiment weights), plus event classes predicted from
the cached embeddings (``stage1event_*``, keyed by the event model). Re-runs (e.g. re-binning
impact) only repeat stage 2.
Impact bins are fitted on the burn-in period (configs/impact.yaml) from a first stage-2 pass, then
stage 2 runs again with them.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

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


def _base_key() -> str:
    """Fingerprint of what links and embeds documents: universe/linking config, the tweet
    original-ticker map and the embedding model."""
    h = hashlib.sha1()
    for rel in ("configs/universe.yaml", "data/replay/tweet_original_ticker.csv"):
        path = repo_root() / rel
        if path.exists():
            h.update(path.read_bytes())
    h.update(load_config("app")["events"]["embedding_model"].encode())
    return h.hexdigest()[:8]


def _cache(name: str) -> Path:
    return data_path("processed", "cache", name)


def base_stage(docs: list, feed_key: str, scorer: NLPScorer, batch_size: int = 128):
    """Links, regions and embeddings per document (cached; no sentiment or event model)."""
    key = f"{feed_key}_{_base_key()}"
    meta_p, emb_p = _cache(f"stage1base_{key}.parquet"), _cache(f"stage1base_{key}_emb.npy")
    if meta_p.exists() and emb_p.exists():
        meta = pd.read_parquet(meta_p)
        log.info(f"Stage 1 base: loaded cache for {len(meta)} docs")
        return meta, np.load(emb_p).astype(np.float32)
    links, regions, embs = [], [], []
    t0 = time.perf_counter()
    for i in range(0, len(docs), batch_size):
        chunk = docs[i : i + batch_size]
        for lk, rg in (scorer.link(d) for d in chunk):
            links.append(json.dumps(lk))
            regions.append(json.dumps(rg))
        embs.append(scorer.embedder.encode([d.title or d.text for d in chunk]).astype(np.float16))
        if (i // batch_size) % 100 == 0:
            log.info(
                f"Stage 1 base: {i + len(chunk)}/{len(docs)} ({time.perf_counter() - t0:.0f}s)"
            )
    meta = pd.DataFrame({"doc_id": [d.doc_id for d in docs], "links": links, "regions": regions})
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_p, index=False)
    emb = np.vstack(embs)
    np.save(emb_p, emb)
    return meta, emb.astype(np.float32)


def sentiment_stage(
    docs: list,
    base: pd.DataFrame,
    feed_key: str,
    variant: str | None = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Document and entity sentiment per document, cached per sentiment weights (D-043), so the
    cache survives event-model changes and can be precomputed for another variant."""
    from riskpulse.engine.sentiment import FinBertScorer, weights_fingerprint

    path = _cache(f"stage1sent_{feed_key}_{weights_fingerprint(variant)}.parquet")
    if path.exists():
        log.info(f"Stage 1 sentiment: loaded cache {path.name}")
        return pd.read_parquet(path)
    scorer = NLPScorer(finbert=FinBertScorer(variant=variant))
    links = [json.loads(x) for x in base["links"]]
    doc_s, ent_s = [], []
    t0 = time.perf_counter()
    for i in range(0, len(docs), batch_size):
        texts = [d.title or d.text for d in docs[i : i + batch_size]]
        for s, ent in scorer.sentiments(texts, links[i : i + batch_size]):
            doc_s.append(s)
            ent_s.append(json.dumps(ent))
        if (i // batch_size) % 100 == 0:
            rate = len(doc_s) / max(time.perf_counter() - t0, 1e-9)
            log.info(f"Stage 1 sentiment: {len(doc_s)}/{len(docs)} docs ({rate:.0f} docs/s)")
    out = pd.DataFrame(
        {"doc_id": base["doc_id"], "doc_sentiment": doc_s, "entity_sentiment": ent_s}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log.info(
        f"Stage 1 sentiment: {len(out)} docs in {time.perf_counter() - t0:.0f}s -> {path.name}"
    )
    return out


def _event_key() -> str:
    from riskpulse.engine.pipeline import EVENT_MODEL_PATH

    path = repo_root() / EVENT_MODEL_PATH
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8] if path.exists() else "none"


def event_stage(emb: np.ndarray, base: pd.DataFrame, feed_key: str, scorer: NLPScorer) -> list:
    """Event class per document, cached per event-model file (fast to recompute from the cached
    embeddings, but cached so results are exactly reproducible)."""
    path = _cache(f"stage1event_{feed_key}_{_event_key()}.parquet")
    if path.exists():
        ev = pd.read_parquet(path)
        log.info(f"Stage 1 events: loaded cache {path.name}")
        return [
            EventPrediction(r.event_class, float(r.event_conf), json.loads(r.event_probs))
            for r in ev.itertuples(index=False)
        ]
    preds = scorer.event_clf.predict_from_embeddings(emb)
    pd.DataFrame(
        {
            "doc_id": base["doc_id"],
            "event_class": [p.event_class for p in preds],
            "event_conf": [p.confidence for p in preds],
            "event_probs": [
                json.dumps({k: round(v, 4) for k, v in p.probs.items()}) for p in preds
            ],
        }
    ).to_parquet(path, index=False)
    return preds


def run_stage1(batch_size: int = 128, limit: int | None = None) -> list[DocScore]:
    """Score every feed document from three caches: links/embeddings, sentiment (per weights)
    and event classes (per event model, predicted from the cached embeddings)."""
    feed_key = _feed_hash() + (f"_n{limit}" if limit else "")
    docs = list(iter_feed())
    if limit:
        docs = docs[:limit]
    scorer = NLPScorer()
    base, emb = base_stage(docs, feed_key, scorer, batch_size)
    sent = sentiment_stage(docs, base, feed_key, batch_size=batch_size)
    assert (base["doc_id"].to_numpy() == sent["doc_id"].to_numpy()).all()
    events = event_stage(emb, base, feed_key, scorer)
    return [
        DocScore(
            doc=d,
            links=json.loads(b.links),
            regions=json.loads(b.regions),
            doc_sentiment=float(s.doc_sentiment),
            entity_sentiment=json.loads(s.entity_sentiment),
            event=ev,
            embedding=e,
        )
        for d, b, s, ev, e in zip(
            docs,
            base.itertuples(index=False),
            sent.itertuples(index=False),
            events,
            emb,
            strict=True,
        )
    ]


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
    from riskpulse.engine.sentiment import FinBertScorer

    versions = {
        "sentiment": FinBertScorer().version,
        "event": "embed-lr@" + load_config("app")["events"]["embedding_model"],
        "impact": engine.impact_version,
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
