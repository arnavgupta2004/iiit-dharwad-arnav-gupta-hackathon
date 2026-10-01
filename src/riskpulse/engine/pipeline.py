"""End-to-end engine: documents -> scored mentions -> entity and event signals.

Two stages so the expensive models run once:

- ``NLPScorer`` (stage 1): entity linking, FinBERT doc and entity sentiment, MiniLM
  embeddings and event-class probabilities. Batched; results can be cached.
- ``SignalEngine`` (stage 2): strictly time-ordered. Story clustering and novelty, velocity,
  breadth, impact v1, then entity/event aggregation. Cheap, so impact can be re-binned or
  re-weighted without re-running the models.

The same code path serves batch (whole feed), replay (paced feed) and the live ``/analyze``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from riskpulse.common.config import load_config, repo_root
from riskpulse.common.schemas import Document, Signal
from riskpulse.engine.aggregate import EntityAggregator, EventAggregator, ScoredMention
from riskpulse.engine.clustering import StoryClusterer
from riskpulse.engine.entities import MKT, EntityLinker
from riskpulse.engine.events import Embedder, EmbeddingClassifier, EventPrediction
from riskpulse.engine.impact import (
    BreadthTracker,
    ImpactBins,
    ImpactFeatures,
    VelocityTracker,
    score_impact,
)
from riskpulse.engine.sentiment import FinBertScorer, entity_sentiment

EVENT_MODEL_PATH = "data/processed/models/event_clf.pkl"


@dataclass
class DocScore:
    """Stage-1 output for one document."""

    doc: Document
    links: dict[str, float]  # ticker -> relevance (MKT allowed)
    regions: list[str]
    doc_sentiment: float
    entity_sentiment: dict[str, float]
    event: EventPrediction
    embedding: np.ndarray = field(repr=False)


def outlet_of(doc: Document) -> str:
    if doc.source.value == "gdelt":
        return str(doc.meta.get("domain") or "gdelt")
    return doc.source.value


class NLPScorer:
    """Stage 1: models (FinBERT, MiniLM, event classifier) + entity linking."""

    def __init__(
        self,
        linker: EntityLinker | None = None,
        finbert: FinBertScorer | None = None,
        embedder: Embedder | None = None,
        event_clf: EmbeddingClassifier | None = None,
    ) -> None:
        self.linker = linker or EntityLinker()
        self.finbert = finbert or FinBertScorer()
        self.embedder = embedder or Embedder()
        self.event_clf = event_clf or EmbeddingClassifier.load(
            repo_root() / EVENT_MODEL_PATH, embedder=self.embedder
        )

    def link(self, doc: Document) -> tuple[dict[str, float], list[str]]:
        meta_tickers = list(doc.meta.get("tickers") or [])
        prior = meta_tickers[0] if doc.source_type == "social" and meta_tickers else None
        ms = self.linker.link(
            doc.text if doc.title is None else "", title=doc.title, prior_ticker=prior
        )
        links = {m.ticker: m.relevance for m in ms}
        for t in meta_tickers:  # labels from the feed builder are kept even if not re-found
            links.setdefault(t, 0.3)
        if len(links) > 1:
            links.pop(MKT, None)
        regions = list(
            doc.meta.get("regions") or self.linker.regions_of(f"{doc.title or ''} {doc.text}")
        )
        return links, regions

    def score(self, docs: list[Document]) -> list[DocScore]:
        texts = [d.title or d.text for d in docs]
        sent = self.finbert.score(texts)
        emb = self.embedder.encode(texts)
        events = self.event_clf.predict_from_embeddings(emb)
        out = []
        for d, s, e, x, text in zip(docs, sent, events, emb, texts, strict=True):
            links, regions = self.link(d)
            companies = [t for t in links if t != MKT]
            if len(companies) >= 2:
                ent = entity_sentiment(text, companies, self.finbert, self.linker, s.score)
            else:
                ent = dict.fromkeys(links, s.score)
            out.append(DocScore(d, links, regions, s.score, ent, e, x))
        return out

    @property
    def versions(self) -> dict[str, str]:
        return {"sentiment": self.finbert.version, "event": self.event_clf.version, "impact": "v1"}


class SignalEngine:
    """Stage 2: time-ordered impact scoring and aggregation."""

    def __init__(self, bins: ImpactBins | None = None) -> None:
        self.impact_cfg = load_config("impact")
        self.bins = bins if bins is not None else ImpactBins.load()
        self.clusterer = StoryClusterer()
        self.velocity = VelocityTracker(self.impact_cfg)
        self.breadth = BreadthTracker(self.impact_cfg)
        self.entities = EntityAggregator()
        self.events = EventAggregator()
        self.lookback = float(self.impact_cfg["novelty"]["lookback_hours"])
        self.cred = self.impact_cfg["credibility"]
        self.priors = self.impact_cfg["type_prior"]

    def process(self, scored: list[DocScore]) -> tuple[list[ScoredMention], list[Signal]]:
        """Consume stage-1 scores in time order; return mentions and emitted signals."""
        mentions: list[ScoredMention] = []
        signals: list[Signal] = []
        touched: dict[str, datetime] = {}
        for ds in sorted(scored, key=lambda x: x.doc.published_at):
            d, ts = ds.doc, ds.doc.published_at
            event_id, _, _, novelty = self.clusterer.observe(
                d.doc_id, ds.embedding, ts, self.lookback
            )
            outlet = outlet_of(d)
            for ticker, rel in ds.links.items():
                # MKT velocity/breadth are tracked per primary region (first in config order).
                key = (
                    ticker if ticker != MKT else f"MKT:{ds.regions[0] if ds.regions else 'GLOBAL'}"
                )
                v, _ = self.velocity.observe(key, ts)
                b, _ = self.breadth.observe(key, outlet, ts)
                s = ds.entity_sentiment.get(ticker, ds.doc_sentiment)
                feats = ImpactFeatures(
                    type_prior=float(self.priors[ds.event.event_class]),
                    sentiment=abs(s),
                    velocity=v,
                    breadth=b,
                    novelty=novelty,
                    credibility=float(self.cred.get(d.source.value, 0.5)),
                    relevance=float(rel),
                )
                imp = score_impact(feats, self.bins, self.impact_cfg)
                m = ScoredMention(
                    doc_id=d.doc_id,
                    source=d.source.value,
                    outlet=outlet,
                    published_at=ts,
                    title=d.title or d.text[:200],
                    url=d.url,
                    ticker=ticker,
                    relevance=float(rel),
                    sentiment=float(s),
                    event_class=ds.event.event_class,
                    event_confidence=float(ds.event.confidence),
                    event_id=event_id,
                    impact_score=imp.score,
                    impact_raw=imp.raw,
                    drivers=imp.drivers,
                    regions=ds.regions,
                )
                mentions.append(m)
                self.entities.update(m)
                ev = self.events.update(m)
                if ev is not None:
                    signals.append(ev)
                if ticker != MKT:
                    touched[ticker] = ts
        for ticker, ts in touched.items():
            sig = self.entities.signal(ticker, ts)
            if sig is not None:
                signals.append(sig)
        return mentions, signals


def model_versions(scorer: NLPScorer) -> dict[str, str]:
    return scorer.versions


def stamp_versions(signals: list[Signal], versions: dict[str, str]) -> list[Signal]:
    return [s.model_copy(update={"model_versions": versions}) for s in signals]


def event_model_exists() -> bool:
    return Path(repo_root() / EVENT_MODEL_PATH).exists()
