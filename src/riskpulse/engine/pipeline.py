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
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path

import numpy as np

from riskpulse.common.config import load_config, repo_root
from riskpulse.common.schemas import Document, Signal
from riskpulse.engine.aggregate import EntityAggregator, EventAggregator, ScoredMention
from riskpulse.engine.clustering import StoryClusterer
from riskpulse.engine.entities import MKT, EntityLinker
from riskpulse.engine.events import (
    Embedder,
    EmbeddingClassifier,
    EventPrediction,
    KeywordClassifier,
)
from riskpulse.engine.impact import (
    BreadthTracker,
    ImpactBins,
    ImpactFeatures,
    ImpactScore,
    ImpactV2,
    VelocityTracker,
    score_impact,
)
from riskpulse.engine.sentiment import FinBertScorer, entity_sentiment, variant_for

EVENT_MODEL_PATH = "data/trained/event_clf_round2.pkl"  # committed (D-053, D-054)


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
    """Distinct-outlet key for breadth and n_sources: the publishing domain when known."""
    return str(doc.meta.get("domain") or doc.source.value)


def social_links(
    doc: Document, links: dict[str, float], original: dict[str, str], universe: set[str]
) -> dict[str, float]:
    """Original-ticker rule for tweets (D-033): a tweet links only to the ticker it was
    originally scraped for; copies and cross-ticker duplicates never move other names."""
    from riskpulse.ingestion.normalize import dedupe_key

    orig = original.get(dedupe_key(doc.text))
    if orig is None:  # not from the Kaggle corpus (e.g. a live social post): keep first link
        orig = next(iter(links), None)
    if orig is None or orig not in universe:
        return {}
    return {orig: links.get(orig, 0.3)}


def apply_social_link_policy(scored: list[DocScore]) -> tuple[list[DocScore], dict[str, int]]:
    """Apply the original-ticker rule to cached stage-1 scores (no model re-run)."""
    from riskpulse.ingestion.kaggle_tweets import original_ticker_lookup

    original = original_ticker_lookup()
    universe = set(load_config("universe")["tickers"])
    out, stats = [], {"social_docs": 0, "relinked": 0, "dropped": 0}
    for ds in scored:
        if ds.doc.source_type != "social":
            out.append(ds)
            continue
        stats["social_docs"] += 1
        new = social_links(ds.doc, ds.links, original, universe)
        if not new:
            stats["dropped"] += 1
            continue
        if set(new) != set(ds.links):
            stats["relinked"] += 1
        (t,) = new
        ent = {t: ds.entity_sentiment.get(t, ds.doc_sentiment)}
        out.append(DocScore(ds.doc, new, ds.regions, ds.doc_sentiment, ent, ds.event, ds.embedding))
    return out, stats


class NLPScorer:
    """Stage 1: models (FinBERT, MiniLM, event classifier) + entity linking."""

    def __init__(
        self,
        linker: EntityLinker | None = None,
        finbert: FinBertScorer | None = None,
        embedder: Embedder | None = None,
        event_clf: EmbeddingClassifier | None = None,
    ) -> None:
        # Models are created on first use, so linking alone (e.g. building caches) loads none.
        self.linker = linker or EntityLinker()
        self._finbert, self._embedder, self._event_clf = finbert, embedder, event_clf

    @cached_property
    def finbert(self) -> FinBertScorer:
        """Scorer for social text; also for all text when one scorer was passed explicitly."""
        return self._finbert or FinBertScorer(variant=variant_for("social"))

    @cached_property
    def finbert_news(self) -> FinBertScorer:
        """Scorer for news and other non-social text (D-058)."""
        if self._finbert is not None:
            return self._finbert
        if variant_for("news") == variant_for("social"):
            return self.finbert
        return FinBertScorer(variant=variant_for("news"))

    def scorer_for(self, source_type: str) -> FinBertScorer:
        return self.finbert if source_type == "social" else self.finbert_news

    @cached_property
    def embedder(self) -> Embedder:
        return self._embedder or Embedder()

    @cached_property
    def event_clf(self) -> EmbeddingClassifier | KeywordClassifier:
        """The trained classifier, or the keyword baseline if it is not trained here (a fresh
        clone: model files are not committed). Train it with `riskpulse train events`."""
        if self._event_clf is not None:
            return self._event_clf
        path = repo_root() / EVENT_MODEL_PATH
        if not path.exists():
            from riskpulse.common.logging import get_logger

            get_logger().warning(
                f"FALLBACK IN USE: event classifier {EVENT_MODEL_PATH} not found; event classes "
                "come from the keyword baseline, not the reported model (`riskpulse train events`)"
            )
            return KeywordClassifier()
        return EmbeddingClassifier.load(path, embedder=self.embedder)

    @cached_property
    def _tweet_original(self) -> dict[str, str]:
        from riskpulse.ingestion.kaggle_tweets import original_ticker_lookup

        return original_ticker_lookup()

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
        if doc.source_type == "social":
            links = social_links(doc, links, self._tweet_original, set(self.linker.tickers))
        regions = list(
            doc.meta.get("regions") or self.linker.regions_of(f"{doc.title or ''} {doc.text}")
        )
        return links, regions

    def sentiments(
        self,
        texts: list[str],
        links: list[dict[str, float]],
        source_types: list[str] | None = None,
    ) -> list[tuple[float, dict[str, float]]]:
        """(document score, entity scores) per text; entity scores are clause-level when two or
        more companies are linked, else the document score for every link. Each text is scored by
        the weights for its source type (D-058)."""
        types = source_types or ["news"] * len(texts)
        out: list = [None] * len(texts)
        for st in sorted(set(types)):
            idx = [i for i, t in enumerate(types) if t == st]
            fb = self.scorer_for(st)
            for i, s in zip(idx, fb.score([texts[i] for i in idx]), strict=True):
                companies = [t for t in links[i] if t != MKT]
                if len(companies) >= 2:
                    ent = entity_sentiment(texts[i], companies, fb, self.linker, s.score)
                else:
                    ent = dict.fromkeys(links[i], s.score)
                out[i] = (s.score, ent)
        return out

    def score(self, docs: list[Document]) -> list[DocScore]:
        texts = [d.title or d.text for d in docs]
        linked = [self.link(d) for d in docs]
        sent = self.sentiments(texts, [lk for lk, _ in linked], [d.source_type for d in docs])
        emb = self.embedder.encode(texts)
        clf = self.event_clf
        events = (
            clf.predict(texts)
            if isinstance(clf, KeywordClassifier)
            else clf.predict_from_embeddings(emb)
        )
        return [
            DocScore(d, links, regions, s, ent, e, x)
            for d, (links, regions), (s, ent), e, x in zip(
                docs, linked, sent, events, emb, strict=True
            )
        ]

    @property
    def versions(self) -> dict[str, str]:
        sent = f"news:{self.finbert_news.version}; social:{self.finbert.version}"
        return {"sentiment": sent, "event": self.event_clf.version, "impact": "v1"}


class SignalEngine:
    """Stage 2: time-ordered impact scoring and aggregation."""

    def __init__(self, bins: ImpactBins | None = None, v2: ImpactV2 | bool | None = None) -> None:
        """``v2``: None = as configured (configs/impact.yaml), False = v1 for all mentions."""
        self.impact_cfg = load_config("impact")
        self.bins = bins if bins is not None else ImpactBins.load()
        self.v2 = ImpactV2.load() if v2 is None else (v2 or None)
        self.impact_version = "v2-company+v1-market" if self.v2 else "v1-burnin-quantile"
        self.clusterer = StoryClusterer()
        self.velocity = VelocityTracker(self.impact_cfg)
        self.breadth = BreadthTracker(self.impact_cfg)
        self.entities = EntityAggregator()
        self.events = EventAggregator()
        self.lookback = float(self.impact_cfg["novelty"]["lookback_hours"])
        self.cred = self.impact_cfg["credibility"]
        self.priors = self.impact_cfg["type_prior"]
        every = load_config("app")["aggregation"]["entity_emit_every_minutes"]
        self.emit_every = timedelta(minutes=float(every))
        self._next_emit: datetime | None = None
        self._touched: dict[str, datetime] = {}

    def _emit_entities(self, as_of: datetime) -> list[Signal]:
        out = []
        for ticker in sorted(self._touched):
            sig = self.entities.signal(ticker, as_of)
            if sig is not None:
                out.append(sig)
        self._touched.clear()
        return out

    def process(self, scored: list[DocScore]) -> tuple[list[ScoredMention], list[Signal]]:
        """Consume stage-1 scores in time order; return mentions and emitted signals."""
        mentions: list[ScoredMention] = []
        signals: list[Signal] = []
        for ds in sorted(scored, key=lambda x: x.doc.published_at):
            d, ts = ds.doc, ds.doc.published_at
            if self._next_emit is None:
                self._next_emit = ts + self.emit_every
            elif ts >= self._next_emit:  # stream-time cadence for entity signals
                signals.extend(self._emit_entities(self._next_emit))
                while self._next_emit <= ts:
                    self._next_emit += self.emit_every
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
                group = "market" if ticker == MKT else "company"
                imp = score_impact(feats, self.bins, self.impact_cfg, group)
                if group == "company" and self.v2 is not None:
                    raw2 = round(self.v2.observe(ticker, ts, feats), 6)
                    imp = ImpactScore(
                        raw2, self.bins.score(raw2, group), imp.drivers, imp.contributions
                    )
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
                    self._touched[ticker] = ts
        if self._touched and scored:
            last = max(x.doc.published_at for x in scored)
            signals.extend(self._emit_entities(last))
        return mentions, signals


def model_versions(scorer: NLPScorer) -> dict[str, str]:
    return scorer.versions


def stamp_versions(signals: list[Signal], versions: dict[str, str]) -> list[Signal]:
    return [s.model_copy(update={"model_versions": versions}) for s in signals]


def model_origins() -> dict[str, str]:
    """Where each model the engine uses comes from (logged at start-up; fallbacks flagged)."""
    from riskpulse.engine.sentiment import resolve_model

    impact_cfg = load_config("impact").get("v2", {})
    v2_path = repo_root() / impact_cfg.get("model_path", "")
    out = {}
    for st in ("news", "social"):
        name, origin = resolve_model(variant_for(st))
        out[f"sentiment_{st}"] = f"{origin}:{name}"
    out |= {
        "event_classifier": f"repo:{EVENT_MODEL_PATH}"
        if event_model_exists()
        else "FALLBACK:keyword-baseline",
        "impact_company": f"repo:{impact_cfg.get('model_path')}"
        if impact_cfg.get("enabled") and v2_path.exists()
        else "FALLBACK:v1",
        "embeddings": f"hub:{load_config('app')['events']['embedding_model']}",
    }
    from riskpulse.common.logging import get_logger

    log = get_logger()
    for k, v in out.items():
        (log.warning if "FALLBACK" in v or v.startswith("base_fallback") else log.info)(
            f"model origin | {k}: {v}"
        )
    return out


def event_model_exists() -> bool:
    return Path(repo_root() / EVENT_MODEL_PATH).exists()
