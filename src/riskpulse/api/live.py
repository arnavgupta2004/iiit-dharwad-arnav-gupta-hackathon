"""Live engine used by the API: /analyze, /inject and paced replay share one stateful engine."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from riskpulse.common.schemas import Document, Signal, Source
from riskpulse.engine.aggregate import ScoredMention
from riskpulse.engine.pipeline import NLPScorer, SignalEngine, stamp_versions


class LiveEngine:
    """Stateful wrapper: stage 1 (models) + stage 2 (time-ordered aggregation) under a lock."""

    def __init__(self, scorer: NLPScorer | None = None, engine: SignalEngine | None = None) -> None:
        self.scorer = scorer or NLPScorer()
        self.engine = engine or SignalEngine()
        self._lock = threading.Lock()

    def process(self, docs: list[Document]) -> tuple[list[ScoredMention], list[Signal]]:
        with self._lock:
            scored = self.scorer.score(docs)
            mentions, signals = self.engine.process(scored)
        return mentions, stamp_versions(signals, self.scorer.versions)

    def analyze(self, text: str, source: str = "synthetic_demo") -> dict:
        """Score one headline without mutating engine state (a dry run for the jury demo)."""
        doc = Document.build(Source(source), text, datetime.now(UTC), title=text)
        with self._lock:
            ds = self.scorer.score([doc])[0]
            scratch = SignalEngine(bins=self.engine.bins)
            mentions, _ = scratch.process([ds])
        return {
            "text": text,
            "source": source,
            "sentiment_score": round(ds.doc_sentiment, 4),
            "event_class": ds.event.event_class,
            "event_confidence": round(ds.event.confidence, 4),
            "event_probs": {
                k: round(v, 4) for k, v in sorted(ds.event.probs.items(), key=lambda kv: -kv[1])
            },
            "regions": ds.regions,
            "entities": [
                {
                    "ticker": m.ticker,
                    "relevance": round(m.relevance, 3),
                    "sentiment_score": round(m.sentiment, 4),
                    "impact_score": m.impact_score,
                    "impact_raw": round(m.impact_raw, 4),
                    "drivers": m.drivers,
                }
                for m in mentions
            ],
            "note": "Dry run: velocity/breadth computed for this item alone (no stream context).",
            "model_versions": self.scorer.versions,
        }

    def inject(
        self, text: str, published_at: datetime | None = None, outlet: str = "demo"
    ) -> list[Signal]:
        """Push a clearly labelled synthetic headline into the live stream."""
        doc = Document.build(
            Source.SYNTHETIC_DEMO,
            text,
            published_at or datetime.now(UTC),
            title=text,
            meta={
                "domain": f"synthetic:{outlet}",
                "label": "SYNTHETIC DEMO HEADLINE - not real news",
            },
        )
        _, signals = self.process([doc])
        return signals
