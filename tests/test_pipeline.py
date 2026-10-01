"""End-to-end engine test with tiny fake models (no downloads)."""

from datetime import UTC, datetime, timedelta

import numpy as np

from riskpulse.common.schemas import Document, Source
from riskpulse.engine.events import EventPrediction
from riskpulse.engine.impact import ImpactBins
from riskpulse.engine.pipeline import NLPScorer, SignalEngine
from riskpulse.engine.sentiment import SentimentResult

T0 = datetime(2022, 2, 24, 6, tzinfo=UTC)


class FakeFinbert:
    version = "fake"

    def score(self, texts):
        out = []
        for t in texts:
            s = -0.8 if any(w in t.lower() for w in ("invade", "miss", "falls")) else 0.6
            out.append(SentimentResult(s, "negative" if s < 0 else "positive", 0, 0, 0))
        return out


class FakeEmbedder:
    model_name = "fake"

    def encode(self, texts):
        vecs = []
        for t in texts:
            v = np.array([1.0, 0.0, 0.0]) if "ukraine" in t.lower() else np.array([0.0, 1.0, 0.0])
            vecs.append(v / np.linalg.norm(v))
        return np.vstack(vecs)


class FakeEventClf:
    version = "fake"

    def predict_from_embeddings(self, x):
        return [EventPrediction("GEOPOLITICAL" if v[0] > 0.5 else "EARNINGS", 0.9, {}) for v in x]


def doc(text: str, minutes: int, domain: str, source: Source = Source.GDELT, tickers=None):
    return Document.build(
        source,
        text,
        T0 + timedelta(minutes=minutes),
        title=text if source == Source.GDELT else None,
        url=f"https://{domain}/{minutes}",
        meta={"domain": domain, "tickers": tickers or []},
    )


def test_engine_end_to_end_produces_entity_and_event_signals() -> None:
    scorer = NLPScorer(finbert=FakeFinbert(), embedder=FakeEmbedder(), event_clf=FakeEventClf())
    docs = [
        doc("Russia invades Ukraine as sanctions loom", 0, "a.com", tickers=["MKT"]),
        doc("Russia invades Ukraine; sanctions announced", 10, "b.com", tickers=["MKT"]),
        doc("Ukraine invasion: sanctions on Russian banks", 20, "c.com", tickers=["MKT"]),
        doc("JPMorgan earnings beat estimates", 30, "d.com", tickers=["JPM"]),
        doc("$TSLA deliveries strong, shares up", 40, "x", Source.KAGGLE_TWEETS, ["TSLA"]),
    ]
    scored = scorer.score(docs)
    assert scored[0].links == {"MKT": 1.0} and "RUSSIA_UKRAINE" in scored[0].regions
    mentions, signals = SignalEngine(bins=ImpactBins(None)).process(scored)
    assert len(mentions) == 5
    mkt_events = {m.event_id for m in mentions if m.ticker == "MKT"}
    assert len(mkt_events) == 1  # the three invasion headlines form one story
    (mkt_id,) = mkt_events
    ev = [s for s in signals if s.signal_type == "event" and s.event_id == mkt_id]
    assert ev and ev[-1].event_class.value == "GEOPOLITICAL" and ev[-1].n_sources == 3
    assert "RUSSIA_UKRAINE" in ev[-1].regions
    ent = {s.entity.ticker: s for s in signals if s.signal_type == "entity"}
    assert set(ent) == {"JPM", "TSLA"}
    assert ent["JPM"].sentiment_score > 0
    assert all(1 <= m.impact_score <= 10 for m in mentions)
    # first-in-story items are novel; later ones are not
    first, later = mentions[0], mentions[2]
    assert first.drivers["novelty"] == 1.0 and later.drivers["novelty"] < 0.1
