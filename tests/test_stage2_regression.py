"""Regression: the optimised stage-2 code must reproduce the original implementation.

Old = verbatim snapshot at commit 74716fb^ (tests/legacy). New = current src. Same 1,000 synthetic
stage-1 documents (hour-aligned GDELT-style timestamps, second-resolution tweets, clustered
embeddings, several outlets and tickers) go through both; story ids, impact scores, impact raws
and every driver must match (float tolerance 1e-9), as must the emitted event signals.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from riskpulse.common.schemas import Document, Source
from riskpulse.engine.events import EventPrediction
from riskpulse.engine.impact import ImpactBins
from riskpulse.engine.pipeline import DocScore, SignalEngine
from tests.legacy.pipeline_v0 import SignalEngine as LegacySignalEngine

TICKERS = ["AAPL", "TSLA", "JPM", "XOM", "MKT"]
CLASSES = ["GEOPOLITICAL", "MACROECONOMIC", "CREDIT_EVENT", "EARNINGS", "OTHER"]


def synthetic_stage1(n: int = 1000, seed: int = 7) -> list[DocScore]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(40, 384))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    t0 = datetime(2022, 2, 14, tzinfo=UTC)
    out = []
    for i in range(n):
        hours = int(rng.integers(0, 24 * 30))
        if rng.random() < 0.3:  # bursts: several items in the same hour
            hours = int(rng.choice([24 * 10, 24 * 10 + 1, 24 * 11]))
        social = rng.random() < 0.3
        ts = t0 + timedelta(hours=hours, seconds=int(rng.integers(0, 3600)) if social else 0)
        c = int(rng.integers(0, len(centers)))
        emb = centers[c] + 0.02 * rng.normal(size=384)  # pairwise cosine ~0.86 > 0.80
        emb /= np.linalg.norm(emb)
        tick = TICKERS[c % len(TICKERS)]
        src = Source.KAGGLE_TWEETS if social else Source.GDELT
        doc = Document.build(
            src,
            f"doc {i} cluster {c}",
            ts,
            title=None if social else f"doc {i}",
            # tweets carry no publishing domain (as in the real feed); news items do
            meta={} if social else {"domain": f"outlet{int(rng.integers(0, 12))}.com"},
        )
        s = float(rng.uniform(-1, 1))
        cls = CLASSES[c % len(CLASSES)]
        out.append(
            DocScore(
                doc=doc,
                links={tick: float(rng.uniform(0.2, 1.0))},
                regions=["US"] if tick != "MKT" else ["RUSSIA_UKRAINE"],
                doc_sentiment=s,
                entity_sentiment={tick: s},
                event=EventPrediction(cls, float(rng.uniform(0.3, 1)), {}),
                embedding=emb,
            )
        )
    return out


@pytest.fixture(scope="module")
def outputs():
    scored = synthetic_stage1()
    bins = ImpactBins.fit(np.linspace(0.0, 0.6, 500))
    new = SignalEngine(bins=bins).process(scored)
    old = LegacySignalEngine(bins=bins).process(scored)
    return new, old


def test_mentions_identical(outputs) -> None:
    (new_m, _), (old_m, _) = outputs
    assert len(new_m) == len(old_m) == 1000
    for a, b in zip(new_m, old_m, strict=True):
        assert (a.doc_id, a.ticker, a.event_id, a.impact_score) == (
            b.doc_id,
            b.ticker,
            b.event_id,
            b.impact_score,
        )
        assert a.impact_raw == pytest.approx(b.impact_raw, abs=1e-9)
        assert a.drivers.keys() == b.drivers.keys()
        for k in a.drivers:
            assert a.drivers[k] == pytest.approx(b.drivers[k], abs=1e-9), k


def test_event_signals_identical(outputs) -> None:
    (_, new_s), (_, old_s) = outputs
    ev_new = [s for s in new_s if s.signal_type == "event"]
    ev_old = [s for s in old_s if s.signal_type == "event"]
    assert len(ev_new) == len(ev_old) > 0
    for a, b in zip(ev_new, ev_old, strict=True):
        assert (a.event_id, a.impact_score, a.n_sources, a.event_class) == (
            b.event_id,
            b.impact_score,
            b.n_sources,
            b.event_class,
        )


def test_fixture_exercises_clustering_and_velocity(outputs) -> None:
    (new_m, _), _ = outputs
    assert len({m.event_id for m in new_m}) < 900  # stories actually merge
    assert max(m.drivers["velocity"] for m in new_m) > 0.9  # bursts register
