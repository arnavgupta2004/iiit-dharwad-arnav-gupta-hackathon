from datetime import UTC, datetime, timedelta

import pytest

from riskpulse.engine.aggregate import EntityAggregator, EventAggregator, ScoredMention

T0 = datetime(2022, 2, 24, 14, tzinfo=UTC)
CFG = {
    "entity_half_life_hours": 6,
    "entity_confidence_w0": 3.0,
    "event_breadth_bonus": [{"min_sources": 3, "bonus": 1}, {"min_sources": 5, "bonus": 2}],
    "event_emit_min_docs": 2,
    "evidence_max": 3,
}


def m(
    i: int,
    ticker: str,
    s: float,
    *,
    hours: float = 0,
    outlet: str = "a.com",
    impact: int = 5,
    cls: str = "GEOPOLITICAL",
    evt: str = "evt_1",
    rel: float = 1.0,
    conf: float = 1.0,
) -> ScoredMention:
    return ScoredMention(
        doc_id=f"d{i}",
        source="gdelt",
        outlet=outlet,
        published_at=T0 + timedelta(hours=hours),
        title=f"t{i}",
        url=None,
        ticker=ticker,
        relevance=rel,
        sentiment=s,
        event_class=cls,
        event_confidence=conf,
        event_id=evt,
        impact_score=impact,
        impact_raw=impact / 10,
        drivers={},
        regions=["RUSSIA_UKRAINE"],
    )


def test_entity_ewma_halves_old_evidence() -> None:
    agg = EntityAggregator(CFG)
    agg.update(m(1, "JPM", +1.0, hours=0))
    agg.update(m(2, "JPM", -1.0, hours=6))  # older item now has half the weight
    sig = agg.signal("JPM", T0 + timedelta(hours=6))
    assert sig.sentiment_score == pytest.approx((0.5 * 1 - 1) / 1.5, abs=1e-4)
    assert sig.signal_type == "entity" and sig.entity.ticker == "JPM"


def test_entity_confidence_grows_with_evidence_and_decays() -> None:
    agg = EntityAggregator(CFG)
    agg.update(m(1, "AAPL", 0.5))
    c1 = agg.signal("AAPL", T0).confidence
    for i in range(2, 8):
        agg.update(m(i, "AAPL", 0.5))
    c7 = agg.signal("AAPL", T0).confidence
    stale = agg.signal("AAPL", T0 + timedelta(hours=48)).confidence
    assert c1 < c7 and stale < c7
    assert agg.signal("MSFT", T0) is None


def test_event_signal_vote_bonus_and_emission() -> None:
    agg = EventAggregator(CFG)
    assert agg.update(m(1, "MKT", -0.8, outlet="a.com", impact=7)) is None  # below min docs
    s2 = agg.update(m(2, "MKT", -0.6, outlet="b.com", impact=8))
    assert s2 is not None and s2.impact_score == 8 and s2.n_sources == 2
    assert agg.update(m(3, "MKT", -0.7, outlet="b.com", impact=6)) is None  # impact unchanged
    s4 = agg.update(m(4, "MKT", -0.9, outlet="c.com", impact=6))  # 3 outlets -> +1 bonus
    assert s4.impact_score == 9 and s4.event_class == "GEOPOLITICAL"
    assert s4.regions == ["RUSSIA_UKRAINE"] and len(s4.evidence) == 3
    assert s4.sentiment_label == "negative"


def test_event_class_is_confidence_weighted_majority() -> None:
    agg = EventAggregator(CFG)
    agg.update(m(1, "MKT", -0.5, cls="MACROECONOMIC", conf=0.9))
    agg.update(m(2, "MKT", -0.5, cls="GEOPOLITICAL", conf=0.3))
    sig = agg.update(m(3, "MKT", -0.5, cls="GEOPOLITICAL", conf=0.3, impact=9))
    assert sig.event_class == "MACROECONOMIC"
