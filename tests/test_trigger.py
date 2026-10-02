from datetime import UTC, datetime, timedelta

from riskpulse.common.schemas import EventClass, Signal
from riskpulse.moduleB.trigger import StressTrigger

T0 = datetime(2022, 2, 24, 10, tzinfo=UTC)


def ev(
    impact=9,
    conf=0.8,
    n_sources=3,
    hours=0,
    cls=EventClass.GEOPOLITICAL,
    regions=("RUSSIA_UKRAINE",),
):
    return Signal(
        signal_id=f"s{impact}{hours}",
        signal_type="event",
        as_of=T0 + timedelta(hours=hours),
        sentiment_score=-0.7,
        sentiment_label="negative",
        event_class=cls,
        event_confidence=conf,
        impact_score=impact,
        impact_raw=0.8,
        confidence=0.7,
        n_docs=10,
        n_sources=n_sources,
        regions=list(regions),
    )


def test_trigger_conditions() -> None:
    t = StressTrigger()
    assert not t.check(ev(impact=7)).fired
    assert not t.check(ev(conf=0.5)).fired
    assert not t.check(ev(n_sources=1)).fired
    assert t.check(ev()).fired


def test_cooldown_per_class_and_region() -> None:
    t = StressTrigger()
    assert t.check(ev()).fired
    assert not t.check(ev(hours=5)).fired  # same key within 24 h
    assert t.check(ev(hours=5, regions=("US",))).fired  # different region key
    assert t.check(ev(hours=25)).fired  # cooldown elapsed
    assert any("cooldown" in d.reason for d in t.log)


def test_macro_region_keys() -> None:
    t = StressTrigger()
    assert t.key_of(ev(regions=("RUSSIA_UKRAINE",))) == "GEOPOLITICAL|Europe"
    assert t.key_of(ev(regions=("UK", "EUROPE"))) == "GEOPOLITICAL|Europe"
    assert t.key_of(ev(regions=("US", "CHINA"))) == "GEOPOLITICAL|Global"  # cross-regional
    assert t.key_of(ev(regions=("EM",))) == "GEOPOLITICAL|Global"
    assert t.key_of(ev(regions=())) == "GEOPOLITICAL|Global"


def test_escalation_within_cooldown() -> None:
    t = StressTrigger()
    assert t.check(ev(impact=8)).fired
    assert not t.check(ev(impact=8, hours=2)).fired  # same impact: cooldown
    d = t.check(ev(impact=9, hours=3))
    assert d.fired and d.reason.startswith("escalation")
    assert not t.check(ev(impact=9, hours=4)).fired  # new reference is 9
    assert t.check(ev(impact=10, hours=5)).fired
