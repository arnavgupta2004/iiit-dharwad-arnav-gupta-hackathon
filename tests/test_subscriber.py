from datetime import UTC, datetime

from riskpulse.common.schemas import EventClass, Signal
from riskpulse.store.signal_store import JsonlSignalWriter
from riskpulse.subscribe.subscriber import StorePollingSubscriber


def sig(i: int, kind: str, impact: int) -> Signal:
    return Signal(
        signal_id=f"sig_{i}",
        signal_type=kind,
        as_of=datetime(2022, 2, 24, tzinfo=UTC),
        sentiment_score=0.0,
        sentiment_label="neutral",
        event_class=EventClass.OTHER,
        event_confidence=0.5,
        impact_score=impact,
        impact_raw=0.1,
        confidence=0.5,
        n_docs=1,
        n_sources=1,
    )


def test_store_polling_filters_by_type_and_impact(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    JsonlSignalWriter(path).write([sig(1, "entity", 3), sig(2, "event", 5), sig(3, "event", 9)])
    sub = StorePollingSubscriber(path, signal_type="event", min_impact=8)
    assert [s.signal_id for s in sub] == ["sig_3"]


def test_store_polling_ignores_partial_trailing_line(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    JsonlSignalWriter(path).write([sig(1, "event", 9)])
    with path.open("a") as fh:
        fh.write('{"partial": ')  # a writer mid-flush
    assert [s.signal_id for s in StorePollingSubscriber(path)] == ["sig_1"]
