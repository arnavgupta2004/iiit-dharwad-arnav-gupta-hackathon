from datetime import UTC, datetime

import duckdb

from riskpulse.common.schemas import EntityRef, EventClass, Signal
from riskpulse.store.signal_store import (
    JsonlSignalWriter,
    query_signals,
    read_signals_jsonl,
    signals_frame,
    write_duckdb,
)


def sig(i: int, ticker: str, impact: int, cls: EventClass, day: int) -> Signal:
    return Signal(
        signal_id=f"sig_{i}",
        signal_type="event",
        as_of=datetime(2022, 2, day, tzinfo=UTC),
        entity=EntityRef(ticker=ticker, name=ticker),
        sentiment_score=-0.5,
        sentiment_label="negative",
        event_class=cls,
        event_confidence=0.8,
        impact_score=impact,
        impact_raw=0.5,
        confidence=0.7,
        n_docs=3,
        n_sources=2,
    )


def test_jsonl_roundtrip_and_query(tmp_path) -> None:
    w = JsonlSignalWriter(tmp_path / "s.jsonl")
    sigs = [
        sig(1, "JPM", 9, EventClass.CREDIT_EVENT, 24),
        sig(2, "AAPL", 3, EventClass.EARNINGS, 25),
    ]
    w.write(sigs)
    back = read_signals_jsonl(tmp_path / "s.jsonl")
    assert [s.signal_id for s in back] == ["sig_1", "sig_2"]
    df = signals_frame(back)
    assert list(query_signals(df, min_impact=8)["ticker"]) == ["JPM"]
    assert list(query_signals(df, since="2022-02-25")["ticker"]) == ["AAPL"]
    assert list(query_signals(df, event_class="EARNINGS")["signal_id"]) == ["sig_2"]


def test_duckdb_store(tmp_path) -> None:
    path = write_duckdb([], [sig(1, "JPM", 9, EventClass.CREDIT_EVENT, 24)], tmp_path / "x.duckdb")
    with duckdb.connect(str(path), read_only=True) as con:
        assert con.execute("SELECT ticker, impact_score FROM signals").fetchall() == [("JPM", 9)]
