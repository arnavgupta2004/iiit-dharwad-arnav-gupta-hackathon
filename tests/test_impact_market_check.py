"""Market-wide impact check: session assignment and max-per-session aggregation (D-042)."""

from datetime import UTC, datetime

import pandas as pd

from riskpulse.common.schemas import EntityRef, EventClass, Signal
from riskpulse.eval import impact_market_check as imc


def _sig(i: int, ts: datetime, impact: int, ticker: str | None) -> Signal:
    return Signal(
        signal_id=f"s{i}",
        signal_type="event",
        as_of=ts,
        entity=EntityRef(ticker=ticker, name="x") if ticker else None,
        sentiment_score=0.0,
        sentiment_label="neutral",
        event_class=EventClass.MACROECONOMIC,
        event_confidence=0.9,
        impact_score=impact,
        impact_raw=0.5,
        confidence=0.5,
        n_docs=3,
        n_sources=2,
    )


def test_max_impact_per_session_with_close_cutoff(monkeypatch) -> None:
    days = pd.DatetimeIndex(["2022-06-14", "2022-06-15", "2022-06-16"])
    monkeypatch.setattr(imc, "trading_days", lambda: days)
    sigs = [
        _sig(1, datetime(2022, 6, 14, 15, tzinfo=UTC), 7, "MKT"),  # 11:00 ET -> 06-14
        _sig(2, datetime(2022, 6, 14, 21, tzinfo=UTC), 9, "MKT"),  # 17:00 ET -> 06-15
        _sig(3, datetime(2022, 6, 15, 14, tzinfo=UTC), 8, "MKT"),  # 10:00 ET -> 06-15
        _sig(4, datetime(2022, 6, 15, 14, tzinfo=UTC), 10, "AAPL"),  # company story: ignored
        _sig(5, datetime(2022, 6, 15, 14, tzinfo=UTC), 10, None),  # mixed story: ignored
    ]
    out = imc.market_event_days(sigs)["max_impact"].to_dict()
    assert out == {pd.Timestamp("2022-06-14"): 7, pd.Timestamp("2022-06-15"): 9}
