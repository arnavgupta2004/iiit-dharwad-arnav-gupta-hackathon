"""Market-time helpers: UTC storage, US/Eastern market logic, next-trading-day mapping."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)


def market_date(ts: datetime | pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Trading day on which information published at ``ts`` can first be acted on at the close.

    Items published at or after 16:00 US/Eastern (or on a non-trading day) map to the next
    trading day; earlier items map to the same day. ``trading_days`` is a sorted, tz-naive
    index of session dates (e.g. from cached SPY prices).
    """
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    local = t.tz_convert(ET)
    day = pd.Timestamp(local.date())
    if local.time() >= MARKET_CLOSE:
        day += pd.Timedelta(days=1)
    pos = trading_days.searchsorted(day, side="left")
    if pos >= len(trading_days):
        raise ValueError(f"{ts} is after the last known trading day")
    return trading_days[pos]
