from datetime import UTC, datetime

import pandas as pd

from riskpulse.common.timeutils import market_date
from riskpulse.ingestion.dedupe import dedupe_frame
from riskpulse.ingestion.normalize import clean_text, dedupe_key

TRADING_DAYS = pd.DatetimeIndex(
    ["2022-02-24", "2022-02-25", "2022-02-28", "2022-03-11", "2022-03-14", "2022-03-15"]
)


def test_clean_text_unescapes_and_strips_urls() -> None:
    assert clean_text("AT&amp;T  beats  https://t.co/abc") == "AT&T beats"
    assert clean_text(None) == ""


def test_dedupe_key_collapses_retweets_handles_and_urls() -> None:
    a = dedupe_key("RT @trader: $TSLA breaks out! https://t.co/x")
    b = dedupe_key("$TSLA breaks out https://t.co/y")
    assert a == b


def test_market_date_before_and_after_close() -> None:
    # 2022-02-24 15:59 ET (EST, UTC-5) -> same day; 16:00 ET -> next trading day
    assert market_date(datetime(2022, 2, 24, 20, 59, tzinfo=UTC), TRADING_DAYS) == pd.Timestamp(
        "2022-02-24"
    )
    assert market_date(datetime(2022, 2, 24, 21, 0, tzinfo=UTC), TRADING_DAYS) == pd.Timestamp(
        "2022-02-25"
    )


def test_market_date_weekend_and_dst() -> None:
    # Saturday -> Monday
    assert market_date(datetime(2022, 2, 26, 12, tzinfo=UTC), TRADING_DAYS) == pd.Timestamp(
        "2022-02-28"
    )
    # After the 2022-03-13 DST switch the close is 20:00 UTC (EDT, UTC-4)
    assert market_date(datetime(2022, 3, 14, 19, 59, tzinfo=UTC), TRADING_DAYS) == pd.Timestamp(
        "2022-03-14"
    )
    assert market_date(datetime(2022, 3, 14, 20, 0, tzinfo=UTC), TRADING_DAYS) == pd.Timestamp(
        "2022-03-15"
    )


def test_dedupe_exact_and_near_keeps_earliest() -> None:
    t0 = pd.Timestamp("2022-02-24 10:00", tz="UTC")
    df = pd.DataFrame(
        {
            "published_at": [t0, t0 + pd.Timedelta("1h"), t0 + pd.Timedelta("2h"), t0],
            "text": [
                "Russia invades Ukraine, markets tumble",
                "Russia invades Ukraine, markets tumble!",  # exact after canonicalisation
                "Markets tumble as Russia invades Ukraine",  # near-duplicate (token set)
                "Oil jumps above $100",
            ],
            "primary_ticker": ["MKT", "MKT", "MKT", "MKT"],
        }
    )
    out, stats = dedupe_frame(df)
    assert stats == {"input": 4, "exact_dupes": 1, "near_dupes": 1, "kept": 2}
    assert out["published_at"].min() == t0
    assert set(out["text"]) == {"Russia invades Ukraine, markets tumble", "Oil jumps above $100"}


def test_short_subset_text_is_not_a_near_duplicate() -> None:
    t0 = pd.Timestamp("2022-02-24 10:00", tz="UTC")
    df = pd.DataFrame(
        {
            "published_at": [t0, t0 + pd.Timedelta("1h")],
            "text": ["$TSLA to the moon after record deliveries this quarter", "$TSLA deliveries"],
            "primary_ticker": ["TSLA", "TSLA"],
        }
    )
    out, stats = dedupe_frame(df)
    assert stats["kept"] == 2
