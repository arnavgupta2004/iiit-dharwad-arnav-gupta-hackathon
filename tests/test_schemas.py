from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from riskpulse.common.schemas import Document, EventClass, Signal, Source, make_doc_id


def test_doc_id_is_stable_and_source_scoped() -> None:
    a = make_doc_id("gdelt", "https://x.com/a", "text")
    assert a == make_doc_id("gdelt", "https://x.com/a", "other text")
    assert a != make_doc_id("newsapi", "https://x.com/a", "text")
    assert make_doc_id("kaggle_tweets", None, "t1") != make_doc_id("kaggle_tweets", None, "t2")


def test_document_build_normalises_to_utc_and_sets_source_type() -> None:
    est = timezone(timedelta(hours=-4))
    doc = Document.build(
        Source.KAGGLE_TWEETS, "hello $AAPL", datetime(2022, 1, 3, 9, 30, tzinfo=est)
    )
    assert doc.source_type == "social"
    assert doc.published_at.utcoffset() == timedelta(0)
    assert doc.published_at.hour == 13


def test_naive_timestamps_are_treated_as_utc() -> None:
    doc = Document.build("gdelt", "x", datetime(2022, 2, 24, 12, 0))
    assert doc.published_at.tzinfo is not None
    assert doc.source_type == "news"


def _signal(**overrides) -> dict:
    base = dict(
        signal_id="sig_1",
        signal_type="event",
        as_of=datetime(2022, 2, 24, tzinfo=UTC),
        sentiment_score=-0.6,
        sentiment_label="negative",
        event_class=EventClass.GEOPOLITICAL,
        event_confidence=0.8,
        impact_score=9,
        impact_raw=0.81,
        confidence=0.7,
        n_docs=12,
        n_sources=3,
    )
    base.update(overrides)
    return base


def test_signal_valid() -> None:
    sig = Signal(**_signal())
    assert sig.schema_version == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("sentiment_score", 1.5),
        ("impact_score", 0),
        ("impact_score", 11),
        ("event_confidence", 1.2),
    ],
)
def test_signal_rejects_out_of_range(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Signal(**_signal(**{field: value}))
