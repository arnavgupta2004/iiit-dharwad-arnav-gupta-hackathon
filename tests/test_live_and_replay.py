from datetime import UTC, datetime

import pytest

from riskpulse.common.schemas import Document, Source
from riskpulse.ingestion import gdelt_doc, replay


class _Resp:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("empty body")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, responses: list[_Resp]) -> None:
        self.responses = responses
        self.calls = 0

    def get(self, *args, **kwargs) -> _Resp:
        self.calls += 1
        return self.responses.pop(0)


def test_gdelt_doc_backs_off_on_429_then_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdelt_doc.time, "sleep", lambda s: None)
    article = {
        "url": "https://x.com/a",
        "title": "Fed raises rates",
        "seendate": "20220615T180000Z",
        "domain": "x.com",
        "sourcecountry": "United States",
    }
    session = _Session([_Resp(429), _Resp(200, {"articles": [article]})])
    client = gdelt_doc.GdeltDocClient(session=session)
    arts = client.search("Federal Reserve")
    assert session.calls == 2 and len(arts) == 1
    docs = gdelt_doc.to_documents(arts)
    assert docs[0].source == Source.GDELT
    assert docs[0].published_at == datetime(2022, 6, 15, 18, tzinfo=UTC)


def test_gdelt_doc_empty_body_is_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdelt_doc.time, "sleep", lambda s: None)
    client = gdelt_doc.GdeltDocClient(session=_Session([_Resp(200, None)]))
    assert client.search("nothing matches") == []


def test_replay_paces_in_simulated_time(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = [
        Document.build("gdelt", "a", datetime(2022, 2, 24, 0, tzinfo=UTC)),
        Document.build("gdelt", "b", datetime(2022, 2, 24, 12, tzinfo=UTC)),  # +0.5 day
        Document.build("gdelt", "c", datetime(2022, 2, 25, 0, tzinfo=UTC)),  # +1 day
    ]
    monkeypatch.setattr(replay, "iter_feed", lambda start, end: iter(docs))
    clock = {"t": 0.0}
    monkeypatch.setattr(replay.time, "monotonic", lambda: clock["t"])
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock["t"] += s

    out = list(
        replay.replay("2022-02-24", "2022-02-26", seconds_per_market_day=10, sleep=fake_sleep)
    )
    assert [d.text for d in out] == ["a", "b", "c"]
    assert sleeps == pytest.approx([5.0, 5.0])
