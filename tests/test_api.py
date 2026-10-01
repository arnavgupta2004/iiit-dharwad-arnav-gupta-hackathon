import json
from datetime import UTC, datetime

import httpx
import pytest
from fastapi.testclient import TestClient

from riskpulse.api.app import create_app
from riskpulse.api.hub import SignalHub
from riskpulse.common.schemas import EntityRef, EventClass, Signal


def sig(i: int, kind: str, ticker: str | None, impact: int, cls=EventClass.GEOPOLITICAL) -> Signal:
    return Signal(
        signal_id=f"sig_{i}",
        signal_type=kind,
        as_of=datetime(2022, 2, 24, i, tzinfo=UTC),
        entity=EntityRef(ticker=ticker, name=ticker) if ticker else None,
        sentiment_score=-0.4,
        sentiment_label="negative",
        event_class=cls,
        event_confidence=0.8,
        impact_score=impact,
        impact_raw=0.6,
        confidence=0.7,
        n_docs=4,
        n_sources=3,
        regions=["RUSSIA_UKRAINE"],
    )


class FakeLive:
    def analyze(self, text, source):
        return {"text": text, "event_class": "GEOPOLITICAL", "entities": []}

    def inject(self, text, published_at=None, outlet="demo"):
        return [sig(9, "event", None, 9)]


@pytest.fixture()
def hub(tmp_path):
    h = SignalHub(tmp_path / "s.jsonl", load_existing=False)
    h.publish([sig(1, "entity", "JPM", 3), sig(2, "entity", "JPM", 4), sig(3, "event", None, 9)])
    return h


def test_rest_endpoints_fast_mode(hub) -> None:
    c = TestClient(create_app("fast", hub=hub))
    assert c.get("/health").json()["n_signals"] == 3
    assert [s["signal_id"] for s in c.get("/signals", params={"ticker": "JPM"}).json()] == [
        "sig_2",
        "sig_1",
    ]
    assert [s["signal_id"] for s in c.get("/events", params={"min_impact": 8}).json()] == ["sig_3"]
    assert [s["signal_id"] for s in c.get("/signals/latest").json()] == ["sig_2"]
    assert any(e["ticker"] == "MKT" for e in c.get("/entities").json())
    assert c.post("/analyze", json={"text": "Fed hikes"}).status_code == 503


def test_analyze_and_inject_full_mode(hub, tmp_path) -> None:
    c = TestClient(create_app("full", hub=hub, live=FakeLive()))
    assert (
        c.post("/analyze", json={"text": "Russia invades"}).json()["event_class"] == "GEOPOLITICAL"
    )
    r = c.post("/inject", json={"text": "Synthetic shock", "outlets": 2}).json()
    assert r["source"] == "synthetic_demo" and len(r["signals"]) == 2
    assert len(hub.signals) == 5
    lines = (tmp_path / "s.jsonl").read_text().strip().splitlines()
    assert len(lines) == 5 and json.loads(lines[-1])["signal_id"] == "sig_9"


def test_sse_stream_delivers_published_signals(hub) -> None:
    """Real uvicorn server on a free port: the in-process ASGI transport buffers streams."""
    import socket
    import threading
    import time

    import uvicorn

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app("fast", hub=hub), host="127.0.0.1", port=port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)

    def publish_later() -> None:
        time.sleep(0.5)
        hub.publish([sig(4, "event", None, 5), sig(5, "event", None, 10)], persist=False)

    threading.Thread(target=publish_later, daemon=True).start()
    got = []
    try:
        with httpx.stream(
            "GET", f"http://127.0.0.1:{port}/stream", params={"min_impact": 8}, timeout=10
        ) as resp:
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    got.append(json.loads(line[5:])["signal_id"])
                    break
    finally:
        server.should_exit = True
    assert got == ["sig_5"]  # the impact-5 signal was filtered out by min_impact
