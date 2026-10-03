"""The demo replay and /inject must never write to the batch signals file (Module B and evaluation
read it); they append to the session file instead."""

import hashlib
import time
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from riskpulse.api import app as app_module
from riskpulse.common.schemas import Document, Source
from riskpulse.ingestion import replay as replay_module
from riskpulse.store.signal_store import JsonlSignalWriter
from tests.test_api import sig


class FakeLive:
    def process(self, docs):
        return [], [sig(10 + i, "event", None, 9) for i, _ in enumerate(docs)]

    def inject(self, text, published_at=None, outlet="demo"):
        return [sig(20, "event", None, 9)]

    def analyze(self, text, source):
        return {}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_replay_and_inject_leave_batch_file_byte_identical(tmp_path, monkeypatch) -> None:
    batch = tmp_path / "signals" / "signals.jsonl"
    JsonlSignalWriter(batch).write([sig(1, "entity", "JPM", 3), sig(2, "event", None, 9)])
    before = _sha(batch)
    session = tmp_path / "session" / "signals.jsonl"
    monkeypatch.setattr(app_module, "serving_signals_path", lambda: batch)
    monkeypatch.setattr(app_module, "session_signals_path", lambda: session)
    docs = [
        Document.build(Source.GDELT, f"replayed headline {i}", datetime(2022, 2, 24, i, tzinfo=UTC))
        for i in range(3)
    ]
    monkeypatch.setattr(replay_module, "replay", lambda start, end, spd: iter(docs))

    app = app_module.create_app("full", live=FakeLive(), replay=("2022-02-24", "2022-02-25", 1.0))
    with TestClient(app) as c:
        for _ in range(100):
            if c.get("/health").json()["replay"] == "finished":
                break
            time.sleep(0.05)
        assert c.get("/health").json()["replay"] == "finished"
        assert (
            c.post("/inject", json={"text": "Synthetic bank default", "outlets": 2}).status_code
            == 200
        )
        assert (
            c.get("/health").json()["n_signals"] == 2 + 3 + 2
        )  # batch + replayed docs + injected copies

    assert _sha(batch) == before
    assert len(session.read_text().splitlines()) == 5  # 3 replayed + 2 injected, all here
