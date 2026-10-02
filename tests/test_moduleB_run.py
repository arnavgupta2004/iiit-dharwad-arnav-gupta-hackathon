"""Module B end-to-end: stored event signals -> trigger -> stress runs (uses the committed book)."""

from datetime import UTC, datetime, timedelta

import pytest

from riskpulse.common.config import data_path
from riskpulse.common.schemas import EventClass, Evidence, Signal
from riskpulse.moduleB import run as runmod
from riskpulse.store.signal_store import JsonlSignalWriter
from riskpulse.subscribe.subscriber import StorePollingSubscriber

T0 = datetime(2022, 2, 24, 5, tzinfo=UTC)
HAVE_BOOK = data_path("portfolio", "positions.csv").exists()


def ev(i: int, cls: EventClass, impact: int, n_sources: int, hours: float) -> Signal:
    return Signal(
        signal_id=f"s{i}",
        signal_type="event",
        as_of=T0 + timedelta(hours=hours),
        sentiment_score=-0.7,
        sentiment_label="negative",
        event_class=cls,
        event_confidence=0.8,
        impact_score=impact,
        impact_raw=0.8,
        confidence=0.7,
        n_docs=12,
        n_sources=n_sources,
        regions=["RUSSIA_UKRAINE"],
        evidence=[
            Evidence(
                doc_id="d",
                source="gdelt",
                title="Russia launches invasion of Ukraine",
                published_at=T0,
            )
        ],
    )


@pytest.mark.skipif(not HAVE_BOOK, reason="synthetic book not generated")
def test_replay_triggers_runs_stress_once_per_cooldown(tmp_path, monkeypatch) -> None:
    path = tmp_path / "signals.jsonl"
    JsonlSignalWriter(path).write(
        [
            ev(1, EventClass.GEOPOLITICAL, 7, 5, 0),  # below threshold
            ev(2, EventClass.GEOPOLITICAL, 9, 1, 1),  # single source
            ev(3, EventClass.GEOPOLITICAL, 9, 4, 2),  # fires
            ev(4, EventClass.GEOPOLITICAL, 10, 9, 3),  # escalation (10 > 9)
            ev(5, EventClass.EARNINGS, 9, 4, 4),  # fires but no market-wide scenario
            ev(6, EventClass.GEOPOLITICAL, 9, 9, 5),  # cooldown (9 < 10)
        ]
    )
    monkeypatch.setattr(runmod, "RUNS_PATH", tmp_path / "runs.jsonl")
    summ = runmod.replay_triggers(StorePollingSubscriber(path, signal_type="event"), write=False)
    assert summ["n_triggers_fired"] == 3 and summ["n_stress_runs"] == 2
    r = summ["runs"][0]
    assert r["scenario"].startswith("GEOPOLITICAL") and r["impact"] == 9
    assert r["cet1_after"] < 0.13
    assert summ["candidates_blocked_by"]["cooldown"] == 1
