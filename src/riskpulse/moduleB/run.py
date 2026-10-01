"""Module B runner: subscribe to event signals, trigger stress tests, persist the runs."""

from __future__ import annotations

import json

from riskpulse.common.config import data_path
from riskpulse.common.metrics import update_metrics
from riskpulse.moduleB.scenarios import build_scenario
from riskpulse.moduleB.stress import load_positions, run_stress, stress_from_signal
from riskpulse.moduleB.trigger import StressTrigger
from riskpulse.subscribe.subscriber import SignalSubscriber, StorePollingSubscriber

RUNS_PATH = data_path("processed", "moduleB", "stress_runs.jsonl")


def replay_triggers(subscriber: SignalSubscriber | None = None, write: bool = True) -> dict:
    """Consume event signals in order; run a stress test whenever the trigger fires."""
    sub = subscriber or StorePollingSubscriber(signal_type="event")
    trig = StressTrigger()
    positions = load_positions()
    runs = []
    n_events = 0
    for s in sub:
        n_events += 1
        d = trig.check(s)
        if not d.fired:
            continue
        res = stress_from_signal(s, positions)
        if res is None:  # class without a market-wide scenario (e.g. EARNINGS)
            continue
        res["trigger_key"] = d.key
        runs.append(res)
    blocked: dict[str, int] = {}
    for d in trig.log:
        if not d.fired:
            reason = d.reason.split(" ")[0] if "cooldown" not in d.reason else "cooldown"
            blocked[reason] = blocked.get(reason, 0) + 1
    summary = {
        "n_event_signals_seen": n_events,
        "n_high_impact_candidates": len(trig.log),
        "n_triggers_fired": sum(d.fired for d in trig.log),
        "n_stress_runs": len(runs),
        "candidates_blocked_by": blocked,
        "runs": [
            {
                "as_of": r["trigger"]["as_of"],
                "event_class": r["scenario"]["event_class"],
                "scenario": r["scenario"]["name"],
                "impact": r["trigger"]["impact_score"],
                "n_sources": r["trigger"]["n_sources"],
                "regions": r["trigger"]["regions"],
                "headline": (r["trigger"]["evidence"] or [{}])[0].get("title"),
                "total_impact_usd": r["total_impact"],
                "total_impact_pct": r["total_impact_pct"],
                "cet1_after": r["capital"]["cet1_ratio_after"],
            }
            for r in runs
        ],
    }
    if write:
        RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RUNS_PATH.open("w", encoding="utf-8") as fh:
            for r in runs:
                fh.write(json.dumps(r, default=str) + "\n")
        update_metrics("moduleB_triggers", summary, script="riskpulse stress --replay")
    return summary


def run_named(event_class: str, impact: int, texts: list[str] | None = None, regions=None) -> dict:
    sc = build_scenario(event_class, impact, texts or [])
    if sc is None:
        raise ValueError(f"No market-wide stress scenario for {event_class}")
    return run_stress(sc, load_positions(), regions)


def load_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    with RUNS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
