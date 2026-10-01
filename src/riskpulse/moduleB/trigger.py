"""Stress-test trigger (spec §7.1): event signal with impact >= threshold, event confidence >=
threshold and >= min distinct sources, with a cooldown per (event class, region) key."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from riskpulse.common.config import load_config
from riskpulse.common.schemas import Signal


@dataclass
class TriggerDecision:
    fired: bool
    reason: str
    key: str
    signal: Signal


@dataclass
class StressTrigger:
    cfg: dict = field(default_factory=lambda: load_config("moduleB")["trigger"])
    last_fired: dict[str, datetime] = field(default_factory=dict)
    log: list[TriggerDecision] = field(default_factory=list)

    @staticmethod
    def key_of(s: Signal) -> str:
        region = sorted(s.regions)[0] if s.regions else "GLOBAL"
        return f"{s.event_class.value}|{region}"

    def check(self, s: Signal) -> TriggerDecision:
        """Evaluate one signal; records every decision (fired or not) with its reason."""
        key = self.key_of(s)
        c = self.cfg
        if s.signal_type != "event":
            d = TriggerDecision(False, "not an event signal", key, s)
        elif s.impact_score < c["min_impact"]:
            d = TriggerDecision(False, f"impact {s.impact_score} < {c['min_impact']}", key, s)
        elif s.event_confidence < c["min_event_confidence"]:
            d = TriggerDecision(False, f"event confidence {s.event_confidence:.2f} too low", key, s)
        elif s.n_sources < c["min_sources"]:
            d = TriggerDecision(False, f"only {s.n_sources} source(s)", key, s)
        elif key in self.last_fired and s.as_of - self.last_fired[key] < timedelta(
            hours=c["cooldown_hours"]
        ):
            d = TriggerDecision(False, f"cooldown active for {key}", key, s)
        else:
            self.last_fired[key] = s.as_of
            d = TriggerDecision(True, "all trigger conditions met", key, s)
        if d.fired or s.impact_score >= c["min_impact"]:
            self.log.append(d)
        return d
