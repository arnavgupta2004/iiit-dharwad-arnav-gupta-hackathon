"""Stress-test trigger (spec §7.1, GATE C D-037).

Fires for an event signal with impact >= threshold, event confidence >= threshold and >= min
distinct sources. Cooldown is per (event class, macro-region): within it, a signal re-runs the
stress test only if its impact exceeds the impact that last fired for that key (escalation), and
it then becomes the new reference.
"""

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
    last_fired: dict[str, tuple[datetime, int]] = field(default_factory=dict)
    log: list[TriggerDecision] = field(default_factory=list)

    def key_of(self, s: Signal) -> str:
        """(event class, macro-region); several macro-regions or EM only -> Global."""
        mapping = self.cfg.get("macro_regions", {})
        macros = {mapping.get(r, "Global") for r in s.regions}
        macro = macros.pop() if len(macros) == 1 else "Global"
        return f"{s.event_class.value}|{macro}"

    def check(self, s: Signal) -> TriggerDecision:
        """Evaluate one signal; logs every high-impact decision (fired or not) with its reason."""
        key = self.key_of(s)
        c = self.cfg
        prev = self.last_fired.get(key)
        in_cooldown = prev is not None and s.as_of - prev[0] < timedelta(hours=c["cooldown_hours"])
        if s.signal_type != "event":
            d = TriggerDecision(False, "not an event signal", key, s)
        elif s.impact_score < c["min_impact"]:
            d = TriggerDecision(False, f"impact {s.impact_score} < {c['min_impact']}", key, s)
        elif s.event_confidence < c["min_event_confidence"]:
            d = TriggerDecision(False, f"event confidence {s.event_confidence:.2f} too low", key, s)
        elif s.n_sources < c["min_sources"]:
            d = TriggerDecision(False, f"only {s.n_sources} source(s)", key, s)
        elif in_cooldown and not (c.get("escalation", False) and s.impact_score > prev[1]):
            d = TriggerDecision(False, f"cooldown active for {key}", key, s)
        else:
            reason = (
                f"escalation: impact {s.impact_score} > {prev[1]} within cooldown"
                if in_cooldown
                else "all trigger conditions met"
            )
            self.last_fired[key] = (s.as_of, s.impact_score)
            d = TriggerDecision(True, reason, key, s)
        if d.fired or s.impact_score >= c["min_impact"]:
            self.log.append(d)
        return d
