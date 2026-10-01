"""Scenario library: event signal -> risk-factor shock vector (spec §7.3, DECISIONS D-013).

For an event class the scenario uses its analogue set (configs/scenarios.yaml). Per factor it
takes the move with the largest magnitude across the set (sign kept), so every factor shock traces
to one named historical episode. Shocks are scaled by m(impact) and concurrent scenarios are
combined per factor by max severity (no double counting).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from riskpulse.common.config import load_config
from riskpulse.moduleB.calibration import load_calibration

HAWKISH = re.compile(r"inflation|hike|hawkish|tighten|yields? (jump|surge|rise)|cpi", re.I)
GROWTH = re.compile(
    r"recession|contraction|slowdown|downturn|jobless|layoffs|cut rates|rate cut", re.I
)


@dataclass
class Scenario:
    name: str
    event_class: str
    severity: float
    shocks: dict[str, float]
    drivers: dict[str, str] = field(default_factory=dict)  # factor -> analogue that set it
    analogues: list[str] = field(default_factory=list)
    explanation: str = ""


def _calibration_table(cal: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
    cal = cal if cal is not None else load_calibration()
    out: dict[str, dict[str, float]] = {}
    for r in cal.itertuples(index=False):
        if pd.notna(r.value):
            out.setdefault(r.analogue, {})[r.factor] = float(r.value)
    return out


def sub_scenario(event_class: str, texts: list[str], sentiment: float) -> str:
    """MACRO splits into inflation/hawkish vs growth scare by evidence keywords and tone."""
    if event_class != "MACROECONOMIC":
        return "default"
    joined = " ".join(texts)
    hawk, growth = len(HAWKISH.findall(joined)), len(GROWTH.findall(joined))
    if hawk > growth:
        return "inflation_hawkish"
    if growth > hawk:
        return "growth_scare"
    return "default"


def severity_multiplier(impact: int) -> float:
    m = load_config("moduleB")["severity_multiplier"]
    if impact < min(m):
        return 0.0
    return float(m[min(max(impact, min(m)), max(m))])


def build_scenario(
    event_class: str,
    impact: int,
    texts: list[str] | None = None,
    sentiment: float = 0.0,
    cal: pd.DataFrame | None = None,
) -> Scenario | None:
    """Scenario for one event signal, or None if the class has no market-wide stress mapping."""
    rules = load_config("scenarios")["mapping"].get(event_class, {})
    sub = sub_scenario(event_class, texts or [], sentiment)
    key = "severe" if impact >= 10 and rules.get("severe") else sub if rules.get(sub) else "default"
    analogues = rules.get(key)
    if not analogues:
        return None
    table = _calibration_table(cal)
    m = severity_multiplier(impact)
    shocks, drivers = {}, {}
    for a in analogues:
        for f, v in table.get(a, {}).items():
            if f not in shocks or abs(v) > abs(shocks[f] / m if m else 0):
                shocks[f] = v * m
                drivers[f] = a
    expl = (
        f"{event_class} (impact {impact}, severity x{m:g}) mapped to {key} analogues "
        f"{', '.join(analogues)}; per factor the largest historical move is applied."
    )
    return Scenario(f"{event_class}:{key}", event_class, m, shocks, drivers, list(analogues), expl)


def combine(scenarios: list[Scenario]) -> Scenario | None:
    """Max-severity combination per factor across concurrent scenarios."""
    scenarios = [s for s in scenarios if s is not None]
    if not scenarios:
        return None
    if len(scenarios) == 1:
        return scenarios[0]
    shocks, drivers = {}, {}
    for s in scenarios:
        for f, v in s.shocks.items():
            if f not in shocks or abs(v) > abs(shocks[f]):
                shocks[f], drivers[f] = v, f"{s.name}/{s.drivers.get(f, '')}"
    return Scenario(
        "+".join(s.name for s in scenarios),
        "COMBINED",
        max(s.severity for s in scenarios),
        shocks,
        drivers,
        sorted({a for s in scenarios for a in s.analogues}),
        "Concurrent events combined per factor by maximum severity (no summation).",
    )


def custom_scenario(shocks: dict[str, float], name: str = "custom") -> Scenario:
    """What-if scenario from user-supplied factor shocks (dashboard sliders)."""
    return Scenario(
        name, "CUSTOM", 1.0, dict(shocks), dict.fromkeys(shocks, "user"), [], "User-defined shocks."
    )
