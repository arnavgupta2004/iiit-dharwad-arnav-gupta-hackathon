"""Run a stress test: scenario -> revaluation -> breakdowns, dEL and CET1 (spec §7.4 outputs)."""

from __future__ import annotations

import pandas as pd

from riskpulse.common.config import data_path, load_config
from riskpulse.common.schemas import Signal
from riskpulse.moduleB.capital import capital_view
from riskpulse.moduleB.scenarios import Scenario, build_scenario
from riskpulse.moduleB.valuation import revalue


def load_positions() -> pd.DataFrame:
    return pd.read_csv(data_path("portfolio", "positions.csv"))


def _by(df: pd.DataFrame, col: str) -> dict[str, float]:
    g = df.groupby(df[col].fillna("n/a"))[["pnl_mtm", "d_el"]].sum()
    return {k: round(float(v.pnl_mtm - v.d_el), 2) for k, v in g.iterrows()}


def run_stress(
    scenario: Scenario,
    positions: pd.DataFrame | None = None,
    event_regions: list[str] | None = None,
    trigger: Signal | None = None,
) -> dict:
    """Full stress result as a JSON-friendly dict. Amounts in USD."""
    pos = positions if positions is not None else load_positions()
    val = revalue(pos, scenario.shocks, event_regions)
    v = val.positions
    before = float(v["market_value"].sum())
    mtm = float(v["pnl_mtm"].sum())
    loan_fv = float(v["loan_fv_change_info"].sum())
    d_el = float(v["d_el"].sum())
    total_impact = mtm - d_el
    v["impact"] = v["pnl_mtm"] - v["d_el"]
    n = int(load_config("moduleB")["stress"]["top_n_positions"])
    worst = v.nsmallest(n, "impact")[
        [
            "position_id",
            "asset_class",
            "instrument",
            "cp_id",
            "sector",
            "region",
            "rating",
            "notional",
            "impact",
        ]
    ]
    heat = (
        v.pivot_table(index="sector", columns="asset_class", values="impact", aggfunc="sum")
        .fillna(0.0)
        .round(2)
    )
    return {
        "scenario": {
            "name": scenario.name,
            "event_class": scenario.event_class,
            "severity": scenario.severity,
            "analogues": scenario.analogues,
            "shocks": {k: round(x, 6) for k, x in scenario.shocks.items()},
            "drivers": scenario.drivers,
            "explanation": scenario.explanation,
            "pd_multiplier": round(val.pd_multiplier, 4),
            "event_regions": event_regions or [],
        },
        "trigger": None if trigger is None else trigger.model_dump(mode="json"),
        "value_before": round(before, 2),
        "value_after": round(before + mtm, 2),
        "pnl_mtm": round(mtm, 2),
        "pnl_mtm_pct": round(mtm / before, 6) if before else None,
        "d_el": round(d_el, 2),
        "total_impact": round(total_impact, 2),
        "total_impact_pct": round(total_impact / before, 6) if before else None,
        "loan_fair_value_change_info": round(loan_fv, 2),
        "by_asset_class": _by(v, "asset_class"),
        "by_sector": _by(v, "sector"),
        "by_rating": _by(v, "rating"),
        "by_region": _by(v, "region"),
        "heatmap_sector_x_asset": heat.to_dict(),
        "top_worst_positions": worst.round(2).to_dict(orient="records"),
        "capital": {
            k: (round(x, 6) if isinstance(x, float) else x) for k, x in capital_view(v).items()
        },
    }


def stress_from_signal(signal: Signal, positions: pd.DataFrame | None = None) -> dict | None:
    """Build the scenario implied by an event signal and run it (None if no stress mapping)."""
    texts = [e.title for e in signal.evidence if e.title]
    sc = build_scenario(
        signal.event_class.value, signal.impact_score, texts, signal.sentiment_score
    )
    if sc is None:
        return None
    return run_stress(sc, positions, signal.regions, signal)
