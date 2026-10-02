"""Out-of-sample check of the scenario library on the 2022 episodes (D-013; GATE A item 3).

For each validation episode in configs/scenarios.yaml:
- predicted = the shock vector of the highest-impact stress test that actually fired on the event
  date for the mapped class (from the trigger replay, so prediction uses only live information);
- realised = the same proxies measured from the close before the event date over the next
  ``horizon_trading_days`` sessions (same measurement code as calibration);
- per-factor sign agreement and errors, and the book's P&L under predicted vs realised shocks.
Nothing is tuned on these episodes: they are reported only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskpulse.common.config import load_config
from riskpulse.common.metrics import update_metrics
from riskpulse.ingestion.prices import trading_days
from riskpulse.moduleB.calibration import load_fred_extract, measure_window
from riskpulse.moduleB.run import load_runs
from riskpulse.moduleB.scenarios import custom_scenario
from riskpulse.moduleB.stress import load_positions, run_stress

EPISODE_CLASS = {"russia_ukraine_2022": "GEOPOLITICAL", "fomc_june_2022": "MACROECONOMIC"}
KEY_FACTORS = [
    "eq_SPY",
    "eq_Energy",
    "eq_Financials",
    "eq_Information Technology",
    "eq_EUROPE",
    "rates_10y_bp",
    "rates_3m_bp",
    "credit_bbb_bp",
    "oil",
    "fx_EUR",
    "fx_GBP",
    "fx_JPY",
    "usd_index",
    "vol_vix_pts",
]
SUPPLEMENTARY_LOOKBACK_DAYS = 7


def realised_window(event_date: str, horizon: int) -> tuple[str, str]:
    days = trading_days()
    d = pd.Timestamp(event_date)
    k = int(days.searchsorted(d, side="left"))  # first session on/after the event date
    return str(days[k - 1].date()), str(days[k - 1 + horizon].date())


def compare(fired: dict, day: str, horizon: int, positions: pd.DataFrame, fred) -> dict:
    """Predicted (the fired run's shocks) vs realised moves over the episode window."""
    pred = fired["scenario"]["shocks"]
    start, end = realised_window(day, horizon)
    real = {k: v for k, v in measure_window(start, end, fred).items() if v is not None}
    rows = []
    for f in KEY_FACTORS:
        if f in pred and f in real:
            p, r = float(pred[f]), float(real[f])
            rows.append(
                {
                    "factor": f,
                    "predicted": round(p, 6),
                    "realised": round(r, 6),
                    "error": round(p - r, 6),
                    "same_sign": bool(np.sign(p) == np.sign(r)),
                }
            )
    pv = run_stress(custom_scenario(pred, "predicted"), positions)
    rv = run_stress(custom_scenario(real, "realised"), positions)
    return {
        "trigger": {
            k: fired["trigger"][k] for k in ("as_of", "impact_score", "n_sources", "regions")
        },
        "headline": (fired["trigger"]["evidence"] or [{}])[0].get("title"),
        "scenario": fired["scenario"]["name"],
        "analogues": fired["scenario"]["analogues"],
        "realised_window": [start, end],
        "factors": rows,
        "sign_agreement": round(float(np.mean([r["same_sign"] for r in rows])), 3)
        if rows
        else None,
        "book_total_impact_predicted": pv["total_impact"],
        "book_total_impact_realised": rv["total_impact"],
        "capital_losses_predicted": pv["capital"]["capital_losses"],
        "capital_losses_realised": rv["capital"]["capital_losses"],
        "cet1_after_predicted": pv["capital"]["cet1_ratio_after"],
        "cet1_after_realised": rv["capital"]["cet1_ratio_after"],
    }


def episode(name: str, spec: dict, runs: list[dict], positions: pd.DataFrame, fred) -> dict:
    cls = EPISODE_CLASS[name]
    day = spec["event_date"]
    horizon = int(spec["horizon_trading_days"])
    of_class = [r for r in runs if r["scenario"]["event_class"] == cls]
    same_day = [r for r in of_class if r["trigger"]["as_of"][:10] == day]
    out: dict = {
        "event_date": day,
        "class": cls,
        "anchor": f"realised window anchored on the pre-registered event date {day}: close of the "
        f"previous session to +{horizon} sessions; the prediction uses the highest-impact run "
        "fired on that date",
    }
    # Reporting only: every run of this class in the 7 days before the event date (timing context).
    lo7 = str((pd.Timestamp(day) - pd.Timedelta(days=SUPPLEMENTARY_LOOKBACK_DAYS)).date())
    out["class_runs_in_prior_week"] = [
        {
            "as_of": r["trigger"]["as_of"],
            "impact_score": r["trigger"]["impact_score"],
            "n_sources": r["trigger"]["n_sources"],
            "scenario": r["scenario"]["name"],
            "headline": (r["trigger"]["evidence"] or [{}])[0].get("title"),
        }
        for r in sorted(of_class, key=lambda r: r["trigger"]["as_of"])
        if lo7 <= r["trigger"]["as_of"][:10] < day
    ]
    if same_day:
        fired = max(
            same_day, key=lambda r: (r["trigger"]["impact_score"], r["trigger"]["n_sources"])
        )
        return (
            out
            | {"status": "stress test fired on the event date"}
            | compare(fired, day, horizon, positions, fred)
        )
    # Missed episode: the pre-registered comparison does not exist. As supplementary context only,
    # compare the latest run of the same class fired in the preceding week (if any).
    lo = str((pd.Timestamp(day) - pd.Timedelta(days=SUPPLEMENTARY_LOOKBACK_DAYS)).date())
    prior = [r for r in of_class if lo <= r["trigger"]["as_of"][:10] < day]
    out["status"] = "missed: no stress test fired for this class on the event date"
    if prior:
        fired = max(prior, key=lambda r: r["trigger"]["as_of"])
        out["supplementary_latest_run_in_prior_week"] = compare(
            fired, day, horizon, positions, fred
        )
    return out


def run() -> dict:
    cfg = load_config("scenarios")
    runs, positions, fred = load_runs(), load_positions(), load_fred_extract()
    eps = {n: episode(n, s, runs, positions, fred) for n, s in cfg["validation_episodes"].items()}
    payload = {
        "method": "predicted = shocks of the highest-impact stress test fired on the event "
        "date for the mapped class; realised = same proxies, close before event -> +10 sessions. "
        "If no run fired on the date the episode is reported as missed; the latest same-class "
        "run of the preceding 7 days is shown as supplementary context only.",
        "episodes": eps,
        "note": "Out-of-sample: calibration used only episodes ending before 2021-09-30. "
        "Reported, not tuned.",
    }
    update_metrics("moduleB_validation", payload, script="riskpulse eval predicted_vs_realised")
    return payload
