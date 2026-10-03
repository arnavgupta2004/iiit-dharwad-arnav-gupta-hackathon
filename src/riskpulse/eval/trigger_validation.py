"""Trigger validation against market stress days (spec §7.5; pre-registered in D-062).

Stress day: SPY adjusted-close return <= -2.0% or a ^VIX close >= 15% above the previous close.
A trigger session s (16:00 ET rule) hits if a stress day falls on s, s+1 or s+2. Precision = hits /
trigger sessions; recall = stress days with a trigger session in {d-2, d-1, d} / stress days.
Baseline: the same number of distinct sessions drawn at random (1,000 draws). Evaluation only.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from riskpulse.common.metrics import update_metrics
from riskpulse.eval.impact_v2 import market_days
from riskpulse.ingestion.prices import wide
from riskpulse.moduleB.run import load_runs

WINDOW = ("2021-09-30", "2022-09-29")
SPY_DROP, VIX_JUMP, LAG = -0.02, 0.15, 2
N_DRAWS, SEED = 1000, 20261003


def stress_days(sessions: pd.DatetimeIndex) -> pd.Series:
    """Boolean per session: SPY return <= -2% or VIX close >= +15% vs the previous close."""
    spy = wide("adj_close", ["SPY"])["SPY"].pct_change()
    vix = wide("close", ["^VIX"])["^VIX"].pct_change()
    return ((spy <= SPY_DROP) | (vix >= VIX_JUMP)).reindex(sessions).fillna(False).astype(bool)


def scores(trigger_pos: set[int], stress_pos: set[int], n: int) -> tuple[float, float]:
    """(precision, recall) for trigger and stress-day positions in a session index of length n."""
    hits = sum(any(s + k in stress_pos for k in range(LAG + 1)) for s in trigger_pos)
    covered = sum(any(d - k in trigger_pos for k in range(LAG + 1)) for d in stress_pos)
    prec = hits / len(trigger_pos) if trigger_pos else float("nan")
    rec = covered / len(stress_pos) if stress_pos else float("nan")
    return prec, rec


def run() -> dict:
    spy_px = wide("adj_close", ["SPY"])["SPY"].dropna()
    sessions = pd.DatetimeIndex(spy_px.index)
    sessions = sessions[(sessions >= WINDOW[0]) & (sessions <= WINDOW[1])]
    stress = stress_days(sessions)
    stress_pos = set(np.flatnonzero(stress.to_numpy()))
    pos_of = {d: i for i, d in enumerate(sessions)}

    runs = load_runs()
    ts = pd.Series(pd.to_datetime([r["trigger"]["as_of"] for r in runs], utc=True))
    days = market_days(ts, pd.DatetimeIndex(spy_px.index))  # full history: no clipping at the edge
    by_class: dict[str, set[int]] = defaultdict(set)
    trig: set[int] = set()
    for r, d in zip(runs, days, strict=True):
        if d in pos_of:
            trig.add(pos_of[d])
            by_class[r["scenario"]["event_class"]].add(pos_of[d])

    prec, rec = scores(trig, stress_pos, len(sessions))
    rng = np.random.default_rng(SEED)
    draws = np.array(
        [
            scores(
                set(rng.choice(len(sessions), len(trig), replace=False)), stress_pos, len(sessions)
            )
            for _ in range(N_DRAWS)
        ]
    )
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]  # noqa: E731
    spy_ret = wide("adj_close", ["SPY"])["SPY"].pct_change().reindex(sessions)
    vix_chg = wide("close", ["^VIX"])["^VIX"].pct_change().reindex(sessions)
    payload = {
        "protocol": "pre-registered in DECISIONS.md D-062 before computing",
        "definition": "stress day: SPY adj-close return <= -2.0% or ^VIX close >= +15% vs previous "
        "close; hit: stress day on the trigger session or the next 2 sessions (16:00 ET rule)",
        "window": [str(sessions[0].date()), str(sessions[-1].date())],
        "n_sessions": int(len(sessions)),
        "n_stress_days": int(len(stress_pos)),
        "n_stress_days_spy": int((spy_ret <= SPY_DROP).sum()),
        "n_stress_days_vix": int((vix_chg >= VIX_JUMP).sum()),
        "n_stress_runs": int(len(runs)),
        "n_trigger_sessions": int(len(trig)),
        "share_of_sessions_with_trigger": round(len(trig) / len(sessions), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "random_baseline": {
            "draws": N_DRAWS,
            "precision_mean": round(float(draws[:, 0].mean()), 4),
            "precision_95": ci(draws[:, 0]),
            "recall_mean": round(float(draws[:, 1].mean()), 4),
            "recall_95": ci(draws[:, 1]),
            "share_of_draws_precision_ge_observed": round(float((draws[:, 0] >= prec).mean()), 4),
        },
        "by_class_reported_only": {
            c: dict(
                zip(
                    ("trigger_sessions", "precision", "recall"),
                    (len(p), *(round(v, 4) for v in scores(p, stress_pos, len(sessions)))),
                    strict=True,
                )
            )
            for c, p in sorted(by_class.items())
        },
        "stress_days": [str(sessions[i].date()) for i in sorted(stress_pos)],
        "note": "Evaluation only (D-062); trigger logic unchanged.",
    }
    update_metrics("trigger_validation", payload, script="riskpulse eval trigger_validation")
    return payload
