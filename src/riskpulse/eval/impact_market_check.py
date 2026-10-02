"""Descriptive check of market-wide impact (D-042; pre-specified, reported only, no tuning).

Market-wide stories keep impact v1 (v2 has no market-wide training target). For each trading
session after the burn-in, take the maximum impact of market-wide event signals (entity MKT)
assigned to that session by the 16:00 ET rule, and correlate it with |SPY return| and |dVIX|
(close to close) on the same session. Days without a market-wide event are not included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from riskpulse.common.config import load_config
from riskpulse.common.metrics import update_metrics
from riskpulse.engine.entities import MKT
from riskpulse.eval.impact_v2 import market_days
from riskpulse.ingestion.prices import trading_days, wide
from riskpulse.store.signal_store import read_signals_jsonl

N_BOOT, SEED = 2000, 20261002


def market_event_days(signals) -> pd.DataFrame:
    """Session -> max impact over market-wide event signals."""
    rows = [
        {"as_of": s.as_of, "impact": s.impact_score}
        for s in signals
        if s.signal_type == "event" and s.entity is not None and s.entity.ticker == MKT
    ]
    df = pd.DataFrame(rows)
    df["day"] = market_days(pd.to_datetime(df["as_of"], utc=True), trading_days())
    return df.groupby("day")["impact"].max().rename("max_impact").to_frame()


def spearman_ci(x: pd.Series, y: pd.Series) -> dict:
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(SEED)
    xv, yv, n = x.to_numpy(), y.to_numpy(), len(x)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        boots.append(spearmanr(xv[i], yv[i])[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {
        "spearman_rho": round(float(rho), 4),
        "p_value": float(f"{p:.3g}"),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
    }


def run() -> dict:
    burn_in_end = pd.Timestamp(load_config("impact")["binning"]["burn_in"][1])
    days = market_event_days(read_signals_jsonl())
    days = days[days.index >= burn_in_end]
    spy = wide("adj_close", ["SPY"])["SPY"].pct_change().abs().rename("abs_spy_ret")
    vix = wide("close", ["^VIX"])["^VIX"].diff().abs().rename("abs_dvix_pts")
    df = days.join(spy, how="inner").join(vix, how="inner").dropna()
    by_level = (
        df.groupby("max_impact")[["abs_spy_ret", "abs_dvix_pts"]].agg(["mean", "count"]).round(4)
    )
    payload = {
        "definition": "per session after the burn-in: max impact of market-wide (MKT) event "
        "signals (16:00 ET rule) vs |SPY close-to-close return| and |dVIX| (points), same session",
        "n_days": int(len(df)),
        "window": [str(df.index.min().date()), str(df.index.max().date())],
        "vs_abs_spy_return": spearman_ci(df["max_impact"], df["abs_spy_ret"]),
        "vs_abs_dvix": spearman_ci(df["max_impact"], df["abs_dvix_pts"]),
        "by_max_impact": {
            int(k): {
                "n_days": int(r[("abs_spy_ret", "count")]),
                "mean_abs_spy_ret": float(r[("abs_spy_ret", "mean")]),
                "mean_abs_dvix_pts": float(r[("abs_dvix_pts", "mean")]),
            }
            for k, r in by_level.iterrows()
        },
        "note": "Descriptive only (D-042); nothing is tuned on it.",
    }
    update_metrics("impact_market_check", payload, script="riskpulse eval impact_market_check")
    return payload
