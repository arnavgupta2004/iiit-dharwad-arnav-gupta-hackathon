"""Performance and signal-quality metrics for Module A (rf = 0; 252 trading days a year)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from riskpulse.moduleA.backtest import BacktestResult

ANN = 252


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def performance(res: BacktestResult) -> dict:
    r = res.returns
    vol = float(r.std(ddof=1) * np.sqrt(ANN))
    ann_ret = float((1 + r).prod() ** (ANN / len(r)) - 1)
    return {
        "cumulative_return": round(float((1 + r).prod() - 1), 4),
        "annualised_return": round(ann_ret, 4),
        "annualised_vol": round(vol, 4),
        "sharpe_rf0": round(float(r.mean() / r.std(ddof=1) * np.sqrt(ANN)), 3)
        if r.std() > 0
        else None,
        "max_drawdown": round(max_drawdown(r), 4),
        "avg_daily_one_way_turnover": round(float(res.turnover.mean()), 4),
        "total_cost_drag": round(float(res.costs.sum()), 5),
        "n_days": int(len(r)),
    }


def information_coefficient(s: pd.DataFrame, rets: pd.DataFrame, min_names: int = 5) -> dict:
    """Daily cross-sectional Spearman IC of s_t against next-day returns r_{t+1}.

    Only names with a non-zero signal on day t are used.
    """
    ics, dates = [], []
    for k in range(len(s.index) - 1):
        sig = s.iloc[k]
        nxt = rets.iloc[k + 1]
        mask = (sig != 0) & nxt.notna()
        if mask.sum() < min_names:
            continue
        rho, _ = spearmanr(sig[mask], nxt[mask])
        if np.isfinite(rho):
            ics.append(rho)
            dates.append(s.index[k])
    ic = pd.Series(ics, index=pd.DatetimeIndex(dates), dtype=float)
    if ic.empty:
        return {"n_days": 0, "series": ic}
    sd = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    t_stat = float(ic.mean() / (sd / np.sqrt(len(ic)))) if sd > 0 else None
    return {
        "n_days": int(len(ic)),
        "mean_ic": round(float(ic.mean()), 4),
        "ic_std": round(float(ic.std(ddof=1)), 4),
        "t_stat": round(t_stat, 2) if t_stat is not None else None,
        "hit_rate": round(float((ic > 0).mean()), 4),
        "series": ic,
    }
