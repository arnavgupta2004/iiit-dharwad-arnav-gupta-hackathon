"""Module A daily backtest (spec §6). No look-ahead by construction:

- Entity sentiment for day t is the EWMA evaluated at the 16:00 US/Eastern close of t using only
  mentions published at or before that instant.
- Weights decided at the close of t earn the close-to-close return from t to t+1.
- Between rebalances weights drift with returns.

Strategies: sentiment tilt (ours), equal-weight buy-and-hold, equal-weight rebalanced daily
(isolates the tilt from rebalancing), naive sign rule (+/- step per positive/negative headline).
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from riskpulse.moduleA.rebalancer import cap_weights, one_way_turnover, rebalance, target_weights

ET = ZoneInfo("America/New_York")


def close_times(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """16:00 US/Eastern of each session date, in UTC."""
    local = pd.DatetimeIndex([pd.Timestamp(d.date()).replace(hour=16) for d in days]).tz_localize(
        ET
    )
    return local.tz_convert("UTC")


def daily_sentiment(
    mentions: pd.DataFrame,
    days: pd.DatetimeIndex,
    tickers: list[str],
    half_life_hours: float,
    w0_conf: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """EWMA sentiment s, confidence c and same-day headline balance at each close.

    Returns (s, c, net_count) as day x ticker frames. ``net_count`` = (#positive - #negative)
    labelled mentions whose market day is t (used by the naive sign rule).
    """
    closes = close_times(days)
    s = pd.DataFrame(0.0, index=days, columns=tickers)
    c = pd.DataFrame(0.0, index=days, columns=tickers)
    net = pd.DataFrame(0.0, index=days, columns=tickers)
    lam = np.log(2) / (half_life_hours * 3600.0)
    m = mentions[mentions["ticker"].isin(tickers)].sort_values("published_at")
    for t, g in m.groupby("ticker"):
        # explicit ns unit: pandas 3 may store these as us/s, which breaks epoch maths
        ts = g["published_at"].dt.as_unit("ns").astype("int64").to_numpy() / 1e9
        w = (g["relevance"].clip(lower=1e-3) * g["event_confidence"].clip(lower=0.1)).to_numpy()
        x = g["sentiment"].to_numpy()
        lab = np.sign(np.where(np.abs(x) > 0.15, x, 0.0))
        num = den = 0.0
        last = None
        j = 0
        prev_close = -np.inf
        for i, close in enumerate(closes):
            ct = close.timestamp()
            day_net = 0.0
            while j < len(ts) and ts[j] <= ct:
                if last is not None:
                    f = np.exp(-lam * (ts[j] - last))
                    num, den = num * f, den * f
                num += w[j] * x[j]
                den += w[j]
                last = ts[j]
                if ts[j] > prev_close:
                    day_net += lab[j]
                j += 1
            if last is not None and den > 0:
                f = np.exp(-lam * (ct - last))
                s.iat[i, s.columns.get_loc(t)] = float(np.clip(num / den, -1, 1))
                c.iat[i, c.columns.get_loc(t)] = float(1 - np.exp(-den * f / w0_conf))
            net.iat[i, net.columns.get_loc(t)] = day_net
            prev_close = ct
    return s, c, net


@dataclass
class BacktestResult:
    returns: pd.Series  # net daily returns, indexed by the return date (t+1)
    weights: pd.DataFrame  # weights held over (t, t+1], indexed by decision date t
    turnover: pd.Series
    costs: pd.Series


def _run(
    rets: pd.DataFrame,
    decide,  # callable(t_index, w_drifted) -> (w_new, turnover, cost)
) -> BacktestResult:
    days = rets.index
    n = rets.shape[1]
    w = np.full(n, 1.0 / n)
    out_r, out_w, out_t, out_c = [], [], [], []
    for k in range(len(days) - 1):
        w, turn, cost = decide(k, w)
        r_next = rets.iloc[k + 1].to_numpy()
        gross = float(w @ r_next)
        out_r.append(gross - cost)
        out_w.append(w.copy())
        out_t.append(turn)
        out_c.append(cost)
        w = w * (1 + r_next) / (1 + gross)  # drift to the next close
    idx_dec, idx_ret = days[:-1], days[1:]
    return BacktestResult(
        pd.Series(out_r, index=idx_ret),
        pd.DataFrame(out_w, index=idx_dec, columns=rets.columns),
        pd.Series(out_t, index=idx_dec),
        pd.Series(out_c, index=idx_dec),
    )


def run_strategies(
    rets: pd.DataFrame,
    s: pd.DataFrame,
    c: pd.DataFrame,
    net: pd.DataFrame,
    cfg: dict,
    kappa: float | None = None,
) -> dict[str, BacktestResult]:
    """Run all strategies on aligned day x ticker frames (rets[t] = return from t-1 to t)."""
    n = rets.shape[1]
    w0 = np.full(n, 1.0 / n)
    tilt, cons, turn = cfg["tilt"], cfg["constraints"], cfg["turnover"]
    k = tilt["kappa"] if kappa is None else kappa
    bounds = (cons["w_min"], cons["w_max"])

    def sentiment_tilt(i: int, w: np.ndarray):
        tgt = target_weights(
            w0, s.iloc[i].to_numpy(), c.iloc[i].to_numpy(), k, tilt["deadband"], *bounds
        )
        r = rebalance(w, tgt, turn["tau_max"], turn["cost_bps"])
        return r.weights, r.turnover, r.cost

    def ew_rebalanced(i: int, w: np.ndarray):
        r = rebalance(w, w0, 1.0, turn["cost_bps"])
        return r.weights, r.turnover, r.cost

    def ew_buy_hold(i: int, w: np.ndarray):
        return w, 0.0, 0.0

    step = float(cfg["backtest"]["naive_sign_rule_step"])

    def naive_sign(i: int, w: np.ndarray):
        tgt = cap_weights(np.clip(w0 + step * net.iloc[i].to_numpy(), 1e-6, None), *bounds)
        cost = turn["cost_bps"] * 1e-4 * one_way_turnover(w, tgt)
        return tgt, one_way_turnover(w, tgt), cost

    return {
        "sentiment_tilt": _run(rets, sentiment_tilt),
        "equal_weight_buy_hold": _run(rets, ew_buy_hold),
        "equal_weight_rebalanced": _run(rets, ew_rebalanced),
        "naive_sign_rule": _run(rets, naive_sign),
    }
