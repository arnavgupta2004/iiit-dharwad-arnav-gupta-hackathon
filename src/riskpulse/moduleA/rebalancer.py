"""Module A: sentiment-tilted index weights (spec §6).

target:   w~_i = w0_i * exp(kappa * s_i * c_i)   (s_i := 0 when |s_i| < deadband)
bounds:   project onto {w : sum w = 1, w_min <= w_i <= w_max} (exact capped-simplex projection
          by iterative clipping and proportional redistribution)
turnover: one-way turnover T = 0.5 * sum |w~ - w_old|; move w_new = w_old + lam * (w~ - w_old)
          with lam = min(1, tau_max / T)
cost:     cost_bps * 1e-4 * realised turnover
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cap_weights(w: np.ndarray, w_min: float, w_max: float, max_iter: int = 100) -> np.ndarray:
    """Scale to sum 1 and enforce bounds, redistributing excess proportionally to free names."""
    n = len(w)
    if not (n * w_min <= 1 + 1e-12 and n * w_max >= 1 - 1e-12):
        raise ValueError(f"Infeasible bounds for n={n}: [{w_min}, {w_max}]")
    w = np.clip(np.asarray(w, dtype=float), 1e-12, None)
    w = w / w.sum()
    fixed = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        free = ~fixed
        budget = 1.0 - w[fixed].sum()
        w[free] = w[free] / w[free].sum() * budget
        lo, hi = w < w_min - 1e-12, w > w_max + 1e-12
        if not (lo | hi).any():
            break
        w[lo], w[hi] = w_min, w_max
        fixed |= lo | hi
    return w


def target_weights(
    w0: np.ndarray,
    s: np.ndarray,
    c: np.ndarray,
    kappa: float,
    deadband: float,
    w_min: float,
    w_max: float,
    impact: np.ndarray | None = None,
) -> np.ndarray:
    """Sentiment-tilted, bounded target weights. ``impact`` (1..10) optionally scales kappa."""
    s = np.where(np.abs(s) < deadband, 0.0, s)
    k = kappa * (impact / 5.0) if impact is not None else kappa
    return cap_weights(w0 * np.exp(k * s * c), w_min, w_max)


def one_way_turnover(w_old: np.ndarray, w_new: np.ndarray) -> float:
    return float(0.5 * np.abs(w_new - w_old).sum())


@dataclass(frozen=True)
class RebalanceResult:
    weights: np.ndarray
    turnover: float
    cost: float
    lam: float


def rebalance(
    w_old: np.ndarray, w_target: np.ndarray, tau_max: float, cost_bps: float
) -> RebalanceResult:
    """Partial move toward target under a turnover cap; returns new weights and cost."""
    t = one_way_turnover(w_old, w_target)
    lam = 1.0 if t <= tau_max or t == 0 else tau_max / t
    w_new = w_old + lam * (w_target - w_old)
    realised = one_way_turnover(w_old, w_new)
    return RebalanceResult(w_new, realised, cost_bps * 1e-4 * realised, lam)
