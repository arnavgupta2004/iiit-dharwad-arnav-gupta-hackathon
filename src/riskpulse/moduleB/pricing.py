"""Closed-form pricing analytics used to build and revalue the synthetic book.

All rates are decimals (0.025 = 2.5%), times in years. Bonds pay ``freq`` coupons a year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BondAnalytics:
    price: float  # per 100 face
    mod_duration: float  # years
    convexity: float  # years^2


def bond_analytics(coupon: float, ytm: float, maturity: float, freq: int = 2) -> BondAnalytics:
    """Price, modified duration and convexity of a fixed-coupon bullet bond (per 100 face)."""
    n = max(1, int(round(maturity * freq)))
    t = np.arange(1, n + 1) / freq
    cf = np.full(n, 100 * coupon / freq)
    cf[-1] += 100
    y = ytm / freq
    df = (1 + y) ** (-np.arange(1, n + 1))
    price = float((cf * df).sum())
    mac = float((t * cf * df).sum() / price)
    mod = mac / (1 + y)
    conv = float((cf * df * t * (t + 1 / freq)).sum() / (price * (1 + y) ** 2))
    return BondAnalytics(price, mod, conv)


def bond_price_change(mod_duration: float, convexity: float, dy: float) -> float:
    """dP/P ~= -D_mod * dy + 0.5 * C * dy^2 (spec §7.4)."""
    return -mod_duration * dy + 0.5 * convexity * dy * dy


def annuity(rate: float, maturity: float, freq: int = 2) -> float:
    """PV of 1 per year paid ``freq`` times a year (the swap fixed-leg annuity)."""
    n = max(1, int(round(maturity * freq)))
    return float(sum((1 + rate / freq) ** (-k) for k in range(1, n + 1)) / freq)


def swap_dv01(notional: float, rate: float, maturity: float) -> float:
    """USD value change of the fixed leg per 1 bp (positive number)."""
    return notional * annuity(rate, maturity) * 1e-4


def cds_risky_duration(
    spread: float, lgd: float, rate: float, maturity: float, freq: int = 4
) -> float:
    """Risky annuity with a flat hazard rate h = spread / LGD (credit triangle)."""
    h = spread / max(lgd, 1e-6)
    n = max(1, int(round(maturity * freq)))
    t = np.arange(1, n + 1) / freq
    return float((np.exp(-(rate + h) * t)).sum() / freq)


def black_scholes(
    s: float, k: float, t: float, r: float, sigma: float, call: bool
) -> dict[str, float]:
    """Price and greeks per unit of underlying (vega per 1.00 of vol, i.e. 100 vol points)."""
    t = max(t, 1e-6)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if call:
        price = s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (s * sigma * math.sqrt(t))
    vega = s * norm.pdf(d1) * math.sqrt(t)
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega}


def curve_rate(maturity: float, r3m: float, r10y: float) -> float:
    """Treasury yield by linear interpolation between 3m and 10y; flat beyond 10y (assumption)."""
    if maturity <= 0.25:
        return r3m
    if maturity >= 10:
        return r10y
    return r3m + (r10y - r3m) * (maturity - 0.25) / (10 - 0.25)
