import math

import pytest

from riskpulse.moduleB.pricing import (
    annuity,
    black_scholes,
    bond_analytics,
    bond_price_change,
    cds_risky_duration,
    curve_rate,
    swap_dv01,
)


def test_par_bond_prices_at_100() -> None:
    b = bond_analytics(0.05, 0.05, 10)
    assert b.price == pytest.approx(100.0, abs=1e-8)


def test_duration_convexity_match_finite_differences() -> None:
    c, y, m = 0.04, 0.05, 7
    b = bond_analytics(c, y, m)
    h = 1e-4
    up, dn = bond_analytics(c, y + h, m).price, bond_analytics(c, y - h, m).price
    assert b.mod_duration == pytest.approx(-(up - dn) / (2 * h * b.price), rel=1e-4)
    assert b.convexity == pytest.approx((up + dn - 2 * b.price) / (h * h * b.price), rel=1e-3)


def test_duration_convexity_approximation_close_to_repricing() -> None:
    b = bond_analytics(0.03, 0.03, 10)
    dy = 0.02  # +200 bp
    exact = bond_analytics(0.03, 0.05, 10).price / b.price - 1
    assert bond_price_change(b.mod_duration, b.convexity, dy) == pytest.approx(exact, abs=2e-3)
    assert bond_price_change(b.mod_duration, b.convexity, dy) < 0


def test_zero_coupon_style_checks() -> None:
    assert annuity(0.0, 5) == pytest.approx(5.0)
    assert swap_dv01(1e8, 0.0, 5) == pytest.approx(50_000.0)


def test_cds_risky_duration_falls_with_spread() -> None:
    lo = cds_risky_duration(0.005, 0.6, 0.02, 5)
    hi = cds_risky_duration(0.05, 0.6, 0.02, 5)
    assert 0 < hi < lo < 5


def test_black_scholes_put_call_parity_and_greeks() -> None:
    s, k, t, r, v = 100, 95, 0.5, 0.02, 0.25
    c, p = black_scholes(s, k, t, r, v, True), black_scholes(s, k, t, r, v, False)
    assert c["price"] - p["price"] == pytest.approx(s - k * math.exp(-r * t), abs=1e-9)
    assert 0 < c["delta"] < 1 and -1 < p["delta"] < 0
    assert c["gamma"] == pytest.approx(p["gamma"]) and c["vega"] == pytest.approx(p["vega"])


def test_curve_interpolation() -> None:
    assert curve_rate(0.1, 0.01, 0.03) == 0.01 and curve_rate(20, 0.01, 0.03) == 0.03
    assert 0.01 < curve_rate(5, 0.01, 0.03) < 0.03
