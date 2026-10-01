"""Valuation formula and sign tests (spec §7.4: 'unit-test the signs')."""

import pandas as pd
import pytest

from riskpulse.moduleB.capital import capital_view
from riskpulse.moduleB.scenarios import Scenario, build_scenario, combine, severity_multiplier
from riskpulse.moduleB.valuation import revalue


def pos(**kw) -> dict:
    base = dict(
        asset_class="derivative",
        instrument="irs",
        sector="Rates",
        region="US",
        rating="A",
        notional=1e8,
        market_value=0.0,
        maturity_years=10.0,
    )
    base.update(kw)
    return base


def run(rows: list[dict], shocks: dict, regions=None) -> pd.DataFrame:
    return revalue(pd.DataFrame(rows), shocks, regions).positions


def test_irs_signs_pay_fixed_gains_when_rates_rise() -> None:
    df = run(
        [pos(direction="pay_fixed", dv01=10_000), pos(direction="receive_fixed", dv01=10_000)],
        {"rates_3m_bp": 100, "rates_10y_bp": 100},
    )
    assert df["pnl_mtm"].tolist() == pytest.approx([1_000_000, -1_000_000])


def test_bond_loses_when_yields_and_spreads_rise() -> None:
    b = pos(
        asset_class="bond",
        instrument="corporate_bond",
        rating="BBB",
        market_value=1e8,
        mod_duration=7.0,
        convexity=60.0,
        maturity_years=10.0,
    )
    df = run([b], {"rates_10y_bp": 100, "rates_3m_bp": 100, "credit_bbb_bp": 0})
    assert df["pnl_mtm"].iloc[0] == pytest.approx(1e8 * (-7.0 * 0.01 + 0.5 * 60 * 0.0001))
    widen = run([b], {"credit_bbb_bp": 100})["pnl_mtm"].iloc[0]
    assert widen < 0


def test_sovereign_ignores_credit_spread() -> None:
    s = pos(
        asset_class="bond",
        instrument="sovereign_bond",
        rating="AA",
        market_value=1e8,
        mod_duration=8.0,
        convexity=70.0,
    )
    assert run([s], {"credit_bbb_bp": 200})["pnl_mtm"].iloc[0] == 0


def test_cds_buyer_gains_seller_loses_on_widening() -> None:
    rows = [
        pos(instrument="cds", rating="BBB", direction="protection_bought", spread_dv01=5_000),
        pos(instrument="cds", rating="BBB", direction="protection_sold", spread_dv01=5_000),
    ]
    df = run(rows, {"credit_bbb_bp": 40})
    assert df["pnl_mtm"].tolist() == pytest.approx([200_000, -200_000])


def test_fx_trs_equity_and_option_signs() -> None:
    rows = [
        pos(instrument="fx_forward", currency="EUR", direction="long_foreign"),
        pos(instrument="fx_forward", currency="EUR", direction="short_foreign"),
        pos(instrument="equity_trs", sector="Energy", direction="receive_equity"),
        pos(asset_class="equity", instrument="sector_basket", sector="Energy", market_value=1e8),
        pos(instrument="equity_option", sector="Energy", delta=1000.0, gamma=10.0, vega=500.0),
    ]
    df = run(rows, {"fx_EUR": -0.05, "eq_Energy": -0.10, "eq_SPY": -0.05, "vol_vix_pts": 10})
    fx_long, fx_short, trs, eq, opt = df["pnl_mtm"].tolist()
    assert fx_long == pytest.approx(-5e6) and fx_short == pytest.approx(5e6)
    assert trs == pytest.approx(-1e7) and eq == pytest.approx(-1e7)
    ds = -10.0  # 100 * -10%
    assert opt == pytest.approx(1000 * ds + 0.5 * 10 * ds * ds + 500 * 0.10)


def test_loan_expected_loss_formula_and_regional_factor() -> None:
    loan = pos(
        asset_class="loan",
        instrument="term_loan",
        rating="BBB",
        ead=1e8,
        pd=0.0014,
        lgd=0.45,
        spread_duration=4.0,
        region="EUROPE",
    )
    # dS_BBB = +185 bp doubles the BBB spread (185 bp level) -> spread-implied multiplier 2
    df = run([loan], {"credit_bbb_bp": 185})
    assert df["pd_stressed"].iloc[0] == pytest.approx(0.0028)
    assert df["d_el"].iloc[0] == pytest.approx((0.0028 - 0.0014) * 0.45 * 1e8)
    assert df["pnl_mtm"].iloc[0] == 0  # banking-book loans: credit hits via dEL, not MtM
    hit = run([loan], {"credit_bbb_bp": 185}, regions=["EUROPE"])
    assert hit["pd_stressed"].iloc[0] == pytest.approx(0.0014 * 2 * 1.5)


def test_capital_view_ratio_falls_with_losses() -> None:
    rows = [
        pos(
            asset_class="loan",
            instrument="term_loan",
            rating="BBB",
            ead=1e9,
            pd=0.0014,
            lgd=0.45,
            spread_duration=4.0,
            market_value=1e9,
        ),
        pos(
            asset_class="equity",
            instrument="sector_basket",
            sector="Energy",
            market_value=1e8,
            rating=None,
        ),
    ]
    v = revalue(pd.DataFrame(rows), {"credit_bbb_bp": 185, "eq_Energy": -0.2}).positions
    c = capital_view(v)
    assert c["rwa_before"] == pytest.approx(1e9 * 0.75 + 1e8 * 2.5)
    assert c["cet1_ratio_after"] < c["cet1_ratio_before"]
    assert c["capital_losses"] == pytest.approx(2e7 + v["d_el"].sum())


def test_severity_scaling_and_max_combination() -> None:
    assert (
        severity_multiplier(7) == 0
        and severity_multiplier(9) == 0.8
        and severity_multiplier(12) == 1.0
    )
    a = Scenario("a", "X", 1, {"eq_SPY": -0.1, "oil": 0.05})
    b = Scenario("b", "Y", 1, {"eq_SPY": -0.2, "oil": 0.01})
    c = combine([a, b])
    assert c.shocks == {"eq_SPY": -0.2, "oil": 0.05}  # max severity per factor, no summation


def test_non_market_classes_have_no_stress_scenario() -> None:
    assert build_scenario("EARNINGS", 10) is None
