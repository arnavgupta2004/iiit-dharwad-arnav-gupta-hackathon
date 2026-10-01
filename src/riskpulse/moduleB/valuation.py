"""Revaluation of the synthetic book under a factor shock vector (spec §7.4).

Rates:   parallel-plus-slope shift interpolated by maturity between the 3m and 10y shocks.
Credit:  dS_r = dS_BBB * (spread_r / spread_BBB)  (proportional widening by rating; the ratios
         are the verified ICE OAS ratios in data/market/credit_spread_levels.csv).
PD:      PD_s = min(1, PD * m),  m = (1 + dS_BBB / S_BBB) * regional factor  (spread-implied).

| asset          | change in value                                                     |
|----------------|---------------------------------------------------------------------|
| bond           | MV * (-D_mod * dy + 0.5 * C * dy^2), dy = d_rate(T) + dS_r          |
| loan (FV info) | -SpreadDuration * dS_r * EAD;  credit loss dEL = (PD_s - PD) * LGD * EAD |
| IRS            | -DV01 * d_bp (receive fixed), +DV01 * d_bp (pay fixed)              |
| FX forward     | notional * d_fx * (+1 long foreign / -1 short foreign)             |
| option         | delta * dS + 0.5 * gamma * dS^2 + vega * d_sigma  (d_sigma = dVIX/100) |
| TRS            | notional * r_sector * (+1 receive equity / -1 pay equity)          |
| CDS            | +SpreadDV01 * dS_r (protection bought), -SpreadDV01 * dS_r (sold) |
| equity         | MV * r_sector                                                       |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskpulse.common.config import data_path, load_config
from riskpulse.moduleB.pricing import bond_price_change, curve_rate


@dataclass
class ValuationResult:
    positions: pd.DataFrame  # input positions + pnl, d_el, pd_stressed columns
    pd_multiplier: float
    credit_bbb_bp: float


def spread_ratios() -> dict[str, float]:
    sp = pd.read_csv(data_path("market", "credit_spread_levels.csv")).set_index("rating")
    return sp["oas_ratio_to_bbb_median"].to_dict()


def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
    v = row.get(col, default)
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def revalue(
    positions: pd.DataFrame,
    shocks: dict[str, float],
    event_regions: list[str] | None = None,
) -> ValuationResult:
    """Apply shocks to every position. Returns per-position P&L (USD) and credit-loss changes."""
    ratios = spread_ratios()
    bbb_level_bp = float(
        pd.read_csv(data_path("market", "credit_spread_levels.csv"))
        .set_index("rating")
        .loc["BBB", "spread_bp_2021_09_30"]
    )
    d_bbb = float(shocks.get("credit_bbb_bp", 0.0))
    m_base = max(0.0, 1.0 + d_bbb / bbb_level_bp)
    reg_factor = float(load_config("moduleB")["stress"]["regional_pd_factor"])
    hit_regions = set(event_regions or []) - {"US"}
    d3m, d10 = float(shocks.get("rates_3m_bp", 0.0)), float(shocks.get("rates_10y_bp", 0.0))
    eq_mkt = float(shocks.get("eq_SPY", 0.0))
    dvol = float(shocks.get("vol_vix_pts", 0.0)) / 100.0

    pnl, fv_info, d_el, pd_s = [], [], [], []
    for _, p in positions.iterrows():
        cls, ins = p["asset_class"], p["instrument"]
        mat = _num(p, "maturity_years", 1.0)
        d_rate_bp = curve_rate(mat, d3m, d10)
        rating = p.get("rating")
        d_s_bp = d_bbb * ratios.get(rating, 1.0) if isinstance(rating, str) else 0.0
        eq = float(shocks.get(f"eq_{p['sector']}", eq_mkt))
        v = fv = el = 0.0
        ps = _num(p, "pd")
        if cls == "bond":
            if ins == "sovereign_bond":
                d_s_bp = 0.0
            dy = (d_rate_bp + d_s_bp) / 1e4
            v = p["market_value"] * bond_price_change(p["mod_duration"], p["convexity"], dy)
        elif cls == "loan":
            fv = -p["spread_duration"] * d_s_bp / 1e4 * p["ead"]
            m = m_base * (reg_factor if p["region"] in hit_regions else 1.0)
            ps = min(1.0, p["pd"] * m)
            el = (ps - p["pd"]) * p["lgd"] * p["ead"]
        elif ins == "irs":
            sign = -1.0 if p["direction"] == "receive_fixed" else 1.0
            v = sign * p["dv01"] * d_rate_bp
        elif ins == "fx_forward":
            fx = float(shocks.get(f"fx_{p['currency']}", 0.0))
            v = p["notional"] * fx * (1.0 if p["direction"] == "long_foreign" else -1.0)
        elif ins == "equity_option":
            ds = 100.0 * eq
            v = p["delta"] * ds + 0.5 * p["gamma"] * ds * ds + p["vega"] * dvol
        elif ins == "equity_trs":
            v = p["notional"] * eq * (1.0 if p["direction"] == "receive_equity" else -1.0)
        elif ins == "cds":
            v = p["spread_dv01"] * d_s_bp * (1.0 if p["direction"] == "protection_bought" else -1.0)
        elif cls == "equity":
            v = p["market_value"] * eq
        pnl.append(v)
        fv_info.append(fv)
        d_el.append(el)
        pd_s.append(ps)
    out = positions.copy()
    out["pnl_mtm"] = pnl
    out["loan_fv_change_info"] = fv_info
    out["d_el"] = d_el
    out["pd_stressed"] = pd_s
    return ValuationResult(out, m_base, d_bbb)
