"""Simplified CET1 view (spec §7.4 capital view; risk weights per Basel CRE20, DECISIONS D-024).

RWA (credit risk only):
  loans, corporate bonds  exposure * RW_corporate(rating)      CRE20.42-43 Table 10
  sovereign bonds         exposure * RW_sovereign(rating)      CRE20.7 Table 1
  equity                  exposure * 250%                      CRE20.57
  derivatives             max(MtM, 0) * RW_corporate(cp rating) (current exposure only; no add-on)
CET1 capital = start ratio * base RWA. Stress losses hitting CET1 (pre-tax, no offsets):
  -(MtM P&L of bonds, derivatives, equity) + dEL on loans (provisions).
Ratings are held fixed (no migration: a P2 stretch), so stressed RWA differs only through
derivative current exposure.
"""

from __future__ import annotations

import pandas as pd

from riskpulse.common.config import load_config


def _rw(row: pd.Series, mtm: float, cfg: dict) -> float:
    corp, sov = cfg["risk_weights"], cfg["sovereign_risk_weights"]
    cls, ins = row["asset_class"], row["instrument"]
    rating = row.get("rating") if isinstance(row.get("rating"), str) else "UNRATED"
    if cls == "loan":
        return row["ead"] * corp.get(rating, corp["UNRATED"])
    if ins == "sovereign_bond":
        return (row["market_value"] + mtm) * sov.get(rating, 1.0)
    if cls == "bond":
        return (row["market_value"] + mtm) * corp.get(rating, corp["UNRATED"])
    if cls == "equity":
        return (row["market_value"] + mtm) * cfg["equity_risk_weight"]
    current_exposure = max(row.get("market_value", 0.0) + mtm, 0.0)
    return current_exposure * corp.get(rating, corp["UNRATED"])


def capital_view(valued: pd.DataFrame) -> dict:
    cfg = load_config("moduleB")["capital"]
    rwa_base = sum(_rw(r, 0.0, cfg) for _, r in valued.iterrows())
    rwa_stress = sum(_rw(r, r["pnl_mtm"], cfg) for _, r in valued.iterrows())
    cet1 = cfg["cet1_ratio_start"] * rwa_base
    fv = valued[valued["asset_class"] != "loan"]["pnl_mtm"].sum()
    d_el = valued["d_el"].sum()
    losses = -fv + d_el
    return {
        "rwa_before": float(rwa_base),
        "rwa_after": float(rwa_stress),
        "cet1_capital_before": float(cet1),
        "cet1_capital_after": float(cet1 - losses),
        "cet1_ratio_before": float(cfg["cet1_ratio_start"]),
        "cet1_ratio_after": float((cet1 - losses) / rwa_stress) if rwa_stress else None,
        "capital_losses": float(losses),
        "losses_from_mtm": float(-fv),
        "losses_from_d_el": float(d_el),
    }
