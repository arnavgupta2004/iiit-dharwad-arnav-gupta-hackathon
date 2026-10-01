"""Synthetic wholesale-banking book seeded from transaction data (spec §7.2).

Every record is synthetic and labelled ``source = synthetic_demo``; counterparties are named
CP_0001.. and carry no real names. Seeding (DECISIONS/ASSUMPTIONS):
- obligors = merchants with >= ``min_merchant_months`` of flows, chosen per MCC sector
  (allocation proportional to sqrt of sector volume, minimum per sector) with a non-US share;
- sector from MCC (configs/mcc_sector_map.yaml); region from merchant geography;
- size weight proportional to log(total transaction volume);
- rating by rank of monthly-flow volatility (coefficient of variation) into the target rating mix;
- PD = S&P long-run one-year default rate of the rating; spread = rating spread level at as_of.
Instrument analytics (duration, convexity, DV01, risky duration, greeks, beta) are computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskpulse.common.config import data_path, load_config
from riskpulse.ingestion.prices import wide
from riskpulse.moduleB.pricing import (
    black_scholes,
    bond_analytics,
    cds_risky_duration,
    curve_rate,
    swap_dv01,
)

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
SOURCE = "synthetic_demo"


def market_snapshot(as_of: str, lookback: int) -> dict:
    """Rates, VIX, sector ETF levels, realised vols and betas known at the close of ``as_of``."""
    px = wide("adj_close")
    close = wide("close")
    d = pd.Timestamp(as_of)
    px, close = px[px.index <= d], close[close.index <= d]
    rets = px.pct_change().iloc[-lookback:]
    sectors = load_config("mcc_sector_map")["sector_etf"]
    etf_vol, etf_beta = {}, {}
    for sec, etf in sectors.items():
        r = rets[etf].dropna()
        etf_vol[sec] = float(r.std() * np.sqrt(252))
        cov = np.cov(r, rets["SPY"].loc[r.index])
        etf_beta[sec] = float(cov[0, 1] / cov[1, 1])
    return {
        "as_of": d,
        "r3m": float(close["^IRX"].dropna().iloc[-1]) / 100,
        "r10y": float(close["^TNX"].dropna().iloc[-1]) / 100,
        "vix": float(close["^VIX"].dropna().iloc[-1]),
        "etf_vol": etf_vol,
        "etf_beta": etf_beta,
    }


def rating_tables() -> tuple[dict[str, float], dict[str, float]]:
    """(spread in decimals, one-year PD in decimals) by rating, from data/market."""
    sp = pd.read_csv(data_path("market", "credit_spread_levels.csv")).set_index("rating")
    pdf = pd.read_csv(data_path("market", "default_rates_by_rating.csv")).set_index("rating")
    spreads = (sp["spread_bp_2021_09_30"] / 1e4).to_dict()
    pds = (pdf["one_year_default_rate_pct"] / 100).to_dict()
    return spreads, pds


def region_of(geo: str | None, region_map: dict[str, str]) -> str:
    if geo is None or (isinstance(geo, float) and np.isnan(geo)):
        return "US"
    if len(geo) == 2 and geo.isupper():
        return "US"
    return region_map.get(geo, "EM")


def select_obligors(agg: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    mcc_map = {int(k): v for k, v in load_config("mcc_sector_map")["mcc_to_sector"].items()}
    a = agg[(agg["n_months"] >= cfg["min_merchant_months"]) & agg["flow_cv"].notna()].copy()
    a["sector"] = a["mcc"].astype(int).map(mcc_map)
    a = a.dropna(subset=["sector"])
    a["region"] = [region_of(g, cfg["region_map"]) for g in a["merchant_state"]]
    n_total = int(cfg["n_obligors"])
    n_non_us = int(round(n_total * cfg["non_us_obligor_share"]))
    non_us = a[a["region"] != "US"].sort_values("total_volume", ascending=False).head(n_non_us)
    us = a[a["region"] == "US"]
    weight = np.sqrt(us.groupby("sector")["total_volume"].sum())
    alloc = (weight / weight.sum() * (n_total - len(non_us))).round().astype(int)
    alloc = alloc.clip(lower=int(cfg["min_obligors_per_sector"]))
    picks = [
        us[us["sector"] == sec].sort_values("total_volume", ascending=False).head(k)
        for sec, k in alloc.items()
    ]
    ob = pd.concat([*picks, non_us]).drop_duplicates("merchant_id")
    ob = ob.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    # rating: rank by flow volatility into the configured mix (lower volatility -> better rating)
    mix = cfg["rating_mix"]
    order = ob["flow_cv"].rank(method="first").to_numpy() - 1
    cuts = np.cumsum([mix[r] for r in RATINGS]) * len(ob)
    ob["rating"] = [RATINGS[int(np.searchsorted(cuts, i, side="right"))] for i in order]
    ob["size_weight"] = np.log(ob["total_volume"].to_numpy())
    ob["cp_id"] = [f"{cfg['counterparty_prefix']}{i:04d}" for i in range(1, len(ob) + 1)]
    return ob


def generate_portfolio(seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (counterparties, positions) and write them to data/portfolio/."""
    cfg = load_config("moduleB")["portfolio"]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    agg = pd.read_parquet(data_path("portfolio", "merchant_aggregates.parquet"))
    mkt = market_snapshot(cfg["as_of"], int(cfg["equity_vol_lookback_days"]))
    spreads, pds = rating_tables()
    ob = select_obligors(agg, cfg, rng)
    total = float(cfg["total_exposure_usd"])
    mix = cfg["mix"]
    lgd = cfg["lgd"]
    w = ob["size_weight"] / ob["size_weight"].sum()
    rows: list[dict] = []

    def add(**kw) -> None:
        rows.append({"source": SOURCE, **kw})

    # --- loans (45%) ---
    loan_budget = total * mix["loan"]
    for (_, o), wi in zip(ob.iterrows(), w, strict=True):
        n_loans = 2 if rng.random() < cfg["share_revolvers"] else 1
        for j in range(n_loans):
            ead = loan_budget * wi / n_loans
            mat = float(rng.uniform(*cfg["loan_tenor_years"]))
            secured = rng.random() < cfg["share_secured_loans"]
            s = spreads[o["rating"]]
            r = curve_rate(mat, mkt["r3m"], mkt["r10y"])
            add(
                position_id=f"LN_{o['cp_id']}_{j + 1}",
                asset_class="loan",
                instrument="revolver" if j == 1 else "term_loan",
                cp_id=o["cp_id"],
                sector=o["sector"],
                region=o["region"],
                rating=o["rating"],
                notional=ead,
                ead=ead,
                market_value=ead,
                maturity_years=round(mat, 2),
                rate_type="floating" if rng.random() < 0.7 else "fixed",
                spread_bp=round(s * 1e4, 1),
                spread_duration=round(
                    cds_risky_duration(s, lgd["loan_senior_unsecured"], r, mat), 4
                ),
                pd=pds[o["rating"]],
                lgd=lgd["loan_secured"] if secured else lgd["loan_senior_unsecured"],
            )
    # --- bonds (30%): corporate for ~60% of obligors + US Treasuries ---
    bond_budget = total * mix["bond"]
    ust_share = 0.25
    corp = ob.sample(frac=0.6, random_state=int(rng.integers(1 << 31)))
    wc = corp["size_weight"] / corp["size_weight"].sum()
    for (_, o), wi in zip(corp.iterrows(), wc, strict=True):
        mat = float(rng.uniform(*cfg["bond_tenor_years"]))
        ytm = curve_rate(mat, mkt["r3m"], mkt["r10y"]) + spreads[o["rating"]]
        coupon = round(ytm * 8 * 100) / 800  # issued near par, coupon on a 1/8% grid
        b = bond_analytics(coupon, ytm, mat)
        face = bond_budget * (1 - ust_share) * wi
        add(
            position_id=f"BD_{o['cp_id']}",
            asset_class="bond",
            instrument="corporate_bond",
            cp_id=o["cp_id"],
            sector=o["sector"],
            region=o["region"],
            rating=o["rating"],
            notional=face,
            ead=face * b.price / 100,
            market_value=face * b.price / 100,
            maturity_years=round(mat, 2),
            coupon=coupon,
            ytm=round(ytm, 6),
            mod_duration=round(b.mod_duration, 4),
            convexity=round(b.convexity, 4),
            spread_bp=round(spreads[o["rating"]] * 1e4, 1),
            pd=pds[o["rating"]],
            lgd=lgd["bond_senior_unsecured"],
        )
    for k in range(int(cfg["n_ust_bonds"])):
        mat = [2, 3, 5, 7, 10, 20][k % 6]
        ytm = curve_rate(mat, mkt["r3m"], mkt["r10y"])
        coupon = round(ytm * 8 * 100) / 800
        b = bond_analytics(coupon, ytm, mat)
        face = bond_budget * ust_share / int(cfg["n_ust_bonds"])
        add(
            position_id=f"UST_{mat}Y",
            asset_class="bond",
            instrument="sovereign_bond",
            cp_id="SOV_US",
            sector="Sovereign",
            region="US",
            rating="AA",
            notional=face,
            ead=face * b.price / 100,
            market_value=face * b.price / 100,
            maturity_years=mat,
            coupon=coupon,
            ytm=round(ytm, 6),
            mod_duration=round(b.mod_duration, 4),
            convexity=round(b.convexity, 4),
            spread_bp=0.0,
            pd=0.0,
            lgd=0.0,
        )
    # --- derivatives (15% notional-equivalent) ---
    der_budget = total * mix["derivative"]
    shares = {"irs": 0.4, "fx": 0.2, "cds": 0.2, "opt": 0.1, "trs": 0.1}
    n_irs = int(cfg["n_irs"])
    for k in range(n_irs):
        mat = float(rng.choice([2, 3, 5, 7, 10]))
        rate = curve_rate(mat, mkt["r3m"], mkt["r10y"])
        notional = der_budget * shares["irs"] / n_irs
        pay_fixed = rng.random() < 0.5
        add(
            position_id=f"IRS_{k + 1:02d}",
            asset_class="derivative",
            instrument="irs",
            cp_id=ob["cp_id"].iloc[k % len(ob)],
            sector="Rates",
            region="US",
            rating="A",
            notional=notional,
            market_value=0.0,
            maturity_years=mat,
            fixed_rate=round(rate, 6),
            direction="pay_fixed" if pay_fixed else "receive_fixed",
            dv01=round(swap_dv01(notional, rate, mat), 2),
        )
    n_fx = int(cfg["n_fx_forwards"])
    for k in range(n_fx):
        ccy = ["EUR", "GBP", "JPY"][k % 3]
        add(
            position_id=f"FXF_{ccy}_{k + 1:02d}",
            asset_class="derivative",
            instrument="fx_forward",
            cp_id=ob["cp_id"].iloc[(k * 7) % len(ob)],
            sector="FX",
            region="US",
            rating="A",
            notional=der_budget * shares["fx"] / n_fx,
            market_value=0.0,
            currency=ccy,
            direction="long_foreign" if rng.random() < 0.5 else "short_foreign",
            maturity_years=float(rng.choice([0.25, 0.5, 1.0])),
        )
    n_cds = int(cfg["n_cds"])
    refs = ob.sample(n=min(n_cds, len(ob)), random_state=int(rng.integers(1 << 31)))
    for _, o in refs.iterrows():
        mat = 5.0
        s = spreads[o["rating"]]
        rd = cds_risky_duration(
            s, lgd["bond_senior_unsecured"], curve_rate(mat, mkt["r3m"], mkt["r10y"]), mat
        )
        notional = der_budget * shares["cds"] / n_cds
        add(
            position_id=f"CDS_{o['cp_id']}",
            asset_class="derivative",
            instrument="cds",
            cp_id=o["cp_id"],
            sector=o["sector"],
            region=o["region"],
            rating=o["rating"],
            notional=notional,
            market_value=0.0,
            maturity_years=mat,
            spread_bp=round(s * 1e4, 1),
            direction="protection_bought" if rng.random() < 0.7 else "protection_sold",
            spread_dv01=round(notional * rd * 1e-4, 2),
            pd=pds[o["rating"]],
        )
    sectors = list(load_config("mcc_sector_map")["sector_etf"])
    n_opt = int(cfg["n_equity_options"])
    for k in range(n_opt):
        sec = sectors[k % len(sectors)]
        call = rng.random() < 0.5
        t = float(rng.choice([0.25, 0.5, 1.0]))
        moneyness = float(rng.choice([0.9, 1.0, 1.1]))
        g = black_scholes(100.0, 100.0 * moneyness, t, mkt["r3m"], mkt["etf_vol"][sec], call)
        notional = der_budget * shares["opt"] / n_opt  # underlying notional (index units of 100)
        units = notional / 100.0 * (1 if rng.random() < 0.6 else -1)
        add(
            position_id=f"OPT_{sec[:4].upper()}_{k + 1:02d}",
            asset_class="derivative",
            instrument="equity_option",
            cp_id=ob["cp_id"].iloc[(k * 11) % len(ob)],
            sector=sec,
            region="US",
            rating="A",
            notional=notional,
            market_value=round(units * g["price"], 2),
            option_type="call" if call else "put",
            strike_pct=moneyness,
            maturity_years=t,
            implied_vol=round(mkt["etf_vol"][sec], 4),
            units=units,
            delta=round(units * g["delta"], 4),
            gamma=round(units * g["gamma"], 6),
            vega=round(units * g["vega"], 4),
        )
    n_trs = int(cfg["n_equity_trs"])
    for k in range(n_trs):
        sec = sectors[(k * 3) % len(sectors)]
        add(
            position_id=f"TRS_{sec[:4].upper()}_{k + 1:02d}",
            asset_class="derivative",
            instrument="equity_trs",
            cp_id=ob["cp_id"].iloc[(k * 13) % len(ob)],
            sector=sec,
            region="US",
            rating="A",
            notional=der_budget * shares["trs"] / n_trs,
            market_value=0.0,
            direction="receive_equity" if rng.random() < 0.6 else "pay_equity",
            maturity_years=1.0,
        )
    # --- equity book (10%) ---
    eq_budget = total * mix["equity"]
    for sec in sectors:
        add(
            position_id=f"EQ_{sec[:6].upper().replace(' ', '')}",
            asset_class="equity",
            instrument="sector_basket",
            cp_id=None,
            sector=sec,
            region="US",
            rating=None,
            notional=eq_budget / len(sectors),
            market_value=eq_budget / len(sectors),
            beta=round(mkt["etf_beta"][sec], 3),
        )
    positions = pd.DataFrame(rows)
    cps = ob[
        [
            "cp_id",
            "sector",
            "region",
            "merchant_state",
            "mcc",
            "rating",
            "flow_cv",
            "total_volume",
            "n_months",
        ]
    ].copy()
    cps["pd"] = cps["rating"].map(pds)
    cps["spread_bp"] = cps["rating"].map(spreads) * 1e4
    cps = cps.rename(
        columns={
            "merchant_state": "seed_geo",
            "mcc": "seed_mcc",
            "flow_cv": "seed_flow_cv",
            "total_volume": "seed_tx_volume",
            "n_months": "seed_months",
        }
    )
    cps["source"] = SOURCE
    out = data_path("portfolio")
    out.mkdir(parents=True, exist_ok=True)
    positions.to_csv(out / "positions.csv", index=False)
    cps.to_csv(out / "counterparties.csv", index=False)
    return cps, positions
