"""Measure risk-factor moves over historical analogue windows (spec §7.3, DECISIONS D-013).

Factors (unit):
  eq_SPY, eq_<sector>, eq_EUROPE, eq_CHINA   equity return (fraction)
  rates_10y_bp, rates_3m_bp                  yield change (bp)
  credit_bbb_bp, credit_aaa_bp               Moody's Baa/Aaa minus 10y Treasury change (bp)
  oil, usd_index, fx_EUR, fx_GBP, fx_JPY     return of asset / foreign currency vs USD (fraction)
  vol_vix_pts                                VIX change (index points)
Only analogues ending on or before ``calibration_cutoff`` are measured for the shock library;
``measure_window`` is reused for the out-of-sample realised moves.
"""

from __future__ import annotations

import pandas as pd

from riskpulse.common.config import data_path, load_config
from riskpulse.ingestion.prices import wide

FRED_EXTRACT = data_path("raw", "_downloads", "fred", "moodys_treasury_windows.csv")


def _ret(series: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float | None:
    s = series.dropna()
    s0, s1 = s[s.index <= a], s[s.index <= b]
    if s0.empty or s1.empty or s.index.min() > a:
        return None
    return float(s1.iloc[-1] / s0.iloc[-1] - 1)


def _diff(series: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float | None:
    s = series.dropna()
    s0, s1 = s[s.index <= a], s[s.index <= b]
    if s0.empty or s1.empty:
        return None
    return float(s1.iloc[-1] - s0.iloc[-1])


def measure_window(start: str, end: str, fred: pd.DataFrame | None = None) -> dict[str, float]:
    """Factor moves from the close on/before ``start`` to the close on/before ``end``."""
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    px = wide("adj_close")
    out: dict[str, float] = {}
    spy = _ret(px["SPY"], a, b)
    out["eq_SPY"] = spy
    sectors = load_config("mcc_sector_map")["sector_etf"]
    for sector, etf in sectors.items():
        r = _ret(px[etf], a, b) if etf in px else None
        out[f"eq_{sector}"] = r if r is not None else spy  # ETF not yet listed -> market proxy
    out["eq_EUROPE"] = _ret(px["VGK"], a, b)
    out["eq_CHINA"] = _ret(px["FXI"], a, b)
    close = wide("close")
    out["rates_10y_bp"] = _diff(close["^TNX"], a, b) * 100
    out["rates_3m_bp"] = _diff(close["^IRX"], a, b) * 100
    out["oil"] = _ret(px["CL=F"], a, b)
    out["usd_index"] = _ret(px["DX-Y.NYB"], a, b)
    out["fx_EUR"] = _ret(px["EURUSD=X"], a, b)
    out["fx_GBP"] = _ret(px["GBPUSD=X"], a, b)
    jpy = _ret(px["JPY=X"], a, b)  # JPY=X is JPY per USD -> JPY value in USD moves 1/(1+r)-1
    out["fx_JPY"] = None if jpy is None else 1 / (1 + jpy) - 1
    out["vol_vix_pts"] = _diff(close["^VIX"], a, b)
    if fred is not None:
        f = fred.set_index("date")
        out["credit_bbb_bp"] = _diff(f["DBAA"] - f["DGS10"], a, b) * 100
        out["credit_aaa_bp"] = _diff(f["DAAA"] - f["DGS10"], a, b) * 100
    return {k: (round(v, 6) if v is not None else None) for k, v in out.items()}


def load_fred_extract() -> pd.DataFrame | None:
    if not FRED_EXTRACT.exists():
        return None
    df = pd.read_csv(FRED_EXTRACT, parse_dates=["date"])
    return df.sort_values("date")


def calibrate() -> pd.DataFrame:
    """Measure every pre-cutoff analogue; write data/scenarios/calibration.csv (long format)."""
    cfg = load_config("scenarios")
    cutoff = pd.Timestamp(cfg["calibration_cutoff"])
    fred = load_fred_extract()
    if fred is None:
        raise FileNotFoundError(
            f"{FRED_EXTRACT} missing; see data/market/README.md (credit spreads need it)."
        )
    rows = []
    for name, spec in cfg["analogues"].items():
        start, end = spec["window"]
        if pd.Timestamp(end) > cutoff:
            raise ValueError(f"Analogue {name} ends after the calibration cutoff")
        for factor, value in measure_window(start, end, fred).items():
            rows.append(
                {"analogue": name, "start": start, "end": end, "factor": factor, "value": value}
            )
    df = pd.DataFrame(rows)
    out = data_path("scenarios", "calibration.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def load_calibration() -> pd.DataFrame:
    return pd.read_csv(data_path("scenarios", "calibration.csv"))
