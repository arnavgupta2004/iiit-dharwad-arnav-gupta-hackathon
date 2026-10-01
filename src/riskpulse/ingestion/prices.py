"""Daily price cache (yfinance), so demos and evaluation never depend on Yahoo uptime.

Stored long-format in ``data/prices/daily.parquet``:
date, symbol, open, high, low, close, adj_close, volume.
"""

from __future__ import annotations

from functools import cache

import pandas as pd

from riskpulse.common.config import load_config, repo_root


def price_symbols() -> list[str]:
    """Universe tickers plus market/risk-factor proxies from config."""
    cfg = load_config("app")["prices"]
    return list(load_config("universe")["tickers"]) + list(cfg["extra_symbols"])


def download_prices(symbols: list[str] | None = None) -> pd.DataFrame:
    """Download daily OHLCV (unadjusted close + adjusted close) and write the parquet cache."""
    import yfinance as yf

    cfg = load_config("app")["prices"]
    symbols = symbols or price_symbols()
    raw = yf.download(
        symbols,
        start=cfg["start"],
        end=cfg["end"],
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    long = (
        raw.stack(level=1, future_stack=True)
        .rename_axis(["date", "symbol"])
        .reset_index()
        .rename(columns=str.lower)
        .rename(columns={"adj close": "adj_close"})
    )
    long = long.dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    cols = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    out = long[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    path = repo_root() / cfg["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


@cache
def load_prices() -> pd.DataFrame:
    """Long-format cached prices."""
    return pd.read_parquet(repo_root() / load_config("app")["prices"]["path"])


def wide(field: str = "adj_close", symbols: list[str] | None = None) -> pd.DataFrame:
    """Date x symbol matrix of one field."""
    p = load_prices()
    if symbols is not None:
        p = p[p["symbol"].isin(symbols)]
    return p.pivot(index="date", columns="symbol", values=field).sort_index()


def trading_days() -> pd.DatetimeIndex:
    """NYSE session dates, taken from SPY's cached history."""
    return pd.DatetimeIndex(wide("close", ["SPY"]).dropna().index)
