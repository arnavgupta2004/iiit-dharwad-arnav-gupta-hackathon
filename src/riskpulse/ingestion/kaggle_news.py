"""Benzinga historical headlines (miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests).

1.40M headlines for 6,192 tickers, 2009-02 -> 2020-06, minute timestamps with explicit
UTC offset. Used cross-sectionally for the impact event study (not per-name history; see
data/SOURCES.md coverage caveat). Malformed rows (unparseable date / missing ticker) dropped.
"""

from __future__ import annotations

import zipfile

import pandas as pd

from riskpulse.common.config import load_config, repo_root
from riskpulse.ingestion.normalize import clean_text


def load_benzinga() -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (frame[published_at UTC, title, ticker, source], counts)."""
    cfg = load_config("app")["kaggle_news"]
    with zipfile.ZipFile(repo_root() / cfg["zip"]) as zf, zf.open(cfg["member"]) as fh:
        df = pd.read_csv(fh, usecols=["title", "date", "stock"])
    stats = {"raw": len(df)}
    df["published_at"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at", "stock", "title"])
    stats["valid"] = len(df)
    df["title"] = df["title"].map(clean_text)
    df = df[df["title"].str.len() > 0]
    out = df.rename(columns={"stock": "ticker"})[["published_at", "title", "ticker"]]
    out["source"] = "kaggle_news"
    return out.sort_values("published_at").reset_index(drop=True), stats
