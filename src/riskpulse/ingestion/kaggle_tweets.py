"""Kaggle stock-tweets adapter (equinxx/stock-tweets-for-sentiment-analysis-and-prediction).

Social source: 80,793 tweets, 2021-09-30 -> 2022-09-29 UTC, labelled with the scraped ticker.
We keep tweets for universe tickers, drop cashtag lists (more than ``max_cashtags`` $TICKERs)
and, by default, tweets whose text never mentions the company they are attributed to.
"""

from __future__ import annotations

import re
import zipfile

import pandas as pd

from riskpulse.common.config import load_config, repo_root
from riskpulse.engine.entities import EntityLinker
from riskpulse.ingestion.normalize import clean_text

_CASHTAG = re.compile(r"\$[A-Za-z]{1,5}\b")


def load_raw() -> pd.DataFrame:
    """Raw tweets with parsed UTC timestamps."""
    cfg = load_config("app")["kaggle_tweets"]
    with zipfile.ZipFile(repo_root() / cfg["zip"]) as zf, zf.open(cfg["member"]) as fh:
        df = pd.read_csv(fh)
    df["published_at"] = pd.to_datetime(df["Date"], utc=True)
    df["label_ticker"] = df["Stock Name"].replace(cfg.get("ticker_map", {}))
    return df


def load_tweets(linker: EntityLinker | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Universe tweets after quality filters. Returns (frame, filter counts).

    Output columns: published_at, text, label_ticker, tickers (linked, label first),
    relevance (of label ticker), n_cashtags, source.
    """
    cfg = load_config("app")["kaggle_tweets"]
    linker = linker or EntityLinker()
    raw = load_raw()
    stats = {"raw": len(raw)}
    df = raw[raw["label_ticker"].isin(linker.tickers)].copy()
    stats["universe"] = len(df)
    df["text"] = df["Tweet"].map(clean_text)
    df["n_cashtags"] = df["Tweet"].str.count(_CASHTAG)
    df = df[df["n_cashtags"] <= int(cfg["max_cashtags"])]
    stats["after_cashtag_filter"] = len(df)

    tickers_col, rel_col, keep = [], [], []
    for text, label in zip(df["text"], df["label_ticker"], strict=True):
        mentions = linker.link(text)
        found = {m.ticker: m.relevance for m in mentions}
        ok = label in found or not cfg.get("require_text_link", True)
        keep.append(ok)
        others = [t for t in found if t != label and t != "MKT"]
        tickers_col.append([label, *others])
        rel_col.append(found.get(label, 0.0))
    df["tickers"] = tickers_col
    df["relevance"] = rel_col
    df = df[keep]
    stats["after_text_link_filter"] = len(df)
    df["source"] = "kaggle_tweets"
    cols = ["published_at", "text", "label_ticker", "tickers", "relevance", "n_cashtags", "source"]
    return df[cols].reset_index(drop=True), stats
