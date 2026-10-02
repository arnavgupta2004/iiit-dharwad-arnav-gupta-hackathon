"""Benzinga event-study inputs for impact v2 (DECISIONS D-014): tickers, prices, stage-1 scores.

- Tickers: the 300 most-covered Benzinga tickers that have yfinance history (delisted names fail
  to download and are dropped; survivorship is a stated limitation).
- Prices: daily adjusted close and volume 2008-06 -> 2020-07, cached in
  data/raw/_downloads/event_study/prices.parquet (gitignored, reproducible here).
- Stage 1 (cached): FinBERT sentiment, MiniLM embedding and event class for every headline of
  those tickers, using the same models as the live engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskpulse.common.config import data_path
from riskpulse.common.logging import get_logger
from riskpulse.ingestion.kaggle_news import load_benzinga

log = get_logger()
CACHE = data_path("raw", "_downloads", "event_study")
N_TICKERS = 300


def headlines() -> pd.DataFrame:
    df, _ = load_benzinga()
    return df


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Long frame date, symbol, adj_close, volume (cached)."""
    path = CACHE / "prices.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import yfinance as yf

    raw = yf.download(
        [*tickers, "SPY"],
        start="2008-06-01",
        end="2020-07-31",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    long = (
        raw[["Adj Close", "Volume"]]
        .stack(level=1, future_stack=True)
        .rename_axis(["date", "symbol"])
        .reset_index()
        .rename(columns={"Adj Close": "adj_close", "Volume": "volume"})
        .dropna(subset=["adj_close"])
    )
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    CACHE.mkdir(parents=True, exist_ok=True)
    long.to_parquet(path, index=False)
    return long


def select_universe(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    counts = df["ticker"].value_counts()
    cand = list(counts.index[:N_TICKERS])
    px = download_prices(cand)
    have = px.groupby("symbol")["date"].count()
    keep = [t for t in cand if have.get(t, 0) >= 250]
    log.info(f"Event-study tickers: {len(keep)} of {len(cand)} have >= 250 days of prices")
    return keep, px


def stage1(df: pd.DataFrame) -> pd.DataFrame:
    """FinBERT score, event class/confidence and embeddings for each headline (cached)."""
    meta_p, emb_p = CACHE / "stage1.parquet", CACHE / "stage1_emb.npy"
    if meta_p.exists() and emb_p.exists():
        meta = pd.read_parquet(meta_p)
        if len(meta) == len(df):
            return meta
    from riskpulse.common.config import repo_root
    from riskpulse.engine.events import Embedder, EmbeddingClassifier
    from riskpulse.engine.pipeline import EVENT_MODEL_PATH
    from riskpulse.engine.sentiment import FinBertScorer

    titles = df["title"].tolist()
    emb_model = Embedder()
    clf = EmbeddingClassifier.load(repo_root() / EVENT_MODEL_PATH, embedder=emb_model)
    fb = FinBertScorer()
    sent, cls, conf, embs = [], [], [], []
    step = 4096
    for i in range(0, len(titles), step):
        chunk = titles[i : i + step]
        p = fb.probs(chunk)
        sent.extend((p[:, 0] - p[:, 1]).tolist())
        e = emb_model.encode(chunk)
        for pred in clf.predict_from_embeddings(e):
            cls.append(pred.event_class)
            conf.append(pred.confidence)
        embs.append(e.astype(np.float16))
        log.info(f"event-study stage 1: {min(i + step, len(titles)):,}/{len(titles):,}")
    meta = df[["published_at", "ticker", "title"]].copy()
    meta["sentiment"], meta["event_class"], meta["event_confidence"] = sent, cls, conf
    CACHE.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_p, index=False)
    np.save(emb_p, np.vstack(embs))
    return meta


def embeddings() -> np.ndarray:
    return np.load(CACHE / "stage1_emb.npy").astype(np.float64)
