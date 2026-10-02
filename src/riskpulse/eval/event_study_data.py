"""Benzinga event-study inputs for impact v2 (DECISIONS D-014): tickers, prices, stage-1 scores.

- Tickers: the 300 most-covered Benzinga tickers that have yfinance history (delisted names fail
  to download and are dropped; survivorship is a stated limitation).
- Prices: daily adjusted close and volume 2008-06 -> 2020-07, cached in
  data/raw/_downloads/event_study/prices.parquet (gitignored, reproducible here).
- Stage 1 (cached in parts): MiniLM embeddings; sentiment per sentiment weights; event class per
  event model, using the same models as the live engine.
"""

from __future__ import annotations

import hashlib

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


def universe_headlines() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Headlines of the event-study tickers in time order, plus their long price frame."""
    df = headlines()
    keep, px_long = select_universe(df)
    sub = df[df["ticker"].isin(keep)].sort_values("published_at").reset_index(drop=True)
    return sub, px_long


def _titles_key(titles: list[str]) -> str:
    return hashlib.sha1("\n".join(titles).encode()).hexdigest()[:10]


def _base(df: pd.DataFrame) -> pd.DataFrame:
    """Headline rows plus MiniLM embeddings (cached in stage1.parquet / stage1_emb.npy)."""
    meta_p, emb_p = CACHE / "stage1.parquet", CACHE / "stage1_emb.npy"
    cols = ["published_at", "ticker", "title"]
    if meta_p.exists() and emb_p.exists():
        meta = pd.read_parquet(meta_p)
        if len(meta) == len(df) and (meta["title"].to_numpy() == df["title"].to_numpy()).all():
            return meta[cols].copy()
    from riskpulse.engine.events import Embedder

    emb_model, embs = Embedder(), []
    titles = df["title"].tolist()
    for i in range(0, len(titles), 4096):
        embs.append(emb_model.encode(titles[i : i + 4096]).astype(np.float16))
        log.info(f"event-study embeddings: {min(i + 4096, len(titles)):,}/{len(titles):,}")
    meta = df[cols].copy()
    CACHE.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_p, index=False)
    np.save(emb_p, np.vstack(embs))
    return meta


def sentiment(titles: list[str], variant: str | None = None) -> np.ndarray:
    """Document score P(pos) - P(neg) per headline, cached per sentiment weights (D-043). Benzinga
    is news, so the default is the news weights (D-058)."""
    from riskpulse.engine.sentiment import FinBertScorer, variant_for, weights_fingerprint

    variant = variant or variant_for("news")

    path = CACHE / f"sentiment_{_titles_key(titles)}_{weights_fingerprint(variant)}.npy"
    if path.exists():
        return np.load(path)
    fb, out = FinBertScorer(variant=variant), []
    for i in range(0, len(titles), 4096):
        p = fb.probs(titles[i : i + 4096])
        out.append(p[:, 0] - p[:, 1])
        log.info(f"event-study sentiment: {min(i + 4096, len(titles)):,}/{len(titles):,}")
    res = np.concatenate(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, res)
    return res


def events(titles: list[str]) -> pd.DataFrame:
    """Event class and confidence per headline from cached embeddings; cached per event model."""
    from riskpulse.common.config import repo_root
    from riskpulse.engine.events import EmbeddingClassifier
    from riskpulse.engine.pipeline import EVENT_MODEL_PATH

    model_path = repo_root() / EVENT_MODEL_PATH
    ek = hashlib.sha1(model_path.read_bytes()).hexdigest()[:8]
    path = CACHE / f"events_{_titles_key(titles)}_{ek}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    clf = EmbeddingClassifier.load(model_path)
    preds = clf.predict_from_embeddings(embeddings().astype(np.float32))
    out = pd.DataFrame(
        {
            "event_class": [p.event_class for p in preds],
            "event_confidence": [p.confidence for p in preds],
        }
    )
    out.to_parquet(path, index=False)
    return out


def stage1(df: pd.DataFrame, variant: str | None = None) -> pd.DataFrame:
    """Per headline: sentiment (configured or given weights), event class/confidence, with the
    embeddings in ``embeddings()``. Each part is cached under its own model key."""
    meta = _base(df)
    titles = meta["title"].tolist()
    meta["sentiment"] = sentiment(titles, variant)
    ev = events(titles)
    meta["event_class"], meta["event_confidence"] = (
        ev["event_class"].to_numpy(),
        ev["event_confidence"].to_numpy(),
    )
    return meta


def embeddings() -> np.ndarray:
    return np.load(CACHE / "stage1_emb.npy").astype(np.float64)
