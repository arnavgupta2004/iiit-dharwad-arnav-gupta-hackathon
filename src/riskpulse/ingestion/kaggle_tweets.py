"""Kaggle stock-tweets adapter (equinxx/stock-tweets-for-sentiment-analysis-and-prediction).

Social source: 80,793 tweets, 2021-09-30 -> 2022-09-29 UTC, labelled with the scraped ticker.
We keep tweets for universe tickers, drop cashtag lists (more than ``max_cashtags`` $TICKERs)
and, by default, tweets whose text never mentions the company they are attributed to.

Original-ticker rule (DECISIONS D-033): identical tweet texts scraped under several tickers are
kept once and linked ONLY to their original ticker. Label sets that are wholesale copies of
another label's set (PG and MSFT copy AMZN in this dataset) are never treated as originals; among
genuine labels the first-mentioned cashtag wins. The text -> original-ticker map is committed in
``data/replay/tweet_original_ticker.csv`` so the live path can apply it without the raw data.
"""

from __future__ import annotations

import re
import zipfile

import pandas as pd

from riskpulse.common.config import load_config, repo_root
from riskpulse.engine.entities import EntityLinker
from riskpulse.ingestion.normalize import clean_text, dedupe_key

_CASHTAG = re.compile(r"\$[A-Za-z]{1,5}\b")


def load_raw() -> pd.DataFrame:
    """Raw tweets with parsed UTC timestamps."""
    cfg = load_config("app")["kaggle_tweets"]
    with zipfile.ZipFile(repo_root() / cfg["zip"]) as zf, zf.open(cfg["member"]) as fh:
        df = pd.read_csv(fh)
    df["published_at"] = pd.to_datetime(df["Date"], utc=True)
    df["label_ticker"] = df["Stock Name"].replace(cfg.get("ticker_map", {}))
    return df


ORIGINAL_MAP = "data/replay/tweet_original_ticker.csv"
COPY_OVERLAP = 0.95  # share of a label's texts also present under another label


def copy_sets(raw: pd.DataFrame) -> dict[str, str]:
    """Labels whose tweet set is (>= 95%) a copy of another label's set -> that source label.

    The copy is the label with the lower share of tweets containing its own cashtag.
    """
    texts = {t: set(g["key"]) for t, g in raw.groupby("label_ticker")}
    own = {
        t: g["Tweet"].str.contains(rf"\${re.escape(str(t))}\b", case=False, regex=True).mean()
        for t, g in raw.groupby("label_ticker")
    }
    out = {}
    for a, sa in texts.items():
        for b, sb in texts.items():
            if (
                a != b
                and len(sa) >= 50
                and len(sa & sb) / len(sa) >= COPY_OVERLAP
                and own[a] < own[b]
            ):
                out[a] = b
    return out


def _cashtag_pos(text: str, ticker: str, cashtags: dict[str, list[str]]) -> int:
    tags = cashtags.get(ticker, [f"${ticker}"])
    low = text.lower()
    pos = [low.find(t.lower()) for t in tags]
    pos = [p for p in pos if p >= 0]
    return min(pos) if pos else 1 << 30


def build_original_map(raw: pd.DataFrame | None = None, write: bool = True) -> pd.DataFrame:
    """One row per distinct tweet text: key, original_ticker, all scraped labels."""
    raw = load_raw() if raw is None else raw
    raw = raw.assign(key=raw["Tweet"].map(lambda t: dedupe_key(clean_text(t))))
    copies = copy_sets(raw)
    cashtags = {t: s["cashtags"] for t, s in load_config("universe")["tickers"].items()}
    rows = []
    for key, g in raw.groupby("key"):
        labels = sorted(set(g["label_ticker"]))
        genuine = [lab for lab in labels if lab not in copies] or [copies[labels[0]]]
        text = g["Tweet"].iloc[0]
        orig = min(genuine, key=lambda lab: (_cashtag_pos(text, lab, cashtags), lab))
        rows.append((key, orig, ",".join(labels)))
    out = pd.DataFrame(rows, columns=["key", "original_ticker", "scraped_labels"])
    if write:
        out.to_csv(repo_root() / ORIGINAL_MAP, index=False)
    return out


def original_ticker_lookup() -> dict[str, str]:
    """key -> original ticker, from the committed map (empty if absent)."""
    path = repo_root() / ORIGINAL_MAP
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["key", "original_ticker"])
    return dict(zip(df["key"], df["original_ticker"], strict=True))


def load_tweets(linker: EntityLinker | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Universe tweets after quality filters. Returns (frame, filter counts).

    Output columns: published_at, text, label_ticker, tickers (linked, label first),
    relevance (of label ticker), n_cashtags, source.
    """
    cfg = load_config("app")["kaggle_tweets"]
    linker = linker or EntityLinker()
    raw = load_raw()
    stats = {"raw": len(raw)}
    raw["key"] = raw["Tweet"].map(lambda t: dedupe_key(clean_text(t)))
    orig = build_original_map(raw).set_index("key")["original_ticker"]
    raw["original_ticker"] = raw["key"].map(orig)
    raw = raw[raw["label_ticker"] == raw["original_ticker"]].drop_duplicates("key")
    stats["after_original_ticker_rule"] = len(raw)
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
        tickers_col.append([label])  # original ticker only (D-033)
        rel_col.append(found.get(label, 0.0))
    df["tickers"] = tickers_col
    df["relevance"] = rel_col
    df = df[keep]
    stats["after_text_link_filter"] = len(df)
    df["source"] = "kaggle_tweets"
    cols = ["published_at", "text", "label_ticker", "tickers", "relevance", "n_cashtags", "source"]
    return df[cols].reset_index(drop=True), stats
