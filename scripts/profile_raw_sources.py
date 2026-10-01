"""Profile downloaded raw datasets: schema, dtypes, nulls, sample rows, date ranges.

Reads CSVs directly from the Kaggle zips in ``data/raw/_downloads/kaggle`` (no
extraction). Output is printed; the verified facts are transcribed into
``data/SOURCES.md``. Run after ``scripts/download_kaggle.py``.

    python scripts/profile_raw_sources.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "data" / "raw" / "_downloads" / "kaggle"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 80)


def _read(zip_name: str, member: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(KAGGLE / zip_name) as zf, zf.open(member) as fh:
        return pd.read_csv(fh, **kw)


def _profile(df: pd.DataFrame, title: str) -> None:
    print(f"\n===== {title}: shape={df.shape}")
    print("dtypes:", {c: str(t) for c, t in df.dtypes.items()})
    print("nulls:", df.isna().sum().to_dict())
    print(df.head(3).to_string())


def stock_tweets() -> None:
    z = "stock-tweets-for-sentiment-analysis-and-prediction.zip"
    df = _read(z, "stock_tweets.csv")
    _profile(df, "stock_tweets.csv")
    print("Date range:", df["Date"].min(), "->", df["Date"].max())
    print("Tickers:", df["Stock Name"].value_counts().to_dict())
    _profile(_read(z, "stock_yfinance_data.csv"), "stock_yfinance_data.csv")


def tweet_impact() -> None:
    z = "tweet-sentiment-s-impact-on-stock-returns.zip"
    df = _read(z, "full_dataset-release.csv", nrows=200_000)
    _profile(df, "full_dataset-release.csv (first 200k rows)")
    print("Date range (sample):", df["DATE"].min(), "->", df["DATE"].max())
    print("Top stocks (sample):", df["STOCK"].value_counts().head(40).to_dict())
    full = _read(z, "full_dataset-release.csv", usecols=["DATE", "STOCK"])
    print("FULL rows:", len(full), "date range:", full["DATE"].min(), "->", full["DATE"].max())
    print("FULL stock counts:", full["STOCK"].value_counts().to_dict())


def benzinga_news() -> None:
    z = "massive-stock-news-analysis-db-for-nlpbacktests.zip"
    df = _read(z, "analyst_ratings_processed.csv")
    _profile(df, "analyst_ratings_processed.csv")
    print("Date range:", df["date"].min(), "->", df["date"].max())
    print("Unique stocks:", df["stock"].nunique())
    for name in ["raw_analyst_ratings.csv", "raw_partner_headlines.csv"]:
        raw = _read(z, name)
        _profile(raw, name)
        print("Date range:", raw["date"].min(), "->", raw["date"].max())
        if "publisher" in raw:
            print("Top publishers:", raw["publisher"].value_counts().head(10).to_dict())
    globals()["_benzinga_counts"] = df["stock"].value_counts()


def transactions() -> None:
    z = "transactions-fraud-datasets.zip"
    tx = _read(z, "transactions_data.csv", nrows=500_000)
    _profile(tx, "transactions_data.csv (first 500k rows)")
    print("Date range (sample):", tx["date"].min(), "->", tx["date"].max())
    print("merchant_state top:", tx["merchant_state"].value_counts().head(15).to_dict())
    print(
        "unique merchants (sample):",
        tx["merchant_id"].nunique(),
        "unique mcc:",
        tx["mcc"].nunique(),
    )
    _profile(_read(z, "users_data.csv"), "users_data.csv")
    _profile(_read(z, "cards_data.csv"), "cards_data.csv")
    with zipfile.ZipFile(KAGGLE / z) as zf:
        mcc = json.loads(zf.read("mcc_codes.json"))
    print(f"\nmcc_codes.json: {len(mcc)} codes; sample:", dict(list(mcc.items())[:8]))


DATASETS = {
    "stock_tweets": stock_tweets,
    "tweet_impact": tweet_impact,
    "benzinga_news": benzinga_news,
    "transactions": transactions,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    for n in names:
        DATASETS[n]()
