"""Download the public Kaggle datasets used by RiskPulse into data/raw/_downloads/kaggle.

Kaggle's public download endpoint serves these (all public) datasets without an API key
(verified 2026-10-01). Zips are kept as-is; loaders read CSVs straight from the zip.
Raw zips are gitignored; cleaned samples derived from them are committed under data/.

    python scripts/download_kaggle.py [slug ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "_downloads" / "kaggle"
URL = "https://www.kaggle.com/api/v1/datasets/download/{slug}"

DATASETS: dict[str, str] = {
    # social: stock tweets, 25 tickers, 2021-09-30..2022-09-29 (CC0)
    "equinxx/stock-tweets-for-sentiment-analysis-and-prediction": "stock tweets",
    # news: Benzinga headlines with tickers, 2009..2020 (CC0)
    "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests": "benzinga news",
    # transactions: seeds the Module B synthetic wholesale book (Apache 2.0)
    "computingvictor/transactions-fraud-datasets": "transactions",
}


def download(slug: str, chunk: int = 1 << 20) -> Path:
    """Stream one dataset zip to disk; skip if it already exists."""
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{slug.split('/')[1]}.zip"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip  {slug} (exists, {dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    tmp = dest.with_suffix(".part")
    with requests.get(URL.format(slug=slug), stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for block in r.iter_content(chunk):
                fh.write(block)
    tmp.rename(dest)
    print(f"done  {slug} -> {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


if __name__ == "__main__":
    for s in sys.argv[1:] or list(DATASETS):
        download(s)
