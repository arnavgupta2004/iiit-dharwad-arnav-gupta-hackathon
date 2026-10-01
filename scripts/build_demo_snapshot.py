"""Copy compact versions of the batch outputs to data/demo/ for `demo --fast` on a fresh clone.

Inputs come from `riskpulse process`, `riskpulse backtest` and `riskpulse stress --replay`.
Every output file must stay under 50 MB (repo rule); the script fails loudly otherwise.

    python scripts/build_demo_snapshot.py
"""

from __future__ import annotations

import gzip
import shutil

import pandas as pd

from riskpulse.common.config import data_path, load_config, repo_root

LIMIT_MB = 50
DEMO = data_path("demo")
MENTION_COLS = [
    "doc_id",
    "source",
    "outlet",
    "published_at",
    "title",
    "url",
    "ticker",
    "relevance",
    "sentiment",
    "event_class",
    "event_confidence",
    "event_id",
    "impact_score",
    "impact_raw",
    "regions",
]


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(data_path("processed", "mentions.parquet"), columns=MENTION_COLS)
    m["title"] = m["title"].str.slice(0, 200)
    m.to_parquet(DEMO / "mentions.parquet", index=False, compression="zstd")
    src = repo_root() / load_config("app")["paths"]["signals_jsonl"]
    with src.open("rb") as fi, gzip.open(DEMO / "signals.jsonl.gz", "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)
    a_src, a_dst = data_path("processed", "moduleA"), DEMO / "moduleA"
    a_dst.mkdir(exist_ok=True)
    for f in a_src.glob("*.parquet"):
        shutil.copy(f, a_dst / f.name)
    runs = data_path("processed", "moduleB", "stress_runs.jsonl")
    if runs.exists():
        shutil.copy(runs, DEMO / "stress_runs.jsonl")
    total = 0.0
    for f in sorted(DEMO.rglob("*")):
        if f.is_file():
            mb = f.stat().st_size / 1e6
            total += mb
            print(f"{mb:8.2f} MB  {f.relative_to(repo_root())}")
            if mb > LIMIT_MB:
                raise SystemExit(f"{f} exceeds {LIMIT_MB} MB")
    print(f"total {total:.1f} MB")


if __name__ == "__main__":
    main()
