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
from riskpulse.common.schemas import Signal

LIMIT_MB = 50
DEMO = data_path("demo")
# Only the columns the dashboard reads (repo-size budget): random-hash ids, outlets, drivers and
# raw scores stay in the full batch output. event_id becomes a compact story number.
MENTION_COLS = [
    "source",
    "published_at",
    "title",
    "ticker",
    "sentiment",
    "event_class",
    "event_confidence",
    "event_id",
    "impact_score",
]


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(data_path("processed", "mentions.parquet"), columns=MENTION_COLS)
    m["title"] = m["title"].str.slice(0, 200)
    m["event_id"] = pd.factorize(m["event_id"])[0].astype("int32")
    m["sentiment"] = m["sentiment"].astype("float32")
    m["event_confidence"] = m["event_confidence"].astype("float32")
    m = m.sort_values("published_at", kind="stable")
    m.to_parquet(DEMO / "mentions.parquet", index=False, compression="zstd", compression_level=19)
    # Signals: event signals with impact >= 6 (all stress candidates) and one entity snapshot per
    # ticker per day (the last of the day); evidence trimmed to 3 items.
    src = repo_root() / load_config("app")["paths"]["signals_jsonl"]
    last_entity: dict[tuple[str, str], str] = {}
    keep_events: list[str] = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            s = Signal.model_validate_json(line)
            s = s.model_copy(update={"evidence": s.evidence[:3]})
            if s.signal_type == "event":
                if s.impact_score >= 6:
                    keep_events.append(s.model_dump_json())
            elif s.entity is not None:
                last_entity[(s.entity.ticker, s.as_of.date().isoformat())] = s.model_dump_json()
    with gzip.open(DEMO / "signals.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as fo:
        for line in [*keep_events, *last_entity.values()]:
            fo.write(line + "\n")
    a_src, a_dst = data_path("processed", "moduleA"), DEMO / "moduleA"
    a_dst.mkdir(exist_ok=True)
    for f in a_src.glob("*.parquet"):
        shutil.copy(f, a_dst / f.name)
    runs = data_path("processed", "moduleB", "stress_runs.jsonl")
    if runs.exists():
        with (
            runs.open("rb") as fi,
            gzip.open(DEMO / "stress_runs.jsonl.gz", "wb", compresslevel=9) as fo,
        ):
            shutil.copyfileobj(fi, fo)
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
