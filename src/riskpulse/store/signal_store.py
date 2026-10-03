"""Signal and mention persistence: append-only JSONL (the file output) + DuckDB (queries).

- ``signals.jsonl``: one Signal (schema_version 1) per line; downstream apps can tail it.
- DuckDB ``mentions``: one row per scored (document, entity) pair.
- DuckDB ``signals``: key signal columns + full JSON payload, for filtering by ticker / class /
  impact / time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

import duckdb
import pandas as pd

from riskpulse.common.config import load_config, repo_root
from riskpulse.common.schemas import Signal
from riskpulse.engine.aggregate import ScoredMention


def default_paths() -> tuple[Path, Path]:
    p = load_config("app")["paths"]
    return repo_root() / p["signals_jsonl"], repo_root() / p["duckdb"]


class JsonlSignalWriter:
    """Append-only JSONL writer for signals."""

    def __init__(self, path: Path | None = None, truncate: bool = False) -> None:
        self.path = path or default_paths()[0]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if truncate and self.path.exists():
            self.path.unlink()

    def write(self, signals: list[Signal]) -> None:
        if not signals:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for s in signals:
                fh.write(s.model_dump_json() + "\n")


def read_signals_jsonl(path: Path | None = None) -> list[Signal]:
    """Read signals from a .jsonl or .jsonl.gz file (empty list if missing)."""
    import gzip

    path = path or default_paths()[0]
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [Signal.model_validate_json(line) for line in fh if line.strip()]


def session_signals_path() -> Path:
    """Where the API appends signals it produces (demo replay, /inject): never the batch file,
    which Module B and evaluation read."""
    return repo_root() / load_config("app")["paths"]["session_signals"]


def serving_signals_path() -> Path:
    """Signals to serve: the batch output if present, else the committed demo snapshot.

    ``RISKPULSE_FORCE_DEMO=1`` always uses the snapshot (fresh-clone behaviour).
    """
    import os

    batch = default_paths()[0]
    demo = repo_root() / "data" / "demo" / "signals.jsonl.gz"
    if os.environ.get("RISKPULSE_FORCE_DEMO") == "1" or not batch.exists():
        return demo
    return batch


def signals_frame(signals: list[Signal]) -> pd.DataFrame:
    """Flatten signals into a query-friendly frame (payload kept as JSON)."""
    rows = []
    for s in signals:
        rows.append(
            {
                "signal_id": s.signal_id,
                "signal_type": s.signal_type,
                "as_of": pd.Timestamp(s.as_of),
                "ticker": s.entity.ticker if s.entity else None,
                "event_id": s.event_id,
                "sentiment_score": s.sentiment_score,
                "event_class": s.event_class.value,
                "event_confidence": s.event_confidence,
                "impact_score": s.impact_score,
                "confidence": s.confidence,
                "n_docs": s.n_docs,
                "n_sources": s.n_sources,
                "regions": ",".join(s.regions),
                "payload": s.model_dump_json(),
            }
        )
    return pd.DataFrame(rows)


def mentions_frame(mentions: list[ScoredMention]) -> pd.DataFrame:
    cols = [f.name for f in fields(ScoredMention)]
    df = pd.DataFrame([asdict(m) for m in mentions], columns=cols)
    if not df.empty:
        df["drivers"] = df["drivers"].map(json.dumps)
        df["regions"] = df["regions"].map(lambda r: ",".join(r))
    return df


def write_duckdb(
    mentions: list[ScoredMention], signals: list[Signal], path: Path | None = None
) -> Path:
    """(Re)create the DuckDB store from a batch run."""
    path = path or default_paths()[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    m_df, s_df = mentions_frame(mentions), signals_frame(signals)
    with duckdb.connect(str(path)) as con:
        con.execute("DROP TABLE IF EXISTS mentions")
        con.execute("DROP TABLE IF EXISTS signals")
        con.register("m_df", m_df)
        con.register("s_df", s_df)
        con.execute("CREATE TABLE mentions AS SELECT * FROM m_df")
        con.execute("CREATE TABLE signals AS SELECT * FROM s_df")
    return path


def query_signals(
    df: pd.DataFrame,
    ticker: str | None = None,
    event_class: str | None = None,
    min_impact: int | None = None,
    since: str | None = None,
    signal_type: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Filter a signals frame (used by the API and dashboard)."""
    q = df
    if ticker:
        q = q[q["ticker"] == ticker]
    if event_class:
        q = q[q["event_class"] == event_class]
    if min_impact is not None:
        q = q[q["impact_score"] >= min_impact]
    if since:
        q = q[q["as_of"] >= pd.Timestamp(since, tz="UTC")]
    if signal_type:
        q = q[q["signal_type"] == signal_type]
    return q.sort_values("as_of", ascending=False).head(limit)
