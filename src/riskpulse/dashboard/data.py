"""Dashboard data access: cached outputs first (works with the API down: `demo --fast`),
the live API for /analyze, /inject and the latest replay signals when it is running."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from riskpulse.common.config import data_path, load_config, repo_root, reports_path
from riskpulse.common.schemas import Signal

API = os.environ.get("RISKPULSE_API", "http://127.0.0.1:8000")
DEMO_DIR = data_path("demo")


def _first(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    p = reports_path("metrics.json")
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def feed_stats() -> dict:
    p = reports_path("feed_stats.json")
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner="Loading signals")
def signals_df() -> pd.DataFrame:
    """Flattened signals from the batch store or the committed demo snapshot."""
    p = _first(
        repo_root() / load_config("app")["paths"]["signals_jsonl"], DEMO_DIR / "signals.jsonl.gz"
    )
    if p is None:
        return pd.DataFrame()
    opener = gzip.open if p.suffix == ".gz" else open
    rows = []
    with opener(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            s = Signal.model_validate_json(line)
            rows.append(
                {
                    "signal_id": s.signal_id,
                    "signal_type": s.signal_type,
                    "as_of": pd.Timestamp(s.as_of),
                    "ticker": s.entity.ticker if s.entity else "MKT",
                    "event_id": s.event_id,
                    "event_class": s.event_class.value,
                    "event_confidence": s.event_confidence,
                    "sentiment": s.sentiment_score,
                    "impact": s.impact_score,
                    "confidence": s.confidence,
                    "n_docs": s.n_docs,
                    "n_sources": s.n_sources,
                    "regions": ", ".join(s.regions),
                    "explanation": s.explanation,
                    "drivers": s.drivers,
                    "evidence": [e.model_dump(mode="json") for e in s.evidence],
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Loading scored mentions")
def mentions_df() -> pd.DataFrame:
    p = _first(data_path("processed", "mentions.parquet"), DEMO_DIR / "mentions.parquet")
    if p is None:
        return pd.DataFrame()
    m = pd.read_parquet(p)
    m["published_at"] = pd.to_datetime(m["published_at"], utc=True)
    return m


@st.cache_data(show_spinner=False)
def module_a_outputs() -> dict[str, pd.DataFrame]:
    base = _first(data_path("processed", "moduleA"), DEMO_DIR / "moduleA")
    out = {}
    if base is not None:
        for name in ["weights_tilt", "returns", "daily_sentiment", "daily_confidence", "ic"]:
            f = base / f"{name}.parquet"
            if f.exists():
                out[name] = pd.read_parquet(f)
    return out


@st.cache_data(show_spinner=False)
def stress_runs() -> list[dict]:
    p = _first(
        data_path("processed", "moduleB", "stress_runs.jsonl"), DEMO_DIR / "stress_runs.jsonl"
    )
    if p is None:
        return []
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def api_up() -> bool:
    try:
        return httpx.get(f"{API}/health", timeout=1.5).status_code == 200
    except httpx.HTTPError:
        return False


def api_health() -> dict | None:
    try:
        return httpx.get(f"{API}/health", timeout=1.5).json()
    except httpx.HTTPError:
        return None


def api_post(path: str, payload: dict, timeout: float = 60.0) -> tuple[int, dict]:
    try:
        r = httpx.post(f"{API}{path}", json=payload, timeout=timeout)
        return r.status_code, r.json()
    except httpx.HTTPError as exc:
        return 0, {"detail": f"API not reachable at {API}: {exc}"}


def api_get(path: str, params: dict | None = None) -> list | dict | None:
    try:
        r = httpx.get(f"{API}{path}", params=params, timeout=5)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None
