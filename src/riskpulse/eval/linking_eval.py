"""Entity-linking precision from Arnav's marks in data/gold/entity_check.csv."""

from __future__ import annotations

import math

import pandas as pd

from riskpulse.common.config import data_path
from riskpulse.common.metrics import update_metrics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def run() -> dict:
    df = pd.read_csv(data_path("gold", "entity_check.csv"), dtype={"correct": str})
    marked = df[df["correct"].fillna("").str.strip().str.lower().isin(["y", "n"])].copy()
    if marked.empty:
        payload = {
            "status": "PENDING - entity_check.csv not yet marked by Arnav",
            "n_links": len(df),
        }
    else:
        marked["ok"] = marked["correct"].str.strip().str.lower() == "y"
        k, n = int(marked["ok"].sum()), len(marked)
        per = {
            t: {"n": len(g), "precision": round(float(g["ok"].mean()), 4)}
            for t, g in marked.groupby("ticker")
        }
        payload = {
            "status": "complete" if n == len(df) else f"partial: {n} of {len(df)} marked",
            "n_marked": n,
            "precision": round(k / n, 4),
            "precision_95ci_wilson": wilson(k, n),
            "per_ticker": per,
            "note": "News headline links, 5 per ticker, so per-ticker intervals are wide.",
        }
    update_metrics("entity_linking", payload, script="riskpulse eval linking")
    return payload
