"""Impact score (field 3), v1 heuristic (spec §5.5).

raw = T * (w_S*S + w_V*V + w_B*B + w_N*N + w_C*C) * (f + (1 - f) * R)

T type prior of the event class, S |entity sentiment|, V velocity (sigmoid of the z-score of
mentions in the last N hours vs a trailing per-bucket baseline), B breadth (log-scaled distinct
outlets in the last N hours), N novelty, C source credibility, R relevance, f relevance floor.
All inputs are known at publication time (no look-ahead). ``raw`` maps to 1..10 with quantile
bin edges fitted on the historical batch; every score carries per-driver contributions.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from riskpulse.common.config import load_config, repo_root


@dataclass(frozen=True)
class ImpactFeatures:
    type_prior: float
    sentiment: float
    velocity: float
    breadth: float
    novelty: float
    credibility: float
    relevance: float


@dataclass(frozen=True)
class ImpactScore:
    raw: float
    score: int
    drivers: dict[str, float]
    contributions: dict[str, float]


def raw_impact(f: ImpactFeatures, cfg: dict | None = None) -> tuple[float, dict[str, float]]:
    """Return (raw score in [0, 1], per-driver contributions to raw)."""
    cfg = cfg or load_config("impact")
    w = cfg["weights"]
    floor = float(cfg["relevance_floor"])
    rel_mult = floor + (1 - floor) * f.relevance
    parts = {
        "sentiment": w["sentiment"] * f.sentiment,
        "velocity": w["velocity"] * f.velocity,
        "breadth": w["breadth"] * f.breadth,
        "novelty": w["novelty"] * f.novelty,
        "credibility": w["credibility"] * f.credibility,
    }
    scale = f.type_prior * rel_mult
    contributions = {k: round(v * scale, 4) for k, v in parts.items()}
    return float(scale * sum(parts.values())), contributions


class ImpactBins:
    """Quantile bin edges mapping raw impact to 1..10."""

    def __init__(self, edges: list[float] | None = None) -> None:
        self.edges = edges  # n_bins - 1 interior edges, ascending

    @classmethod
    def fit(cls, raws: np.ndarray, n_bins: int = 10) -> ImpactBins:
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        return cls([float(x) for x in np.quantile(np.asarray(raws, dtype=float), qs)])

    def score(self, raw: float) -> int:
        if self.edges is None:  # unfitted fallback: linear
            return int(min(10, max(1, math.ceil(raw * 10))))
        return int(np.searchsorted(self.edges, raw, side="right")) + 1

    def save(self, path: Path | None = None) -> Path:
        path = path or repo_root() / load_config("impact")["binning"]["bins_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"edges": self.edges}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> ImpactBins:
        path = path or repo_root() / load_config("impact")["binning"]["bins_path"]
        return cls(json.loads(path.read_text())["edges"]) if path.exists() else cls(None)


def score_impact(f: ImpactFeatures, bins: ImpactBins, cfg: dict | None = None) -> ImpactScore:
    raw, contrib = raw_impact(f, cfg)
    drivers = {k: round(v, 4) for k, v in asdict(f).items()}
    return ImpactScore(round(raw, 6), bins.score(raw), drivers, contrib)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class VelocityTracker:
    """Per-entity mention velocity: z-score of the last-N-hours count vs trailing buckets.

    recent   = mentions in the sliding window (t - N h, t]          (exact, deque)
    baseline = counts in the 28 aligned N-hour buckets before the current bucket (7 days for
               N = 6), kept incrementally per key, so each observation is O(28).
    """

    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or load_config("impact"))["velocity"]
        self.window = float(c["window_hours"]) * 3600
        self.scale = float(c["sigmoid_scale"])
        self.min_std = float(c["min_std"])
        self.n_buckets = int(float(c["baseline_days"]) * 86400 // self.window)
        self.recent: dict[str, deque[float]] = defaultdict(deque)
        self.buckets: dict[str, dict[int, int]] = defaultdict(dict)

    def observe(self, entity: str, ts: datetime) -> tuple[float, float]:
        """Record a mention and return (V in (0,1), z). Times must be non-decreasing."""
        t = ts.timestamp()
        q = self.recent[entity]
        q.append(t)
        while q and q[0] <= t - self.window:
            q.popleft()
        b = int(t // self.window)
        bk = self.buckets[entity]
        bk[b] = bk.get(b, 0) + 1
        if len(bk) > 4 * self.n_buckets:  # drop buckets older than the baseline
            for old in [k for k in bk if k < b - self.n_buckets]:
                del bk[old]
        counts = np.array([bk.get(k, 0) for k in range(b - self.n_buckets, b)], dtype=float)
        z = (len(q) - counts.mean()) / max(counts.std(), self.min_std)
        return _sigmoid(z / self.scale), float(z)


class BreadthTracker:
    """Distinct outlets mentioning an entity / story in the last N hours, log-scaled to [0,1]."""

    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or load_config("impact"))["breadth"]
        self.window = timedelta(hours=float(c["window_hours"]))
        self.sat = float(c["saturation_outlets"])
        self.events: dict[str, deque[tuple[datetime, str]]] = defaultdict(deque)

    def observe(self, key: str, outlet: str, ts: datetime) -> tuple[float, int]:
        q = self.events[key]
        q.append((ts, outlet))
        while q and q[0][0] < ts - self.window:
            q.popleft()
        n = len({o for _, o in q})
        return min(1.0, math.log1p(n) / math.log1p(self.sat)), n
