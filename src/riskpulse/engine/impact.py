"""Impact score (field 3), v1 heuristic (spec §5.5).

raw = T * (w_S*S + w_V*V + w_B*B + w_N*N + w_C*C) * (f + (1 - f) * R)

T type prior of the event class, S |entity sentiment|, V velocity (sigmoid of the z-score of
mentions in the last N hours vs a trailing per-bucket baseline), B breadth (log-scaled distinct
outlets in the last N hours), N novelty, C source credibility, R relevance, f relevance floor.
All inputs are known at publication time (no look-ahead). ``raw`` maps to 1..10 with quantile
bin edges fitted on the historical batch; every score carries per-driver contributions.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
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
    """Quantile edges mapping raw impact to 1..10, optionally per population.

    ``edges`` is either one ascending list (shared) or ``{"company": [...], "market": [...]}``.
    """

    def __init__(self, edges: list[float] | dict[str, list[float]] | None = None) -> None:
        self.edges = edges

    @staticmethod
    def _quantiles(n_bins: int, edge_quantiles: list[float] | None) -> np.ndarray:
        if edge_quantiles is not None:
            q = np.asarray(edge_quantiles, dtype=float)
            if len(q) != n_bins - 1 or np.any(np.diff(q) <= 0):
                raise ValueError("edge_quantiles must be n_bins - 1 strictly increasing values")
            return q
        return np.linspace(0, 1, n_bins + 1)[1:-1]

    @classmethod
    def fit(
        cls,
        raws: np.ndarray,
        n_bins: int = 10,
        edge_quantiles: list[float] | None = None,
        groups: np.ndarray | None = None,
    ) -> ImpactBins:
        qs = cls._quantiles(n_bins, edge_quantiles)
        raws = np.asarray(raws, dtype=float)
        if groups is None:
            return cls([float(x) for x in np.quantile(raws, qs)])
        groups = np.asarray(groups)
        return cls(
            {g: [float(x) for x in np.quantile(raws[groups == g], qs)] for g in np.unique(groups)}
        )

    def score(self, raw: float, group: str = "company") -> int:
        if self.edges is None:  # unfitted fallback: linear
            return int(min(10, max(1, math.ceil(raw * 10))))
        edges = self.edges[group] if isinstance(self.edges, dict) else self.edges
        return int(np.searchsorted(edges, raw, side="right")) + 1

    def save(self, path: Path | None = None) -> Path:
        path = path or repo_root() / load_config("impact")["binning"]["bins_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"edges": self.edges}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> ImpactBins:
        path = path or repo_root() / load_config("impact")["binning"]["bins_path"]
        return cls(json.loads(path.read_text())["edges"]) if path.exists() else cls(None)


def score_impact(
    f: ImpactFeatures, bins: ImpactBins, cfg: dict | None = None, group: str = "company"
) -> ImpactScore:
    raw, contrib = raw_impact(f, cfg)
    drivers = {k: round(v, 4) for k, v in asdict(f).items()}
    return ImpactScore(round(raw, 6), bins.score(raw, group), drivers, contrib)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_US = timedelta(microseconds=1)


def _micros(ts: datetime) -> int:
    """Exact integer microseconds since the epoch (no float rounding at bucket edges)."""
    return (ts - _EPOCH) // _US


class VelocityTracker:
    """Per-entity mention velocity: z-score of the last-N-hours count vs trailing buckets.

    recent   = mentions in (t - N h, t]
    baseline = counts in the n = baseline_days / N consecutive N-hour buckets covering
               [t - N h - baseline, t - N h] (relative to t; the last bucket includes its end).
    Timestamps are kept per key as a sorted list of integer microseconds with a moving head; each
    observation costs n + 2 binary searches instead of a scan of the whole baseline.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or load_config("impact"))["velocity"]
        self.window = int(timedelta(hours=float(c["window_hours"])) // _US)
        self.baseline = int(timedelta(days=float(c["baseline_days"])) // _US)
        self.scale = float(c["sigmoid_scale"])
        self.min_std = float(c["min_std"])
        self.n_buckets = int(self.baseline // self.window)
        self.times: dict[str, list[int]] = defaultdict(list)
        self.head: dict[str, int] = defaultdict(int)

    def observe(self, entity: str, ts: datetime) -> tuple[float, float]:
        """Record a mention and return (V in (0,1), z). Times must be non-decreasing."""
        t = _micros(ts)
        q = self.times[entity]
        q.append(t)
        base_start = t - self.baseline - self.window
        h = bisect.bisect_left(q, base_start, lo=self.head[entity])
        if h > 4096 and h > len(q) // 2:  # occasional compaction of expired timestamps
            del q[:h]
            h = 0
        self.head[entity] = h
        cut = t - self.window
        recent = len(q) - bisect.bisect_right(q, cut, lo=h)
        edges = [
            bisect.bisect_left(q, base_start + k * self.window, lo=h) for k in range(self.n_buckets)
        ]
        end = bisect.bisect_right(q, cut, lo=h)
        counts = np.diff(np.array([*edges, end], dtype=float))
        z = (recent - counts.mean()) / max(counts.std(), self.min_std)
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
