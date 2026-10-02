# VERBATIM SNAPSHOT of src/riskpulse/engine/clustering.py at commit 74716fb^ (pre-optimisation), kept only as the
# reference implementation for tests/test_stage2_regression.py. Do not edit.
# ruff: noqa
"""Online story clustering (spec §5.6).

Each incoming document (with a unit-norm embedding) joins the most similar *active* story if
cosine(doc, story centroid) >= threshold, else it starts a new story. A story is active while
its last document is within ``window_hours``. Centroids are running means, re-normalised.
Story ids are stable: ``evt_`` + first 12 hex chars of the founding document id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from riskpulse.common.config import load_config


@dataclass
class Story:
    event_id: str
    centroid: np.ndarray
    first_seen: datetime
    last_seen: datetime
    n_docs: int = 1
    doc_ids: list[str] = field(default_factory=list)


class StoryClusterer:
    """Greedy single-pass clusterer over a rolling time window."""

    def __init__(
        self,
        threshold: float | None = None,
        window_hours: float | None = None,
        retention_hours: float | None = None,
    ) -> None:
        cfg = load_config("app")["clustering"]
        self.threshold = float(cfg["cosine_threshold"] if threshold is None else threshold)
        self.window = timedelta(
            hours=float(cfg["window_hours"] if window_hours is None else window_hours)
        )
        # Stories are retained longer than the assignment window so novelty can look back
        # over impact.novelty.lookback_hours (72 h) even though assignment uses 48 h.
        lookback = load_config("impact")["novelty"]["lookback_hours"]
        retention = float(lookback if retention_hours is None else retention_hours)
        self.retention = max(self.window, timedelta(hours=retention))
        self.active: list[Story] = []
        self._mat: np.ndarray | None = None  # stacked centroids of active stories

    def _expire(self, now: datetime) -> None:
        keep = [s for s in self.active if now - s.last_seen <= self.retention]
        if len(keep) != len(self.active):
            self.active = keep
            self._mat = np.vstack([s.centroid for s in keep]) if keep else None

    def assign(self, doc_id: str, embedding: np.ndarray, ts: datetime) -> tuple[str, float, bool]:
        """Assign one document. Returns (event_id, similarity, is_new_story).

        Documents must arrive in non-decreasing time order.
        """
        self._expire(ts)
        if self._mat is not None:
            sims = self._mat @ embedding
            in_window = np.array([ts - s.last_seen <= self.window for s in self.active])
            sims = np.where(in_window, sims, -np.inf)
            j = int(np.argmax(sims))
            if sims[j] >= self.threshold:
                s = self.active[j]
                c = s.centroid * s.n_docs + embedding
                s.centroid = c / np.linalg.norm(c)
                s.n_docs += 1
                s.last_seen = ts
                s.doc_ids.append(doc_id)
                self._mat[j] = s.centroid
                return s.event_id, float(sims[j]), False
        story = Story(f"evt_{doc_id[:12]}", embedding.copy(), ts, ts, 1, [doc_id])
        self.active.append(story)
        self._mat = (
            story.centroid[None, :] if self._mat is None else np.vstack([self._mat, story.centroid])
        )
        return story.event_id, 1.0, True

    def novelty(self, embedding: np.ndarray, ts: datetime, lookback_hours: float) -> float:
        """1 - max cosine to stories seen within ``lookback_hours`` (before assignment)."""
        horizon = timedelta(hours=lookback_hours)
        recent = [s.centroid for s in self.active if ts - s.last_seen <= horizon]
        if not recent:
            return 1.0
        return float(max(0.0, 1.0 - np.max(np.vstack(recent) @ embedding)))
