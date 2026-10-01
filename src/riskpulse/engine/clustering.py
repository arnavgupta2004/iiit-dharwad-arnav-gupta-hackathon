"""Online story clustering (spec §5.6).

Each incoming document (with a unit-norm embedding) joins the most similar *active* story if
cosine(doc, story centroid) >= threshold, else it starts a new story. A story accepts new
documents while its last document is within ``window_hours``; stories are retained for
``retention_hours`` (>= window) so novelty can look back over the impact lookback.
Centroids are running means, re-normalised. Story ids: ``evt_`` + first 12 hex chars of the
founding document id.

Implementation: centroids, last-seen times and counts live in growable numpy arrays, and expiry
compaction runs at most once per ``compact_every`` of stream time, so each assignment is one
matrix-vector product.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from riskpulse.common.config import load_config


class StoryClusterer:
    """Greedy single-pass clusterer over a rolling time window."""

    def __init__(
        self,
        threshold: float | None = None,
        window_hours: float | None = None,
        retention_hours: float | None = None,
        compact_every: timedelta = timedelta(minutes=30),
    ) -> None:
        cfg = load_config("app")["clustering"]
        self.threshold = float(cfg["cosine_threshold"] if threshold is None else threshold)
        self.window = float(cfg["window_hours"] if window_hours is None else window_hours) * 3600
        lookback = load_config("impact")["novelty"]["lookback_hours"]
        retention = float(lookback if retention_hours is None else retention_hours) * 3600
        self.retention = max(self.window, retention)
        self.compact_every = compact_every.total_seconds()
        self._dim: int | None = None
        self._c = np.zeros((0, 0))  # centroids
        self._last = np.zeros(0)  # last-seen epoch seconds
        self._n = np.zeros(0, dtype=int)
        self._ids: list[str] = []
        self._size = 0
        self._last_compact = -np.inf

    def _ensure(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            self._c = np.zeros((256, dim), dtype=np.float32)
            self._last = np.zeros(256)
            self._n = np.zeros(256, dtype=int)

    def _append(self, event_id: str, emb: np.ndarray, t: float) -> None:
        if self._size == len(self._last):
            cap = 2 * len(self._last)
            self._c = np.vstack([self._c, np.zeros((cap - len(self._c), self._dim), np.float32)])
            self._last = np.concatenate([self._last, np.zeros(cap - len(self._last))])
            self._n = np.concatenate([self._n, np.zeros(cap - len(self._n), dtype=int)])
        i = self._size
        self._c[i], self._last[i], self._n[i] = emb, t, 1
        self._ids.append(event_id)
        self._size += 1

    def _compact(self, now: float) -> None:
        if now - self._last_compact < self.compact_every:
            return
        self._last_compact = now
        keep = np.nonzero(now - self._last[: self._size] <= self.retention)[0]
        if len(keep) == self._size:
            return
        k = len(keep)
        self._c[:k] = self._c[keep]
        self._last[:k] = self._last[keep]
        self._n[:k] = self._n[keep]
        self._ids = [self._ids[j] for j in keep]
        self._size = k

    @property
    def active(self) -> list[str]:
        return list(self._ids)

    def assign(self, doc_id: str, embedding: np.ndarray, ts: datetime) -> tuple[str, float, bool]:
        """Assign one document. Returns (event_id, similarity, is_new_story).

        Documents must arrive in non-decreasing time order.
        """
        event_id, sim, new, _ = self.observe(doc_id, embedding, ts, lookback_hours=None)
        return event_id, sim, new

    def novelty(self, embedding: np.ndarray, ts: datetime, lookback_hours: float) -> float:
        """1 - max cosine to stories seen within ``lookback_hours`` (no state change)."""
        if not self._size:
            return 1.0
        t = ts.timestamp()
        sims = self._c[: self._size] @ embedding.astype(np.float32)
        recent = t - self._last[: self._size] <= lookback_hours * 3600
        return float(max(0.0, 1.0 - sims[recent].max())) if recent.any() else 1.0

    def observe(
        self, doc_id: str, embedding: np.ndarray, ts: datetime, lookback_hours: float | None
    ) -> tuple[str, float, bool, float]:
        """Novelty (before assignment) and assignment with one similarity computation.

        Returns (event_id, similarity, is_new_story, novelty).
        """
        t = ts.timestamp()
        emb = embedding.astype(np.float32)
        self._ensure(len(emb))
        self._compact(t)
        novelty = 1.0
        if self._size:
            age = t - self._last[: self._size]
            sims = self._c[: self._size] @ emb
            if lookback_hours is not None:
                recent = age <= lookback_hours * 3600
                if recent.any():
                    novelty = float(max(0.0, 1.0 - sims[recent].max()))
            sims = np.where(age <= self.window, sims, -np.inf)
            j = int(np.argmax(sims))
            if sims[j] >= self.threshold:
                c = self._c[j] * self._n[j] + emb
                self._c[j] = c / np.linalg.norm(c)
                self._n[j] += 1
                self._last[j] = t
                return self._ids[j], float(sims[j]), False, novelty
        event_id = f"evt_{doc_id[:12]}"
        self._append(event_id, emb, t)
        return event_id, 1.0, True, novelty
