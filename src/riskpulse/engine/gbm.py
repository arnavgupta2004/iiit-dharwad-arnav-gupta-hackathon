"""Dependency-free evaluator for a LightGBM regression model exported with ``dump_model()``.

Why: on macOS, LightGBM's and PyTorch's OpenMP runtimes cannot share a process (one order hangs,
the other segfaults; reproduced 2026-10-02). The engine, API and evals run torch models, so they
never import lightgbm: the model is trained in a spawned subprocess (``eval.impact_v2.train``),
exported as JSON, and evaluated here with plain Python/NumPy. Only numerical ``<=`` splits are
supported (all impact-v2 features are numeric and never missing).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TreeEnsemble:
    """Sum of regression trees; ``predict`` matches ``lightgbm.Booster.predict`` (raw score)."""

    def __init__(self, trees: list[dict], n_features: int) -> None:
        self.n_features = n_features
        # Flatten each tree to parallel lists: (feature, threshold, left, right, value); leaves
        # have feature -1. Node 0 is the root.
        self.flat = [self._flatten(t["tree_structure"]) for t in trees]

    @staticmethod
    def _flatten(root: dict) -> tuple[list[int], list[float], list[int], list[int], list[float]]:
        feat, thr, left, right, val = [], [], [], [], []

        def add(node: dict) -> int:
            i = len(feat)
            feat.append(-1)
            thr.append(0.0)
            left.append(-1)
            right.append(-1)
            val.append(0.0)
            if "leaf_value" in node:
                val[i] = float(node["leaf_value"])
                return i
            if node.get("decision_type", "<=") != "<=":
                raise ValueError(f"unsupported split {node.get('decision_type')}")
            feat[i] = int(node["split_feature"])
            thr[i] = float(node["threshold"])
            left[i] = add(node["left_child"])
            right[i] = add(node["right_child"])
            return i

        add(root)
        return feat, thr, left, right, val

    @classmethod
    def from_dump(cls, dump: dict) -> TreeEnsemble:
        return cls(dump["tree_info"], int(dump["max_feature_idx"]) + 1)

    @classmethod
    def load(cls, path: Path) -> TreeEnsemble:
        return cls.from_dump(json.loads(Path(path).read_text()))

    def predict_one(self, x) -> float:
        """Single row (fast path for the streaming engine)."""
        total = 0.0
        for feat, thr, left, right, val in self.flat:
            i = 0
            while feat[i] >= 0:
                i = left[i] if x[feat[i]] <= thr[i] else right[i]
            total += val[i]
        return total

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Batch prediction, vectorised per tree."""
        x = np.asarray(x, dtype=float)
        out = np.zeros(len(x))
        for feat, thr, left, right, val in self.flat:
            f, t = np.array(feat), np.array(thr)
            lft, rgt, v = np.array(left), np.array(right), np.array(val)
            idx = np.zeros(len(x), dtype=int)
            active = f[idx] >= 0
            while active.any():
                rows = np.nonzero(active)[0]
                node = idx[rows]
                go_left = x[rows, f[node]] <= t[node]
                idx[rows] = np.where(go_left, lft[node], rgt[node])
                active = f[idx] >= 0
            out += v[idx]
        return out
