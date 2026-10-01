"""Event classification (field 2): keyword baseline, zero-shot NLI, embedding classifier.

Layers (spec §5.4):
1. ``KeywordClassifier``: taxonomy keyword hits; naive baseline and high-precision weak labeller.
2. ``ZeroShotClassifier``: NLI zero-shot over the taxonomy's label descriptions.
3. ``EmbeddingClassifier`` (primary): logistic regression on MiniLM sentence embeddings, trained on
   weak labels + taxonomy seed examples, with isotonic-calibrated probabilities; predictions
   below ``min_confidence`` become OTHER.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from riskpulse.common.config import load_config
from riskpulse.common.schemas import EventClass

CLASSES: list[str] = [c.value for c in EventClass]
# Tie-break order for the keyword baseline: more market-moving classes first.
PRIORITY: list[str] = [
    "CREDIT_EVENT",
    "GEOPOLITICAL",
    "MACROECONOMIC",
    "MERGER_ACQUISITION",
    "REGULATORY_LEGAL",
    "EARNINGS",
    "OPERATIONAL_ESG",
    "MANAGEMENT_CHANGE",
    "PRODUCT_LAUNCH",
    "OTHER",
]


@dataclass(frozen=True)
class EventPrediction:
    event_class: str
    confidence: float
    probs: dict[str, float]


class KeywordClassifier:
    """Counts taxonomy keyword hits per class; most hits wins (priority breaks ties)."""

    version = "keywords@v1"

    def __init__(self, taxonomy: dict | None = None) -> None:
        tax = taxonomy or load_config("taxonomy")
        self.patterns: dict[str, re.Pattern] = {}
        for cls, spec in tax["classes"].items():
            kws = sorted(set(spec.get("keywords", [])), key=len, reverse=True)
            if kws:
                body = "|".join(re.escape(k) for k in kws)
                self.patterns[cls] = re.compile(rf"(?<![\w-])(?:{body})(?![\w-])", re.IGNORECASE)

    def hits(self, text: str) -> dict[str, int]:
        return {c: len(p.findall(text)) for c, p in self.patterns.items()}

    def predict(self, texts: list[str]) -> list[EventPrediction]:
        out = []
        for t in texts:
            h = {c: n for c, n in self.hits(t).items() if n > 0}
            if not h:
                out.append(EventPrediction("OTHER", 0.0, {"OTHER": 1.0}))
                continue
            best = max(h, key=lambda c: (h[c], -PRIORITY.index(c)))
            total = sum(h.values())
            out.append(EventPrediction(best, h[best] / total, {c: n / total for c, n in h.items()}))
        return out


class ZeroShotClassifier:
    """NLI zero-shot classifier over taxonomy label descriptions (single-label)."""

    def __init__(self, model_name: str | None = None, taxonomy: dict | None = None) -> None:
        self.model_name = model_name or load_config("app")["events"]["zero_shot_model"]
        tax = taxonomy or load_config("taxonomy")
        self.label_text = {c: s["zero_shot_label"] for c, s in tax["classes"].items()}
        self.text_to_class = {v: k for k, v in self.label_text.items()}

    @property
    def version(self) -> str:
        return f"zeroshot@{self.model_name}"

    @cached_property
    def _pipe(self):
        from transformers import pipeline

        return pipeline("zero-shot-classification", model=self.model_name, device=-1)

    def predict(self, texts: list[str], batch_size: int = 8) -> list[EventPrediction]:
        res = self._pipe(
            texts,
            candidate_labels=list(self.label_text.values()),
            hypothesis_template="This news is about {}.",
            multi_label=False,
            batch_size=batch_size,
        )
        if isinstance(res, dict):
            res = [res]
        out = []
        for r in res:
            probs = {
                self.text_to_class[lbl]: float(s)
                for lbl, s in zip(r["labels"], r["scores"], strict=True)
            }
            best = max(probs, key=probs.get)
            out.append(EventPrediction(best, probs[best], probs))
        return out


class Embedder:
    """Sentence embeddings (all-MiniLM-L6-v2, 384-d, L2-normalised)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or load_config("app")["events"]["embedding_model"]

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )


class EmbeddingClassifier:
    """Calibrated logistic regression on sentence embeddings; low confidence -> OTHER."""

    def __init__(self, embedder: Embedder | None = None, min_confidence: float | None = None):
        self.embedder = embedder or Embedder()
        cfg = load_config("app")["events"]
        self.min_confidence = cfg["min_confidence"] if min_confidence is None else min_confidence
        self.model = None
        self.classes_: list[str] = []
        self.version = "embed-lr@untrained"

    def fit(self, texts: list[str], labels: list[str], seed: int = 0) -> EmbeddingClassifier:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression

        x = self.embedder.encode(texts)
        base = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed)
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.model.fit(x, labels)
        self.classes_ = list(self.model.classes_)
        self.version = f"embed-lr@{self.embedder.model_name}"
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(self.embedder.encode(texts))

    def predict(self, texts: list[str]) -> list[EventPrediction]:
        return self.predict_from_proba(self.predict_proba(texts))

    def predict_from_embeddings(self, x: np.ndarray) -> list[EventPrediction]:
        """Classify precomputed embeddings (avoids re-encoding in the pipeline)."""
        return self.predict_from_proba(self.model.predict_proba(x))

    def predict_from_proba(self, p: np.ndarray) -> list[EventPrediction]:
        out = []
        for row in p:
            i = int(row.argmax())
            cls, conf = self.classes_[i], float(row[i])
            probs = {c: float(v) for c, v in zip(self.classes_, row, strict=True)}
            if conf < self.min_confidence:
                cls = "OTHER"
            out.append(EventPrediction(cls, conf, probs))
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {"model": self.model, "classes": self.classes_, "version": self.version}, fh
            )

    @classmethod
    def load(cls, path: Path, embedder: Embedder | None = None) -> EmbeddingClassifier:
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls(embedder=embedder)
        obj.model, obj.classes_, obj.version = blob["model"], blob["classes"], blob["version"]
        return obj
