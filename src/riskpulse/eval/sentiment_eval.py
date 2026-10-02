"""Sentiment evaluation on Twitter Financial News Sentiment (zeroshot, MIT).

Protocol (DECISIONS D-004, D-015):
- Test set = HF `valid` split (2,388 tweets). FinBERT never saw it; nothing is tuned on it.
- Each method maps a continuous score to 3 classes via a symmetric neutral band. We report
  (a) the method's conventional default band and (b) a band tuned on HF `train` only.
- FinBERT is also reported with plain argmax over its 3 probabilities.
- Majority-class (neutral) reference included.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from riskpulse.common.config import data_path, load_config, reports_path
from riskpulse.common.metrics import update_metrics
from riskpulse.engine.sentiment import (
    LABELS,
    FinBertScorer,
    LoughranMcDonaldScorer,
    VaderScorer,
)

HF_DIR = data_path("raw", "_downloads", "hf")
LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}  # bearish, bullish, neutral
CLASSES = ["negative", "neutral", "positive"]
GRID = np.round(np.arange(0.0, 0.96, 0.05), 2)
DEFAULT_BAND = {"finbert": None, "vader": 0.05, "lm": 0.0}  # finbert default from config


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(HF_DIR / f"sent_{name}.csv")
    df["gold"] = df["label"].map(LABEL_MAP)
    return df


def to_labels(scores: np.ndarray, band: float) -> np.ndarray:
    return np.where(scores > band, "positive", np.where(scores < -band, "negative", "neutral"))


def _metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    f1s = f1_score(gold, pred, labels=CLASSES, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "macro_f1": round(float(f1_score(gold, pred, labels=CLASSES, average="macro")), 4),
        "f1_per_class": {c: round(float(f), 4) for c, f in zip(CLASSES, f1s, strict=True)},
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": confusion_matrix(gold, pred, labels=CLASSES).tolist(),
        },
    }


def tune_band(scores: np.ndarray, gold: np.ndarray) -> float:
    """Neutral band maximising macro-F1 on the given (training) data."""
    best = max(GRID, key=lambda b: f1_score(gold, to_labels(scores, b), average="macro"))
    return float(best)


def _finbert_probs(scorer: FinBertScorer, texts: list[str], tag: str) -> tuple[np.ndarray, float]:
    """FinBERT probabilities with an on-disk cache keyed by model + texts."""
    h = hashlib.sha1((scorer.model_name + "\n".join(texts)).encode()).hexdigest()[:16]
    path = data_path("processed", "cache", f"finbert_{tag}_{h}.npy")
    if path.exists():
        return np.load(path), float("nan")
    t0 = time.perf_counter()
    p = scorer.probs(texts)
    secs = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, p)
    return p, len(texts) / secs


def run() -> dict:
    train, test = load_split("train"), load_split("valid")
    # Base FinBERT baseline table (the fine-tuned model's one test score is `sentiment_finetune`).
    fb = FinBertScorer(variant="finbert")
    p_train, _ = _finbert_probs(fb, train["text"].tolist(), "train")
    p_test, _ = _finbert_probs(fb, test["text"].tolist(), "valid")
    scores = {
        "finbert": (p_train[:, 0] - p_train[:, 1], p_test[:, 0] - p_test[:, 1]),
        "vader": tuple(np.array(VaderScorer().score(d["text"].tolist())) for d in (train, test)),
        "lm": tuple(
            np.array(LoughranMcDonaldScorer().score(d["text"].tolist())) for d in (train, test)
        ),
    }
    cfg_band = float(load_config("app")["sentiment"]["positive_threshold"])
    results: dict = {}
    gold_tr, gold_te = train["gold"].to_numpy(), test["gold"].to_numpy()
    for name, (s_tr, s_te) in scores.items():
        default = cfg_band if DEFAULT_BAND[name] is None else DEFAULT_BAND[name]
        tuned = tune_band(s_tr, gold_tr)
        results[name] = {
            "default_band": default,
            "default": _metrics(gold_te, to_labels(s_te, default)),
            "train_tuned_band": tuned,
            "train_tuned": _metrics(gold_te, to_labels(s_te, tuned)),
        }
    argmax = np.array(LABELS)[p_test.argmax(axis=1)]
    results["finbert"]["argmax"] = _metrics(gold_te, argmax)
    results["majority_class_neutral"] = _metrics(gold_te, np.full(len(gold_te), "neutral"))

    payload = {
        "dataset": "zeroshot/twitter-financial-news-sentiment",
        "test_split": "valid",
        "n_test": int(len(test)),
        "n_train_for_tuning": int(len(train)),
        "test_label_counts": test["gold"].value_counts().to_dict(),
        "results": results,
        # Throughput is measured in the pipeline benchmark (eval pipeline), not here: this
        # run may share the CPU with other jobs.
        "note": "Bands tuned on HF train only. FinBERT not evaluated on PhraseBank (leakage).",
    }
    update_metrics("sentiment", payload, script="riskpulse eval sentiment")
    _plot_confusions(results)
    return payload


def _plot_confusions(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [("finbert", "train_tuned"), ("vader", "train_tuned"), ("lm", "train_tuned")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (m, k) in zip(axes, methods, strict=True):
        cm = np.array(results[m][k]["confusion_matrix"]["matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "#1f2937",
                )
        ax.set_xticks(range(3), CLASSES, fontsize=9)
        ax.set_yticks(range(3), CLASSES, fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        f1 = results[m][k]["macro_f1"]
        ax.set_title(f"{m.upper()}  macro-F1 {f1:.3f}", fontsize=10, color="#1f2a44")
    fig.suptitle("Sentiment on Twitter Financial News (valid split, n=2,388)", fontsize=11)
    fig.tight_layout()
    out = reports_path("figures", "sentiment_confusion.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
