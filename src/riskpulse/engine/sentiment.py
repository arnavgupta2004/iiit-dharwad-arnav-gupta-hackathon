"""Sentiment (field 1): FinBERT document- and entity-level scores, plus lexicon baselines.

Document score  s = P(positive) - P(negative)  in [-1, 1] (spec §5.3).
Entity score    = relevance-weighted mean of FinBERT scores over the clauses that mention
                  the entity; falls back to the document score when no clause mentions it.
Baselines       = VADER compound score and Loughran-McDonald polarity (P - N) / (P + N).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np

from riskpulse.common.config import load_config, repo_root
from riskpulse.common.logging import get_logger
from riskpulse.engine.entities import MKT, EntityLinker

log = get_logger()

LABELS = ("positive", "negative", "neutral")


def torch_device() -> str:
    """Inference device: CPU by default (the project must run on CPU). ``RISKPULSE_DEVICE=mps``
    (or ``cuda``) is an opt-in accelerator for one-off batch precomputation; outputs match CPU
    within float tolerance (checked: max |dp| 5e-6, 100% argmax agreement on 512 tweets)."""
    return os.environ.get("RISKPULSE_DEVICE", "cpu")


def label_from_score(score: float, pos: float | None = None, neg: float | None = None) -> str:
    """Map a score in [-1, 1] to positive / negative / neutral using config thresholds."""
    cfg = load_config("app")["sentiment"]
    pos = cfg["positive_threshold"] if pos is None else pos
    neg = cfg["negative_threshold"] if neg is None else neg
    if score > pos:
        return "positive"
    if score < neg:
        return "negative"
    return "neutral"


def _has_weights(path) -> bool:
    return (path / "config.json").exists() and any(path.glob("*.safetensors"))


@cache
def resolve_model(variant: str | None = None) -> tuple[str, str]:
    """Weights to score with, as (name or path, origin). Logs the choice.

    ``finbert``: the base model. ``finetuned`` (D-043): Hugging Face Hub repo, else the local copy
    built by ``scripts/build_finetuned_sentiment.py`` (run automatically if
    ``RISKPULSE_BUILD_SENTIMENT=1``; about 25 CPU minutes), else the base model with a warning.
    """
    cfg = load_config("app")["sentiment"]
    variant = variant or cfg.get("variant", "finbert")
    base = cfg["model"]
    if variant == "finbert":
        log.info(f"sentiment model: base {base}")
        return base, "base"
    if variant != "finetuned":
        raise ValueError(f"unknown sentiment variant {variant!r}")
    ft = cfg["finetuned"]
    if ft.get("hub_repo"):
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(ft["hub_repo"])
            log.info(f"sentiment model: fine-tuned from the Hub ({ft['hub_repo']})")
            return path, "hub"
        except Exception as exc:  # offline, missing repo, rate limit
            log.warning(
                f"sentiment model: Hub download failed ({type(exc).__name__}); trying local"
            )
    local = repo_root() / ft["local_dir"]
    if not _has_weights(local) and os.environ.get("RISKPULSE_BUILD_SENTIMENT") == "1":
        from riskpulse.eval.finetune_sentiment import build

        log.info("sentiment model: rebuilding the fine-tuned model locally (train split only)")
        build()
    if _has_weights(local):
        log.info(f"sentiment model: fine-tuned, local copy {ft['local_dir']}")
        return str(local), "local"
    log.warning(
        "sentiment model: fine-tuned weights unavailable; FALLING BACK to base FinBERT. "
        "Run `python scripts/build_finetuned_sentiment.py` to rebuild them."
    )
    return base, "base_fallback"


@cache
def weights_fingerprint(variant: str | None = None) -> str:
    """Short hash identifying the sentiment weights (cache key). The same fine-tuned weights give
    the same key whether they came from the Hub or the local rebuild."""
    import hashlib
    from pathlib import Path

    name, origin = resolve_model(variant)
    h = hashlib.sha1()
    if origin in ("base", "base_fallback"):
        h.update(f"base|{name}".encode())
    else:
        root = Path(name)
        for f in sorted([root / "config.json", *root.glob("*.safetensors")]):
            with f.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 22), b""):
                    h.update(block)
    return h.hexdigest()[:10]


@dataclass(frozen=True)
class SentimentResult:
    score: float
    label: str
    p_pos: float
    p_neg: float
    p_neu: float


class FinBertScorer:
    """Batched CPU FinBERT scorer (ProsusAI/finbert, id2label verified as pos/neg/neu)."""

    def __init__(self, model_name: str | None = None, variant: str | None = None) -> None:
        cfg = load_config("app")["sentiment"]
        if model_name:
            self.model_name, self.origin = model_name, "explicit"
        else:
            self.model_name, self.origin = resolve_model(variant)
        self.batch_size = int(cfg["batch_size"])
        self.max_length = int(cfg["max_length"])
        self.device = torch_device()

    @cached_property
    def _model(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
        model = model.to(self.device)
        id2label = {i: lbl.lower() for i, lbl in model.config.id2label.items()}
        order = [next(i for i, lbl in id2label.items() if lbl == name) for name in LABELS]
        return tok, model, order

    @property
    def version(self) -> str:
        if self.origin in ("hub", "local"):
            return "finbert-tweets-ft"
        return f"finbert@{self.model_name}"

    def probs(self, texts: list[str]) -> np.ndarray:
        """Return an (n, 3) array of [P(pos), P(neg), P(neu)]."""
        import torch

        tok, model, order = self._model
        # Length-sorted batches minimise padding; results are restored to input order.
        idx = sorted(range(len(texts)), key=lambda k: len(texts[k]))
        texts_sorted = [texts[k] for k in idx]
        out = []
        for i in range(0, len(texts_sorted), self.batch_size):
            batch = [t if t else " " for t in texts_sorted[i : i + self.batch_size]]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = enc.to(self.device)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out.append(p[:, order])
        if not out:
            return np.zeros((0, 3))
        res = np.empty((len(texts), 3))
        res[idx] = np.vstack(out)
        return res

    def score(self, texts: list[str]) -> list[SentimentResult]:
        p = self.probs(texts)
        res = []
        for pos, neg, neu in p:
            s = float(pos - neg)
            res.append(SentimentResult(s, label_from_score(s), float(pos), float(neg), float(neu)))
        return res


def split_clauses(text: str) -> list[str]:
    """Split text into clauses on sentence ends and contrastive connectives."""
    pattern = load_config("app")["sentiment"]["clause_split_regex"]
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p.strip(" ,") for p in parts if p and p.strip(" ,")]


def entity_sentiment(
    text: str,
    tickers: list[str],
    scorer: FinBertScorer,
    linker: EntityLinker,
    doc_score: float | None = None,
) -> dict[str, float]:
    """Entity-level scores: FinBERT over clauses mentioning each ticker, relevance-weighted.

    A ticker never mentioned in any clause (e.g. linked via a dataset label) gets the
    document score. MKT always gets the document score.
    """
    clauses = split_clauses(text)
    if doc_score is None:
        doc_score = scorer.score([text])[0].score
    if len(clauses) <= 1:
        return dict.fromkeys(tickers, doc_score)
    clause_links = [
        {m.ticker: m.relevance for m in linker.link(c) if m.ticker != MKT} for c in clauses
    ]
    needed = sorted({i for i, links in enumerate(clause_links) if set(links) & set(tickers)})
    scores = dict(
        zip(needed, (r.score for r in scorer.score([clauses[i] for i in needed])), strict=True)
    )
    out = {}
    for t in tickers:
        pairs = [(scores[i], clause_links[i][t]) for i in needed if t in clause_links[i]]
        if not pairs or t == MKT:
            out[t] = doc_score
            continue
        w = sum(r for _, r in pairs)
        out[t] = float(sum(s * r for s, r in pairs) / w) if w > 0 else doc_score
    return out


# ---------------- baselines ----------------
class VaderScorer:
    """VADER compound score in [-1, 1] (social-media lexicon baseline)."""

    @cached_property
    def _analyzer(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()

    def score(self, texts: list[str]) -> list[float]:
        return [float(self._analyzer.polarity_scores(t)["compound"]) for t in texts]


@cache
def _lm_lists() -> tuple[frozenset[str], frozenset[str]]:
    import pandas as pd
    import pysentiment2

    path = os.path.join(os.path.dirname(pysentiment2.__file__), "static", "LM.csv")
    d = pd.read_csv(path, usecols=["Word", "Positive", "Negative"])
    pos = frozenset(d.loc[d["Positive"] != 0, "Word"].str.lower())
    neg = frozenset(d.loc[d["Negative"] != 0, "Word"].str.lower())
    return pos, neg


_TOKEN = re.compile(r"[a-z]+")


class LoughranMcDonaldScorer:
    """Loughran-McDonald polarity (P - N) / (P + N); 0 when no sentiment words."""

    def score(self, texts: list[str]) -> list[float]:
        pos, neg = _lm_lists()
        out = []
        for t in texts:
            toks = _TOKEN.findall(t.lower())
            p = sum(tok in pos for tok in toks)
            n = sum(tok in neg for tok in toks)
            out.append((p - n) / (p + n) if p + n else 0.0)
        return out
