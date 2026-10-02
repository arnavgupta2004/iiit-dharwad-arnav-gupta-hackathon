"""Pre-registered sentiment check on live-feed text (D-041): FinBERT vs the fine-tuned model.

Protocol fixed in DECISIONS.md D-041 before the gold labels existed:
- items: data/gold/labels.csv, split by source into news headlines (gdelt) and tweets;
- prediction (both models): entity-level score for the first ticker in ``linked_tickers``
  (clause-level ``entity_sentiment`` when 2+ companies are linked, document score otherwise and
  for MKT), score = P(pos) - P(neg), labelled with the configured +-0.15 band;
- metric: macro-F1; paired bootstrap of F1(fine-tuned) - F1(FinBERT), 2,000 resamples per
  subset, seed 20261002, 95% percentile CI;
- decision: fine-tuned everywhere unless the news CI lies entirely below 0, in which case FinBERT
  for news and fine-tuned for tweets.
VADER and Loughran-McDonald (their train-tuned bands) are reported for reference only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from riskpulse.common.config import data_path
from riskpulse.common.metrics import load_metrics, update_metrics
from riskpulse.engine.entities import MKT, EntityLinker
from riskpulse.engine.sentiment import (
    FinBertScorer,
    LoughranMcDonaldScorer,
    VaderScorer,
    entity_sentiment,
    label_from_score,
)
from riskpulse.eval.sentiment_eval import CLASSES

N_BOOT, SEED = 2000, 20261002
SUBSETS = {"news_headlines": "gdelt", "tweets": "kaggle_tweets"}


def load_gold() -> pd.DataFrame:
    df = pd.read_csv(data_path("gold", "labels.csv"))
    df["tickers"] = df["linked_tickers"].astype(str).str.split(",")
    df["gold"] = df["label_sentiment"].str.strip().str.lower()
    return df


def model_scores(df: pd.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """(deployed-rule entity score for the first ticker, document argmax label) per item."""
    fb, linker = FinBertScorer(variant=variant), EntityLinker()
    docs = fb.score(df["text"].tolist())
    ent_scores, argmax = [], []
    for text, tickers, d in zip(df["text"], df["tickers"], docs, strict=True):
        first = tickers[0]
        companies = [t for t in tickers if t != MKT]
        if first != MKT and len(companies) >= 2:
            ent_scores.append(entity_sentiment(text, companies, fb, linker, d.score)[first])
        else:
            ent_scores.append(d.score)
        argmax.append(
            ["positive", "negative", "neutral"][int(np.argmax([d.p_pos, d.p_neg, d.p_neu]))]
        )
    return np.array(ent_scores), np.array(argmax)


def macro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(gold, pred, labels=CLASSES, average="macro", zero_division=0))


def paired_ci(gold: np.ndarray, a: np.ndarray, b: np.ndarray) -> list[float]:
    """95% percentile CI of macro-F1(a) - macro-F1(b) by paired bootstrap."""
    rng = np.random.default_rng(SEED)
    n = len(gold)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        diffs.append(macro_f1(gold[i], a[i]) - macro_f1(gold[i], b[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def run() -> dict:
    df = load_gold()
    preds, argmaxes = {}, {}
    for v in ("finbert", "finetuned"):
        s, am = model_scores(df, v)
        preds[v] = np.array([label_from_score(x) for x in s])
        argmaxes[v] = am
    bands = load_metrics()["sentiment"]["results"]
    texts = df["text"].tolist()
    for name, scorer in (("vader", VaderScorer()), ("lm", LoughranMcDonaldScorer())):
        b = float(bands[name]["train_tuned_band"])
        preds[name] = np.array([label_from_score(x, b, -b) for x in scorer.score(texts)])
    out: dict = {}
    for subset, src in SUBSETS.items():
        m = (df["source"] == src).to_numpy()
        g = df["gold"].to_numpy()[m]
        out[subset] = {
            "n": int(m.sum()),
            "label_counts": pd.Series(g).value_counts().to_dict(),
            "macro_f1": {k: round(macro_f1(g, p[m]), 4) for k, p in preds.items()},
            "macro_f1_argmax_secondary": {
                k: round(macro_f1(g, p[m]), 4) for k, p in argmaxes.items()
            },
            "diff_finetuned_minus_finbert": round(
                macro_f1(g, preds["finetuned"][m]) - macro_f1(g, preds["finbert"][m]), 4
            ),
            "diff_95ci": paired_ci(g, preds["finetuned"][m], preds["finbert"][m]),
        }
    news_worse = out["news_headlines"]["diff_95ci"][1] < 0
    decision = (
        {"news": "finbert", "tweets": "finetuned"}
        if news_worse
        else {"news": "finetuned", "tweets": "finetuned"}
    )
    ft = load_metrics().get("sentiment_finetune", {})
    payload = {
        "protocol": "pre-registered in DECISIONS.md D-041 before the gold labels were scored",
        "prediction_rule": "entity score, first linked ticker, +-0.15 band (deployed rule)",
        "results": out,
        "decision_rule": "fine-tuned everywhere unless the news-headline CI of "
        "F1(fine-tuned) - F1(FinBERT) lies entirely below 0",
        "news_ci_entirely_below_zero": bool(news_worse),
        "decision": decision,
        "in_domain_test_macro_f1_finetuned": ft.get("test_macro_f1"),
        "labels": {
            "in_domain": "HF twitter-financial-news-sentiment test split (training's dataset)",
            "live_feed": "gold spot check: GDELT headlines and equinxx tweets from the replay",
        },
    }
    update_metrics("sentiment_gold", payload, script="riskpulse eval sentiment_gold")
    return payload


def run_news_pooled() -> dict:
    """D-057: pooled gold-1 + gold-2 news check, computed once (refuses a second run)."""
    import json

    from riskpulse.common.config import reports_path

    out_path = reports_path("sentiment_news_pooled.json")
    if out_path.exists():
        raise RuntimeError("D-057 check already computed; it is run once")
    parts = []
    for name in ("labels.csv", "labels_2.csv"):
        g = pd.read_csv(data_path("gold", name))
        parts.append(g[g["source"] == "gdelt"].assign(gold_file=name))
    df = pd.concat(parts, ignore_index=True)
    df["tickers"] = df["linked_tickers"].astype(str).str.split(",")
    gold = df["label_sentiment"].str.strip().str.lower().to_numpy()
    preds = {}
    for v in ("finetuned", "finbert"):
        s, _ = model_scores(df, v)
        preds[v] = np.array([label_from_score(x) for x in s])
    ci = paired_ci(gold, preds["finetuned"], preds["finbert"])
    below = bool(ci[1] < 0)
    payload = {
        "protocol": "pre-registered in DECISIONS.md D-057 (commit 251a8e7) before computing",
        "n": int(len(df)),
        "n_by_file": df["gold_file"].value_counts().to_dict(),
        "macro_f1": {k: round(macro_f1(gold, p), 4) for k, p in preds.items()},
        "diff_finetuned_minus_finbert": round(
            macro_f1(gold, preds["finetuned"]) - macro_f1(gold, preds["finbert"]), 4
        ),
        "diff_95ci": ci,
        "ci_entirely_below_zero": below,
        "decision": {"news": "finbert", "tweets": "finetuned"}
        if below
        else {"news": "finetuned", "tweets": "finetuned"},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    update_metrics("sentiment_news_pooled", payload, script="riskpulse eval sentiment_news_pooled")
    return payload
