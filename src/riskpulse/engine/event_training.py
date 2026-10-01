"""Build the weak-label training set, train the primary event classifier, evaluate (provisional).

Training sources (never the gold set):
  hf_topic  - zeroshot/twitter-financial-news-topic *train*, mapped via taxonomy `hf_topic_map`
  seed      - taxonomy example sentences (synthetic, written for the config)
  kw        - feed headlines with keyword hits for exactly one non-OTHER class
  zs        - feed headlines whose zero-shot confidence >= zero_shot_min_conf
Provisional evaluation (until Arnav's gold labels exist):
  hf_topic_valid - human-labelled, 7 of 10 classes, disjoint split from training
  weak_holdout   - held-out slice of kw + zs labels (all classes; weak by construction)
"""

from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from riskpulse.common.config import data_path, load_config, repo_root
from riskpulse.common.logging import get_logger
from riskpulse.common.metrics import update_metrics
from riskpulse.engine.events import (
    CLASSES,
    Embedder,
    EmbeddingClassifier,
    KeywordClassifier,
    ZeroShotClassifier,
)
from riskpulse.engine.pipeline import EVENT_MODEL_PATH

log = get_logger()
HF_DIR = data_path("raw", "_downloads", "hf")


def hf_topic(split: str) -> pd.DataFrame:
    tax = load_config("taxonomy")
    names = tax["hf_topic_labels"]
    m = {t: c for c, spec in tax["classes"].items() for t in spec["hf_topic_map"]}
    df = pd.read_csv(HF_DIR / f"topic_{split}.csv")
    df["label"] = df["label"].map(names).map(m)
    return df.dropna(subset=["label"])[["text", "label"]].assign(source=f"hf_topic_{split}")


def seed_examples() -> pd.DataFrame:
    tax = load_config("taxonomy")
    rows = [(ex, c) for c, spec in tax["classes"].items() for ex in spec["examples"]]
    return pd.DataFrame(rows, columns=["text", "label"]).assign(source="seed")


def feed_headlines(n: int, seed: int) -> pd.DataFrame:
    path = repo_root() / load_config("app")["paths"]["replay_feed"]
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            texts.append(d["title"] or d["text"])
    s = pd.Series(texts).drop_duplicates()
    return pd.DataFrame({"text": s.sample(min(n, len(s)), random_state=seed).to_numpy()})


def sample_per_group(df: pd.DataFrame, col: str, k: int, seed: int) -> pd.DataFrame:
    """Up to k rows per group, keeping the group column (pandas 3 groupby.apply drops it)."""
    parts = [g.sample(min(k, len(g)), random_state=seed) for _, g in df.groupby(col)]
    return pd.concat(parts) if parts else df.iloc[0:0]


def load_gold_texts() -> set[str]:
    """Texts in the gold labelling file (excluded from all training data)."""
    path = data_path("gold", "to_label.csv")
    return set(pd.read_csv(path)["text"]) if path.exists() else set()


def keyword_labels(texts: pd.Series, cap: int, seed: int) -> pd.DataFrame:
    kw = KeywordClassifier()
    rows = []
    for t in texts:
        h = {c: n for c, n in kw.hits(t).items() if n > 0}
        if len(h) == 1:
            (c,) = h
            if c != "OTHER":
                rows.append((t, c))
    df = pd.DataFrame(rows, columns=["text", "label"])
    return sample_per_group(df, "label", cap, seed).assign(source="kw")


def zero_shot_labels(texts: pd.Series, min_conf: float) -> pd.DataFrame:
    key = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:16]
    cache = data_path("processed", "cache", f"zeroshot_feed_{key}.parquet")
    if cache.exists():
        z = pd.read_parquet(cache)
    else:
        preds = ZeroShotClassifier().predict(list(texts))
        z = pd.DataFrame(
            {
                "text": list(texts),
                "label": [p.event_class for p in preds],
                "conf": [p.confidence for p in preds],
            }
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        z.to_parquet(cache)
    return z[z["conf"] >= min_conf][["text", "label"]].assign(source="zs")


def _scores(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    return {
        "n": len(gold),
        "macro_f1": round(
            float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)), 4
        ),
        "f1_per_class": {
            c: round(float(f), 4)
            for c, f in zip(
                labels,
                f1_score(gold, pred, labels=labels, average=None, zero_division=0),
                strict=True,
            )
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
        },
    }


def train_and_evaluate() -> dict:
    cfg = load_config("app")["event_training"]
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    hf_tr = hf_topic("train")
    other = hf_tr[hf_tr["label"] == "OTHER"]
    hf_tr = pd.concat(
        [
            hf_tr[hf_tr["label"] != "OTHER"],
            other.sample(min(int(cfg["hf_other_cap"]), len(other)), random_state=seed),
        ]
    )
    feed = feed_headlines(int(cfg["feed_sample_for_keywords"]), seed)
    feed = feed[~feed["text"].isin(load_gold_texts())]
    kw = keyword_labels(feed["text"], int(cfg["keyword_cap_per_class"]), seed)
    gold_texts = load_gold_texts()
    pool = feed["text"][~feed["text"].isin(set(kw["text"]) | gold_texts)]
    zs_pool = pool.sample(int(cfg["zero_shot_sample"]), random_state=seed)
    zs = zero_shot_labels(zs_pool.reset_index(drop=True), float(cfg["zero_shot_min_conf"]))
    weak = pd.concat([kw, zs], ignore_index=True).drop_duplicates("text")
    hold_mask = rng.random(len(weak)) < float(cfg["weak_holdout_frac"])
    weak_train, weak_hold = weak[~hold_mask], weak[hold_mask]

    train = pd.concat([hf_tr, seed_examples(), weak_train], ignore_index=True).drop_duplicates(
        "text"
    )
    if train["label"].isna().any():
        raise ValueError("Training rows without labels; check weak-label construction")
    # The gold set is a TEST set: never train (or hold out weak labels) on its texts.
    n_before = len(train)
    train = train[~train["text"].isin(gold_texts)]
    weak_hold = weak_hold[~weak_hold["text"].isin(gold_texts)]
    log.info(f"Excluded {n_before - len(train)} gold texts from training")
    assert not train["text"].isin(gold_texts).any(), "gold text leaked into training"
    train.to_parquet(data_path("processed", "event_training_set.parquet"), index=False)
    by_src = train["source"].value_counts().to_dict()
    log.info(f"Event training set: {len(train)} rows; by source {by_src}")

    clf = EmbeddingClassifier(Embedder()).fit(
        train["text"].tolist(), train["label"].tolist(), seed=seed
    )
    clf.save(repo_root() / EVENT_MODEL_PATH)

    hf_va = hf_topic("valid")
    hf_labels = sorted(hf_va["label"].unique())
    kwc = KeywordClassifier()
    results: dict = {}
    for name, df, labels in (
        ("hf_topic_valid", hf_va, hf_labels),
        ("weak_holdout", weak_hold, CLASSES),
    ):
        gold = df["label"].tolist()
        results[name] = {
            "keyword_baseline": _scores(
                gold, [p.event_class for p in kwc.predict(df["text"].tolist())], labels
            ),
            "primary_embed_lr": _scores(
                gold, [p.event_class for p in clf.predict(df["text"].tolist())], labels
            ),
        }
    # Zero-shot is slow on CPU: scored on a stratified subsample of HF-topic valid.
    n_zs = int(cfg["zero_shot_eval_sample"])
    sub = sample_per_group(hf_va, "label", max(1, n_zs // len(hf_labels)), seed)
    zsc = ZeroShotClassifier()
    gold = sub["label"].tolist()
    results["hf_topic_valid_subsample"] = {
        "zero_shot": _scores(
            gold, [p.event_class for p in zsc.predict(sub["text"].tolist())], hf_labels
        ),
        "keyword_baseline": _scores(
            gold, [p.event_class for p in kwc.predict(sub["text"].tolist())], hf_labels
        ),
        "primary_embed_lr": _scores(
            gold, [p.event_class for p in clf.predict(sub["text"].tolist())], hf_labels
        ),
    }
    payload = {
        "status": "PROVISIONAL - final evaluation on Arnav's gold set pending",
        "training_rows": int(len(train)),
        "training_rows_by_source": train["source"].value_counts().to_dict(),
        "training_rows_by_class": train["label"].value_counts().to_dict(),
        "model_version": clf.version,
        "min_confidence_for_non_other": clf.min_confidence,
        "results": results,
        "notes": [
            "hf_topic_valid covers 7 of 10 classes "
            "(no CREDIT_EVENT, PRODUCT_LAUNCH, OPERATIONAL_ESG).",
            "The primary model trains on hf_topic *train*, so hf_topic_valid is in-distribution "
            "for it but not for the keyword baseline or zero-shot.",
            "weak_holdout labels come from keyword rules and zero-shot, so they favour those "
            "methods' biases.",
        ],
    }
    update_metrics("events", payload, script="riskpulse eval events")
    return payload
