"""Create data/gold/to_label_2.csv: 200 replay items, the round-2 event-classification test (D-052).

Pool: replay-feed items (25-280 characters, unique text) that are
- not in gold-1 (doc id or text), not a near-duplicate of a gold-1 text (token-set ratio >= 90)
  and not from the same story as a gold-1 item (MiniLM cosine >= 0.80, the clustering threshold);
- not in the event classifier's training set (its weak labels were drawn from the feed).
Selection (seeded): 160 news + 40 tweets. First, at least 5 items per class predicted by the keyword
baseline and at least 5 per class predicted by the deployed classifier (where available), then the
rest at random within source x month strata (proportional to the pool). No two picks are
near-duplicates or same-story. Predicted classes are kept out of the labelling file (no anchoring);
they are written to data/processed/gold2_sampling_meta.parquet (gitignored) for the record.

    python scripts/make_gold2.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from riskpulse.common.config import data_path, repo_root
from riskpulse.engine.batch import _base_key, _cache, _event_key, _feed_hash
from riskpulse.engine.events import CLASSES, KeywordClassifier

sys.path.insert(0, str(repo_root()))
from scripts.make_gold_template import load_feed  # noqa: E402

SEED = 20261003
N_NEWS, N_TWEETS = 160, 40
MIN_PER_CLASS = 5
FUZZ, COS = 90.0, 0.80


def main() -> None:
    rng = np.random.default_rng(SEED)
    feed = load_feed()
    fk = _feed_hash()
    base = pd.read_parquet(_cache(f"stage1base_{fk}_{_base_key()}.parquet"), columns=["doc_id"])
    emb = np.load(_cache(f"stage1base_{fk}_{_base_key()}_emb.npy")).astype(np.float32)
    ev = pd.read_parquet(_cache(f"stage1event_{fk}_{_event_key()}.parquet"))
    assert (base["doc_id"].to_numpy() == feed["doc_id"].to_numpy()).all()
    assert (ev["doc_id"].to_numpy() == feed["doc_id"].to_numpy()).all()
    feed["row"] = np.arange(len(feed))
    feed["model_class"] = ev["event_class"].to_numpy()

    g1 = pd.read_csv(data_path("gold", "to_label.csv"))
    train_texts = set(pd.read_parquet(data_path("processed", "event_training_set.parquet"))["text"])
    pool = feed[feed["text"].str.len().between(25, 280)].drop_duplicates("text")
    n0 = len(pool)
    pool = pool[~pool["doc_id"].isin(set(g1["doc_id"])) & ~pool["text"].isin(set(g1["text"]))]
    pool = pool[~pool["text"].isin(train_texts)]
    n_excl_train = n0 - len(pool)
    # Same story as a gold-1 item: embedding cosine >= 0.80 (embeddings are L2-normalised).
    g1_rows = feed.set_index("doc_id").loc[g1["doc_id"], "row"].to_numpy()
    sims = emb[pool["row"].to_numpy()] @ emb[g1_rows].T
    pool = pool[sims.max(axis=1) < COS]
    # Near-duplicate text of a gold-1 item.
    fz = process.cdist(
        pool["text"].tolist(),
        g1["text"].tolist(),
        scorer=fuzz.token_set_ratio,
        processor=default_process,
        workers=-1,
        dtype=np.uint8,
    )
    pool = pool[fz.max(axis=1) < FUZZ].copy()
    pool["kw_class"] = [p.event_class for p in KeywordClassifier().predict(pool["text"].tolist())]
    pool["month"] = pd.to_datetime(pool["published_at"], utc=True).dt.strftime("%Y-%m")
    pool = pool.iloc[rng.permutation(len(pool))]
    print(
        f"pool: {len(pool):,} items after exclusions (from {n0:,}; training texts, gold-1 removed)"
    )

    picked: list[int] = []  # positions in pool
    caps = {"gdelt": N_NEWS, "kaggle_tweets": N_TWEETS}
    count = {"gdelt": 0, "kaggle_tweets": 0}
    reason: dict[int, str] = {}

    def ok(i: int) -> bool:
        r = pool.iloc[i]
        if count.get(r["source"], 0) >= caps.get(r["source"], 0) or i in reason:
            return False
        if picked:
            prev = pool.iloc[picked]
            if (emb[prev["row"].to_numpy()] @ emb[r["row"]]).max() >= COS:
                return False
            if (
                max(
                    fuzz.token_set_ratio(r["text"], t, processor=default_process)
                    for t in prev["text"]
                )
                >= FUZZ
            ):
                return False
        return True

    def take(i: int, why: str) -> None:
        picked.append(i)
        reason[i] = why
        count[pool.iloc[i]["source"]] += 1

    for col in ("kw_class", "model_class"):
        for c in CLASSES:
            have = sum(pool.iloc[i][col] == c for i in picked)
            for i in np.flatnonzero(pool[col].to_numpy() == c):
                if have >= MIN_PER_CLASS:
                    break
                if ok(int(i)):
                    take(int(i), f"min5:{col}:{c}")
                    have += 1
    # Fill the rest within source x month strata, proportional to the pool.
    for src, cap in caps.items():
        sub = pool[pool["source"] == src]
        need = cap - count[src]
        share = sub["month"].value_counts(normalize=True)
        target = (share * need).round().astype(int)
        for month, k in target.items():
            for i in np.flatnonzero(
                (pool["source"] == src).to_numpy() & (pool["month"] == month).to_numpy()
            ):
                if k <= 0 or count[src] >= cap:
                    break
                if ok(int(i)):
                    take(int(i), f"fill:{src}:{month}")
                    k -= 1
        for i in np.flatnonzero((pool["source"] == src).to_numpy()):  # rounding remainder
            if count[src] >= cap:
                break
            if ok(int(i)):
                take(int(i), f"fill:{src}:remainder")

    sel = pool.iloc[picked].assign(reason=[reason[i] for i in picked])
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "item_id": [f"H{i:03d}" for i in range(1, len(sel) + 1)],
            "doc_id": sel["doc_id"],
            "source": sel["source"],
            "published_at": sel["published_at"],
            "linked_tickers": sel["linked_tickers"],
            "text": sel["text"],
            "label_event_class": "",
            "label_sentiment": "",
            "label_entity_correct": "",
            "label_impact_1_10": "",
            "notes": "",
        }
    )
    out.to_csv(data_path("gold", "to_label_2.csv"), index=False)
    meta = sel[["doc_id", "source", "month", "kw_class", "model_class", "reason"]]
    meta.to_parquet(data_path("processed", "gold2_sampling_meta.parquet"), index=False)
    print(f"wrote {len(out)} items: {out['source'].value_counts().to_dict()}")
    print("keyword classes:", sel["kw_class"].value_counts().to_dict())
    print("model classes:", sel["model_class"].value_counts().to_dict())
    print("months:", sel["month"].value_counts().sort_index().to_dict())
    print(f"excluded as training-set texts or gold-1: {n_excl_train:,}")


if __name__ == "__main__":
    main()
