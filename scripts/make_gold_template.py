"""Create data/gold/to_label.csv: ~300 stratified feed items for Arnav to label by hand.

Stratified by source (news / social) and by keyword-baseline class so rare classes
(e.g. CREDIT_EVENT) appear. The gold set is a TEST set: it is never used for training, and
training explicitly drops any text present here.

    python scripts/make_gold_template.py [--n 300]
"""

from __future__ import annotations

import argparse
import gzip
import json

import pandas as pd

from riskpulse.common.config import data_path, load_config, repo_root
from riskpulse.engine.events import KeywordClassifier

SEED = 20261002
NEWS_PER_CLASS = 22
SOCIAL_PER_CLASS = 6


def load_feed() -> pd.DataFrame:
    path = repo_root() / load_config("app")["paths"]["replay_feed"]
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append(
                {
                    "doc_id": d["doc_id"],
                    "source": d["source"],
                    "published_at": d["published_at"],
                    "text": d["title"] or d["text"],
                    "linked_tickers": ",".join(d["meta"].get("tickers", [])),
                }
            )
    return pd.DataFrame(rows)


def main(n: int) -> None:
    feed = load_feed()
    feed = feed[feed["text"].str.len().between(25, 280)].drop_duplicates("text")
    kw = KeywordClassifier()
    feed["kw_class"] = [p.event_class for p in kw.predict(feed["text"].tolist())]
    picks = []
    for (src, cls), g in feed.groupby(["source", "kw_class"]):
        k = NEWS_PER_CLASS if src == "gdelt" else SOCIAL_PER_CLASS
        if cls == "OTHER":
            k *= 2
        picks.append(g.sample(min(k, len(g)), random_state=SEED))
    gold = pd.concat(picks)
    if len(gold) > n:
        gold = gold.sample(n, random_state=SEED)
    elif len(gold) < n:
        rest = feed.drop(gold.index)
        gold = pd.concat([gold, rest.sample(n - len(gold), random_state=SEED)])
    gold = gold.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "item_id": [f"G{i:03d}" for i in range(1, len(gold) + 1)],
            "doc_id": gold["doc_id"],
            "source": gold["source"],
            "published_at": gold["published_at"],
            "linked_tickers": gold["linked_tickers"],
            "text": gold["text"],
            "label_event_class": "",
            "label_sentiment": "",
            "label_entity_correct": "",
            "label_impact_1_10": "",
            "notes": "",
        }
    )
    path = data_path("gold", "to_label.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} items to {path}")
    print(out["source"].value_counts().to_dict())
    print(gold["kw_class"].value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    main(ap.parse_args().n)
