"""Precompute sentiment for the replay feed and the Benzinga event study with given weights.

Compute only: fills the sentiment caches keyed by the weights' fingerprint
(data/processed/cache/stage1sent_*, data/raw/_downloads/event_study/sentiment_*). It changes no
config, trains nothing and evaluates nothing; the pipeline picks the cache up once the configured
variant matches (D-046).

    python scripts/precompute_sentiment.py --variant finetuned
"""

from __future__ import annotations

import argparse
import time

from riskpulse.common.logging import get_logger
from riskpulse.engine.batch import _feed_hash, base_stage, sentiment_stage
from riskpulse.engine.pipeline import NLPScorer
from riskpulse.eval import event_study_data as es
from riskpulse.ingestion.replay import iter_feed

log = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="finetuned", choices=["finbert", "finetuned"])
    ap.add_argument("--only", choices=["feed", "benzinga"], default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.only in (None, "feed"):
        docs = list(iter_feed())
        fk = _feed_hash()
        base, _ = base_stage(docs, fk, NLPScorer())
        sentiment_stage(docs, base, fk, variant=a.variant)
        log.info(f"feed sentiment done at {(time.perf_counter() - t0) / 60:.1f} min")
    if a.only in (None, "benzinga"):
        sub, _ = es.universe_headlines()
        titles = es._base(sub)["title"].tolist()
        es.sentiment(titles, a.variant)
        log.info(f"Benzinga sentiment done at {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
