"""Pipeline benchmark on CPU: batch throughput, single-document latency, dedup rate (spec §9).

Run on an otherwise idle machine; the result records the device and thread count used.
"""

from __future__ import annotations

import os
import platform
import time

import numpy as np

from riskpulse.common.config import load_config, reports_path
from riskpulse.common.metrics import update_metrics
from riskpulse.engine.impact import ImpactBins
from riskpulse.engine.pipeline import NLPScorer, SignalEngine
from riskpulse.engine.sentiment import torch_device
from riskpulse.ingestion.replay import iter_feed


def run(n_batch: int = 1024, n_latency: int = 100) -> dict:
    import json

    import torch

    docs = []
    for d in iter_feed("2022-02-22", "2022-03-01"):
        docs.append(d)
        if len(docs) >= n_batch:
            break
    scorer = NLPScorer()
    scorer.score(docs[:8])  # warm-up: load models
    # Both sentiment models (news and social, D-058) must be loaded before timing.
    scorer.finbert.score(["warm up"])
    scorer.finbert_news.score(["warm up"])
    t0 = time.perf_counter()
    scored = []
    for i in range(0, len(docs), 128):
        scored.extend(scorer.score(docs[i : i + 128]))
    stage1 = time.perf_counter() - t0
    t1 = time.perf_counter()
    SignalEngine(bins=ImpactBins.load()).process(scored)
    stage2 = time.perf_counter() - t1
    lat = []
    engine = SignalEngine(bins=ImpactBins.load())
    for d in docs[:n_latency]:
        t = time.perf_counter()
        engine.process(scorer.score([d]))
        lat.append((time.perf_counter() - t) * 1000)
    fs = json.loads(reports_path("feed_stats.json").read_text())
    nd, sd = fs["news_dedupe"], fs["social_dedupe"]
    payload = {
        "device": torch_device(),
        "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(),
        "n_cpu": os.cpu_count(),
        "batch_docs": len(docs),
        "batch_docs_per_sec": round(len(docs) / (stage1 + stage2), 1),
        "stage1_models_share_of_time": round(stage1 / (stage1 + stage2), 3),
        "single_doc_latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "single_doc_latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "dedup_rate_news": round((nd["exact_dupes"] + nd["near_dupes"]) / nd["input"], 4),
        "dedup_rate_social": round((sd["exact_dupes"] + sd["near_dupes"]) / sd["input"], 4),
        "feed_docs": fs["n_docs"],
        "sample": "replay feed 2022-02-22..2022-03-01 (first docs)",
        "batch_size": int(load_config("app")["sentiment"]["batch_size"]),
    }
    update_metrics("pipeline", payload, script="riskpulse eval pipeline")
    return payload
