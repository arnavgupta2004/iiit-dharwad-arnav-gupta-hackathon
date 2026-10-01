"""Deduplication (spec §5.1): exact canonical-text match, then near-duplicate titles.

Near duplicates are detected with rapidfuzz ``token_set_ratio`` against items already kept
in the same time bucket (default: calendar day) and the same primary entity, keeping the
earliest. Bucketing keeps the comparison O(n * k) with small k.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz, utils

from riskpulse.ingestion.normalize import dedupe_key


def dedupe_frame(
    df: pd.DataFrame,
    text_col: str = "text",
    time_col: str = "published_at",
    group_col: str | None = "primary_ticker",
    near_threshold: float = 95.0,
    bucket: str = "D",
    min_length_ratio: float = 0.7,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return (deduplicated frame sorted by time, stats).

    Args:
        df: items with a text column and a tz-aware timestamp column.
        group_col: near-duplicates are only searched within the same group value.
        near_threshold: rapidfuzz token_set_ratio (0-100) at or above which two texts are
            near-duplicates (95 ~ the spec's cosine > 0.95 intent, on surface form).
        min_length_ratio: token-count ratio required before comparing, because token_set_ratio
            scores 100 when one text is a subset of the other (e.g. "$TSLA" vs a long tweet).
    """
    if df.empty:
        return df, {"input": 0, "exact_dupes": 0, "near_dupes": 0, "kept": 0}
    d = df.sort_values(time_col, kind="stable").copy()
    d["_key"] = d[text_col].map(dedupe_key)
    before = len(d)
    d = d.drop_duplicates("_key", keep="first")
    exact = before - len(d)

    keep_mask = []
    seen: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    buckets = d[time_col].dt.floor(bucket)
    groups = d[group_col] if group_col and group_col in d else pd.Series("", index=d.index)
    for text, b, g in zip(d[text_col].str.lower(), buckets, groups, strict=True):
        pool = seen[(b, g)]
        n = len(text.split())
        dup = any(
            min(n, m) / max(n, m, 1) >= min_length_ratio
            and fuzz.token_set_ratio(text, other, processor=utils.default_process) >= near_threshold
            for other, m in pool
        )
        keep_mask.append(not dup)
        if not dup:
            pool.append((text, n))
    out = d[keep_mask].drop(columns="_key")
    stats = {
        "input": before,
        "exact_dupes": exact,
        "near_dupes": len(d) - len(out),
        "kept": len(out),
    }
    return out, stats
