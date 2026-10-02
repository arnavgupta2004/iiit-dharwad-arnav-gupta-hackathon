"""Old vs new stage-2 code on real cached stage-1 outputs (no model re-run).

Compares the verbatim pre-optimisation implementation (tests/legacy) with the current code on
1,000-document slices of the cached stage-1 scores and prints the largest differences.

    python scripts/check_stage2_equivalence.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from riskpulse.common.config import repo_root
from riskpulse.engine.batch import run_stage1
from riskpulse.engine.impact import ImpactBins
from riskpulse.engine.pipeline import SignalEngine

sys.path.insert(0, str(repo_root()))
from tests.legacy.pipeline_v0 import SignalEngine as LegacySignalEngine  # noqa: E402


def compare(scored: list, label: str) -> dict:
    bins = ImpactBins.load()
    new_m, new_s = SignalEngine(bins=bins).process(scored)
    old_m, old_s = LegacySignalEngine(bins=bins).process(scored)
    assert len(new_m) == len(old_m)
    same_ids = all(
        (a.doc_id, a.ticker, a.event_id, a.impact_score)
        == (b.doc_id, b.ticker, b.event_id, b.impact_score)
        for a, b in zip(new_m, old_m, strict=True)
    )
    raw_diff = max(abs(a.impact_raw - b.impact_raw) for a, b in zip(new_m, old_m, strict=True))
    drv_diff = max(
        abs(a.drivers[k] - b.drivers[k])
        for a, b in zip(new_m, old_m, strict=True)
        for k in a.drivers
    )
    # drivers are stored rounded to 4 dp and impact_raw to 6 dp, so a ~1e-16 difference in a
    # similarity sum can flip the last stored digit: tolerance = one unit of stored precision.
    diff_drv = [
        (a.doc_id, k)
        for a, b in zip(new_m, old_m, strict=True)
        for k in a.drivers
        if a.drivers[k] != b.drivers[k]
    ]
    ev_new = [(s.event_id, s.impact_score, s.n_sources) for s in new_s if s.signal_type == "event"]
    ev_old = [(s.event_id, s.impact_score, s.n_sources) for s in old_s if s.signal_type == "event"]
    res = {
        "slice": label,
        "docs": len(scored),
        "mentions": len(new_m),
        "stories": len({m.event_id for m in new_m}),
        "ids_and_scores_identical": same_ids,
        "max_abs_impact_raw_diff": float(raw_diff),
        "max_abs_driver_diff": float(drv_diff),
        "event_signals_identical": ev_new == ev_old,
        "n_event_signals": len(ev_new),
        "mentions_with_any_driver_diff": len({d for d, _ in diff_drv}),
        "drivers_differing": sorted({k for _, k in diff_drv}),
    }
    print(res)
    assert same_ids and raw_diff <= 1e-6 + 1e-12 and drv_diff <= 1e-4 + 1e-12, label
    assert ev_new == ev_old, label
    return res


def main() -> None:
    scored = run_stage1()
    ts = pd.DatetimeIndex([s.doc.published_at for s in scored])
    first = scored[:1000]
    k = int(np.searchsorted(ts, pd.Timestamp("2022-02-24 03:00", tz="UTC")))
    invasion = scored[k : k + 1000]
    for sl, label in (
        (first, "first 1,000 docs"),
        (invasion, "1,000 docs from 2022-02-24 03:00 UTC"),
    ):
        compare(sl, label)
    print("OK: identical ids, scores and event signals; numeric fields within stored precision")


if __name__ == "__main__":
    main()
