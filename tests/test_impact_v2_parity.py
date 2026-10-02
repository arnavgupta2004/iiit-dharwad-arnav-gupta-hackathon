"""Parity: the JSON tree evaluator reproduces LightGBM's predictions exactly on the full validation
and test sets of impact v2. LightGBM runs only in a child process (OpenMP clash with torch, D-049).
Skipped where the trained model or the event-study data are absent (e.g. a fresh clone)."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from riskpulse.common.config import data_path
from riskpulse.engine.gbm import TreeEnsemble
from riskpulse.eval import impact_v2

NEEDED = [
    impact_v2.MODEL_PATH,
    impact_v2.JSON_PATH,
    data_path("processed", "mentions.parquet"),
    data_path("raw", "_downloads", "event_study", "stage1.parquet"),
]

CHILD = """
import sys, numpy as np, lightgbm as lgb
booster = lgb.Booster(model_file=sys.argv[1])
for name in sys.argv[3:]:
    np.save(f"{sys.argv[2]}/{name}_lgb.npy", booster.predict(np.load(f"{sys.argv[2]}/{name}.npy")))
"""


@pytest.mark.skipif(
    not all(p.exists() for p in NEEDED) or importlib.util.find_spec("lightgbm") is None,
    reason="trained impact v2 model or event-study data not present",
)
def test_json_evaluator_matches_lightgbm_on_full_validation_and_test(tmp_path) -> None:
    mats = impact_v2.feature_matrices()
    for name, x in mats.items():
        np.save(tmp_path / f"{name}.npy", x)
    subprocess.run(
        [sys.executable, "-c", CHILD, str(impact_v2.MODEL_PATH), str(tmp_path), *mats],
        check=True,
        capture_output=True,
    )
    ens = TreeEnsemble.load(impact_v2.JSON_PATH)
    for name, x in mats.items():
        ours, theirs = ens.predict(x), np.load(tmp_path / f"{name}_lgb.npy")
        assert len(ours) == len(theirs) == len(x) > 1000
        assert np.array_equal(ours, theirs), f"{name}: max |diff| {np.abs(ours - theirs).max()}"
        single = np.array([ens.predict_one(r) for r in x[:2000]])
        assert np.array_equal(single, theirs[:2000]), f"{name}: single-row path differs"
