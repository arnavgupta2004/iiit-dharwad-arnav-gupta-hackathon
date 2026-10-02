import numpy as np
import pandas as pd
import pytest

from riskpulse.common.config import load_config
from riskpulse.moduleA.calibrate import calibrate_kappa, median_abs_active


def frames(n_days: int = 40, n: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = pd.DataFrame(rng.uniform(-0.8, 0.8, (n_days, n)))
    c = pd.DataFrame(rng.uniform(0.2, 1.0, (n_days, n)))
    return s, c


def test_median_active_weight_is_monotone_in_kappa() -> None:
    cfg = load_config("moduleA")
    s, c = frames()
    vals = [median_abs_active(s, c, k, cfg) for k in (0.0, 0.5, 1.0, 2.0)]
    assert vals[0] == pytest.approx(0, abs=1e-12) and vals == sorted(vals)


def test_bisection_hits_target() -> None:
    cfg = load_config("moduleA")
    s, c = frames()
    r = calibrate_kappa(s, c, cfg, target=0.015)
    assert r["reachable"] and r["achieved"] == pytest.approx(0.015, abs=2e-4)


def test_unreachable_target_is_flagged() -> None:
    cfg = load_config("moduleA")
    s, c = frames()
    r = calibrate_kappa(s * 0, c, cfg, target=0.015)  # no signal: active weight stays 0
    assert not r["reachable"]
