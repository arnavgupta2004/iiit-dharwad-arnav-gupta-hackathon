from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from riskpulse.engine.impact import (
    BreadthTracker,
    ImpactBins,
    ImpactFeatures,
    VelocityTracker,
    raw_impact,
    score_impact,
)

T0 = datetime(2022, 2, 24, tzinfo=UTC)
CFG = {
    "weights": {
        "sentiment": 0.35,
        "velocity": 0.25,
        "breadth": 0.15,
        "novelty": 0.15,
        "credibility": 0.10,
    },
    "relevance_floor": 0.5,
    "velocity": {"window_hours": 6, "baseline_days": 7, "sigmoid_scale": 1.0, "min_std": 1.0},
    "breadth": {"window_hours": 6, "saturation_outlets": 20},
}


def feats(**kw) -> ImpactFeatures:
    base = dict(
        type_prior=0.9,
        sentiment=0.6,
        velocity=0.8,
        breadth=0.5,
        novelty=0.7,
        credibility=0.8,
        relevance=0.9,
    )
    base.update(kw)
    return ImpactFeatures(**base)


def test_raw_formula_matches_spec() -> None:
    f = feats()
    raw, contrib = raw_impact(f, CFG)
    inner = 0.35 * 0.6 + 0.25 * 0.8 + 0.15 * 0.5 + 0.15 * 0.7 + 0.10 * 0.8
    assert raw == pytest.approx(0.9 * inner * (0.5 + 0.5 * 0.9))
    assert sum(contrib.values()) == pytest.approx(raw, abs=1e-3)


@pytest.mark.parametrize(
    "field",
    ["sentiment", "velocity", "breadth", "novelty", "credibility", "relevance", "type_prior"],
)
def test_raw_is_monotone_in_each_driver(field: str) -> None:
    lo, _ = raw_impact(feats(**{field: 0.2}), CFG)
    hi, _ = raw_impact(feats(**{field: 0.9}), CFG)
    assert hi > lo


def test_bins_map_quantiles_to_1_10() -> None:
    bins = ImpactBins.fit(np.linspace(0, 1, 1001))
    assert bins.score(0.0) == 1 and bins.score(1.0) == 10
    assert bins.score(0.55) == 6
    scores = [bins.score(x) for x in np.linspace(0, 1, 50)]
    assert scores == sorted(scores)


def test_unfitted_bins_fall_back_to_linear() -> None:
    assert ImpactBins(None).score(0.0) == 1 and ImpactBins(None).score(0.73) == 8


def test_score_impact_carries_drivers() -> None:
    s = score_impact(feats(), ImpactBins(None), CFG)
    assert 1 <= s.score <= 10 and set(s.drivers) >= {"type_prior", "velocity", "relevance"}


def test_velocity_spike_vs_quiet_baseline() -> None:
    v = VelocityTracker(CFG)
    for d in range(7):  # one mention per day for a week
        v.observe("JPM", T0 + timedelta(days=d))
    quiet, _ = v.observe("JPM", T0 + timedelta(days=7, hours=12))
    burst = 0.0
    for m in range(10):  # burst of 10 mentions in an hour
        burst, z = v.observe("JPM", T0 + timedelta(days=8, minutes=6 * m))
    assert burst > quiet and z > 3


def test_breadth_counts_distinct_outlets_in_window() -> None:
    b = BreadthTracker(CFG)
    b.observe("evt", "a.com", T0)
    b.observe("evt", "a.com", T0 + timedelta(hours=1))
    val, n = b.observe("evt", "b.com", T0 + timedelta(hours=2))
    assert n == 2 and 0 < val < 1
    _, n_late = b.observe("evt", "c.com", T0 + timedelta(hours=9))
    assert n_late == 1
