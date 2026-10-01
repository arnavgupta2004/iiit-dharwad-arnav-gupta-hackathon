import numpy as np
import pytest

from riskpulse.moduleA.rebalancer import cap_weights, one_way_turnover, rebalance, target_weights

N = 20
W0 = np.full(N, 1 / N)


def test_cap_weights_sum_and_bounds() -> None:
    raw = np.array([10.0] + [1.0] * (N - 1))
    w = cap_weights(raw, 0.02, 0.12)
    assert w.sum() == pytest.approx(1.0)
    assert w.max() == pytest.approx(0.12) and w.min() >= 0.02 - 1e-12
    # excess from the capped name is spread proportionally: all others equal
    assert np.allclose(w[1:], w[1])


def test_cap_weights_infeasible() -> None:
    with pytest.raises(ValueError):
        cap_weights(np.ones(5), 0.25, 0.3)


def test_zero_sentiment_gives_equal_weight() -> None:
    w = target_weights(W0, np.zeros(N), np.ones(N), 1.0, 0.1, 0.02, 0.12)
    assert np.allclose(w, W0)


def test_positive_sentiment_increases_weight_negative_decreases() -> None:
    s = np.zeros(N)
    s[0], s[1] = 0.8, -0.8
    w = target_weights(W0, s, np.ones(N), 1.0, 0.1, 0.02, 0.12)
    assert w[0] > W0[0] > w[1]
    assert w.sum() == pytest.approx(1.0)


def test_deadband_and_confidence() -> None:
    s = np.zeros(N)
    s[0] = 0.05  # inside deadband
    assert np.allclose(target_weights(W0, s, np.ones(N), 1.0, 0.1, 0.02, 0.12), W0)
    s[0] = 0.8
    c = np.ones(N)
    c[0] = 0.0  # zero confidence -> no tilt
    assert np.allclose(target_weights(W0, s, c, 1.0, 0.1, 0.02, 0.12), W0)


def test_turnover_cap_moves_partially() -> None:
    w_target = cap_weights(np.r_[np.full(10, 2.0), np.ones(10)], 0.02, 0.12)
    full = one_way_turnover(W0, w_target)
    r = rebalance(W0, w_target, tau_max=full / 2, cost_bps=5)
    assert r.lam == pytest.approx(0.5)
    assert r.turnover == pytest.approx(full / 2)
    assert r.cost == pytest.approx(5e-4 * full / 2)
    assert r.weights.sum() == pytest.approx(1.0)
    no_cap = rebalance(W0, w_target, tau_max=1.0, cost_bps=5)
    assert np.allclose(no_cap.weights, w_target) and no_cap.lam == 1.0
