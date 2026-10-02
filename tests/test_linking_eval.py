import pytest

from riskpulse.eval.linking_eval import wilson


def test_wilson_interval_brackets_proportion() -> None:
    lo, hi = wilson(80, 100)
    assert lo < 0.8 < hi and hi - lo < 0.2
    assert wilson(5, 5)[1] == pytest.approx(1.0)
