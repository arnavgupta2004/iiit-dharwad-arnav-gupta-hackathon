"""Module A backtest: look-ahead guards and benchmark sanity."""

import numpy as np
import pandas as pd
import pytest

from riskpulse.common.config import load_config
from riskpulse.moduleA.backtest import close_times, daily_sentiment, run_strategies
from riskpulse.moduleA.metrics import information_coefficient, max_drawdown, performance

DAYS = pd.DatetimeIndex(["2022-03-01", "2022-03-02", "2022-03-03", "2022-03-04"])
TICKERS = ["AAA", "BBB", "CCC"]


def mention(ts: str, ticker: str, s: float) -> dict:
    return {
        "published_at": pd.Timestamp(ts, tz="UTC"),
        "ticker": ticker,
        "sentiment": s,
        "relevance": 1.0,
        "event_confidence": 1.0,
    }


def test_close_times_are_4pm_new_york() -> None:
    assert close_times(DAYS)[0] == pd.Timestamp("2022-03-01 21:00", tz="UTC")  # EST


def test_item_after_close_only_affects_next_day() -> None:
    m = pd.DataFrame([mention("2022-03-01 21:30", "AAA", 0.9)])  # 16:30 ET, after the close
    s, c, net = daily_sentiment(m, DAYS, TICKERS, half_life_hours=6)
    assert s.loc["2022-03-01", "AAA"] == 0.0 and c.loc["2022-03-01", "AAA"] == 0.0
    assert s.loc["2022-03-02", "AAA"] == pytest.approx(0.9)
    assert net.loc["2022-03-02", "AAA"] == 1 and net.loc["2022-03-01", "AAA"] == 0


def test_weights_from_day_t_earn_day_t_plus_1_return() -> None:
    cfg = {**load_config("moduleA"), "constraints": {"w_min": 0.1, "w_max": 0.6, "max_iter": 50}}
    rets = pd.DataFrame(0.0, index=DAYS, columns=TICKERS)
    rets.loc["2022-03-02", "AAA"] = 0.10  # only AAA moves, on day 2
    s = pd.DataFrame(0.0, index=DAYS, columns=TICKERS)
    s.loc["2022-03-01", "AAA"] = 0.9  # known at the close of day 1
    c = pd.DataFrame(1.0, index=DAYS, columns=TICKERS)
    res = run_strategies(rets, s, c, s * 0, cfg)
    tilt = res["sentiment_tilt"]
    w_day1 = tilt.weights.loc["2022-03-01", "AAA"]
    assert w_day1 > 1 / 3
    gross = tilt.returns.loc["2022-03-02"] + tilt.costs.loc["2022-03-01"]
    assert gross == pytest.approx(w_day1 * 0.10)
    # buy-and-hold never trades
    assert res["equal_weight_buy_hold"].turnover.sum() == 0
    assert res["equal_weight_buy_hold"].returns.loc["2022-03-02"] == pytest.approx(0.10 / 3)


def test_ic_positive_for_perfect_foresight_and_zero_without_signal() -> None:
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2022-01-03", periods=60)
    rets = pd.DataFrame(rng.normal(0, 0.01, (60, 8)), index=days, columns=list("ABCDEFGH"))
    s = rets.shift(-1).fillna(0)  # s_t equals r_{t+1}: perfect foresight
    assert information_coefficient(s, rets)["mean_ic"] == pytest.approx(1.0)
    assert information_coefficient(s * 0, rets)["n_days"] == 0


def test_performance_metrics() -> None:
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)
    from riskpulse.moduleA.backtest import BacktestResult

    res = BacktestResult(r, pd.DataFrame(), pd.Series([0.0, 0.1, 0.0]), pd.Series([0.0, 0.0, 0.0]))
    p = performance(res)
    assert p["cumulative_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1, abs=1e-4)
