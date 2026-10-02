import numpy as np
import pandas as pd

from riskpulse.common.timeutils import market_date
from riskpulse.eval.impact_v2 import compare, market_days, ticker_days

DAYS = pd.DatetimeIndex(pd.bdate_range("2022-03-01", "2022-03-31"))


def test_vectorised_market_days_matches_engine_rule() -> None:
    rng = np.random.default_rng(0)
    ts = pd.Series(
        pd.to_datetime("2022-03-01", utc=True)
        + pd.to_timedelta(rng.integers(0, 20 * 24 * 60, 500), unit="min")
    )
    fast = market_days(ts, DAYS)
    slow = pd.Series([market_date(t, DAYS) for t in ts], index=ts.index)
    assert (fast == slow).all()


def test_ticker_days_aggregates_max_and_count() -> None:
    items = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "published_at": pd.to_datetime(
                ["2022-03-02 14:00", "2022-03-02 15:00", "2022-03-02 22:00"], utc=True
            ),
            "S": [0.2, 0.9, 0.5],
            "T": [0.5, 0.9, 0.3],
            "V": [0.1, 0.2, 0.3],
            "B": [0.2] * 3,
            "N": [1.0, 0.1, 0.5],
            "C": [0.8] * 3,
            "R": [1.0] * 3,
            "v1": [0.1, 0.4, 0.2],
        }
    )
    out = ticker_days(items, DAYS).set_index("ticker")
    assert out.loc["A", "S"] == 0.9 and out.loc["A", "log_n"] == np.log1p(2)
    assert out.loc["B", "day0"] == pd.Timestamp("2022-03-03")  # after 16:00 ET -> next session


def test_compare_detects_better_score() -> None:
    rng = np.random.default_rng(1)
    y = rng.random(3000)
    df = pd.DataFrame(
        {
            "abs_car01": y,
            "v2": y + rng.normal(0, 0.1, 3000),
            "v1": rng.random(3000),
            "S": y + rng.normal(0, 1.0, 3000),
        }
    )
    r = compare(df, "abs_car01", n_boot=200)
    assert r["impact_v2"]["spearman_rho"] > r["abs_sent"]["spearman_rho"]
    assert r["rho_diff_v2_minus_abs_sent_95ci"][0] > 0 and r["rho_diff_v2_minus_v1_95ci"][0] > 0
    assert r["impact_v2"]["top_decile_hit_rate"] > 0.5
