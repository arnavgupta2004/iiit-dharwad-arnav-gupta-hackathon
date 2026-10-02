"""Live impact v2: JSON tree evaluator, session rule, running ticker-session aggregate, fallback."""

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, date, datetime

import pytest

from riskpulse.engine import impact as imp
from riskpulse.engine.gbm import TreeEnsemble
from riskpulse.engine.impact import ImpactFeatures, ImpactV2


def _leaf(v: float) -> dict:
    return {"leaf_value": v}


def _split(f: int, t: float, left: dict, right: dict) -> dict:
    return {
        "split_feature": f,
        "threshold": t,
        "decision_type": "<=",
        "left_child": left,
        "right_child": right,
    }


# Two trees: one on S (feature 0), one on log_n (feature 7); output rises with both.
DUMP = {
    "max_feature_idx": 7,
    "tree_info": [
        {"tree_structure": _split(0, 0.5, _leaf(0.1), _split(0, 0.8, _leaf(0.3), _leaf(0.5)))},
        {"tree_structure": _split(7, 1.0, _leaf(0.0), _leaf(0.2))},
    ],
}


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(DUMP))
    return path


def _f(s: float) -> ImpactFeatures:
    return ImpactFeatures(0.5, s, 0.5, 0.2, 1.0, 0.8, 1.0)


def test_tree_ensemble_single_and_batch_agree() -> None:
    ens = TreeEnsemble.from_dump(DUMP)
    rows = [
        [0.2, 0, 0, 0, 0, 0, 0, 0.5],
        [0.9, 0, 0, 0, 0, 0, 0, 2.0],
        [0.6, 0, 0, 0, 0, 0, 0, 1.0],
    ]
    assert [ens.predict_one(r) for r in rows] == pytest.approx([0.1, 0.7, 0.3])
    assert ens.predict(rows).tolist() == pytest.approx([0.1, 0.7, 0.3])


def test_session_rule(model_path) -> None:
    v2 = ImpactV2(model_path, [date(2022, 6, 14), date(2022, 6, 15), date(2022, 6, 17)])
    assert v2.session(datetime(2022, 6, 14, 15, tzinfo=UTC)) == date(2022, 6, 14)  # 11:00 ET
    assert v2.session(datetime(2022, 6, 14, 20, 30, tzinfo=UTC)) == date(2022, 6, 15)  # 16:30 ET
    assert v2.session(datetime(2022, 6, 15, 21, tzinfo=UTC)) == date(2022, 6, 17)  # holiday skipped


def test_running_aggregate_is_monotone_and_resets(model_path) -> None:
    v2 = ImpactV2(model_path, [date(2022, 6, 14), date(2022, 6, 15)])
    t = datetime(2022, 6, 14, 14, tzinfo=UTC)
    first = v2.observe("AAPL", t, _f(0.9))  # S 0.9, n 1 -> 0.5 + 0.0
    second = v2.observe("AAPL", t, _f(0.1))  # max S stays 0.9, n 2 -> 0.5 + 0.2
    assert (first, second) == pytest.approx((0.5, 0.7))
    nxt = v2.observe("AAPL", datetime(2022, 6, 15, 14, tzinfo=UTC), _f(0.1))
    assert v2.state["AAPL"][2] == 1 and nxt == pytest.approx(0.1)  # new session: reset


def test_missing_model_falls_back_to_v1(monkeypatch, tmp_path) -> None:
    cfg = {"v2": {"enabled": True, "model_path": "nope/missing.json"}}
    monkeypatch.setattr(imp, "load_config", lambda name: cfg)
    monkeypatch.setattr(imp, "repo_root", lambda: tmp_path)
    assert ImpactV2.load() is None


CHECK = """
import json, numpy as np, lightgbm as lgb
from riskpulse.engine.gbm import TreeEnsemble
rng = np.random.default_rng(0)
x = rng.random((3000, 8)); y = x[:, 0] + 0.5 * x[:, 7] + 0.1 * rng.random(3000)
m = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, monotone_constraints=[1] * 8, verbose=-1)
m = m.fit(x, y)
ens = TreeEnsemble.from_dump(m.booster_.dump_model())
diff = np.abs(ens.predict(x) - m.booster_.predict(x)).max()
one = max(abs(ens.predict_one(r) - p) for r, p in zip(x[:200], m.booster_.predict(x[:200])))
print(json.dumps({"diff": float(diff), "one": float(one)}))
"""


def test_tree_ensemble_matches_lightgbm_in_subprocess() -> None:
    """lightgbm runs only in a child process (it must not share a process with torch), so its
    availability is checked without importing it here."""
    if importlib.util.find_spec("lightgbm") is None:
        pytest.skip("lightgbm not installed")
    out = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, check=True)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["diff"] < 1e-9 and res["one"] < 1e-9
