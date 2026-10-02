"""Risk-budget calibration of the tilt strength kappa (GATE C, D-038).

kappa is chosen so that the median absolute active weight |w~ - w0| of the *target* weights equals
a risk budget (default 1.5%), computed only from the signal distribution (s, c) over the first
calibration months of the window. Returns are never used. The value is frozen to
``data/processed/moduleA_calibration.json`` and the evaluation starts after the calibration window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskpulse.common.config import data_path, load_config
from riskpulse.moduleA.rebalancer import target_weights

CALIB_PATH = data_path("processed", "moduleA_calibration.json")


def median_abs_active(s: pd.DataFrame, c: pd.DataFrame, kappa: float, cfg: dict) -> float:
    """Median |w~ - w0| over all name-days of the given signal frames."""
    n = s.shape[1]
    w0 = np.full(n, 1.0 / n)
    t, b = cfg["tilt"], cfg["constraints"]
    act = [
        np.abs(
            target_weights(
                w0,
                s.iloc[i].to_numpy(),
                c.iloc[i].to_numpy(),
                kappa,
                t["deadband"],
                b["w_min"],
                b["w_max"],
            )
            - w0
        )
        for i in range(len(s))
    ]
    return float(np.median(np.concatenate(act)))


def calibrate_kappa(
    s: pd.DataFrame, c: pd.DataFrame, cfg: dict, target: float, hi: float = 100.0, tol: float = 1e-5
) -> dict:
    """Bisection for kappa with median |active weight| = target (monotone in kappa)."""
    f_hi = median_abs_active(s, c, hi, cfg)
    if f_hi < target:
        return {"kappa": hi, "achieved": f_hi, "target": target, "reachable": False}
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if median_abs_active(s, c, mid, cfg) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    k = 0.5 * (lo + hi)
    return {
        "kappa": round(k, 4),
        "achieved": median_abs_active(s, c, k, cfg),
        "target": target,
        "reachable": True,
    }


def save(calib: dict, path: Path = CALIB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2, default=str))


def frozen_kappa(path: Path = CALIB_PATH) -> float:
    """Calibrated kappa if frozen, else the config default."""
    if path.exists():
        return float(json.loads(path.read_text())["kappa"])
    return float(load_config("moduleA")["tilt"]["kappa"])
