"""Measure analogue-window factor moves into data/scenarios/calibration.csv.

python scripts/calibrate_scenarios.py
"""

from riskpulse.moduleB.calibration import calibrate

if __name__ == "__main__":
    df = calibrate()
    print(df.pivot(index="factor", columns="analogue", values="value").round(4).to_string())
