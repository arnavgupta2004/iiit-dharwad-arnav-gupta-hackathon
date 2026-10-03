"""Hosted dashboard entrypoint (Streamlit Community Cloud), demo --fast mode.

Serves the dashboard from the committed demo snapshot and reports/metrics.json: no models are
loaded and there is no /analyze. Run locally with `streamlit run deploy/streamlit_app.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # the riskpulse package, without installing it
os.environ.setdefault("RISKPULSE_FORCE_DEMO", "1")  # committed snapshot only
os.environ.setdefault("RISKPULSE_API", "http://127.0.0.1:9")  # no API in the hosted demo

import streamlit as st  # noqa: E402

DASH = ROOT / "src" / "riskpulse" / "dashboard"
A = "Module A: Index Rebalancer"
pages = [
    st.Page(str(DASH / "app.py"), title="RiskPulse in 60 seconds", default=True),
    st.Page(str(DASH / "pages" / "1_Signal_Monitor.py"), title="Signal Monitor"),
    st.Page(str(DASH / "pages" / "2_Module_A_Index_Rebalancer.py"), title=A),
    st.Page(str(DASH / "pages" / "3_Module_B_Stress_Testing.py"), title="Module B: Stress Testing"),
    st.Page(str(DASH / "pages" / "4_Model_Quality.py"), title="Model Quality"),
]
st.navigation(pages).run()
