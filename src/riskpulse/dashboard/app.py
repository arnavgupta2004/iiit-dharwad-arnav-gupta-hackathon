"""RiskPulse dashboard entry point (Streamlit multipage: pages/ holds the four views)."""

from __future__ import annotations

import streamlit as st

from riskpulse.dashboard import data
from riskpulse.dashboard.theme import kpi, setup_page

setup_page("Overview")

st.title("RiskPulse")
st.markdown(
    "AI/NLP risk engine that turns news and social posts into structured risk signals "
    "(sentiment, event class, impact), feeding a **sentiment-tilted index** (Module A) and an "
    "**event-driven stress test** of a synthetic wholesale-banking book (Module B)."
)

health = data.api_health()
sig = data.signals_df()
fs = data.feed_stats()
c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi("Documents in replay feed", f"{fs.get('n_docs', 0):,}"), unsafe_allow_html=True)
c2.markdown(
    kpi("Sources", "GDELT news + stock tweets", f"{fs.get('by_source', {})}"),
    unsafe_allow_html=True,
)
c3.markdown(kpi("Signals stored", f"{len(sig):,}"), unsafe_allow_html=True)
c4.markdown(
    kpi(
        "Signal API",
        "online" if health else "offline",
        f"mode: {health['mode']}" if health else "dashboard uses cached outputs",
    ),
    unsafe_allow_html=True,
)

st.subheader("Pages")
st.markdown(
    """
- **Signal Monitor**: feed, signals with evidence and drivers, impact heatmap, *Analyze a headline*.
- **Module A: Index Rebalancer**: weights over time, performance vs benchmarks, turnover, IC.
- **Module B: Stress Testing**: trigger timeline, before/after value, waterfall, CET1, what-if sliders.
- **Model Quality**: sentiment and event metrics against baselines, data and pipeline statistics.
"""
)
st.markdown(
    '<p class="rp-note">All figures shown are produced by scripts in the repository and stored in '
    "reports/metrics.json. The Module B portfolio and any injected headline are synthetic and "
    "labelled as such.</p>",
    unsafe_allow_html=True,
)
