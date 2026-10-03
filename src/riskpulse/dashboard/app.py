"""RiskPulse dashboard landing page: "RiskPulse in 60 seconds" (every number from reports/metrics.json).

The four detailed views live in pages/. Wording rule: the system flagged the escalation; it did
not predict the invasion.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskpulse.common.config import load_config
from riskpulse.dashboard import data
from riskpulse.dashboard.theme import MUTED, NAVY, NEG, POS, kpi, pct, setup_page, usd

setup_page("RiskPulse in 60 seconds")
m = data.metrics()
st.title("RiskPulse in 60 seconds")
st.markdown(
    "An AI/NLP risk engine that reads news and tweets in time order, turns each item into a risk signal "
    "(sentiment, event class, impact) and acts on it: a **sentiment-tilted index** (Module A) and "
    "**event-triggered stress tests** of a synthetic USD 10 bn wholesale book (Module B)."
)
ep = (m.get("moduleB_validation") or {}).get("episodes", {}).get("russia_ukraine_2022")
if not ep:
    st.info("Run `python -m riskpulse eval all` to produce reports/metrics.json.")
    st.stop()

# ---------- 1. the escalation ----------
st.subheader("1 · 22 February 2022: the escalation is flagged")
runs = [
    r
    for r in ep.get("class_runs_in_prior_week", [])
    if any(w in (r["headline"] or "") for w in ("Russia", "Ukrain", "Putin"))
]
if ep.get("trigger"):
    runs.append(
        {
            "as_of": ep["trigger"]["as_of"],
            "impact_score": ep["trigger"]["impact_score"],
            "n_sources": ep["trigger"]["n_sources"],
            "headline": ep.get("headline"),
        }
    )
tl = pd.DataFrame(runs).drop_duplicates("as_of").sort_values("as_of")
tl["t"] = pd.to_datetime(tl["as_of"], utc=True)
first = tl.iloc[0]
fig = go.Figure(
    go.Scatter(
        x=tl["t"],
        y=tl["impact_score"],
        mode="markers",
        marker={"symbol": "diamond", "size": 9 + 2 * tl["n_sources"].clip(upper=10), "color": NAVY},
        customdata=tl[["n_sources", "headline"]].to_numpy(),
        hovertemplate="%{x|%d %b %H:%M} UTC · impact %{y} · %{customdata[0]} outlets"
        "<br>%{customdata[1]}<extra></extra>",
    )
)
inv = pd.Timestamp(ep["event_date"], tz="UTC")
fig.add_vline(x=inv, line={"color": NEG, "dash": "dot", "width": 1.5})
fig.add_annotation(
    x=inv,
    y=10.3,
    text="24 Feb: full-scale invasion",
    showarrow=False,
    font={"color": NEG, "size": 12},
    xanchor="left",
)
fig.update_layout(
    height=280,
    yaxis={"title": "impact (1-10)", "range": [6.5, 10.8]},
    xaxis={"title": "UTC"},
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    f"**{pd.Timestamp(first['as_of']).strftime('%d %b %Y, %H:%M')} UTC:** a geopolitical stress test fires on "
    f"*“{first['headline']}”* (impact {first['impact_score']}), two days before the full-scale "
    "invasion. Each diamond is a stress test, sized by the number of outlets. Real news, replayed in time "
    "order with no look-ahead; the system flagged the **escalation**, it did not predict the invasion."
)

# ---------- 2. the stress run ----------
st.subheader("2 · The 24 February stress run")
start = float(load_config("moduleB")["capital"]["cet1_ratio_start"])
c = st.columns(3)
c[0].markdown(
    kpi(
        "Predicted P&L, USD 10 bn book",
        usd(ep["book_total_impact_predicted"]),
        negative=ep["book_total_impact_predicted"] < 0,
    ),
    unsafe_allow_html=True,
)
c[1].markdown(
    kpi(
        "CET1 ratio",
        f"{pct(start, 2)} → {pct(ep['cet1_after_predicted'], 2)}",
        "start is a configured assumption",
    ),
    unsafe_allow_html=True,
)
c[2].markdown(
    kpi("Scenario", ep["scenario"], "analogues: " + ", ".join(ep["analogues"])),
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="rp-note">Shocks are measured on historical analogues that ended before September 2021, so 2022 '
    "is out of sample. The book and its counterparties are synthetic.</p>",
    unsafe_allow_html=True,
)

# ---------- 3. predicted vs realised ----------
UNITS = {"rates_10y_bp": "bp", "rates_3m_bp": "bp", "credit_bbb_bp": "bp", "vol_vix_pts": "pts"}


def _fmt(f: str, x: float) -> str:
    return f"{x:+.1f} {UNITS[f]}" if f in UNITS else f"{x * 100:+.1f}%"


n_right = sum(f["same_sign"] for f in ep["factors"])
st.subheader(f"3 · Predicted vs realised: {n_right} of {len(ep['factors'])} directions right")
fac = pd.DataFrame(
    [
        {
            "factor": f["factor"],
            "predicted": _fmt(f["factor"], f["predicted"]),
            "realised": _fmt(f["factor"], f["realised"]),
            "direction": "right" if f["same_sign"] else "wrong",
        }
        for f in ep["factors"]
    ]
)
st.dataframe(
    fac.style.apply(
        lambda r: ["", "", "", f"color: {POS if r['direction'] == 'right' else NEG}"], axis=1
    ),
    use_container_width=True,
    hide_index=True,
    height=300,
)
st.caption(
    f"Realised: close of {ep['realised_window'][0]} to {ep['realised_window'][1]} (10 sessions). Biggest miss: oil, "
    "because the analogues mix risk-off with commodity-supply shocks."
)

# ---------- 4. headline results ----------
st.subheader("4 · Headline results (and what didn't work)")
ic = m["moduleA"]["headline_information_coefficient"]
t = m["impact_v2"]["test_post_burn_in"]["abs_car01"]
sg = m["sentiment_gold"]["results"]["news_headlines"]["macro_f1"]
lk, pl = m["entity_linking"], m["pipeline"]
c = st.columns(5)
c[0].markdown(
    kpi("Module A IC", f"{ic['mean_ic']:+.3f}", f"t = {ic['t_stat']}, {ic['n_days']} days"),
    unsafe_allow_html=True,
)
c[1].markdown(
    kpi(
        "Impact v2 vs v1",
        f"{t['impact_v2']['spearman_rho']:.3f} vs {t['impact_v1']['spearman_rho']:.3f}",
        "Spearman with abnormal returns",
    ),
    unsafe_allow_html=True,
)
c[2].markdown(
    kpi("News sentiment", f"{sg['finbert']:.3f}", f"vs VADER {sg['vader']:.3f} (macro-F1)"),
    unsafe_allow_html=True,
)
c[3].markdown(
    kpi("Entity linking", pct(lk["precision"]), f"{lk['n_marked']} human-checked links"),
    unsafe_allow_html=True,
)
c[4].markdown(
    kpi(
        "Throughput",
        f"{pl['batch_docs_per_sec']}/s",
        f"p95 {pl['single_doc_latency_ms_p95']} ms, CPU",
    ),
    unsafe_allow_html=True,
)
tv = m.get("trigger_validation")
misses = [
    "event classification is no better than a keyword baseline",
    "market-wide impact does not predict SPY or VIX moves",
    "the June 2022 FOMC hike was missed",
]
if tv:
    misses.insert(
        2,
        f"stress-trigger timing is no better than random against market stress days "
        f"(precision {pct(tv['precision'])} vs {pct(tv['random_baseline']['precision_mean'])})",
    )
st.markdown(
    "**What didn't work:** " + "; ".join(misses) + ". Details in Model Quality and the README."
)
st.markdown(
    f'<p class="rp-note" style="color:{MUTED}">Every number on this page is read from reports/metrics.json '
    "(regenerate with <code>python -m riskpulse eval all</code>). Explore: Signal Monitor, Module A, Module B, "
    "Model Quality (sidebar).</p>",
    unsafe_allow_html=True,
)
