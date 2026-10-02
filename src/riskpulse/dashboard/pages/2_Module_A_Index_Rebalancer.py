"""Page 2: Module A: sentiment-tilted index of 20 S&P 100 names."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskpulse.common.config import load_config
from riskpulse.dashboard import data
from riskpulse.dashboard.theme import MUTED, NAVY, NEG, POS, SERIES, kpi, pct, setup_page
from riskpulse.moduleA.rebalancer import target_weights

setup_page("Module A")
st.title("Module A: Tactical Index Rebalancer")
st.markdown(
    '<p class="rp-note">Equal-weight base tilted by decayed entity sentiment: '
    "w = w0 · exp(κ · s · c), deadband |s| &lt; 0.10, weights in [2%, 12%], turnover cap, 5 bps costs. "
    "Weights decided at the 16:00 ET close of day t earn the return of day t+1.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="rp-note"><b>Signal coverage.</b> Each tweet links only to its original ticker: the tweet dataset\'s '
    "PG and MSFT sets are copies of the AMZN set, so AMZN posts never move PG or MSFT. PG has thin news coverage "
    "and no tweets of its own. Between sparse items its confidence c decays toward zero, so its own tilt exp(κ·s·c) "
    "is usually negligible (the ±0.10 deadband removes weak scores too); remaining weight changes come from "
    "renormalisation as other names tilt. Intended: weak or stale evidence, little or no tilt.</p>",
    unsafe_allow_html=True,
)

out = data.module_a_outputs()
met = data.metrics().get("moduleA", {})
if not out or not met:
    st.warning("No Module A outputs yet. Run `python -m riskpulse backtest`.")
    st.stop()

w = out["weights_tilt"]
rets = out["returns"]
perf = met["performance_secondary"]
icm = met["headline_information_coefficient"]
cal = met["calibration"]
cfg = load_config("moduleA")
names = {
    "sentiment_tilt": "Sentiment tilt",
    "equal_weight_buy_hold": "Equal weight, buy & hold",
    "equal_weight_rebalanced": "Equal weight, rebalanced",
    "naive_sign_rule": "Naive sign rule",
}

# ---------- headline: information coefficient ----------
st.subheader("Headline: does sentiment rank next-day returns? (information coefficient)")
h = st.columns(4)
h[0].markdown(
    kpi("Mean daily IC", f"{icm['mean_ic']:+.3f}", f"Spearman, {icm['n_days']} days"),
    unsafe_allow_html=True,
)
h[1].markdown(kpi("IC t-stat", f"{icm['t_stat']}", "mean / (sd / √n)"), unsafe_allow_html=True)
h[2].markdown(
    kpi("Days with IC > 0", pct(icm["pct_positive_days"]), "50% = no skill"), unsafe_allow_html=True
)
h[3].markdown(
    kpi(
        "Tilt strength κ (frozen)",
        f"{cal['kappa']:.2f}",
        f"median |active weight| {pct(cal['achieved'], 2)} target",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="rp-note">κ was calibrated without returns: the value giving a median absolute active weight of '
    f"{pct(cal['target'], 1)} on the signals of {cal['calibration_window'][0]} → {cal['calibration_window'][1]}. "
    f"Everything here is evaluated on {met['evaluation_window'][0]} → {met['evaluation_window'][1]}.</p>",
    unsafe_allow_html=True,
)

st.subheader("Returns (secondary)")
k = st.columns(4)
for col, key in zip(k, names, strict=True):
    p = perf[key]
    col.markdown(
        kpi(
            names[key],
            pct(p["cumulative_return"]),
            f"Sharpe {p['sharpe_rf0']} · MDD {pct(p['max_drawdown'])}",
            negative=p["cumulative_return"] < 0,
        ),
        unsafe_allow_html=True,
    )
st.dataframe(
    pd.DataFrame(
        [
            {
                "strategy": names[key],
                "gross return": pct(perf[key]["cumulative_return_gross"])
                if "cumulative_return_gross" in perf[key]
                else "n/a",
                "cost drag (sum of daily costs)": pct(perf[key]["total_cost_drag"], 2),
                "net return": pct(perf[key]["cumulative_return"]),
                "avg daily one-way turnover": pct(perf[key]["avg_daily_one_way_turnover"], 2),
            }
            for key in names
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    f"Turnover cap: {met['config']['tau_max']:.0%} one-way per day, set by policy for "
    f"operational realism (D-045), not optimised on returns; costs {met['config']['cost_bps']} bps."
)

# ---------- weights over time (stacked area) with event annotations ----------
st.subheader("Weights over time")
sig = data.signals_df()
fig = go.Figure()
order = w.mean().sort_values(ascending=False).index
palette = (SERIES * 4)[: len(order)]
for t, c in zip(order, palette, strict=True):
    fig.add_trace(
        go.Scatter(
            x=w.index,
            y=w[t],
            name=t,
            stackgroup="w",
            mode="lines",
            line={"width": 0.6, "color": "white"},
            fillcolor=c,
            hovertemplate=f"{t}: %{{y:.1%}}<extra></extra>",
        )
    )
if not sig.empty:
    big = sig[(sig["signal_type"] == "event") & (sig["impact"] >= 9)].copy()
    big["day"] = big["as_of"].dt.tz_convert(None).dt.normalize()
    marks = big.groupby("day").size().sort_values(ascending=False).head(6).index
    for d in marks:
        fig.add_vline(x=d, line_width=1, line_dash="dot", line_color=NAVY)
fig.update_layout(
    height=430,
    yaxis_tickformat=".0%",
    yaxis_range=[0, 1],
    hovermode="x unified",
    legend={"font": {"size": 10}},
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    '<p class="rp-note">Dotted lines: days with the most impact ≥ 9 event signals.</p>',
    unsafe_allow_html=True,
)

# ---------- current weights vs baseline + evidence for a weight change ----------
c1, c2 = st.columns([3, 2])
day = c1.select_slider(
    "Rebalance date",
    options=list(w.index),
    value=w.index[len(w) // 2],
    format_func=lambda d: d.strftime("%Y-%m-%d"),
)
row = w.loc[day].sort_values()
base = 1 / len(row)
fig = go.Figure(
    go.Bar(
        x=row.to_numpy() - base,
        y=row.index,
        orientation="h",
        marker_color=[POS if v > base else NEG if v < base else MUTED for v in row],
        customdata=row.to_numpy(),
        hovertemplate="%{y}: weight %{customdata:.1%} (vs 5.0%)<extra></extra>",
    )
)
fig.update_layout(
    title=f"Weight minus equal weight on {day:%Y-%m-%d}", height=520, xaxis_tickformat="+.1%"
)
c1.plotly_chart(fig, use_container_width=True)
with c2:
    pick = st.selectbox("Why did this weight move? Ticker", list(row.index[::-1]))
    sent = out["daily_sentiment"]
    st.markdown(
        kpi(
            f"{pick} sentiment at close",
            f"{sent.loc[day, pick]:+.2f}",
            f"confidence {out['daily_confidence'].loc[day, pick]:.2f}",
        ),
        unsafe_allow_html=True,
    )
    men = data.mentions_df()
    t1 = pd.Timestamp(day).tz_localize("America/New_York").replace(hour=16).tz_convert("UTC")
    ev = men[
        (men["ticker"] == pick)
        & (men["published_at"] <= t1)
        & (men["published_at"] > t1 - pd.Timedelta("1D"))
    ]
    st.markdown("**Evidence in the 24 h before the close**")
    if ev.empty:
        st.caption("No mentions; the tilt is the decayed residue of earlier news, or zero.")
    for r in ev.sort_values("impact_score", ascending=False).head(8).itertuples():
        st.markdown(f"- `{r.source}` {r.sentiment:+.2f} · {r.title[:110]}")

# ---------- performance vs benchmarks ----------
st.subheader("Performance against benchmarks (net of costs)")
growth = (1 + rets).cumprod()
fig = go.Figure()
for i, key in enumerate(names):
    fig.add_trace(
        go.Scatter(
            x=growth.index,
            y=growth[key],
            name=names[key],
            line={
                "width": 2.2 if key == "sentiment_tilt" else 1.4,
                "color": [NAVY, SERIES[1], SERIES[2], SERIES[3]][i],
            },
        )
    )
fig.update_layout(height=380, hovermode="x unified", yaxis_title="growth of 1")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(pd.DataFrame(perf).T.rename(index=names), use_container_width=True)

# ---------- IC and turnover ----------
c1, c2 = st.columns(2)
ic = out.get("ic")
if ic is not None and not ic.empty:
    fig = go.Figure(go.Bar(x=ic.index, y=ic["ic"], marker_color=np.where(ic["ic"] > 0, POS, NEG)))
    fig.add_trace(
        go.Scatter(
            x=ic.index,
            y=ic["ic"].rolling(20).mean(),
            line={"color": NAVY, "width": 2},
            name="20-day mean",
        )
    )
    fig.update_layout(
        title=f"Daily IC: mean {icm['mean_ic']}, t-stat {icm['t_stat']}, positive {pct(icm['pct_positive_days'])}",
        height=320,
        showlegend=False,
    )
    c1.plotly_chart(fig, use_container_width=True)
turn = w.diff().abs().sum(axis=1) / 2
fig = go.Figure(go.Bar(x=turn.index, y=turn, marker_color=NAVY))
fig.update_layout(
    title=f"Daily one-way turnover (cap τ = {cfg['turnover']['tau_max']:.0%})",
    height=320,
    yaxis_tickformat=".1%",
)
c2.plotly_chart(fig, use_container_width=True)

# ---------- robustness grid ----------
st.subheader(
    "Sensitivity only: κ × half-life (excess cumulative return vs equal weight, rebalanced)"
)
st.caption(
    "Shown for transparency; no parameter is chosen from this grid (κ is frozen by the risk budget above)."
)
grid = pd.DataFrame(met.get("sensitivity_grid", []))
if not grid.empty:
    piv = grid.pivot(index="half_life_hours", columns="kappa", values="excess_vs_ew_rebalanced")
    fig = go.Figure(
        go.Heatmap(
            z=piv.to_numpy(),
            x=[f"κ={c}" for c in piv.columns],
            y=[f"H={r}h" for r in piv.index],
            colorscale=[[0, NEG], [0.5, "#f0efec"], [1, POS]],
            zmid=0,
            xgap=2,
            ygap=2,
            text=np.vectorize(lambda v: f"{v:+.1%}")(piv.to_numpy()),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
st.markdown(f'<p class="rp-note">{met.get("note", "")}</p>', unsafe_allow_html=True)

# ---------- live (replay) view ----------
st.subheader("Live view (replay mode)")
latest = data.api_get("/signals/latest")
if latest:
    u = list(load_config("universe")["tickers"])
    s = (
        pd.Series({x["entity"]["ticker"]: x["sentiment_score"] for x in latest})
        .reindex(u)
        .fillna(0)
    )
    c = pd.Series({x["entity"]["ticker"]: x["confidence"] for x in latest}).reindex(u).fillna(0)
    tw = target_weights(
        np.full(len(u), 1 / len(u)),
        s.to_numpy(),
        c.to_numpy(),
        cal["kappa"],
        cfg["tilt"]["deadband"],
        cfg["constraints"]["w_min"],
        cfg["constraints"]["w_max"],
    )
    st.dataframe(
        pd.DataFrame({"sentiment": s, "confidence": c, "target_weight": tw}, index=u).style.format(
            {"sentiment": "{:+.2f}", "confidence": "{:.2f}", "target_weight": "{:.1%}"}
        ),
        use_container_width=True,
    )
else:
    st.caption(
        "Signal API offline: start `python -m riskpulse demo` to watch weights move as the replay streams."
    )
