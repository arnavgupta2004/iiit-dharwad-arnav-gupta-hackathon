"""Page 3: Module B: event-triggered stress test of a synthetic wholesale-banking book."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskpulse.dashboard import data
from riskpulse.dashboard.theme import (
    NAVY,
    NEG,
    POS,
    SEQ_RED,
    event_color,
    kpi,
    pct,
    setup_page,
    usd,
)
from riskpulse.moduleB.scenarios import build_scenario, custom_scenario
from riskpulse.moduleB.stress import load_positions, run_stress

setup_page("Module B")
st.title("Module B: Strategic Portfolio Stress Testing")
st.markdown(
    '<p class="rp-note">Synthetic book: 121 counterparties (CP_xxxx) seeded from card-transaction '
    "merchant data, 334 positions (loans, bonds, IRS, FX forwards, options, TRS, CDS, equity), "
    "USD 10 bn notional. A stress test fires for event signals with impact ≥ 8, event confidence ≥ 0.6, "
    "≥ 2 sources and adverse sentiment (below −0.15, D-059), with a 24 h cooldown per event class and "
    "macro-region (within a cooldown it re-runs only if impact escalates). Shocks come from pre-2021 "
    "historical analogues.</p>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def positions() -> pd.DataFrame:
    return load_positions()


sig = data.signals_df()
runs = data.stress_runs()

# ---------- trigger timeline ----------
st.subheader("Event timeline and triggers")
if not sig.empty:
    ev = sig[(sig["signal_type"] == "event") & (sig["impact"] >= 7)]
    ev = ev.sort_values("impact").drop_duplicates("event_id", keep="last")
    fig = go.Figure()
    for cls, g in ev.groupby("event_class"):
        fig.add_trace(
            go.Scatter(
                x=g["as_of"],
                y=g["event_class"],
                mode="markers",
                name=cls,
                marker={
                    "size": (g["impact"] - 5) * 4,
                    "color": event_color(cls),
                    "opacity": 0.55,
                    "line": {"width": 1, "color": "white"},
                },
                customdata=g[["impact", "n_sources", "regions"]].to_numpy(),
                hovertext=[(e or [{}])[0].get("title", "") for e in g["evidence"]],
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>impact %{customdata[0]} · %{customdata[1]} sources"
                "<br>%{customdata[2]}<br>%{hovertext}<extra></extra>",
            )
        )
    if runs:
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(r["trigger"]["as_of"]) for r in runs],
                y=[r["trigger"]["event_class"] for r in runs],
                mode="markers",
                name="stress test fired",
                marker={"symbol": "diamond-open", "size": 14, "color": NAVY, "line": {"width": 2}},
                hovertemplate="TRIGGER %{x|%Y-%m-%d %H:%M}<extra></extra>",
            )
        )
    fig.update_layout(height=360, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Event signals with impact ≥ 7 (one marker per story, sized by impact). Diamonds: {len(runs)} triggered stress tests."
    )

# ---------- choose a result to inspect ----------
st.subheader("Stress result")
mode = st.radio(
    "Source",
    ["Triggered run", "Scenario library (what-if)", "Custom shocks"],
    horizontal=True,
    index=0 if runs else 1,
)
pos = positions()
res = None
if mode == "Triggered run":
    if not runs:
        st.info("No triggered runs stored. Run `python -m riskpulse stress --replay`.")
    else:
        i = st.selectbox(
            "Triggered stress test",
            range(len(runs)),
            format_func=lambda j: (
                f"{runs[j]['trigger']['as_of'][:16]} · {runs[j]['scenario']['name']} · "
                f"impact {runs[j]['trigger']['impact_score']} · "
                f"{(runs[j]['trigger']['evidence'] or [{}])[0].get('title', '')[:70]}"
            ),
        )
        res = runs[i]
elif mode == "Scenario library (what-if)":
    c1, c2 = st.columns(2)
    cls = c1.selectbox(
        "Event class", ["CREDIT_EVENT", "GEOPOLITICAL", "MACROECONOMIC", "OPERATIONAL_ESG"]
    )
    imp = c2.slider("Impact", 8, 10, 9)
    texts = []
    if cls == "MACROECONOMIC":
        texts = [
            c1.radio(
                "Sub-scenario", ["inflation hike hawkish", "recession slowdown"], horizontal=True
            )
        ]
    sc = build_scenario(cls, imp, texts)
    res = run_stress(sc, pos)
else:
    st.markdown("Set factor shocks directly (what-if):")
    c = st.columns(6)
    shocks = {
        "eq_SPY": c[0].slider("Equities (all sectors)", -0.5, 0.2, -0.15, 0.01),
        "rates_10y_bp": c[1].slider("10y rates (bp)", -200, 300, 100, 5),
        "rates_3m_bp": c[2].slider("3m rates (bp)", -200, 300, 100, 5),
        "credit_bbb_bp": c[3].slider("BBB spread (bp)", -50, 400, 150, 5),
        "fx_EUR": c[4].slider("EUR vs USD", -0.2, 0.2, -0.05, 0.01),
        "vol_vix_pts": c[5].slider("VIX (pts)", -10, 60, 20, 1),
    }
    res = run_stress(custom_scenario(shocks), pos)

if res:
    cap = res["capital"]
    k = st.columns(6)
    k[0].markdown(kpi("Value before", usd(res["value_before"])), unsafe_allow_html=True)
    k[1].markdown(
        kpi(
            "Value after (MtM)",
            usd(res["value_after"]),
            pct(res["pnl_mtm_pct"], 2),
            negative=res["pnl_mtm"] < 0,
        ),
        unsafe_allow_html=True,
    )
    k[2].markdown(
        kpi(
            "Total impact",
            usd(res["total_impact"]),
            pct(res["total_impact_pct"], 2),
            negative=res["total_impact"] < 0,
        ),
        unsafe_allow_html=True,
    )
    k[3].markdown(
        kpi(
            "Δ expected loss (loans)",
            usd(res["d_el"]),
            f"PD × {res['scenario']['pd_multiplier']:.2f}",
            negative=res["d_el"] > 0,
        ),
        unsafe_allow_html=True,
    )
    k[4].markdown(kpi("CET1 before", pct(cap["cet1_ratio_before"])), unsafe_allow_html=True)
    k[5].markdown(
        kpi(
            "CET1 after",
            pct(cap["cet1_ratio_after"], 2),
            f"{(cap['cet1_ratio_after'] - cap['cet1_ratio_before']) * 1e4:+.0f} bp",
            negative=True,
        ),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 2])
    # waterfall by asset class
    by = res["by_asset_class"]
    order = [a for a in ["loan", "bond", "derivative", "equity"] if a in by]
    fig = go.Figure(
        go.Waterfall(
            x=[a.title() for a in order] + ["Total"],
            y=[by[a] for a in order] + [0],
            measure=["relative"] * len(order) + ["total"],
            decreasing={"marker": {"color": NEG}},
            increasing={"marker": {"color": POS}},
            totals={"marker": {"color": NAVY}},
            text=[usd(by[a]) for a in order] + [usd(res["total_impact"])],
            textposition="outside",
            connector={"line": {"color": "#c0c4cc"}},
        )
    )
    fig.update_layout(
        title="Impact by asset class (MtM, loans via ΔEL)", height=380, yaxis_tickformat="~s"
    )
    c1.plotly_chart(fig, use_container_width=True)
    # scenario explainer
    with c2:
        sc = res["scenario"]
        st.markdown(f"**Scenario explainer:** `{sc['name']}`")
        st.markdown(sc["explanation"])
        if res.get("trigger"):
            t = res["trigger"]
            st.markdown(
                f"**Trigger:** {t['event_class']} · impact {t['impact_score']} · {t['n_sources']} sources · "
                f"regions {', '.join(t['regions']) or 'n/a'}"
            )
            st.markdown(f"_{(t['evidence'] or [{}])[0].get('title', '')}_")
        shocks = pd.DataFrame({"shock": sc["shocks"], "driven by": sc["drivers"]})
        shocks = shocks.reindex(
            [
                f
                for f in [
                    "eq_SPY",
                    "eq_Financials",
                    "eq_Energy",
                    "rates_10y_bp",
                    "rates_3m_bp",
                    "credit_bbb_bp",
                    "oil",
                    "fx_EUR",
                    "fx_GBP",
                    "fx_JPY",
                    "vol_vix_pts",
                ]
                if f in shocks.index
            ]
        )
        st.dataframe(shocks, use_container_width=True, height=300)

    # heatmap sector x asset class (losses in red, sequential)
    hm = pd.DataFrame(res["heatmap_sector_x_asset"]).fillna(0)
    if not hm.empty:
        loss = (-hm).clip(lower=0)
        fig = go.Figure(
            go.Heatmap(
                z=loss.to_numpy(),
                x=list(loss.columns),
                y=list(loss.index),
                xgap=2,
                ygap=2,
                colorscale=[[i / (len(SEQ_RED) - 1), c] for i, c in enumerate(SEQ_RED)],
                customdata=hm.to_numpy(),
                hovertemplate="%{y} · %{x}: %{customdata:,.0f} USD<extra></extra>",
                colorbar={"title": "loss USD"},
            )
        )
        fig.update_layout(title="Loss heatmap: sector × asset class", height=420)
        st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Top 10 worst positions**")
    c1.dataframe(pd.DataFrame(res["top_worst_positions"]), use_container_width=True, height=380)
    c2.markdown("**Impact by rating and region (USD)**")
    c2.dataframe(pd.DataFrame({"by rating": pd.Series(res["by_rating"])}), use_container_width=True)
    c2.dataframe(pd.DataFrame({"by region": pd.Series(res["by_region"])}), use_container_width=True)

# ---------- scenario comparison ----------
st.subheader("Scenario comparison")
rows = []
for cls in ["GEOPOLITICAL", "MACROECONOMIC", "CREDIT_EVENT", "OPERATIONAL_ESG"]:
    for imp in (8, 10):
        r = run_stress(build_scenario(cls, imp), pos)
        rows.append(
            {
                "scenario": r["scenario"]["name"],
                "impact": imp,
                "total impact": usd(r["total_impact"]),
                "% of value": pct(r["total_impact_pct"], 2),
                "CET1 after": pct(r["capital"]["cet1_ratio_after"], 2),
            }
        )
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------- trigger frequency ----------
trig = data.metrics().get("moduleB_triggers", {})
if trig.get("stress_runs_per_month"):
    st.subheader("Trigger frequency")
    months = list(trig["stress_runs_per_month"])
    fig = go.Figure(
        go.Bar(
            x=months,
            y=[trig["stress_runs_per_month"][m] for m in months],
            marker_color=NAVY,
            hovertemplate="%{x}: %{y} stress runs<extra></extra>",
        )
    )
    fig.update_layout(height=260, yaxis_title="stress runs", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    blocked = trig.get("candidates_blocked_by", {})
    st.caption(
        f"{trig['n_triggers_fired']} triggers fired ({trig.get('n_escalations', 0)} escalations), "
        f"{trig['n_stress_runs']} stress runs over the replay year. Candidates blocked: "
        + ", ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in blocked.items())
        + ". Sep 2021 holds one replay day. Source: reports/metrics.json → moduleB_triggers."
    )

# ---------- out-of-sample validation ----------
FACTOR_UNITS = {
    "rates_10y_bp": "bp",
    "rates_3m_bp": "bp",
    "credit_bbb_bp": "bp",
    "vol_vix_pts": "pts",
}


def _fmt_factor(f: str, x: float) -> str:
    unit = FACTOR_UNITS.get(f)
    return f"{x:+.1f} {unit}" if unit else f"{x * 100:+.1f}%"


val = data.metrics().get("moduleB_validation", {})
if val.get("episodes"):
    st.subheader("Out-of-sample check: 2022 predicted vs realised")
    st.markdown(
        '<p class="rp-note">Scenario shocks are calibrated only on episodes before Sep 2021. For each '
        "2022 episode, the prediction is the stress test the trigger actually fired on the event date; "
        "realised moves are measured from the close before the event over the next 10 sessions. "
        "Reported, not tuned.</p>",
        unsafe_allow_html=True,
    )
    for name, e in val["episodes"].items():
        x = e if "factors" in e else e.get("supplementary_latest_run_in_prior_week")
        title = f"{name.replace('_', ' ')} ({e['event_date']}, {e['class']})"
        with st.expander(title, expanded=True):
            if "factors" not in e:
                st.markdown(f"**Missed:** {e['status'].split(': ', 1)[-1]}.")
                if x:
                    st.markdown(
                        "Supplementary context only (not the pre-registered comparison): the latest "
                        f"{e['class']} run in the preceding week, fired {x['trigger']['as_of'][:16]}."
                    )
            if not x:
                continue
            c = st.columns(4)
            c[0].markdown(
                kpi("Sign agreement", pct(x["sign_agreement"], 1)), unsafe_allow_html=True
            )
            c[1].markdown(
                kpi("Book impact, predicted", usd(x["book_total_impact_predicted"])),
                unsafe_allow_html=True,
            )
            c[2].markdown(
                kpi("Book impact, realised", usd(x["book_total_impact_realised"])),
                unsafe_allow_html=True,
            )
            c[3].markdown(
                kpi(
                    "CET1 after: predicted / realised",
                    f"{pct(x['cet1_after_predicted'], 2)} / {pct(x['cet1_after_realised'], 2)}",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(f"Trigger headline: _{x['headline']}_ · scenario `{x['scenario']}`")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "factor": r["factor"],
                            "predicted": _fmt_factor(r["factor"], r["predicted"]),
                            "realised": _fmt_factor(r["factor"], r["realised"]),
                            "same sign": "yes" if r["same_sign"] else "no",
                        }
                        for r in x["factors"]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

# ---------- inject demo event ----------
st.subheader("Inject demo event (synthetic)")
st.markdown(
    '<div class="rp-synth">Pushes a clearly labelled <b>synthetic_demo</b> headline through the live engine '
    "from several simulated outlets. It is not real news.</div>",
    unsafe_allow_html=True,
)
txt = st.text_input(
    "Synthetic headline",
    "SYNTHETIC DEMO: Major European bank defaults on debt; regulators seize lender",
)
n_out = st.slider("Simulated outlets", 2, 6, 3)
if st.button("Inject", type="primary"):
    code, r = data.api_post("/inject", {"text": txt, "outlets": n_out}, timeout=120)
    if code != 200:
        st.error(r.get("detail", "API error"))
    else:
        from riskpulse.common.schemas import Signal
        from riskpulse.moduleB.stress import stress_from_signal
        from riskpulse.moduleB.trigger import StressTrigger

        sigs = [Signal.model_validate(x) for x in r["signals"] if x["signal_type"] == "event"]
        st.write(f"{len(r['signals'])} signals emitted; {len(sigs)} event signals.")
        trig = StressTrigger()
        fired = None
        for s in sigs:
            d = trig.check(s)
            st.markdown(
                f"- {s.event_class.value} · impact {s.impact_score} · {s.n_sources} sources → **{d.reason}**"
            )
            if d.fired:
                fired = s
        if fired is not None:
            out = stress_from_signal(fired, pos)
            if out:
                st.success(
                    f"Stress test fired: {out['scenario']['name']} → total impact {usd(out['total_impact'])} "
                    f"({pct(out['total_impact_pct'], 2)}), CET1 {pct(out['capital']['cet1_ratio_after'], 2)}"
                )
