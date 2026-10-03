"""Page 1: Signal Monitor: feed, signals, evidence, heatmap, Analyze a headline."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from riskpulse.dashboard import data
from riskpulse.dashboard.theme import NAVY, SEQ_BLUE, event_color, kpi, setup_page

setup_page("Signal Monitor")
st.title("Signal Monitor")


@st.fragment(run_every=60)
def live_panel() -> None:
    """Current GDELT news scored by the engine; shown only when the API runs in live mode."""
    health = data.api_health()
    if not health or not health.get("live"):
        return
    res = data.api_get("/live", {"limit": 300}) or {}
    stt, rows = res.get("status", {}), res.get("mentions", [])
    st.subheader("Live feed: current news (live, unvalidated)")
    st.markdown(
        '<p class="rp-synth"><b>LIVE, UNVALIDATED.</b> Current GDELT news headlines scored by the same '
        "engine. News only (no free live social source). These outputs are not used for any metric; "
        "every reported number comes from the 2021-22 replay.</p>",
        unsafe_allow_html=True,
    )
    k = st.columns(4)
    k[0].markdown(
        kpi("Polls", f"{stt.get('polls', 0)}", f"last {str(stt.get('last_poll'))[:19]} UTC"),
        unsafe_allow_html=True,
    )
    k[1].markdown(
        kpi(
            "Headlines scored", f"{stt.get('processed', 0):,}", f"{stt.get('fetched', 0):,} fetched"
        ),
        unsafe_allow_html=True,
    )
    k[2].markdown(kpi("Mentions", f"{stt.get('mentions', 0):,}"), unsafe_allow_html=True)
    k[3].markdown(kpi("Signals emitted", f"{stt.get('signals', 0):,}"), unsafe_allow_html=True)
    if not rows:
        st.info("Waiting for the first poll (GDELT allows one request every 5 seconds).")
        return
    live = pd.DataFrame(rows)
    live["label"] = "live, unvalidated"
    st.dataframe(
        live[
            [
                "label",
                "published_at",
                "outlet",
                "ticker",
                "title",
                "sentiment",
                "event_class",
                "impact_score",
            ]
        ].style.format({"sentiment": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True,
        height=320,
    )
    st.divider()


live_panel()

sig = data.signals_df()
men = data.mentions_df()
if sig.empty or men.empty:
    st.warning("No processed signals found. Run `python -m riskpulse process` or `demo --fast`.")
    st.stop()

# ---------- filters (one row) ----------
lo, hi = men["published_at"].min().date(), men["published_at"].max().date()
f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
d0, d1 = f1.date_input(
    "Date range", (pd.Timestamp("2022-02-14").date(), pd.Timestamp("2022-03-11").date()), lo, hi
)
tickers = sorted(men["ticker"].unique())
pick = f2.multiselect("Entities", tickers, default=[])
classes = sorted(sig["event_class"].unique())
cls_pick = f3.multiselect("Event classes", classes, default=[])
min_imp = f4.slider("Min impact", 1, 10, 6)

t0, t1 = pd.Timestamp(d0, tz="UTC"), pd.Timestamp(d1, tz="UTC") + pd.Timedelta(days=1)
m = men[(men["published_at"] >= t0) & (men["published_at"] < t1)]
s = sig[(sig["as_of"] >= t0) & (sig["as_of"] < t1)]
if pick:
    m, s = m[m["ticker"].isin(pick)], s[s["ticker"].isin(pick)]
if cls_pick:
    m, s = m[m["event_class"].isin(cls_pick)], s[s["event_class"].isin(cls_pick)]
ev = s[(s["signal_type"] == "event") & (s["impact"] >= min_imp)]

k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Scored mentions", f"{len(m):,}"), unsafe_allow_html=True)
k2.markdown(kpi("Distinct stories", f"{m['event_id'].nunique():,}"), unsafe_allow_html=True)
k3.markdown(kpi(f"Event signals impact >= {min_imp}", f"{len(ev):,}"), unsafe_allow_html=True)
k4.markdown(
    kpi("Impact >= 8 (stress candidates)", f"{int((ev['impact'] >= 8).sum()):,}"),
    unsafe_allow_html=True,
)

# ---------- impact heatmap: entity x day ----------
st.subheader("Impact heatmap (max impact per entity per day)")
hm = (
    m.assign(day=m["published_at"].dt.date)
    .pivot_table(index="ticker", columns="day", values="impact_score", aggfunc="max")
    .fillna(0)
)
if not hm.empty:
    fig = go.Figure(
        go.Heatmap(
            z=hm.to_numpy(),
            x=[str(c) for c in hm.columns],
            y=hm.index,
            colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
            zmin=0,
            zmax=10,
            xgap=2,
            ygap=2,
            hovertemplate="%{y} on %{x}<br>max impact %{z}<extra></extra>",
            colorbar={"title": "impact"},
        )
    )
    fig.update_layout(height=max(320, 22 * len(hm)), margin={"t": 10})
    st.plotly_chart(fig, use_container_width=True)

# ---------- event signals with evidence ----------
st.subheader("Event signals")
if ev.empty:
    st.info("No event signals match the filters.")
else:
    top = ev.sort_values(["impact", "as_of"], ascending=[False, True]).drop_duplicates("event_id")
    table = top[
        [
            "as_of",
            "event_class",
            "impact",
            "sentiment",
            "event_confidence",
            "n_docs",
            "n_sources",
            "regions",
            "ticker",
        ]
    ].head(200)
    st.dataframe(
        table.style.format({"sentiment": "{:+.2f}", "event_confidence": "{:.2f}"}),
        use_container_width=True,
        height=300,
    )
    choice = st.selectbox(
        "Evidence drawer: pick a signal",
        top.index[:200],
        format_func=lambda i: (
            f"{top.at[i, 'as_of']:%Y-%m-%d %H:%M} | {top.at[i, 'event_class']} | "
            f"impact {top.at[i, 'impact']} | {(top.at[i, 'evidence'] or [{}])[0].get('title', '')[:90]}"
        ),
    )
    row = top.loc[choice]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Why {row['impact']}/10:** {row['explanation']}")
        for e in row["evidence"]:
            link = (
                f"[{e.get('title') or 'item'}]({e['url']})"
                if e.get("url")
                else (e.get("title") or "")
            )
            st.markdown(f"- {e['published_at'][:16]} · `{e['source']}` · {link}")
    with c2:
        drv = {k: v for k, v in (row["drivers"] or {}).items()}
        if drv:
            fig = go.Figure(
                go.Bar(
                    x=list(drv.values()),
                    y=list(drv.keys()),
                    orientation="h",
                    marker_color=event_color(row["event_class"]),
                    hovertemplate="%{y}: %{x:.2f}<extra></extra>",
                )
            )
            fig.update_layout(title="Impact drivers (0-1)", height=280, xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

# ---------- live document feed ----------
st.subheader("Document feed")
feed = m.sort_values("published_at", ascending=False).head(300)[
    ["published_at", "source", "ticker", "title", "sentiment", "event_class", "impact_score"]
]
st.dataframe(feed.style.format({"sentiment": "{:+.2f}"}), use_container_width=True, height=320)

# ---------- analyze a headline ----------
st.subheader("Analyze a headline")
st.markdown(
    '<p class="rp-note">Calls <code>POST /analyze</code> on the signal API '
    "(start it with <code>python -m riskpulse serve --full</code>).</p>",
    unsafe_allow_html=True,
)
text = st.text_input("Headline", "Moody's downgrades major regional bank to junk as deposits flee")
if st.button("Analyze", type="primary"):
    code, res = data.api_post("/analyze", {"text": text})
    if code != 200:
        st.error(res.get("detail", "API error"))
    else:
        a, b, c = st.columns(3)
        a.markdown(kpi("Sentiment", f"{res['sentiment_score']:+.2f}"), unsafe_allow_html=True)
        b.markdown(
            kpi(
                "Event class",
                res["event_class"].replace("_", " ").title(),
                f"confidence {res['event_confidence']:.2f}",
            ),
            unsafe_allow_html=True,
        )
        imp = max((e["impact_score"] for e in res["entities"]), default=None)
        c.markdown(
            kpi("Impact (1-10)", str(imp) if imp else "n/a", "dry run, no stream context"),
            unsafe_allow_html=True,
        )
        probs = pd.Series(res["event_probs"]).sort_values()
        fig = px.bar(probs, orientation="h", color_discrete_sequence=[NAVY])
        fig.update_layout(title="Event-class probabilities", showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.json(res, expanded=False)
