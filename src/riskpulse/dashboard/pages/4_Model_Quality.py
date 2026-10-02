"""Page 4: Model Quality: every figure here is read from reports/metrics.json."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskpulse.dashboard import data
from riskpulse.dashboard.theme import MUTED, NAVY, SEQ_BLUE, SERIES, setup_page

setup_page("Model Quality")
st.title("Model Quality")
m = data.metrics()
if not m:
    st.warning("No metrics yet. Run `python -m riskpulse eval all`.")
    st.stop()


def provenance(section: dict) -> None:
    p = section.get("_provenance", {})
    st.markdown(
        f'<p class="rp-note">Source: <code>{p.get("script")}</code> · generated {p.get("generated_at")} '
        f"· commit {p.get('git_commit')}</p>",
        unsafe_allow_html=True,
    )


def confusion(cm: dict, title: str) -> go.Figure:
    z = cm["matrix"]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=cm["labels"],
            y=cm["labels"],
            xgap=2,
            ygap=2,
            showscale=False,
            colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
            text=z,
            texttemplate="%{text}",
            hovertemplate="actual %{y} → predicted %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        xaxis_title="predicted",
        yaxis_title="actual",
        yaxis_autorange="reversed",
    )
    return fig


# ---------- sentiment ----------
st.header("Sentiment")
s = m.get("sentiment")
if s:
    st.markdown(
        f"Test set: **{s['dataset']}** ({s['test_split']} split, n = {s['n_test']:,}); "
        f"neutral bands tuned on the train split only. {s.get('note', '')}"
    )
    rows = []
    for name, r in s["results"].items():
        if name == "majority_class_neutral":
            rows.append(
                {
                    "method": "Majority class (neutral)",
                    "macro-F1": r["macro_f1"],
                    "accuracy": r["accuracy"],
                }
            )
            continue
        rows.append(
            {
                "method": f"{name.upper()} (default band {r['default_band']})",
                "macro-F1": r["default"]["macro_f1"],
                "accuracy": r["default"]["accuracy"],
            }
        )
        rows.append(
            {
                "method": f"{name.upper()} (train-tuned band {r['train_tuned_band']})",
                "macro-F1": r["train_tuned"]["macro_f1"],
                "accuracy": r["train_tuned"]["accuracy"],
            }
        )
        if "argmax" in r:
            rows.append(
                {
                    "method": f"{name.upper()} (argmax)",
                    "macro-F1": r["argmax"]["macro_f1"],
                    "accuracy": r["argmax"]["accuracy"],
                }
            )
    df = pd.DataFrame(rows).sort_values("macro-F1")
    fig = go.Figure(
        go.Bar(
            x=df["macro-F1"],
            y=df["method"],
            orientation="h",
            marker_color=[NAVY if "FINBERT" in x else MUTED for x in df["method"]],
            text=df["macro-F1"].map("{:.3f}".format),
            textposition="outside",
        )
    )
    fig.update_layout(title="Macro-F1 on held-out financial tweets", height=380, xaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Table and confusion matrix"):
        st.dataframe(df, use_container_width=True)
        st.plotly_chart(
            confusion(
                s["results"]["finbert"]["train_tuned"]["confusion_matrix"],
                "FinBERT (train-tuned band)",
            ),
            use_container_width=True,
        )
    provenance(s)

# ---------- events ----------
st.header("Event classification")
e = m.get("events")
if e:
    st.markdown(f"**Status:** {e['status']}")
    rows = []
    for ds, res in e["results"].items():
        for meth, r in res.items():
            rows.append(
                {"evaluation set": ds, "method": meth, "n": r["n"], "macro-F1": r["macro_f1"]}
            )
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for i, meth in enumerate(df["method"].unique()):
        d = df[df["method"] == meth]
        fig.add_trace(
            go.Bar(
                x=d["evaluation set"],
                y=d["macro-F1"],
                name=meth,
                marker_color=SERIES[i % len(SERIES)],
                text=d["macro-F1"].map("{:.3f}".format),
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group", height=380, yaxis_range=[0, 1], title="Macro-F1 by evaluation set"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)
    for note in e.get("notes", []):
        st.markdown(f"- {note}")
    provenance(e)

# ---------- impact ----------
st.header("Impact score")
imp, v2 = m.get("impact"), m.get("impact_v2")
SCORES = [
    ("impact_v2", "Impact v2 (learned on Benzinga ≤ 2018)", NAVY),
    ("impact_v1", "Impact v1 (live, spec formula)", SERIES[0]),
    ("abs_sent", "abs(sentiment) baseline", MUTED),
]
if v2:
    car = v2["test_post_burn_in"]["abs_car01"]
    vol = v2["test_post_burn_in"]["abn_volume"]
    st.markdown(
        f'<p class="rp-note">Does a higher score mean a bigger market reaction? Untouched 2021-22 test set: '
        f"{car['n']:,} ticker-days after the burn-in; market-model CAR[0,+1] on SPY. Pre-registered rule: "
        f"{v2['adoption_rule']}.</p>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "score": label,
                    "Spearman vs |CAR|": f"{car[k]['spearman_rho']:.3f}",
                    "top-decile hit rate": f"{car[k]['top_decile_hit_rate'] * 100:.1f}%",
                    "Spearman vs abnormal volume": f"{vol[k]['spearman_rho']:.3f}",
                }
                for k, label, _ in SCORES
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    fig = go.Figure()
    for k, label, color in SCORES:
        fig.add_trace(
            go.Bar(
                x=list(range(1, 11)),
                y=[x * 100 for x in car[k]["decile_means"]],
                name=label,
                marker_color=color,
                hovertemplate="decile %{x}: mean |CAR| %{y:.2f}%<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=340,
        title="Mean |CAR[0,+1]| by score decile (1 = lowest)",
        yaxis_title="%",
        xaxis={"dtick": 1},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"95% bootstrap CI of the Spearman difference: v2 − v1 {car['rho_diff_v2_minus_v1_95ci']}, "
        f"v2 − abs(sentiment) {car['rho_diff_v2_minus_abs_sent_95ci']}. "
        f"**Adopted: {'yes' if v2['adopted'] else 'no'}.** v2 beats v1 but not the sentiment-only "
        "baseline, so the live engine keeps v1 and this result is reported as is (D-039)."
        if not v2["adopted"]
        else "**Adopted.**"
    )
    for lim in v2.get("limitations", []):
        st.markdown(f"- {lim}")
    provenance(v2)
elif imp:
    st.json(imp, expanded=False)
    provenance(imp)
else:
    st.info(
        "Impact validation (event study against abnormal returns) is TBD: produced in the P1 phase."
    )

# ---------- pipeline & data ----------
st.header("Pipeline and data")
fs = data.feed_stats()
if fs:
    c1, c2 = st.columns(2)
    c1.markdown("**Replay feed composition**")
    c1.json(
        {
            k: fs[k]
            for k in [
                "n_docs",
                "by_source",
                "news_filter",
                "news_dedupe",
                "social_filter",
                "social_dedupe",
            ]
        },
        expanded=False,
    )
    bt = pd.Series(fs["by_ticker"]).sort_values()
    fig = go.Figure(go.Bar(x=bt.to_numpy(), y=bt.index, orientation="h", marker_color=NAVY))
    fig.update_layout(title="Documents per entity in the replay feed", height=520, xaxis_type="log")
    c2.plotly_chart(fig, use_container_width=True)
pb = m.get("pipeline")
if pb:
    st.json(pb, expanded=False)
    provenance(pb)
else:
    st.caption("Pipeline throughput and latency: TBD (eval pipeline).")
