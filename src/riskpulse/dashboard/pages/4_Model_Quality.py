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

sg, ft = m.get("sentiment_gold"), m.get("sentiment_finetune")
if sg:
    st.subheader("Live-feed check on the gold set (pre-registered, D-041)")
    rows = []
    for sub, r in sg["results"].items():
        for meth, f1 in r["macro_f1"].items():
            rows.append(
                {"text type": sub.replace("_", " "), "n": r["n"], "model": meth, "macro-F1": f1}
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    news = sg["results"]["news_headlines"]
    st.markdown(
        f"Fine-tuned − FinBERT on news: {news['diff_finetuned_minus_finbert']:+.3f}, 95% CI "
        f"{news['diff_95ci']}. Decision by the pre-registered rule: {sg['decision']}. "
        + (
            f"In-domain test (HF split, same dataset as training): fine-tuned "
            f"{ft['test_macro_f1']:.3f}."
            if ft
            else ""
        )
    )
    provenance(sg)

lk = m.get("entity_linking")
if lk and lk.get("precision") is not None:
    st.subheader("Entity linking")
    st.markdown(
        f"Precision on {lk['n_marked']} hand-checked headline links: "
        f"**{lk['precision'] * 100:.1f}%** (95% Wilson CI "
        f"{lk['precision_95ci_wilson'][0] * 100:.1f}–{lk['precision_95ci_wilson'][1] * 100:.1f}%)."
    )
    provenance(lk)

# ---------- events ----------
st.header("Event classification")
e = m.get("events")
if e:
    st.markdown(f"**Status:** {e['status']}")
    rows = []
    gold = e["results"].get("gold")
    if gold:
        for sub in ("all", "news_headlines", "tweets"):
            for meth, r in gold[sub].items():
                if isinstance(r, dict) and "macro_f1" in r:
                    rows.append(
                        {
                            "evaluation set": f"GOLD {sub.replace('_', ' ')}",
                            "method": meth,
                            "n": r["n"],
                            "macro-F1": r["macro_f1"],
                        }
                    )
    for ds, res in e["results"].items():
        if ds == "gold":
            continue
        for meth, r in res.items():
            rows.append(
                {
                    "evaluation set": {
                        "hf_topic_valid": "in-domain (HF topic data)",
                        "hf_topic_valid_subsample": "in-domain (HF topic data, subsample)",
                        "weak_holdout": "weak-label hold-out (circular)",
                    }.get(ds, ds),
                    "method": meth,
                    "n": r["n"],
                    "macro-F1": r["macro_f1"],
                }
            )
    df = pd.DataFrame(rows)
    if gold:
        st.markdown(
            f"**Gold set (Arnav's labels, n = {gold['n']}, cross-border GEOPOLITICAL):** paired-bootstrap "
            "95% CIs of macro-F1(trained) − macro-F1(baseline): "
            + "; ".join(
                f"{sub.replace('_', ' ')} vs keyword {gold[sub]['primary_minus_keyword_95ci']}, "
                f"vs zero-shot {gold[sub]['primary_minus_zero_shot_95ci']}"
                for sub in ("all", "news_headlines", "tweets")
            )
            + ". HF-topic rows are in-domain for the trained model."
        )
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
    ("impact_v1", "Impact v1 (spec formula; live for market-wide items)", SERIES[0]),
    ("abs_sent", "abs(sentiment) baseline", MUTED),
]
if v2:
    car = v2["test_post_burn_in"]["abs_car01"]
    vol = v2["test_post_burn_in"]["abn_volume"]
    st.markdown(
        f'<p class="rp-note">Does a higher score mean a bigger market reaction? 2021-22 test set: '
        f"{car['n']:,} ticker-days after the burn-in; market-model CAR[0,+1] on SPY. Impact v2 scores "
        "company mentions; market-wide items keep v1.</p>",
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
        f"v2 − abs(sentiment) {car['rho_diff_v2_minus_abs_sent_95ci']}."
    )
    st.markdown(
        f"**Adoption.** Rule written before the first test ({v2.get('pre_registered_rule', '')}): "
        f"**{'met' if v2.get('pre_registered_rule_met') else 'not met'}**. Applied rule, revised after "
        f"the first test was seen ({v2.get('applied_rule', '')}): "
        f"**{'met' if v2.get('adopted') else 'not met'}**, so v2 is "
        f"{'live for company mentions' if v2.get('adopted') else 'not used'} (D-039). In short: v2 is "
        "on par with abs(sentiment) for predicting the market reaction and adds explainable drivers."
    )
    for lim in v2.get("limitations", []):
        st.markdown(f"- {lim}")
    provenance(v2)
elif imp:
    st.json(imp, expanded=False)
    provenance(imp)
mc = m.get("impact_market_check")
if mc:
    st.subheader("Market-wide events (v1): descriptive check, D-042")
    st.markdown(
        f"{mc['n_days']} sessions after the burn-in: Spearman of the day's max market-wide impact vs "
        f"|SPY return| {mc['vs_abs_spy_return']['spearman_rho']:+.3f} "
        f"(CI {mc['vs_abs_spy_return']['ci95']}), vs |ΔVIX| {mc['vs_abs_dvix']['spearman_rho']:+.3f} "
        f"(CI {mc['vs_abs_dvix']['ci95']}). Reported only; nothing tuned."
    )
    provenance(mc)
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
