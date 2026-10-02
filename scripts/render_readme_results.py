"""Fill the README "Key Results" block from reports/metrics.json (no hand-typed numbers).

Replaces everything between <!-- RESULTS:START --> and <!-- RESULTS:END -->.

    python scripts/render_readme_results.py
"""

from __future__ import annotations

import json
import re

from riskpulse.common.config import repo_root


def pct(x: float | None, d: int = 1) -> str:
    return "TBD" if x is None else f"{x * 100:.{d}f}%"


REF_LABELS = {
    "hf_topic_valid": "in-domain (HF topic data)",
    "hf_topic_valid_subsample": "in-domain (HF topic data, subsample)",
    "weak_holdout": "weak-label hold-out (circular)",
}


def usd(x: float) -> str:
    a = abs(x)
    return (
        f"{'-' if x < 0 else '+'}USD {a / 1e9:.2f} bn"
        if a >= 1e9
        else f"{'-' if x < 0 else '+'}USD {a / 1e6:.1f} m"
    )


def build(m: dict) -> str:
    out: list[str] = []
    s = m.get("sentiment")
    if s:
        r = s["results"]
        out += [
            f"**Sentiment** (held-out Twitter Financial News, n = {s['n_test']:,}; bands tuned on train only)",
            "",
            "| Method | Macro-F1 | Accuracy |",
            "|---|---|---|",
            f"| FinBERT (train-tuned band) | {r['finbert']['train_tuned']['macro_f1']:.3f} | "
            f"{r['finbert']['train_tuned']['accuracy']:.3f} |",
            f"| FinBERT (argmax) | {r['finbert']['argmax']['macro_f1']:.3f} | {r['finbert']['argmax']['accuracy']:.3f} |",
            f"| Loughran-McDonald lexicon | {r['lm']['train_tuned']['macro_f1']:.3f} | {r['lm']['train_tuned']['accuracy']:.3f} |",
            f"| VADER | {r['vader']['train_tuned']['macro_f1']:.3f} | {r['vader']['train_tuned']['accuracy']:.3f} |",
            f"| Majority class (neutral) | {r['majority_class_neutral']['macro_f1']:.3f} | "
            f"{r['majority_class_neutral']['accuracy']:.3f} |",
            "",
        ]
    ft, sg = m.get("sentiment_finetune"), m.get("sentiment_gold")
    if ft and sg:
        n, t = sg["results"]["news_headlines"], sg["results"]["tweets"]
        out += [
            "**Sentiment models in use: FinBERT for news, FinBERT fine-tuned on the tweet train split for tweets** "
            f"(D-058). The fine-tuned model scores {ft['test_macro_f1']:.3f} in-domain (same dataset as its training "
            "data). On live-feed text (Arnav's gold-1 labels, rule pre-registered before scoring, D-041):",
            "",
            "| Live-feed gold | n | FinBERT | Fine-tuned | Fine-tuned − FinBERT (95% CI) | VADER | LM |",
            "|---|---|---|---|---|---|---|",
            f"| News headlines | {n['n']} | {n['macro_f1']['finbert']:.3f} | {n['macro_f1']['finetuned']:.3f} | "
            f"{n['diff_finetuned_minus_finbert']:+.3f} {n['diff_95ci']} | {n['macro_f1']['vader']:.3f} | {n['macro_f1']['lm']:.3f} |",
            f"| Tweets | {t['n']} | {t['macro_f1']['finbert']:.3f} | {t['macro_f1']['finetuned']:.3f} | "
            f"{t['diff_finetuned_minus_finbert']:+.3f} {t['diff_95ci']} | {t['macro_f1']['vader']:.3f} | {t['macro_f1']['lm']:.3f} |",
            "",
            "The in-domain gain does not transfer to news.",
            "",
        ]
        pooled = m.get("sentiment_news_pooled")
        if pooled:
            out += [
                f"Pooled news check (gold-1 + gold-2 news, n = {pooled['n']}, rule pre-registered in D-057): fine-tuned "
                f"{pooled['macro_f1']['finetuned']:.3f} vs FinBERT {pooled['macro_f1']['finbert']:.3f}, difference "
                f"{pooled['diff_finetuned_minus_finbert']:+.3f}, 95% CI {pooled['diff_95ci']} (entirely below 0), so news "
                "switched to FinBERT. This was a post-hoc revision prompted by the gold-2 evidence (D-058).",
                "",
            ]
    e, g2, cv = m.get("events"), m.get("events_gold2"), m.get("events_round2_cv")
    if g2:
        r = g2["results"]
        names = {
            "selected_C1": "**Deployed: C1** (weak labels + gold-1)",
            "previous_deployed": "Previous model (weak labels only)",
            "keyword_baseline": "Keyword baseline",
            "zero_shot_base": "Zero-shot NLI (base)",
            "zero_shot_large_reference": "Zero-shot NLI (large, reference; too slow for the feed on CPU)",
        }
        out += [
            f"**Event classification: final test on gold-2** (n = {g2['n']}, Arnav's labels, evaluated once; model "
            "fixed before the labels were read, D-052/D-053). Macro-F1 over 10 classes:",
            "",
            f"| Method | All (n {r['all']['n']}) | News (n {r['news_headlines']['n']}) | Tweets (n {r['tweets']['n']}) | "
            "C1 − method, 95% CI (all) |",
            "|---|---|---|---|---|",
        ]
        for k, label in names.items():
            ci = r["all"]["selected_minus_95ci"].get(k, "")
            out.append(
                f"| {label} | {r['all'][k]['macro_f1']:.3f} | {r['news_headlines'][k]['macro_f1']:.3f} | "
                f"{r['tweets'][k]['macro_f1']:.3f} | {ci if ci else '-'} |"
            )
        out.append("")
        if cv:
            sm = cv["summary"]
            out += [
                f"Selection by cross-validation on gold-1 (15 folds): C1 {sm['C1']['mean']:.3f}, C2 {sm['C2']['mean']:.3f}, "
                f"C3 {sm['C3']['mean']:.3f}; the one-SE rule chose {cv['chosen_by_one_se_rule']} (D-053). "
                "GEOPOLITICAL is cross-border only (D-044).",
                "",
            ]
    if e:
        out += [
            "In-domain reference sets (not the final test):",
            "",
            "| Evaluation set | Method | n | Macro-F1 |",
            "|---|---|---|---|",
        ]
        for ds, res in e["results"].items():
            if ds == "gold":
                continue
            for meth, v in res.items():
                out.append(
                    f"| {REF_LABELS.get(ds, ds)} | {meth} | {v['n']:,} | {v['macro_f1']:.3f} |"
                )
        out.append("")
    lk = m.get("entity_linking")
    if lk and lk.get("precision") is not None:
        out += [
            f"**Entity linking:** precision {pct(lk['precision'])} on {lk['n_marked']} hand-checked headline links "
            f"(95% Wilson CI {pct(lk['precision_95ci_wilson'][0])}–{pct(lk['precision_95ci_wilson'][1])}).",
            "",
        ]
    a = m.get("moduleA")
    if a:
        ic, cal, w = a["headline_information_coefficient"], a["calibration"], a["evaluation_window"]
        out += [
            f"**Module A, headline: information coefficient** (evaluation {w[0]} to {w[1]}): mean daily IC "
            f"{ic['mean_ic']:+.4f}, t-stat {ic['t_stat']}, IC > 0 on {pct(ic['pct_positive_days'])} of "
            f"{ic['n_days']} days (daily Spearman of decayed sentiment vs next-day return).",
            "",
            f"Tilt strength κ = {cal['kappa']:.2f}, frozen by a risk budget (median absolute active weight "
            f"{pct(cal['target'], 1)}) computed from signals only over {cal['calibration_window'][0]} to "
            f"{cal['calibration_window'][1]}; returns were never used to set it.",
            "",
            "Returns (secondary; net of 5 bps costs; a sentiment-tilt demonstration, not an alpha claim):",
            "",
            "| Strategy | Gross return | Cost drag | Net return | Sharpe (rf = 0) | Max drawdown | Avg daily turnover |",
            "|---|---|---|---|---|---|---|",
        ]
        for k, v in a["performance_secondary"].items():
            out.append(
                f"| {k.replace('_', ' ')} | {pct(v.get('cumulative_return_gross', float('nan')))} | "
                f"{pct(v['total_cost_drag'], 2)} | {pct(v['cumulative_return'])} | {v['sharpe_rf0']} | "
                f"{pct(v['max_drawdown'])} | {pct(v['avg_daily_one_way_turnover'], 2)} |"
            )
        out.append("")
        out.append(
            f"Turnover cap {pct(a['config']['tau_max'], 0)} one-way per day, set by policy for operational realism "
            "(D-045), not optimised on returns."
        )
        out.append("")
    b = m.get("moduleB_triggers")
    if b:
        out += [
            f"**Module B** (replay 2021-09 to 2022-09): {b['n_high_impact_candidates']:,} high-impact candidates, "
            f"{b['n_triggers_fired']} triggers fired ({b.get('n_escalations', 0)} escalations), {b['n_stress_runs']} "
            "stress runs. Cooldown: 24 h per event class and macro-region, re-run within it only on higher impact. "
            "Stress runs per month: "
            + ", ".join(f"{k} {v}" for k, v in b.get("stress_runs_per_month", {}).items())
            + ". First runs:",
            "",
            "| Trigger time (UTC) | Scenario | Impact | Sources | Total impact | CET1 after |",
            "|---|---|---|---|---|---|",
        ]
        for r in b["runs"][:12]:
            out.append(
                f"| {r['as_of'][:16]} | {r['scenario']} | {r['impact']} | {r['n_sources']} | "
                f"{pct(r['total_impact_pct'], 2)} | {pct(r['cet1_after'], 2)} |"
            )
        out.append("")
    pl = m.get("pipeline")
    if pl:
        out += [
            f"**Pipeline** ({pl['device']}, {pl['n_cpu']} cores): {pl['batch_docs_per_sec']} docs/s batch; single-document "
            f"latency p50 {pl['single_doc_latency_ms_p50']} ms, p95 {pl['single_doc_latency_ms_p95']} ms; dedup removed "
            f"{pct(pl['dedup_rate_news'])} of news and {pct(pl['dedup_rate_social'])} of social items.",
            "",
        ]
    v2 = m.get("impact_v2")
    imp = m.get("impact")
    if v2:
        t = v2["test_post_burn_in"]
        car, vol = t["abs_car01"], t["abn_volume"]
        rows = [
            ("Impact v2 (learned on Benzinga ≤ 2018)", "impact_v2"),
            ("Impact v1 (spec formula; live for market-wide items)", "impact_v1"),
            ("abs(sentiment) only (baseline)", "abs_sent"),
        ]
        out += [
            f"**Impact score vs realised market reaction** (2021-22 test set, every look at it listed in D-050: ticker-days after the burn-in, "
            f"n = {car['n']:,}; market-model abnormal returns, CAR[0,+1])",
            "",
            "| Score | Spearman vs abs(CAR) | Top-decile hit rate (10% by chance) | Spearman vs abnormal volume |",
            "|---|---|---|---|",
        ]
        for label, k in rows:
            out.append(
                f"| {label} | {car[k]['spearman_rho']:.3f} | {pct(car[k]['top_decile_hit_rate'])} | "
                f"{vol[k]['spearman_rho']:.3f} |"
            )
        out += [
            "",
            f"95% bootstrap CI of the Spearman difference: v2 − v1 {car['rho_diff_v2_minus_v1_95ci']}, "
            f"v2 − abs(sentiment) {car['rho_diff_v2_minus_abs_sent_95ci']}. **Impact v2 is on par with abs(sentiment) for "
            "predicting the market reaction and adds explainable drivers; it clearly beats the hand-set v1.** It scores "
            "company mentions live; market-wide items keep v1. Adoption: the stricter rule written before the first test "
            f"(beat both v1 and abs(sentiment)) was {'met' if v2.get('pre_registered_rule_met') else 'not met'}; the rule "
            'was revised after that test to the spec\'s "beats v1" on validation and test (D-039), which '
            f"{'is met' if v2.get('adopted') else 'is not met'}.",
            "",
        ]
    mc = m.get("impact_market_check")
    if mc:
        out += [
            f"**Market-wide impact (v1), descriptive check (D-042):** over {mc['n_days']} sessions the day's maximum "
            f"market-wide impact has Spearman {mc['vs_abs_spy_return']['spearman_rho']:+.3f} with abs(SPY return) "
            f"(CI {mc['vs_abs_spy_return']['ci95']}) and {mc['vs_abs_dvix']['spearman_rho']:+.3f} with abs(ΔVIX) "
            f"(CI {mc['vs_abs_dvix']['ci95']}): no measurable relation.",
            "",
        ]
    elif imp:
        x = imp["replay_window_post_burn_in"]
        out.append(
            f"**Impact v1 vs abs(sentiment):** Spearman {x['abs_car01']['impact_v1']['spearman_rho']:.3f} vs "
            f"{x['abs_car01']['abs_sent']['spearman_rho']:.3f}."
        )
    else:
        out.append("**Impact score validation:** TBD (event study, P1).")
    val = m.get("moduleB_validation")
    if val:
        out += [
            "**Module B out-of-sample check (2022 episodes; scenarios calibrated before Sep 2021 only)**",
            "",
            "| Episode | Status | Sign agreement | Book impact predicted | Book impact realised |",
            "|---|---|---|---|---|",
        ]
        for name, e in val["episodes"].items():
            x = e if "factors" in e else e.get("supplementary_latest_run_in_prior_week")
            status = (
                "fired on the day" if "factors" in e else "missed (supplementary: prior-week run)"
            )
            if x:
                out.append(
                    f"| {name} ({e['event_date']}) | {status} | {pct(x['sign_agreement'])} | "
                    f"{usd(x['book_total_impact_predicted'])} | {usd(x['book_total_impact_realised'])} |"
                )
            else:
                out.append(f"| {name} ({e['event_date']}) | {status} | n/a | n/a | n/a |")
        out.append("")
    ft = m.get("sentiment_finetune")
    if ft:
        out += [
            f"**Sentiment fine-tune experiment** (FinBERT on tweet train split only, {ft['budget_minutes']:.0f}-min CPU "
            f"budget): test macro-F1 {ft['test_macro_f1']:.3f} vs bar {ft['bar_finbert_train_tuned']:.3f}; {ft['decision']}.",
            "",
        ]
    return "\n".join(out)


def main() -> None:
    m = json.loads((repo_root() / "reports" / "metrics.json").read_text())
    readme = repo_root() / "README.md"
    text = readme.read_text()
    block = f"<!-- RESULTS:START -->\n{build(m)}\n<!-- RESULTS:END -->"
    new = re.sub(
        r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, text, flags=re.S
    )
    readme.write_text(new)
    print("README results block updated")


if __name__ == "__main__":
    main()
