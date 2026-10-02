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
    e = m.get("events")
    if e:
        out += [
            f"**Event classification** ({e['status']})",
            "",
            "| Evaluation set | Method | n | Macro-F1 |",
            "|---|---|---|---|",
        ]
        for ds, res in e["results"].items():
            for meth, v in res.items():
                out.append(f"| {ds} | {meth} | {v['n']:,} | {v['macro_f1']:.3f} |")
        out.append("")
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
            "| Strategy | Cumulative return | Sharpe (rf = 0) | Max drawdown | Avg daily turnover |",
            "|---|---|---|---|---|",
        ]
        for k, v in a["performance_secondary"].items():
            out.append(
                f"| {k.replace('_', ' ')} | {pct(v['cumulative_return'])} | {v['sharpe_rf0']} | "
                f"{pct(v['max_drawdown'])} | {pct(v['avg_daily_one_way_turnover'], 2)} |"
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
            ("Impact v1 (live, spec formula)", "impact_v1"),
            ("abs(sentiment) only (baseline)", "abs_sent"),
        ]
        out += [
            f"**Impact score vs realised market reaction** (untouched 2021-22 test set: ticker-days after the burn-in, "
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
            f"v2 − abs(sentiment) {car['rho_diff_v2_minus_abs_sent_95ci']}. Pre-registered rule: {v2['adoption_rule']}. "
            f"Adopted: {'yes' if v2['adopted'] else 'no'}; the live engine keeps v1 and this comparison is reported as is.",
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
