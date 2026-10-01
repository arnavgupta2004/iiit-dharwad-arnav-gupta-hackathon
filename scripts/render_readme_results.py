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
        p = a["performance"]
        out += [
            f"**Module A** (daily backtest {a['window'][0]} to {a['window'][1]}, net of 5 bps costs; a sentiment-tilt "
            "demonstration, not an alpha claim)",
            "",
            "| Strategy | Cumulative return | Sharpe (rf = 0) | Max drawdown | Avg daily turnover |",
            "|---|---|---|---|---|",
        ]
        for k, v in p.items():
            out.append(
                f"| {k.replace('_', ' ')} | {pct(v['cumulative_return'])} | {v['sharpe_rf0']} | "
                f"{pct(v['max_drawdown'])} | {pct(v['avg_daily_one_way_turnover'], 2)} |"
            )
        ic = a.get("information_coefficient", {})
        out += [
            "",
            f"Information coefficient (daily Spearman, s_t vs r_t+1): mean {ic.get('mean_ic', 'TBD')}, "
            f"t-stat {ic.get('t_stat', 'TBD')}, hit rate {ic.get('hit_rate', 'TBD')} over {ic.get('n_days', 'TBD')} days.",
            "",
        ]
    b = m.get("moduleB_triggers")
    if b:
        out += [
            f"**Module B** (replay 2021-09 to 2022-09): {b['n_high_impact_candidates']:,} high-impact candidates, "
            f"{b['n_triggers_fired']} triggers fired, {b['n_stress_runs']} stress runs.",
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
    imp = m.get("impact")
    out.append(
        "**Impact score validation:** "
        + ("see reports/metrics.json → impact." if imp else "TBD (event study, P1).")
    )
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
