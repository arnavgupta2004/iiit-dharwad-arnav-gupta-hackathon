# RiskPulse: deck content (DRAFT v3, 7 slides, after the feature freeze)

Numbers are taken from `reports/metrics.json` after the final rerun (D-060/D-061) and match the README "Results at a
glance". Re-check them before exporting the PDF. Organisers' outline: Title, Problem and Approach, System Design,
Implementation Highlights, Key Results, Domain Impact, Limitations and Next Steps. The hook (the 2022 event) comes first.

Style: white background, navy/charcoal text, red/green only for negative/positive values, no gradients, no emoji,
one idea per slide, at most about 25 words of body text plus one visual. Percentages to one decimal, "USD 10.7 m",
bp for rates and spreads.

---

## Slide 1: Title + hook

**22 February 2022, 00:00 UTC: RiskPulse triggers a geopolitical stress test.**
Two days later, Russia launches its full-scale invasion of Ukraine.

RiskPulse: news and social sentiment as a real-time risk signal
Code to Connect 2026 (S&P Global and Crisil), Phase 3 · Arnav Gupta, IIIT Dharwad

Visual: Module B trigger timeline for 20-25 Feb 2022 (diamonds on 22 Feb), muted.

Speaker notes (20 s): "This came from replaying a year of real news through the system in time order: at midnight
UTC on 22 February it flagged Russia's recognition of the separatist regions as a high-impact, adverse geopolitical
event and stress-tested a USD 10 bn wholesale book. The troop order followed at 06:00. RiskPulse turns news and posts
into risk signals and acts on them."
Careful wording: the system flagged the **escalation**, not the invasion itself.

---

## Slide 2: Problem and approach

- Markets react to news within hours; risk teams read it in batches.
- A headline isn't a risk signal until it has an **entity**, an **event type** and an **impact level**.

Three fields per signal: **sentiment** (−1…+1, per company), **event class** (10, geopolitical = cross-border),
**impact** (1–10, learned against abnormal returns).
One signal feeds **Module A** (index tilt) and **Module B** (event-triggered stress test).

---

## Slide 3: System design

Visual: `docs/architecture.png`.

- 203,363 real documents (165,782 GDELT news, 37,581 tweets), 2021-09 → 2022-09, replayed in time order.
- Entity linking → FinBERT (news) / tweet-tuned FinBERT (tweets) → event classifier → story clustering → impact
  (learned model for companies, formula for market-wide).
- FastAPI (REST + live stream) and a Streamlit dashboard; CPU only; `python -m riskpulse demo --fast` on a fresh clone.

---

## Slide 4: Implementation highlights (what and why)

- **Entity-level sentiment:** "JPM beats while BAC misses" scores each bank separately. News uses FinBERT (better
  on labelled live news); tweets use the tweet-fine-tuned model (the two score about the same on tweets).
- **Impact v2:** monotone gradient boosting learned on 2009-18 Benzinga headlines vs abnormal returns. Scored live
  from per-session aggregates (no look-ahead), with every score's drivers shown.
- **Stress triggers:** impact ≥ 8, confident class, ≥ 2 outlets, adverse sentiment. Scenarios come from pre-2021
  historical analogues only.
- **Discipline:** decision rules pre-registered before tests, every look at the test window logged, and the small
  trained models committed so a fresh clone runs the reported system.

---

## Slide 5: Key results (with why to trust each)

| Result | Trust |
|---|---|
| 2022 Russia–Ukraine: stress test at 22 Feb 00:00 UTC; 9 of 14 factor directions right | Out-of-sample: calibrated on pre-2021 episodes only |
| Module A information coefficient +0.047, t 2.85 (IC > 0 on 56.9% of 209 days) | Tilt strength fixed without looking at returns |
| Impact v2 vs v1: Spearman 0.111 vs 0.055 (gain CI [0.029, 0.086]); on par with abs(sentiment) 0.110 | 2021-22 test; trained on 2009-18 |
| Sentiment on live news: FinBERT 0.592 vs VADER 0.471, Loughran-McDonald 0.442 | Human-labelled (n 237) |
| Entity linking precision 82.0% (CI 73.3–88.3%) | 100 human-checked links |
| 86.6 docs/s; 21.3 ms p95 per document | Laptop CPU, no GPU |

Efficiency vs naive: 3,382 high-impact items → 345 stress tests a year (adverse, multi-source, deduplicated by
region and class); dedup removes 39.0% of news before scoring.

Visual: `reports/figures/impact_v2_deciles.png`, small.

---

## Slide 6: Domain impact

- **Risk team (Module B):** an adverse geopolitical story triggers a stress test with P&L, ΔEL and CET1 on a USD 10 bn
  synthetic book. On the 24 Feb run: predicted −USD 10.7 m, CET1 13.00% → 12.98%, every shock traced to named
  analogues (Crimea 2014, Brexit 2016, Abqaiq 2019).
- **Portfolio manager (Module A):** sentiment-tilted index weights with the headlines behind each move. Turnover is
  capped at 5% a day; cost drag 0.5%.
- **Analyst:** one screen shows why a story scores 8/10.

Visual: Module B page (ranked stress runs, top 3 per month). All counterparties are synthetic (CP_xxxx).

---

## Slide 7: Limitations and next steps (measured)

1. Event classification: two rounds (300 + 200 hand labels, one blind test) did not beat a keyword baseline.
2. Market-wide impact does not predict SPY or VIX moves (ρ ≈ 0).
3. Oil shock underestimated in 2022: +6.5% predicted vs +18.0% realised (analogues mix risk-off with supply shocks).
4. June 2022 FOMC missed: no adverse trigger, and the reaction was a relief rally (no surprise-vs-consensus measure).
5. Trigger precision against VIX/SPY event days was never measured.

Next: supply-shock vs risk-off scenarios; surprise-vs-consensus for scheduled releases; labelled news for events.

Speaker notes: "We used 2022 to find out where the method breaks, not to tune it until it looks right."
