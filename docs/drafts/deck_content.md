# RiskPulse: deck content (DRAFT v2, 7 slides)

Status: numbers filled from `reports/metrics.json` after the final rerun (D-051, tag `v1.0-baseline`). Items marked
**[R2]** depend on event classification and will be refreshed once after round 2 (gold-2). Every number here must
still match metrics.json at submission time; re-check before exporting the PDF.

Style: white background, navy/charcoal text, red/green only for negative/positive values, no gradients, no emoji, one
idea per slide, at most about 25 words of body text plus one visual. Percentages to one decimal place, currency as
"USD 10.7 m", rates and spreads in bp.

---

## Slide 1: Title

**RiskPulse: news and social sentiment as a real-time risk signal**
Code to Connect 2026 (S&P Global and Crisil), Phase 3 case study
Arnav Gupta, B.Tech Data Science and AI, IIIT Dharwad

Visual: a muted strip of the Signal Monitor page.

Speaker notes (15 s): "RiskPulse turns financial news and social posts into structured risk signals and uses them in
two places: a tactical index rebalancer and an event-triggered stress test of a wholesale banking book."

---

## Slide 2: Problem and approach

Left, the problem:
- Markets react to news within hours; risk teams read it in batches.
- Raw sentiment isn't a risk signal: it needs the entity, the event type and an impact level.

Right, three fields per signal:
1. **Sentiment**, −1 to +1, at entity level (a clause about JPM doesn't score BAC).
2. **Event class**, one of 10 (geopolitical = cross-border only, macro, credit, M&A, …).
3. **Impact**, 1–10, learned against realised abnormal returns.

Bottom: one signal schema feeds **Module A** (index tilt) and **Module B** (stress test).

---

## Slide 3: System design

Visual: `docs/architecture.png`, full width.

Captions:
- Sources: GDELT GKG news (1-in-4 sampled 15-minute files), stock tweets, Benzinga (event study only), prices.
- Engine: entity linking → fine-tuned FinBERT → event classifier → story clustering → impact (v2 for companies,
  v1 for market-wide).
- Delivery: JSONL + DuckDB, FastAPI (REST and SSE), Streamlit dashboard; deterministic replay.
- CPU only, no paid keys; `python -m riskpulse demo --fast` works on a fresh clone (tested: install to running
  dashboard in under 5 minutes).

Replay feed: 203,363 documents (165,782 news, 37,581 tweets), 2021-09-30 → 2022-09-29, 20 S&P 100 names plus
market-wide.

---

## Slide 4: Implementation highlights (what and why)

- **Entity-level sentiment.** Clause split, so "JPM beats while BAC misses" scores both names. FinBERT fine-tuned
  on the tweet train split only.
- **Impact v2.** Monotone gradient boosting on about 10 years (2009-2018) of Benzinga headlines vs abnormal returns.
  Scored live from per-session running aggregates (no look-ahead). Every score keeps its drivers.
- **Leakage discipline.** Time splits, a burn-in, rules written before tests, a frozen κ, 2022 never used for
  calibration, and every look at the test window logged (D-050).
- **Engineering.** LightGBM and torch can't share a process on macOS, so v2 is exported as JSON trees and scored by
  a NumPy evaluator that matches LightGBM exactly on the full validation and test sets.

Speaker notes, if asked "why not an LLM?": CPU-only, deterministic, auditable, and every number is reproducible with
`python -m riskpulse eval all`.

---

## Slide 5: Key results (vs naive baselines)

| Component | RiskPulse | Naive baseline |
|---|---|---|
| Sentiment, live-feed news (gold, n 237) | 0.549 fine-tuned / 0.592 FinBERT | VADER 0.471, Loughran-McDonald 0.442 |
| Sentiment, in-domain test (HF) | 0.844 fine-tuned | FinBERT 0.668, majority 0.264 |
| Event class, gold (n 300) **[R2]** | 0.593 | keyword 0.612, zero-shot 0.612 |
| Entity linking precision | 82.0% (CI 73.3–88.3%) | n/a |
| Impact vs abs(CAR), Spearman (n 3,538) | v2 0.129 (top-decile hit 24.9%) | v1 0.058 (16.7%), abs(sentiment) 0.129 (24.9%) |
| Module A daily IC | +0.038 (t 2.39; 56.5% of days > 0) | 0 = no skill |
| Module B triggers **[R2]** | 349 stress runs from 2,527 high-impact items | every high-impact item |

Efficiency strip (measured):
- 140.4 docs/s on CPU; 33.0 ms p95 per item (idle machine).
- Dedup removes 39.0% of news and 5.3% of social items before scoring.
- Analyst load: 2,527 high-impact items → 349 stress tests a year, each with a written scenario explainer. **[R2]**

Visual: `reports/figures/impact_v2_deciles.png`, small.

Honesty line (small text): "Returns are secondary: net of costs the tilt returned −21.9% vs −23.3% for equal weight
rebalanced and −21.5% buy-and-hold."

---

## Slide 6: Domain impact

- **Portfolio manager (Module A):** sentiment-tilted weights, with the headlines behind each move. Turnover capped at
  5% a day as policy; cost drag 0.5% over the evaluation window.
- **Risk team (Module B):** on 2022-02-24 at 06:00 UTC, an impact-9 story ("Global market plunges, stocks dive after
  Vladimir Putin launches military operations in Ukraine") triggered a stress test of a USD 10 bn synthetic book:
  −USD 10.7 m, CET1 13.00% → 12.98%, top-10 positions and the analogue explainer. **[R2]**
- **Validation habit:** 2022 used only out of sample. Invasion: 9 of 14 factor directions right; June FOMC: 6 of 14.
  **[R2]**

Visual: Module B page (trigger timeline + stress result).

Speaker notes: all counterparties are synthetic (CP_xxxx); no client data.

---

## Slide 7: Limitations and next steps

Limitations (measured):
1. Fine-tuned sentiment: 0.844 in-domain, but on live news it is indistinguishable from FinBERT.
2. Event classifier ≈ keyword and zero-shot baselines on live text. **[R2: round-2 result goes here]**
3. Impact v2 ≈ abs(sentiment); market-wide impact shows no relation to SPY/VIX moves (ρ −0.03).
4. Geopolitical scenarios average risk-off and supply-shock analogues: oil +6.6% predicted vs +18.0% realised in 2022.
5. Scheduled macro events have no surprise measure: the FOMC scenario predicted a sell-off into a relief rally.

Next steps:
- Supply-shock vs risk-off variants of the geopolitical scenario, calibrated pre-2021.
- A surprise-vs-consensus feature for scheduled releases (CPI, payrolls, FOMC).
- Labelled news headlines for the event classifier; rating migration in the capital view; the full GDELT feed.

Speaker notes: "We used 2022 to find where the method breaks, not to tune it until it looks right."
