# RiskPulse: 5-minute demo video script (DRAFT)

Status: DRAFT v2, numbers filled from `reports/metrics.json` after the final rerun (D-051, tag `v1.0-baseline`).
Lines marked **[R2]** depend on event classification and are refreshed once after round 2 (gold-2). Target length 4:30 to 5:00 (spec §2). Record at 1440p or
1080p, browser at 100% zoom, light theme, terminal font at least 16 pt. Upload as YouTube **Unlisted** and test
the link in an incognito window.

Before recording:
- Fresh terminal in the repo with `.venv` active; `data/processed` present (full mode) or a fresh clone (fast mode).
- Start `python -m riskpulse demo` once beforehand so model loading isn't on camera; restart it for the take.
- Close notifications; hide bookmarks; one browser window with two tabs (dashboard, API docs).

---

## 0:00–0:30 · Intro (camera or title slide)

On screen: deck slide 1.

Voice-over:
> "Hi, I'm Arnav Gupta from IIIT Dharwad. This is RiskPulse. It reads financial news and social posts, turns each
> item into a structured risk signal (sentiment, event type and a 1-to-10 impact score), and feeds those signals
> into two tools: a tactical index rebalancer and an event-triggered stress test of a wholesale banking book.
> Everything runs on a laptop CPU with public data."

---

## 0:30–1:00 · Setup and run (terminal)

On screen: terminal, README Quickstart beside it.

Type:
```bash
python -m riskpulse demo
```

Voice-over:
> "Setup is three commands from the README: clone, install requirements and run the demo. `demo --fast` works on a
> fresh clone from precomputed outputs. Here I'm running the full pipeline: it replays February to March 2022
> through the engine at one market day per minute, and starts the API and the dashboard."

Show briefly: the log line naming the sentiment model in use (Hub / local / base fallback), then the API at
`127.0.0.1:8000/docs`.

---

## 1:00–2:00 · Input to signal (Signal Monitor page)

On screen: dashboard → Signal Monitor. Let the replay stream for a few seconds.

Actions (page sections: impact heatmap, Event signals, Document feed, Analyze a headline):
1. Document feed: headlines arrive with entity, sentiment, event class and impact.
2. Event signals: select one story; show the "Impact drivers (0-1)" chart and its evidence headlines.
3. Analyze a headline: type "JPMorgan beats estimates while Bank of America misses" to show entity-level
   sentiment (JPM positive, BAC negative). Rehearse first and use whatever the model actually returns.

Voice-over:
> "Each item is linked to companies or to the market as a whole, scored by a FinBERT model I fine-tuned on
> financial tweets, classified into one of ten event types, and grouped into stories. Impact for company stories
> comes from a model trained on about ten years of headlines against abnormal stock returns; market-wide stories use an
> interpretable formula. Every score shows its drivers and the evidence behind it."

---

## 2:00–2:50 · Module A: index rebalancer

On screen: Module A page.

Actions:
1. Headline KPIs: IC +0.038, t-stat 2.39, positive days 56.5%.
2. Scroll to "Weights over time"; pick a rebalance date; click a ticker under "Why did this weight move?".
3. One sentence on returns, shown as secondary, with the cost breakdown.

Voice-over:
> "Module A tilts an equal-weight index of 20 S&P 100 names toward positive sentiment, within weight bounds and a
> 5% daily turnover cap. The tilt strength was fixed from a risk budget before looking at any returns. The headline
> result is the information coefficient: sentiment ranks next-day returns weakly but significantly, mean IC 0.038
> with a t-stat of 2.4. Net of costs the tilt returned minus 21.9% in a falling market, against minus 23.3% for
> equal weight rebalanced and minus 21.5% for buy-and-hold. I report that as is: this is a signal demonstration,
> not an alpha claim."

---

## 2:50–3:50 · Module B: event-triggered stress test

On screen: Module B page.

Actions:
1. Trigger timeline: point to the cluster on 2022-02-24.
2. Select the triggered run of 2022-02-24 06:00 UTC ("Global market plunges, stocks dive after Vladimir Putin
   launches military operations in Ukraine") **[R2]**: show total impact, CET1 before/after,
   top-10 worst positions, the scenario explainer (analogues and severity).
3. Scroll to "Out-of-sample check: 2022 predicted vs realised".
4. (Optional, if time) Inject demo event (labelled synthetic) to show a live trigger.

Voice-over:
> "Module B watches event signals. When impact is 8 or more, confidence is high and at least two outlets report it,
> it builds a scenario from historical analogues calibrated only on data before September 2021, and revalues a
> USD 10 billion synthetic book: loans, bonds, swaps, FX, options and CDS. Here is the invasion morning: a loss of
> about USD 10.7 million and CET1 from 13.00% to 12.98%. Then the honest part: on the 2022 episodes, which we
> never trained on, the invasion scenario got 9 of 14 factor directions right but badly underestimated the oil
> shock, and the June FOMC scenario predicted a sell-off into what became a relief rally. Both are listed as next
> steps, not tuned away." **[R2]**

---

## 3:50–4:40 · Results and impact (Model Quality page, then deck slide 5)

On screen: Model Quality page (sentiment table, event classifier, impact comparison), then slide 5.

Voice-over:
> "All results are produced by one command, `riskpulse eval all`, and every number in the README comes from it.
> Sentiment: fine-tuning reached 0.84 macro-F1 in-domain, but on hand-labelled live headlines it scores 0.55, level
> with base FinBERT at 0.59 and ahead of the VADER lexicon at 0.47. Event classification: 0.59 on the gold set, level
> with keyword and zero-shot baselines at 0.61 [R2]. Impact: the learned model beats the hand-set formula, 0.13
> against 0.06 Spearman, and is on par with sentiment strength alone for predicting the market reaction. The
> pipeline scores 140 items per second on a laptop CPU and filters 2,527 high-impact items down to 349 stress tests
> a year." [R2]

---

## 4:40–5:00 · Close

On screen: deck slide 7 (limitations and next steps) or the README.

Voice-over:
> "RiskPulse is open source, reproducible on a laptop, and honest about where it breaks. The next steps are
> supply-shock scenarios and a surprise-versus-consensus measure for scheduled macro events. Thank you."

---

Checklist after recording:
- [ ] Length 4:30–5:00.
- [ ] Every number spoken matches `reports/metrics.json` after the rerun.
- [ ] Nothing synthetic is presented as real (the book and injected events are labelled synthetic on screen).
- [ ] Unlisted link plays in an incognito window; link added to README.
