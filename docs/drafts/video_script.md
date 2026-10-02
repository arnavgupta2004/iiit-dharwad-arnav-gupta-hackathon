# RiskPulse: 5-minute demo video script (DRAFT v3, after the feature freeze)

Numbers come from `reports/metrics.json` after the final rerun (D-060/D-061) and match the README "Results at a
glance". Target 4:30–5:00 (spec §2), following the organisers' flow: intro, setup and run, end-to-end walkthrough,
results and impact. The hook is the 2022 event. Record at 1080p or 1440p, browser at 100% zoom, light theme, terminal
font ≥ 16 pt. Upload as YouTube **Unlisted** and test the link in an incognito window.

Before recording:
- Repo with `.venv` active and the full batch outputs present. Start `python -m riskpulse demo` once beforehand so
  model loading isn't on camera, then restart it for the take.
- Close notifications; one browser window with two tabs (dashboard on 8501, API docs on 8000/docs).
- Rehearse the "Analyze a headline" example and use whatever the model actually returns.

---

## 0:00–0:35 · Hook and intro (Module B page, Feb 2022 zoomed in)

On screen: Module B page, trigger timeline zoomed to 20-25 Feb 2022, then the ranked stress-run table (February).

Voice-over:
> "At midnight UTC on the 22nd of February 2022, this system triggered a geopolitical stress test: Russia had
> recognised two separatist regions of Ukraine. Six hours later it fired again on the troop order. The full-scale
> invasion began two days after that. I'm Arnav Gupta from IIIT Dharwad, and this is RiskPulse: it reads a year of
> real news and tweets in time order, turns each item into a risk signal, and acts on it, here with a stress test on a
> ten-billion-dollar synthetic banking book."

Wording rule: "flagged the escalation", never "predicted the invasion".

---

## 0:35–1:05 · Setup and run (terminal)

On screen: README Quickstart, then the terminal.

```bash
python -m riskpulse demo
```

Voice-over:
> "Setup is three commands from the README: clone, install, run. A fresh clone runs the reported system: the small
> trained models are in the repo, and the sentiment model downloads from the Hugging Face Hub. `demo --fast` runs
> without any downloads. Here I'm running the full pipeline: it replays February and March 2022 through the engine and
> starts the API and the dashboard."

Show briefly: the `model origin` log lines (repo / hub, no FALLBACK), then `127.0.0.1:8000/docs`.

---

## 1:05–2:00 · Input to signal (Signal Monitor page)

Actions:
1. Document feed: headlines arriving with entity, sentiment, event class and impact.
2. Event signals: select a story; show the "Impact drivers (0-1)" chart and its evidence.
3. Analyze a headline: "JPMorgan beats estimates while Bank of America misses" shows per-company sentiment.

Voice-over:
> "Each item is linked to companies or to the market, scored for sentiment with FinBERT for news and a tweet-tuned
> FinBERT for tweets, classified into one of ten event types, and grouped into stories. Impact comes from a model
> trained on about ten years of headlines against abnormal stock returns, and every score shows its drivers. Entity linking
> is 82% precise on 100 hand-checked links."

---

## 2:00–2:50 · Module B: event-triggered stress test

Actions:
1. Ranked stress runs (top 3 per month highlighted); point at 22 Feb.
2. Inspect the 24 Feb run: total impact, CET1 before/after, top-10 positions, scenario explainer (analogues).
3. Scroll to "Out-of-sample check: 2022 predicted vs realised".

Voice-over:
> "A stress test fires only for high-impact, adverse stories confirmed by at least two outlets. The scenario is built
> from historical analogues calibrated only on data before September 2021: Crimea, Brexit, the Abqaiq attack. On the
> invasion episode it predicted a 10.7-million-dollar loss and CET1 down from 13.00 to 12.98 percent. Over the next
> ten sessions it got 9 of 14 factor directions right, but it badly underestimated the oil shock: 6.5% predicted
> against 18% realised. Our analogues mix risk-off and supply shocks; separating them is a next step."

---

## 2:50–3:35 · Module A: index rebalancer

Actions: headline IC tiles, "Why did this weight move?", returns table (secondary) with cost breakdown.

Voice-over:
> "Module A tilts an equal-weight index of 20 S&P 100 stocks toward positive sentiment, with a 5% daily turnover
> cap. The tilt strength was fixed before looking at any returns. The headline is the information coefficient:
> sentiment ranks next-day returns weakly but significantly, 0.047 with a t-stat of 2.85. In returns, the tilt is
> level with equal weight in a falling market: minus 24.0 against minus 23.3 percent. That's a signal demonstration,
> not an alpha claim."

---

## 3:35–4:30 · Results and honesty (Model Quality page, then deck slide 5 or the README glance)

Voice-over:
> "Every number comes from one command, `riskpulse eval all`. The learned impact score beats the hand-set formula,
> 0.111 against 0.055 Spearman with abnormal returns, and is on par with sentiment strength alone. On hand-labelled
> live news, FinBERT scores 0.59 macro-F1 against 0.47 for a lexicon. The pipeline handles 87 documents a second on a
> laptop CPU. And what didn't work: two rounds of event classification, including 500 hand labels and a blind test,
> didn't beat a keyword baseline; market-wide impact doesn't predict market moves; and we missed the June 2022 Fed
> hike, where the market rallied in relief. All of that is in the README, with the decision log."

---

## 4:30–4:50 · Close

On screen: deck slide 7 or the README Limitations.

Voice-over:
> "RiskPulse is open source, reproducible on a laptop, and honest about where it breaks. Next: separate supply-shock
> scenarios and a surprise-versus-consensus measure for scheduled releases. Thank you."

---

Checklist after recording:
- [ ] Before recording: the Codespace fresh-clone test (`docs/drafts/codespace_fresh_clone_test.md`) passed. It backs
  the "a fresh clone runs the reported system" line.
- [ ] Length 4:30–5:00.
- [ ] Every spoken number matches `reports/metrics.json` (and the README glance).
- [ ] The 2022 claim says "flagged the escalation", with the 22 Feb 00:00 UTC timestamp.
- [ ] Nothing synthetic is presented as real (the book and injected events are labelled synthetic).
- [ ] The Unlisted link plays in an incognito window and is added to the README.
