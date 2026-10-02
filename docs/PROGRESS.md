# Progress

Status legend: DONE · IN PROGRESS · NEXT · OPEN

## Phases 0-7 (P0): DONE
- **Phase 0:** scaffold, configs, CLI, verification (GATE A approved 2026-10-01).
- **Phase 1:** GDELT GKG stream (6,680 files), tweets (original-ticker rule, D-033), replay feed of 203,363 docs, entity
  linker; GATE B files delivered: `data/gold/to_label.csv`, `data/gold/entity_check.csv`.
- **Phase 2:** FinBERT doc/entity sentiment vs VADER and LM (`metrics.json → sentiment`).
- **Phase 3:** event classifiers + story clustering (provisional metrics until the gold labels arrive).
- **Phase 4:** impact v1 (severity scale per population, D-035), aggregation (hourly entity signals, D-034), JSONL +
  DuckDB, FastAPI REST/SSE/`/analyze`/`/inject`; optimised stage 2 proven equal to the original code (D-032).
- **Phase 5:** Module A rebalancer + backtest + IC + robustness grid (`metrics.json → moduleA`).
- **Phase 6:** Module B synthetic book, CRE20 capital, pre-2021 analogue calibration, trigger replay
  (`metrics.json → moduleB_triggers`).
- **Phase 7:** four-page dashboard, `demo` and `demo --fast` (fresh-clone snapshot in `data/demo/`, 27 MB); every page
  render-tested on batch and snapshot outputs; pipeline benchmark (`metrics.json → pipeline`).
- **Impact evaluation (pre-GATE C item 5):** v1 vs |sentiment| against |CAR[0,+1]| (`metrics.json → impact`, D-036).

## GATE C decisions applied (2026-10-02): DONE
- **Impact v2** (D-039): LightGBM on the Benzinga event study (train <= 2018, validation 2019 → 2020-07). On the
  untouched 2021-22 test set it beats v1 but not |sentiment|, so it is **not adopted**. Reported and stopped.
- **Trigger** (D-037): breadth bonus removed; 24 h cooldown per (event class, macro-region); escalation re-runs.
  Result: 811 triggers, 459 stress runs, per-month counts in `metrics.json → moduleB_triggers` and on the Module B page.
- **Module A** (D-038): kappa = 7.47 frozen by a 1.5% median active-weight budget on the first 2 months (no returns
  used); evaluation from 2021-11-30. IC is the headline (0.0376, t 2.31, 56.5% positive days); returns secondary;
  the kappa x half-life grid is sensitivity only.
- **2022 predicted vs realised** (D-040): invasion fired on the day (sign agreement 64.3%); June FOMC missed (impact
  peaked at 7). `metrics.json → moduleB_validation`, Module B page.
- **Coverage:** 12 of 20 tickers have tweets (SOURCES.md, ASSUMPTIONS.md).
- **Fine-tune** (P1, item 6): DONE. Test macro-F1 0.844 vs bar 0.661, kept (D-043). Hub hosting prepared
  (model card from `scripts/make_model_card.py`, resolver Hub → local rebuild → base FinBERT).

## P1 review and final rerun (2026-10-02): DONE (D-041 to D-051)
- Gold labels and entity check received from Arnav. Sentiment decision by pre-registered rule (D-048): fine-tuned model
  everywhere (no gain on live text). Taxonomy fix (D-044), event classifier retrained and evaluated once on gold.
- Impact v2 retrained and adopted for company mentions (validation check passed; D-039 revision, D-049).
- Re-score, kappa recalibrated (10.98), τ = 5%, triggers (349 stress runs), 2022 check, all evals once, snapshot once.
- Evaluation history: D-050. Final numbers: D-051 and the README results block.
- Engineering: stage-1 caches split by model (D-047); impact v2 scored by a lightgbm-free JSON tree evaluator because
  LightGBM and torch cannot share a process on macOS (D-049).

## Waiting on Arnav
- Decision: gold-2 sentiment (reported only) shows fine-tuned 0.383 vs FinBERT 0.553 on news; revisit the model?

## Next
- Phase 9 hardening: DONE for the fresh-clone test (README install, `pytest -q`, `demo --fast`) and the idle benchmark
  (D-051). Baseline tagged `v1.0-baseline` (fallback submission).
- Event classification round 2: DONE (D-052 to D-055). CV on gold-1 selected C1 (committed blind, 6df41d5); gold-2
  evaluated once (2677491): C1 within noise of all baselines, deployed as pre-registered. Downstream rerun done; v2
  retrained and kept (validation check passed). Models committed under `data/trained/` (D-054).
- HF weights uploaded by Arnav and verified byte-identical; empty-cache fresh-clone test re-run after round 2.
- Repo size: history ~105 MB after the round-2 snapshot (rule: no file > 50 MB, history < ~150 MB).
- Phase 10 (GATE D): drafts filled from metrics after round 2 (`docs/drafts/`); Arnav records the video and
  finalises the deck.
