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
- **Fine-tune** (P1, item 6): IN PROGRESS (45-min CPU budget).

## Waiting on Arnav
- Gold labels: `data/gold/to_label.csv` → `labels.csv` (final event-class evaluation, sentiment spot check).
- Entity check: mark `data/gold/entity_check.csv` → `riskpulse eval linking`.

## Remaining P1 (after the fine-tune)
1. ΔEL and CET1 refinements (rating migration is P2).
2. Final gold-set evaluation of event classification and entity-linking precision (needs the labels above).

## Then
Phase 9 hardening (fresh-clone test in a new venv, README), Phase 10 deck and video script (GATE D).
