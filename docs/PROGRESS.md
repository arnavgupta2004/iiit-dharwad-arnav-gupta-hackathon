# Progress

Status legend: DONE · IN PROGRESS · NEXT · OPEN

## Phases 0-7 (P0): DONE, at GATE C
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

## Waiting on Arnav
- Gold labels: `data/gold/to_label.csv` → `labels.csv` (final event-class evaluation, sentiment spot check).
- Entity check: mark `data/gold/entity_check.csv` → `riskpulse eval linking`.
- GATE C decisions: trigger values (D-035) and the next P1 order.

## P1 order (after GATE C)
1. Impact v2: Benzinga event study (train <= 2018, validate 2019 - mid-2020; 2021-22 out of domain); replaces v1 only
   if it beats v1 and |sentiment| on held-out data.
2. Predicted-vs-realised validation of the 2022 episodes (invasion, June FOMC) on the Module B page and in metrics.
3. Fine-tune a small sentiment model on the HF tweet-sentiment TRAIN split only; 45-minute CPU budget; keep only if it
   beats FinBERT's 0.661 macro-F1 on the same held-out split (Arnav's item 6).
4. ΔEL and CET1 refinements (rating migration is P2).
5. Module A: IC and robustness grid already reported; add the dashboard narrative.
6. Module B trigger validation against VIX spikes / SPY drawdown days (precision).
7. Final gold-set evaluation of event classification and entity-linking precision.

## Then
Phase 9 hardening (fresh-clone test in a new venv, README), Phase 10 deck and video script (GATE D).
