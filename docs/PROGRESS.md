# Progress

Status legend: DONE · IN PROGRESS · NEXT · BLOCKED · OPEN

## Phase 0: Setup (DONE; GATE A approved 2026-10-01)
Scaffold, configs, CLI, docs, verification of every [VERIFY] item (data/SOURCES.md, DECISIONS D-001..D-016).

## Phase 1: Ingestion and entity linking (DONE; GATE B file delivered)
- DONE: rule-based entity linker with ambiguity rules, MKT routing and region tags (tests)
- DONE: GDELT GKG pilot passed (D-017); full-year stream of 6,680 files (0 failures), in-memory only
- DONE: tweets adapter with cashtag-list and text-link filters; PG/MSFT copy defect found (D-018/D-019)
- DONE: Benzinga, GDELT DOC (live), NewsAPI (optional) adapters; normalisation; exact + near dedupe
- DONE: replay feed of 203,363 documents (data/replay/feed.jsonl.gz, reports/feed_stats.json)
- DONE: price cache for 46 symbols (data/prices/daily.parquet)
- DONE: **GATE B**: `data/gold/to_label.csv` (300 stratified items) + `data/gold/README.md` instructions

## Phase 2: Sentiment (DONE)
FinBERT doc- and entity-level (clause split), VADER and LM baselines; evaluation on the HF valid split with bands
tuned on train only (reports/metrics.json → sentiment).

## Phase 3: Event classification and clustering (DONE, provisional evaluation)
Keyword baseline, zero-shot NLI, calibrated embedding LR (primary); online story clustering. Provisional metrics
on the HF topic valid split and a weak holdout; the final gold-set evaluation waits for Arnav's labels.
Gold texts are excluded from training (assertion; fix c706dcf).

## Phase 4: Impact v1, aggregation, store, API (DONE)
Impact v1 with velocity/breadth/novelty and burn-in quantile bins; entity and event aggregation; JSONL + DuckDB
store; FastAPI with /signals, /events, /entities, /stream (SSE), /analyze, /inject; SignalSubscriber (SSE / store).

## Phase 5: Module A (DONE: code; IN PROGRESS: run on batch outputs)
Rebalancer, daily backtest (look-ahead tested), benchmarks, IC, robustness grid, dashboard page.

## Phase 6: Module B (DONE: code; IN PROGRESS: trigger replay on batch outputs)
Book of 121 synthetic obligors and 334 positions seeded from merchant data; CRE20 risk weights (D-024); S&P PDs
and FRED-derived spreads (data/market); pre-2021 analogue calibration; valuation with sign tests; CET1; trigger.

## Phase 7: Dashboard and demo (IN PROGRESS)
Four pages written; `demo` and `demo --fast` commands; demo snapshot script. Running the batch, then smoke-testing
the pages, then GATE C.

## Open items
- Gold labels from Arnav → final event-class evaluation (and entity-linking precision, sentiment spot check)
- P1 (after GATE C): impact v2 event study; predicted-vs-realised validation of 2022 episodes; trigger validation;
  pipeline benchmark
