# Progress

Status legend: DONE · IN PROGRESS · NEXT · BLOCKED

## Phase 0: Setup (status: DONE, at GATE A awaiting Arnav's OK)

- DONE: Repo scaffold per spec §11 (`src/riskpulse/*`, `configs/`, `scripts/`, `data/*`, `reports/`, `tests/`)
- DONE: `.gitignore`, `.env.example`, `pyproject.toml`, pinned `requirements.txt` (Python 3.11.16 venv via uv)
- DONE: Typer CLI skeleton: `python -m riskpulse --help` lists ingest, process, replay, serve, demo, eval, backtest and stress
- DONE: Common layer: pydantic `Document` and `Signal` (schema v1), config loader, logger
- DONE: Configs: universe (proposed), taxonomy, impact, scenarios, moduleA, moduleB, mcc_sector_map, app
- DONE: Verified all [VERIFY] items (see `data/SOURCES.md` and `docs/DECISIONS.md` D-003 to D-012)
- DONE: Download scripts: `scripts/download_kaggle.py`, `scripts/download_hf.py`; profiling script `scripts/profile_raw_sources.py`
- DONE: Tests: 24 passing (CLI, schemas, config consistency)
- OPEN: GDELT DOC API lookback for older dates (not on the critical path)
- OPEN: Basel II risk-weight table to verify against the BCBS text before CET1 is reported (Phase 6)

### Decisions needed at GATE A
1. Universe (D-007)
2. Replay window and GKG news plan (D-006)
3. Push to `origin` (remote exists: github.com/arnavgupta2004/iiit-dharwad-arnav-gupta-hackathon)

## Next: Phase 1 (Ingestion and entity linking)
- Adapters: kaggle_tweets, gdelt_gkg (historical), gdelt_doc (live), kaggle_news (Benzinga), newsapi (optional)
- normalize, dedupe, replay feed builder, entity linking with tests, `data/gold/to_label.csv`
