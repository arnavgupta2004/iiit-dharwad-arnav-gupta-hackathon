# Design decisions and deviations

Newest first within each date. Each entry: decision, rationale, and status (accepted, or proposed pending Arnav's OK at a gate).

## 2026-10-01 (Phase 0)

### D-001 Packaging: `src/` layout installed via `-e .` inside requirements.txt
`python -m riskpulse` needs the package importable. `requirements.txt` ends with `-e .`, so the documented
`pip install -r requirements.txt` both installs dependencies and makes the CLI work. `pyproject.toml` holds
package metadata and ruff/pytest settings. **Accepted.**

### D-002 Python 3.11 via uv-managed interpreter
The machine had 3.12–3.14 only. `uv venv --python 3.11` provided CPython 3.11.16. The README will state
Python 3.11 and the OS tested. **Accepted.**

### D-003 [VERIFY] results for models and libraries (all verified by loading and running on CPU)
| Item | Result |
|---|---|
| `ProsusAI/finbert` | Loads (transformers 5.18). `id2label = {0: positive, 1: negative, 2: neutral}`; a positive headline gave P(pos) 0.63 and a negative one P(neg) 0.97. No `model.safetensors` (pytorch .bin only); loads fine. Licence field empty on the Hub card. |
| `sentence-transformers/all-MiniLM-L6-v2` | Loads; 384-dim embeddings; Apache-2.0. |
| Zero-shot | `facebook/bart-large-mnli` (MIT) exists. **Chosen: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`** (MIT, smaller and newer). Three test headlines were classified correctly at >= 0.99 confidence, about 0.5 s per headline with 5 labels on CPU. |
| SetFit 1.2.0 + transformers 5.18 + sentence-transformers 6.1 | **Compatible.** A 16-example, 4-class training run completed in about 4 s and predicted 4/4 unseen headlines correctly. |
| `sse-starlette` 3.5.0 | `EventSourceResponse` streams `text/event-stream` events correctly under FastAPI 0.142 (tested via httpx ASGI transport). |
| spaCy `en_core_web_sm` 3.8.0 | Installed from the official GitHub release wheel (compatible with spaCy 3.8.x per compatibility.json); NER tags JPMorgan Chase and Bank of America as ORG. |

### D-004 Sentiment evaluation set has no test split
`zeroshot/twitter-financial-news-sentiment` ships only `train` (9,543) and `valid` (2,388). We use `valid` as the
held-out test set (never trained on), and `train` only if we fine-tune (P2). FinBERT was not trained on this dataset,
so the evaluation is fair. **Accepted.**

### D-005 Social source: equinxx stock tweets, not thedevastator
The organisers' example tweet dataset is structurally malformed on parse and uses company names rather than tickers
(see data/SOURCES.md). equinxx is clean, ticker-labelled and UTC-timestamped (80,793 tweets, 2021-09-30 → 2022-09-29).
**Accepted.**

### D-006 Time-period mismatch between news and social sources; proposed resolution (needs OK at GATE A)
Benzinga news covers 2009–2020-06; the tweets cover 2021-09 → 2022-09; the GDELT DOC API is a recent rolling window.
There is no single source of historical news that overlaps the tweets. **Proposal:**
- **Replay feed and Module A backtest window = 2021-09-30 → 2022-09-29**, built from:
  - tweets (social, equinxx), and
  - **GDELT GKG 2.0 raw files** (news) sampled over that window and filtered to finance-relevant records
    (universe org or alias match, or ECON_/EPU_/sanction/war themes). These carry page titles, URL, themes, orgs and tone.
- This window contains real Module B triggers: Russia's invasion of Ukraine (2022-02-24) and the 2022 Fed hikes
  (e.g. 75 bp on 2022-06-15), so the stress-test demo runs on real events rather than only injected ones.
- **Benzinga** is used for the **impact v2 event study** cross-sectionally (thousands of tickers, 2009–2020),
  with a time-based split.
- GDELT DOC API remains the `live` mode source.
- Download cost (measured 2026-10-01): GKG files are 6–7.5 MB zipped and download at 0.8–1.9 MB/s, i.e. 3–10 s each.
  Every hour of every day would be 8,760 files (about 60 GB), which is too heavy. **Proposed sampling:** one
  15-min file per hour for 13:00–21:00 UTC on weekdays (US session ± pre/after-hours) plus one per 6 h on weekends,
  giving about 2,500 files (about 17 GB streamed). That's roughly 1.5–3 h in the background with a few parallel
  workers, filtered on the fly; only the compact filtered rows (a few MB) are kept and committed. It samples
  about 25% of US-session news, which is acknowledged as a limitation. The sampling rate is configurable.

### D-007 Universe (proposed at GATE A)
20 current S&P 100 names, chosen to maximise overlap with the tweet dataset (14 of 20) while keeping Financials,
Energy and Health Care for Module B relevance:
AAPL, MSFT, AMD, INTC, GOOGL, META, NFLX, DIS, AMZN, TSLA, KO, PG, COST, BA (in tweets) + JPM, BAC, GS, XOM, CVX, JNJ.
Changes vs the spec's list: **added AMD, INTC, NFLX, PG, COST** (all in the tweet data); **dropped NVDA, PFE, UNH, WMT, CAT**
(no tweets). CRM and VZ are in the tweets and S&P 100 but were left out (233 and 123 tweets) to keep sector breadth.
Sector mix: IT 4, Communication 4, Consumer Discretionary 2, Staples 3, Industrials 1, Financials 3, Energy 2, Health Care 1.
GOOGL is linked from both $GOOGL and $GOOG.

### D-008 Analogue windows (checked against yfinance data)
- Lehman 2008-09-12 → 2008-10-10: confirmed (filing 2008-09-15 inside the window).
- COVID: **start changed to 2020-02-19** (SPY close peak); trough 2020-03-23 confirmed.
- Russia–Ukraine 2022-02-23 → 2022-03-08: confirmed; the shock is oil-led and the equity index move is modest
  (this is a finding, not a bug; the scenario will reflect measured moves).
- 2022 rate shock: **proposed 2022-06-09 → 2022-06-16** (CPI surprise 06-10, 75 bp hike 06-15, SPY trough 06-16).
- SVB 2023-03-08 → 2023-03-17: confirmed.
Shock magnitudes are not recorded here; they will be produced by `scripts/calibrate_scenarios.py`.

### D-009 GDELT DOC API usage
The rate limit is enforced (HTTP 429 with "limit requests to one every 5 seconds"); bursts trigger a cool-down during
which even spaced requests fail. The adapter uses >= 6 s spacing, backoff on 429 and a cached fallback. Lookback probes
for older dates: see D-010.

### D-010 GDELT DOC lookback
Probe of older dates is pending (the API was throttling). Not on the critical path, because historical news comes from
GKG raw files (D-006) and the DOC API is only used for `live` mode. Result will be appended here.

### D-011 Transactions data: merchant-level only
Only merchant_id, mcc, merchant_state/city, amount and date are used, to build synthetic obligors `CP_####`.
`users_data.csv` and `cards_data.csv` (synthetic but personal-looking fields such as card numbers and CVVs) are not used
or committed. **Accepted.**

### D-012 Credit risk weights for CET1
Basel II standardised corporate risk weights are recalled from memory in `configs/moduleB.yaml` and flagged
[VERIFY]. They must be checked against the BCBS text before any CET1 number is reported (Phase 6). **Open.**
