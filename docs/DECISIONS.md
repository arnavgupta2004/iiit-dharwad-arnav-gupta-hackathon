# Design decisions and deviations

Newest first within each date. Each entry: decision, rationale, and status (accepted, or proposed pending Arnav's OK at a gate).

## 2026-10-02 (Phases 3-7)

### D-030 Provisional event-classification results and how to read them
`reports/metrics.json → events` (provisional until the gold set is labelled). On the human-labelled HF topic valid
split (7 classes) the primary embedding classifier reaches macro-F1 0.800 against 0.502 for the keyword baseline; on a
595-item subsample, zero-shot NLI scores 0.662. On the *weak* holdout the keyword baseline scores higher (0.897 vs
0.760), but that holdout's labels come mainly from the same keyword rules, so the comparison is circular and not used
as evidence. The primary model trains on HF topic *train*, so the HF valid split is in-distribution for it. The gold
set (our own GDELT/tweet items) is the fair final test.

### D-025 Batch inference device
The project runs on CPU (default). The one-off precompute over 203,363 documents ran at about 18 docs/s on a shared
CPU, so `RISKPULSE_DEVICE=mps` (opt-in) was used for that batch only. On 512 tweets the MPS and CPU outputs agree to
max |Δp| 5e-6, with 100% argmax agreement. No result depends on the accelerator; reviewers run on CPU.

### D-026 Credit inputs for the synthetic book (data/market)
- Spreads by rating: BBB anchored at 185 bp on 2021-09-30 (FRED `DBAA` − `DGS10`), other ratings scaled by the median
  ICE BofA OAS-to-BBB ratios over 2023-10-02 → 2026-09-30. FRED publishes only three years of ICE data.
- PDs: S&P 2024 annual default study, Table 24, year-1 column (verified in the PDF; a web-search summary had
  different, wrong figures).
- Spread shocks: Δ(Moody's Baa − 10y) over each window, scaled to ratings by the same ratios (proportional widening).
- The PD stress multiplier is spread-implied: 1 + ΔS_BBB / S_BBB (credit triangle); ×1.5 for obligors in a region
  named on the triggering event (assumption, config).

### D-027 Loan losses via ΔEL, not mark-to-market
Banking-book loans affect capital through provisions (ΔEL = (PD_s − PD)·LGD·EAD). Their spread-driven fair-value
change is reported for information only and excluded from CET1, to avoid double counting credit risk. Bonds,
derivatives and equity are marked to market.

### D-028 Scenario construction
Per event class, take the analogue set; per factor, the largest-magnitude measured move (sign kept); scale by
m(impact) = {8: 0.6, 9: 0.8, 10: 1.0}; combine concurrent scenarios by max per factor. Where a sector ETF did not yet
exist in an analogue window (XLC before 2018-06, XLRE before 2015-10), the market (SPY) move is used. Implied vol for
option revaluation moves 1:1 with VIX points (assumption).

### D-029 Event-signal emission
Event signals are re-emitted when a story's impact **or** distinct-outlet count changes (outlets capped at 20).
Emitting only on impact changes could leave the n_sources ≥ 2 trigger condition permanently unmet (bug found by test).

## 2026-10-02 (Phase 6: Module B)

### D-024 Corporate risk weights verified against BCBS CRE20
Basel Framework **CRE20.42** (rated corporate exposures receive the "base" risk weights of Table 10) and **CRE20.43**
(unrated corporates 100%, Table 10), in the version in force from 2023-01-01, read on bis.org on 2026-10-02:
AAA to AA– 20%, A+ to A– 50%, **BBB+ to BBB– 75%**, BB+ to BB– 100%, below BB– 150%, unrated 100%.
The earlier from-memory table (Basel II) had BBB at 100%. That was wrong for the current framework and is corrected in
`configs/moduleB.yaml`. Simplifications (documented as assumptions): one table for all corporate credit exposures,
no SME (CRE20.47) or specialised-lending treatment, and derivatives risk-weighted on a credit-equivalent exposure.
Also verified on the same page: **CRE20.7 Table 1** for sovereigns (AAA to AA– 0%, A+ to A– 20%, BBB+ to BBB– 50%,
BB+ to B– 100%, below B– 150%, unrated 100%) and **CRE20.57** for equity (250% for equity holdings; 400% only for
speculative unlisted equity, which the book does not hold).

## 2026-10-02 (Phase 2: sentiment)

### D-022 Sentiment evaluation outcome and label thresholds
On HF valid (n = 2,388, never tuned on), FinBERT macro-F1 is 0.621 with the spec band (±0.15), 0.661 with the band
tuned on HF train (±0.65), and 0.668 with argmax. Baselines: Loughran-McDonald 0.460, VADER 0.448 (each with its
train-tuned band); majority class 0.264. Source: `reports/metrics.json` → `sentiment`.
**Engine choice:** the continuous score s = P(pos) − P(neg) is what the modules consume. Label thresholds stay at the
spec's ±0.15 for news, because the ±0.65 band was tuned on tweets and our news headlines are a different distribution.
The tuned result is reported alongside. Entity-level scoring splits on clause connectives (while, but, although, ;)
and is unit-tested ("JPMorgan beats ... while Bank of America misses" → JPM > 0 > BAC).

### D-023 Loughran-McDonald source
The official LM master dictionary link (Google Drive) returned "quota exceeded" on 2026-10-02. We use the LM word
lists bundled in the MIT-licensed `pysentiment2` package (2014 LM release: 2,355 negative and 354 positive words),
with our own tokenizer and polarity (P − N)/(P + N). The dictionary itself is not committed (LM terms: free for
academic research).

## 2026-10-02 (Phase 1)

### D-017 GDELT GKG pilot result and sampling choice
Pilot: trading week 2022-02-22 → 2022-02-28 plus FOMC days 2022-06-14 → 06-16, at 1/4 sampling (minute 00 of
every hour on weekdays, every 6 h at weekends). Full numbers are in `reports/gkg_pilot.json` (`scripts/gkg_pilot.py`).
- 200/200 files fetched, 0 failures; 99.9% of records carry a `<PAGE_TITLE>`, so headline text is usable.
- Under the final (stricter) linking rules: mean **50.8** unique title-linked headlines per ticker per week (target
  ≥ ~5); minimum 3 (PG, see D-019). All other tickers are ≥ 10.
- MKT volume per day: 1,288 (Feb 23) → **3,140 (Feb 24)** → 3,304 (Feb 25). The FOMC decision day 2022-06-15 had
  1,395 MKT items, 513 of them US-tagged (vs 177 the day before).
- A US-session-only scheme (13–21 UTC) left COST at 4/week, so **all weekday hours are kept**.
- Full stream: 6,680 files for 2021-09-30 → 2022-09-29, launched 2026-10-02 in the background. Raw files are never
  written to disk (in-memory parse), with a disk guard at 3 GB free. Kept rows (a superset; see D-020) go to a gitignored
  intermediate store and are reproducible via `python -m riskpulse ingest --source gdelt_gkg`.

### D-018 Tweet dataset defect: PG and MSFT sets are copies of AMZN
In equinxx `stock_tweets.csv`, the 4,089 tweets labelled PG and the 4,089 labelled MSFT are 100% identical to the AMZN
set. Only 1% of "PG" tweets and 17% of "MSFT" tweets contain their own cashtag (vs 90% for AMZN). They are mislabelled
AMZN tweets. The text-link filter (D-019) removes them, so **12** universe names have genuine tweets (AAPL, AMD, AMZN,
BA, COST, DIS, GOOGL, INTC, KO, META, NFLX, TSLA), not 14. MSFT keeps the tweets that genuinely mention Microsoft. META
tweets use the pre-rename cashtag `$FB`, which is in the alias list. The approved universe is unchanged. CRM (233) and
VZ (123) have genuine tweets if a swap is ever wanted.

### D-019 Tweet quality filters
1. Universe tickers only.
2. Drop cashtag lists (> 3 `$TICKER`s): 20.7% of universe tweets were watchlists or "earnings this week" lists.
3. Keep a tweet only if its own text links to its labelled company.
4. Exact and near-duplicate removal.

Yield: 64,793 universe tweets → 51,349 → 40,608 → 38,505 after dedupe. TSLA is 75% of what remains (28,949).
This is a known skew, handled by per-ticker normalisation in signals.

### D-020 Entity-linking precision rules (after pilot inspection)
Pilot samples showed incidental brand mentions ("... in latest Instagram post", "How to take a screenshot on iPhone")
and a word collision ("The Tide News Online" → PG). Rules adopted:
- Brands (and "Facebook", which post-rename usually means the platform) link only with business context words or a
  corroborating company hit.
- The context list excludes brand names and loose words (union, deal, fine, users, supply, plant, strike, production).
- The PG brand "Tide" was removed.
- Consumer how-to, shopping and entertainment-listing titles are not linked (`low_value_title_patterns`).
The streamer keeps a superset of rows, and `gdelt_gkg.relink` re-applies the current rules, so tightening never needs a
re-download. Residual errors seen in samples: "BAC review" (an arts venue), Coca-Cola HBC (a separate bottler), and
consumer-ish Disney park stories. Linking precision will be measured on the gold set (an `entity_correct` column).

### D-021 Near-duplicate detection
rapidfuzz `token_set_ratio` ≥ 95 (punctuation stripped) within the same day and primary entity, only when token counts
are within a 0.7 ratio. token_set_ratio is 100 for subset texts, so without that guard short tweets would be wrongly
merged. Embedding-based near-dup (cosine ≥ 0.95) is applied later in story clustering (Phase 3).

## 2026-10-01 (GATE A review by Arnav)

### D-013 Scenario calibration only on pre-replay analogues; 2022 used as out-of-sample validation
Calibrating shocks on Russia–Ukraine 2022 or June 2022, which sit inside the replay window where the stress test fires,
would be circular. **Decision (Arnav):** the shock library is calibrated only on analogue windows that end before
2021-09-30 (`calibration_cutoff` in `configs/scenarios.yaml`, enforced by a test). Windows were checked against yfinance:
- Lehman 2008-09-12 → 10-10
- US downgrade / euro 2011-07-22 → 08-08 (SPY trough 08-08)
- Taper tantrum 2013-05-21 → 06-24 (SPY peak 05-21, trough 06-24)
- Crimea 2014-02-26 → 03-14
- China devaluation 2015-08-17 → 08-25 (trough 08-25)
- HY/energy 2015-12-01 → 2016-02-11 (trough 02-11)
- Brexit 2016-06-23 → 06-27 (trough 06-27)
- Abqaiq 2019-09-13 → 09-17
- COVID 2020-02-19 → 03-23

Each class maps to a *set* of analogues. Per risk factor, the scenario uses the largest-magnitude move across the set
(sign kept), so every factor shock traces to one named episode. SVB 2023 was dropped (after the cutoff).

**Finding:** pre-2021 geopolitical analogues moved US proxies very little (SPY about −0.1% in Crimea 2014 and
Abqaiq 2019). Russia ETFs (RSX, ERUS) are no longer on Yahoo. Oil moved +8% in Abqaiq. We expect the Feb 2022
out-of-sample check to show the model under-predicting the oil shock, and we will report this rather than tune it away.

**Validation (new P1 item, right after impact v2):** when the stress test fires on the 2022-02-24 invasion news (and the
2022-06-15 FOMC), compare predicted factor shocks with the realised moves of the same proxies over the next 10 trading days.
Shown as "predicted vs realised" on the Module B page and in `reports/metrics.json`.
Because P0 Module B needs shocks and we must not type in magnitudes, `scripts/calibrate_scenarios.py` is built in Phase 6.
Only the out-of-sample validation remains in Phase 8.

### D-014 Impact v2 split (Arnav)
Benzinga event study: train on events dated ≤ 2018-12-31, validate on 2019-01-01 → 2020-06-11. The 2021-09 → 2022-09
replay window (GDELT and tweets) is reported separately as an out-of-domain check.

### D-015 Sentiment tuning discipline (Arnav)
The HF `valid` split is the test set. No thresholds, model choices or calibration are tuned on it; any tuning uses
`train` only.

### D-016 Universe approved (Arnav), with an added check
AMD, INTC, NFLX, PG and COST were re-checked against the S&P 100 constituents table: all present. All 20 universe
sectors match the table. **Accepted.** D-006 (replay window and GDELT GKG news) was approved, conditional on a
one-week pilot (results in D-017).

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
The DOC API returned results for 2026-03-01 (7 months back) and for 2022-02-24 (`Ukraine` query, 10 articles),
but the 2022 query succeeded only on the 5th attempt because of 429 throttling. So the lookback reaches at least 2022,
but the API is impractical for bulk history. Historical news therefore comes from GKG raw files (D-006); the DOC API
is used for `live` mode only. **Accepted.**

### D-011 Transactions data: merchant-level only
Only merchant_id, mcc, merchant_state/city, amount and date are used, to build synthetic obligors `CP_####`.
`users_data.csv` and `cards_data.csv` (synthetic but personal-looking fields such as card numbers and CVVs) are not used
or committed. **Accepted.**

### D-012 Credit risk weights for CET1
Basel II standardised corporate risk weights are recalled from memory in `configs/moduleB.yaml` and flagged
[VERIFY]. They must be checked against the BCBS text before any CET1 number is reported (Phase 6). **Open.**
