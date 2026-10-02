# RiskPulse: AI/NLP Risk Engine with Index Rebalancer and Stress Tester - S&P Global & Crisil Campus Hackathon
**Candidate Name:** Arnav Gupta
**College Email ID:** [college email: to be filled by Arnav]
**College / Campus:** IIIT Dharwad
**Demo Video Link:** [YouTube unlisted link: to be added]
**Slide Deck Link (if hosted externally):** [`docs/presentation.pdf`](docs/presentation.pdf)

## 1. Project Overview / Problem Statement & Approach

Risk desks read thousands of headlines and posts a day, but positioning and stress testing need *structured* inputs.
RiskPulse is a CPU-only platform that ingests unstructured text from two kinds of sources (news from the GDELT
Global Knowledge Graph and social posts from a stock-tweet corpus), and turns each item into a versioned risk signal
with a **sentiment score** (−1 to 1, FinBERT, at document *and* entity level), an **event class** (ten classes
including the five in the brief), and an **impact score** (1 to 10) whose drivers are explained ("why 8/10").
Signals are written to an append-only JSONL file and served by a FastAPI service with a Server-Sent Events stream
that downstream modules subscribe to.

Both downstream modules are implemented. **Module A** maintains a mock index of 20 S&P 100 stocks whose weights tilt
with decayed entity sentiment, `w = w0 · exp(κ · s · c)`, inside weight bounds and a turnover cap, and is backtested
daily without look-ahead against an equal-weight index and a naive sign rule. **Module B** holds a synthetic
wholesale-banking book (loans, bonds, interest-rate swaps, FX forwards, equity options and total-return swaps, CDS and
equity) seeded from card-transaction merchant data. When a high-impact event arrives (impact ≥ 8, confident class,
≥ 2 independent sources) it applies risk-factor shocks *measured* over historical analogue episodes, then reports
portfolio value before and after, P&L by asset class, sector, rating and region, the change in expected loss, and a
CET1 ratio using Basel CRE20 risk weights.

The design choices are aimed at honesty: every evaluation uses held-out data and naive baselines (no FinBERT score on
the PhraseBank data it was trained on), the impact thresholds are fitted on a burn-in period, scenario shocks are
calibrated only on episodes that end *before* the replay window (so 2022 events serve as out-of-sample checks), and
every number in this README is produced by a script and stored in `reports/metrics.json`.

## 2. Architecture & Tech Stack

![Architecture](docs/architecture.png)

| Layer | What it does | Tech |
|---|---|---|
| Ingestion | GDELT GKG 2.0 15-minute files streamed and filtered in memory (2021-09 → 2022-09), stock tweets, GDELT DOC API (live), optional NewsAPI; normalisation; exact and near-duplicate removal; replay feed | requests, pandas, rapidfuzz |
| Entity linking | aliases, cashtags, brands and executives with context rules for ambiguous names; market-wide `MKT` entity with region tags | regex rules, spaCy (available) |
| Sentiment | FinBERT document and clause-level entity sentiment; VADER and Loughran-McDonald baselines | transformers, torch (CPU) |
| Event classification | calibrated logistic regression on MiniLM embeddings trained on weak labels; keyword and zero-shot NLI baselines | sentence-transformers, scikit-learn |
| Impact | class prior × (sentiment, velocity, breadth, novelty, credibility) × relevance; burn-in quantile mapping to 1-10 | numpy |
| Story clustering | online cosine clustering in a 48 h window, novelty over 72 h | sentence-transformers |
| Delivery | JSONL + DuckDB store; REST, SSE stream, `/analyze`, `/inject` | FastAPI, sse-starlette, DuckDB |
| Module A | sentiment tilt, capped weights, turnover cap, costs; daily backtest; IC; robustness grid | numpy, scipy |
| Module B | synthetic book generator, analogue-calibrated scenarios, duration/convexity, DV01, spread DV01, greeks, ΔEL, CET1 | numpy, scipy |
| Dashboard | Signal Monitor, Module A, Module B, Model Quality (light, consulting style) | Streamlit, Plotly |
| Quality | unit tests for formulas, signs, look-ahead and parsers | pytest, ruff |

## 3. Dataset Used

All data is public or synthetic; no proprietary or client data. Full details, licences and verification notes are in
[`data/SOURCES.md`](data/SOURCES.md); modelling assumptions are in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

- **News:** GDELT GKG 2.0 (6,680 sampled 15-minute files, one per hour on weekdays and every 6 h at weekends,
  2021-09-30 → 2022-09-29), filtered to headlines naming the 20 companies or carrying macro, geopolitical or credit
  themes. Raw files are never stored; the filtered, deduplicated replay feed is committed
  (`data/replay/feed.jsonl.gz`, 203,363 documents including social).
- **Social:** Kaggle "Stock Tweets for Sentiment Analysis and Prediction" (CC0, 2021-09-30 → 2022-09-29). Cashtag
  lists and tweets that never mention their company are dropped. The PG and MSFT sets in this dataset are copies of
  the AMZN set; the filter removes them.
- **Labelled evaluation:** Twitter Financial News sentiment and topic datasets (Hugging Face, MIT).
- **Impact calibration (event study):** Benzinga headlines with tickers (Kaggle, CC0, 2009-2020).
- **Prices:** yfinance daily history for the universe and risk-factor proxies, cached in `data/prices/`.
- **Module B seed:** Kaggle "Financial Transactions Dataset" (Apache 2.0), merchant fields only, aggregated to
  synthetic obligors `CP_0001…`. Credit inputs: S&P long-run one-year default rates by rating (2024 study, Table 24)
  and spread levels anchored to Moody's Baa minus the 10-year Treasury (FRED) and scaled by ICE BofA OAS ratios.
- **Human gold set:** 300 stratified items labelled by the author (`data/gold/`), used only for testing.
- **Synthetic content:** the Module B book and any headline injected in the demo are labelled `synthetic_demo`.

Key assumptions: universe of 20 current S&P 100 names chosen for overlap with the tweet data (tech-heavy by
construction); 1/4 sampling of GDELT (velocity is relative, not absolute); starting CET1 ratio 13% and credit-risk RWA
only; ratings held fixed under stress.

## 4. Quickstart & Installation

Runtime: **Python 3.11** on **macOS 27.0 (Apple silicon, arm64)**; CPU only; no API keys needed.

```bash
git clone https://github.com/arnavgupta2004/iiit-dharwad-arnav-gupta-hackathon.git
cd iiit-dharwad-arnav-gupta-hackathon
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m riskpulse demo --fast      # dashboard from precomputed outputs, no model downloads
```

Open http://localhost:8501 (dashboard) and http://127.0.0.1:8000/docs (API).

```bash
python -m riskpulse demo             # loads models; replays Feb-Mar 2022 live through the engine
python -m riskpulse eval all         # regenerates every metric in reports/
pytest -q                            # tests
```

Rebuilding everything from raw sources (optional, about 2-3 hours, mostly downloads and model inference):

```bash
python scripts/download_kaggle.py && python scripts/download_hf.py && python scripts/download_prices.py
python -m riskpulse ingest --source gdelt_gkg      # streams GDELT files; stores only filtered rows
python -m riskpulse ingest --source feed           # builds the replay feed
python -m riskpulse eval events                    # trains the event classifier
python -m riskpulse process                        # engine over the feed -> signals
python -m riskpulse backtest                       # Module A
python -m riskpulse stress --build-portfolio --replay   # Module B
```

## 5. Key Results & Domain Impact

All figures below are generated from `reports/metrics.json` by `scripts/render_readme_results.py`.

<!-- RESULTS:START -->
**Sentiment** (held-out Twitter Financial News, n = 2,388; bands tuned on train only)

| Method | Macro-F1 | Accuracy |
|---|---|---|
| FinBERT (train-tuned band) | 0.661 | 0.739 |
| FinBERT (argmax) | 0.668 | 0.725 |
| Loughran-McDonald lexicon | 0.460 | 0.601 |
| VADER | 0.448 | 0.534 |
| Majority class (neutral) | 0.264 | 0.656 |

**Event classification** (PROVISIONAL - final evaluation on Arnav's gold set pending)

| Evaluation set | Method | n | Macro-F1 |
|---|---|---|---|
| hf_topic_valid | keyword_baseline | 2,900 | 0.502 |
| hf_topic_valid | primary_embed_lr | 2,900 | 0.800 |
| weak_holdout | keyword_baseline | 1,851 | 0.897 |
| weak_holdout | primary_embed_lr | 1,851 | 0.760 |
| hf_topic_valid_subsample | zero_shot | 595 | 0.662 |
| hf_topic_valid_subsample | keyword_baseline | 595 | 0.485 |
| hf_topic_valid_subsample | primary_embed_lr | 595 | 0.814 |

**Module A, headline: information coefficient** (evaluation 2021-11-30 to 2022-09-29): mean daily IC +0.0376, t-stat 2.31, IC > 0 on 56.5% of 209 days (daily Spearman of decayed sentiment vs next-day return).

Tilt strength κ = 7.47, frozen by a risk budget (median absolute active weight 1.5%) computed from signals only over 2021-09-30 to 2021-11-29; returns were never used to set it.

Returns (secondary; net of 5 bps costs; a sentiment-tilt demonstration, not an alpha claim):

| Strategy | Cumulative return | Sharpe (rf = 0) | Max drawdown | Avg daily turnover |
|---|---|---|---|---|
| sentiment tilt | -26.1% | -1.203 | -29.8% | 19.10% |
| equal weight buy hold | -21.5% | -1.083 | -25.2% | 0.00% |
| equal weight rebalanced | -23.3% | -1.044 | -27.0% | 0.67% |
| naive sign rule | -23.4% | -1.055 | -27.5% | 20.61% |

**Module B** (replay 2021-09 to 2022-09): 4,279 high-impact candidates, 811 triggers fired (60 escalations), 459 stress runs. Cooldown: 24 h per event class and macro-region, re-run within it only on higher impact. Stress runs per month: 2021-09 2, 2021-10 41, 2021-11 32, 2021-12 28, 2022-01 29, 2022-02 30, 2022-03 44, 2022-04 46, 2022-05 43, 2022-06 48, 2022-07 38, 2022-08 39, 2022-09 39. First runs:

| Trigger time (UTC) | Scenario | Impact | Sources | Total impact | CET1 after |
|---|---|---|---|---|---|
| 2021-09-30T03:00 | GEOPOLITICAL:default | 8 | 2 | -0.10% | 12.98% |
| 2021-09-30T17:00 | GEOPOLITICAL:default | 8 | 2 | -0.10% | 12.98% |
| 2021-10-01T05:00 | GEOPOLITICAL:default | 8 | 4 | -0.10% | 12.98% |
| 2021-10-01T08:00 | GEOPOLITICAL:default | 9 | 2 | -0.13% | 12.98% |
| 2021-10-01T17:00 | MACROECONOMIC:default | 8 | 2 | -1.19% | 11.83% |
| 2021-10-04T14:00 | GEOPOLITICAL:default | 8 | 4 | -0.10% | 12.98% |
| 2021-10-04T17:00 | OPERATIONAL_ESG:default | 8 | 2 | -3.77% | 9.50% |
| 2021-10-05T01:00 | OPERATIONAL_ESG:default | 8 | 2 | -3.77% | 9.50% |
| 2021-10-05T04:00 | OPERATIONAL_ESG:default | 8 | 2 | -3.77% | 9.50% |
| 2021-10-05T16:00 | CREDIT_EVENT:default | 9 | 2 | -1.49% | 11.82% |
| 2021-10-05T17:00 | OPERATIONAL_ESG:default | 8 | 5 | -3.77% | 9.50% |
| 2021-10-06T08:00 | CREDIT_EVENT:severe | 10 | 2 | -5.68% | 7.34% |

**Pipeline** (cpu, 10 cores): 150.1 docs/s batch; single-document latency p50 18.5 ms, p95 20.7 ms; dedup removed 39.0% of news and 5.3% of social items.

**Impact score vs realised market reaction** (untouched 2021-22 test set: ticker-days after the burn-in, n = 3,538; market-model abnormal returns, CAR[0,+1])

| Score | Spearman vs abs(CAR) | Top-decile hit rate (10% by chance) | Spearman vs abnormal volume |
|---|---|---|---|
| Impact v2 (learned on Benzinga ≤ 2018) | 0.108 | 24.0% | 0.096 |
| Impact v1 (live, spec formula) | 0.046 | 13.8% | 0.080 |
| abs(sentiment) only (baseline) | 0.104 | 25.7% | 0.107 |

95% bootstrap CI of the Spearman difference: v2 − v1 [0.0348, 0.0894], v2 − abs(sentiment) [-0.0197, 0.0303]. Pre-registered rule: adopt v2 only if 95% CIs of rho(v2)-rho(|s|) and rho(v2)-rho(v1) are both > 0. Adopted: no; the live engine keeps v1 and this comparison is reported as is.

**Module B out-of-sample check (2022 episodes; scenarios calibrated before Sep 2021 only)**

| Episode | Status | Sign agreement | Book impact predicted | Book impact realised |
|---|---|---|---|---|
| russia_ukraine_2022 (2022-02-24) | fired on the day | 64.3% | -USD 8.1 m | +USD 1.6 m |
| fomc_june_2022 (2022-06-15) | missed (supplementary: prior-week run) | 57.1% | -USD 122.3 m | +USD 36.7 m |

<!-- RESULTS:END -->

**Domain impact.** For a credit or risk desk the platform shortens the path from headline to portfolio consequence:
an analyst sees which names and stories drive sentiment, why a story scores 8/10, and, when a market-wide event
breaks, an immediate estimate of the P&L, expected-loss and CET1 effect on a wholesale book, with every shock traceable
to a named historical episode. **Limitations:** one year of replay data and 20 names; rule-based entity linking; a
synthetic book with credit-risk RWA only; scenario severity limited by what pre-2021 history contains. See
`docs/DECISIONS.md` for every design choice and deviation.

## License

- **Code:** MIT; see [LICENSE](LICENSE).
- **Fine-tuned sentiment model weights:** CC BY-NC-SA 3.0, hosted separately on the Hugging Face Hub with a model
  card. The reason: the weights derive from `ProsusAI/finbert`, whose sentiment fine-tuning used Financial
  PhraseBank (CC BY-NC-SA 3.0), and the base model's card declares no licence. So the derivative takes the
  conservative, non-commercial share-alike licence. Our own fine-tuning data (`zeroshot/twitter-financial-news-sentiment`)
  is MIT. Third-party datasets keep their own licences; see [data/SOURCES.md](data/SOURCES.md).
