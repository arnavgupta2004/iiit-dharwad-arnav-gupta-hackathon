# RiskPulse: AI/NLP Risk Engine with Index Rebalancer and Stress Tester
### Project spec for Code to Connect: S&P Global & Crisil Hackathon 2026, Phase 3
Candidate: **Arnav Gupta**, IIIT Dharwad. Individual entry. The working name "RiskPulse" can be renamed; if you rename it, update this spec, the package name and the README together.

> This document is the single source of truth for the build. Sections 1 and 2 restate the organisers' requirements and must not be altered. Everything from section 3 onward is the build design. Items tagged **[VERIFY]** are unconfirmed facts that must be checked before anything depends on them.

---

## 0. Goal and winning strategy

**Goal:** be the obvious winner on paper (repo, deck and video), then win the live jury pitch.

Known scoring signals from the guidelines: there are separate scores for **Domain Understanding** (tie-breaker 1) and **Presentation & Communication** (tie-breaker 2), and the jury asks technical questions on *design choices, code, data assumptions and use-case interpretation*. Most candidates will submit a FinBERT notebook with a toy chart. We beat them on five things:

1. **Both modules (A and B)**, each built properly rather than as a token effort.
2. **A validated impact score.** Impact is the hardest and most hand-waved field. We calibrate and validate it against realised market reactions (abnormal returns from an event study), so we can say "impact correlates with actual market moves; here is the evidence."
3. **Real financial-risk domain depth**, which is Crisil and S&P's own language. That means entity-level (aspect) sentiment, credit-relevant event taxonomy, and stress testing with duration/convexity bond repricing, DV01, PD/LGD/EAD expected loss and a CET1 impact. Shock sizes are calibrated from historical analogue episodes rather than invented.
4. **Honest evaluation against naive baselines**, with no leakage. The deck's "Key Results" slide asks for efficiency gains versus a naive approach, so we measure them.
5. **Reviewer experience.** A single command runs the demo offline on CPU with no keys; the README is clean; there's an architecture diagram; and a live "type a headline and see the signal" feature works in the jury demo.

---

## 1. Problem statement (organisers' requirements, restated faithfully)

**Objective:** design and implement a platform that ingests and analyses real-time, unstructured data to generate actionable financial risk signals.

**Primary deliverable:** a unified **AI/NLP Risk Engine**, the central processing unit. It parses text from sources such as news feeds and social media and converts it into structured output. After the core engine, implement **at least one** downstream module: **Module A** (tactical, high-frequency stock index rebalancer) or **Module B** (strategic, event-driven portfolio stress-testing tool). **We implement both.**

### 1.1 Engine requirements
- **Data ingestion:** process text from **at least two different sources** (for example financial news articles and Twitter/X posts).
- **NLP-driven analysis:** for a given company or event, output structured data with:
  - **Sentiment Score:** numeric, positive, negative or neutral, for example **-1.0 to 1.0**.
  - **Event Classification:** a categorical label, for example **Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Product Launch**.
  - **Impact Score:** a predicted severity, for example **1 to 10**, giving the potential market impact.
- **Output:** make the signals available to downstream applications, for example through **a simple API or by writing to a file**. We do both.

### 1.2 Module A: Tactical Index Rebalancing
- A dynamically rebalanced mock index of **10 to 20 stocks from the S&P 100**, driven by real-time sentiment.
- **Subscribes to the Sentiment Score.** Positive sentiment increases a stock's weight; negative sentiment decreases it.
- **Visualisation:** a dashboard showing how the weights change over time.

### 1.3 Module B: Strategic Portfolio Stress Testing
- Simulates the impact of major real-world events on a **synthetic portfolio of wholesale banking assets**.
- **Subscribes to Event Classification and Impact Score.**
- Define the synthetic portfolio **using the provided sample transaction data**, with a **mix of asset types (loans, bonds, derivatives)**.
- When a **high-impact event** is detected (for example Geopolitical with Impact > 7), trigger a **stress test**. A simplified model is acceptable, such as predefined shocks (for example equities -10%, rates +2%) applied for a specific event type.
- **Visualisation:** a dashboard of portfolio value **before and after** the stress test, highlighting the impact of the event.

### 1.4 Suggested open datasets (from the organisers; all free)
| Resource | Organisers' suggested use |
|---|---|
| GDELT Project (global news, 100+ languages, 15-minute updates, events and themes) | Real-time categorised event feed for Event Classification and Impact |
| Financial News Sentiment datasets (Kaggle), for example "Sentiment Analysis for Financial News" | Training or fine-tuning sentiment; back-testing |
| News API (newsapi.org), free developer plan | Live headlines for the demo |
| Historical Stock Tweets (Kaggle), for example "Tweet Sentiment's Impact on Stock Returns" | Sentiment training; social versus returns |
| yfinance (Python) | Historical prices for Module A and Module B |
| Alpha Vantage (free API) | Alternative or supplement to yfinance |
| Salad Money Open Banking Transaction Data (anonymised UK key-worker transactions) | Synthetic portfolio for Module B |
| Financial Transactions Dataset (Kaggle; transactions, customers, cards) | Synthetic portfolio for Module B |

### 1.5 Deliverables (from the problem statement)
1. Source code in a public repository.
2. A live demonstration of **at most 5 minutes**.
3. A presentation of **at most 7 slides** covering architecture, challenges and results.

---

## 2. Submission guidelines (organisers' requirements, restated)

| # | Deliverable | Where | Requirements |
|---|---|---|---|
| 1 | Source code and architecture | **Public** GitHub repo | Dependencies plus clear setup and run commands |
| 2 | Presentation deck, 5 to 7 slides (PDF or PPTX) | `/docs/presentation.pdf`, linked in the README | Problem approach, system design, key results, domain impact |
| 3 | Demo video | YouTube **Unlisted**, linked in the README | End-to-end working demo. The guidelines say "10 mins", but the problem statement caps the demo at 5 minutes and the suggested flow runs about 4 to 5 minutes, so **target 4:30 to 5:00** |
| 4 | All data | `/data` folder | All data and sources used, including synthetic. **No proprietary data** |

**Repo name:** `<college>-<candidate-name>-hackathon`, which gives **`iiit-dharwad-arnav-gupta-hackathon`**.

**Required structure:**
```
iiit-dharwad-arnav-gupta-hackathon/
├── README.md            # overview, setup, architecture, demo link (mandatory template below)
├── requirements.txt     # dependencies
├── LICENSE              # mandatory; MIT
├── src/                 # application, pipelines, models
├── data/                # sample/synthetic data used for the demo
└── docs/
    ├── presentation.pdf # 5–7 slides
    └── architecture.png # high-res architecture diagram
```

**Mandatory README template.** Follow it exactly, section by section:
```
# [Project Title] - S&P Global & Crisil Campus Hackathon
**Candidate Name:** Arnav Gupta
**College Email ID:** [college email]
**College / Campus:** IIIT Dharwad
**Demo Video Link:** [YouTube unlisted]
**Slide Deck Link (if hosted externally):**
## 1. Project Overview / Problem Statement & Approach   (2–3 paragraphs)
## 2. Architecture & Tech Stack   (embed architecture diagram; frameworks, DBs, AI/ML libs)
## 3. Dataset Used   (source/nature; assumptions)
## 4. Quickstart & Installation   (Runtime: Python 3.11 on [OS tested]; exact commands)
## 5. Key Results & Domain Impact
```

**Deck outline (suggested by organisers):** 1 Title; 2 Problem and Approach; 3 System Design (architecture and data flow); 4 Implementation Highlights (modules, tech choices and why); 5 Key Results (outputs, metrics, screenshots, **efficiency gains versus a naive approach**); 6 Domain Impact; 7 Limitations and Next Steps.

**Video flow:** intro (about 30 s), setup and run from the README commands (about 30 s), end-to-end walkthrough from input through processing to dashboard (2 to 3 min), results and impact (30 to 60 s).

**Disqualification:** any of the following leads to disqualification:
- a private or inaccessible repo
- a broken or restricted video or deck link (no Google Drive or OneDrive; test in an incognito window)
- missing the live pitch without notice
- plagiarism or a pre-existing project
- real confidential client data

**Other rules:**
- AI assistance is allowed **with full honesty**.
- Individual entry only.
- Prefer incremental commits.
- No large binary or data dumps.
- Changes made after the deadline may not count.

**Tie-breakers, in order:** Domain Understanding score, then Presentation & Communication score, then earlier submission timestamp.

---

## 3. System architecture

```
                ┌──────────────────────────── INGESTION ────────────────────────────┐
 News:  GDELT DOC API ─┐                                                            │
        NewsAPI (opt) ─┼─► source adapters ─► normalise ─► dedupe ─► Document store │
        Kaggle news   ─┤      (live / batch / replay)                  (DuckDB)     │
 Social: Kaggle tweets ┤                                                            │
        Reddit (opt)  ─┘                                                            │
                └───────────────────────────────────────────────────────────────────┘
                                         │
                ┌──────────────────── NLP RISK ENGINE ─────────────────────┐
                │ 1 Entity linking (ticker universe + aliases + NER)       │
                │ 2 Story clustering (embeddings, rolling 48h window)      │
                │ 3 Sentiment: FinBERT, doc-level + entity-level           │
                │ 4 Event classification (taxonomy, few-shot classifier)   │
                │ 5 Impact score: features → heuristic v1 → calibrated v2  │
                │ 6 Signal aggregation (entity signals + event signals)    │
                └──────────────────────────────────────────────────────────┘
                                         │
             Signal store (DuckDB + JSONL export) ──► FastAPI: REST + SSE stream + /analyze
                                         │
                ┌────────────────────────┴────────────────────────┐
     Module A: subscribes to sentiment          Module B: subscribes to event class + impact
     Tactical index rebalancer                  Strategic stress tester
     (weights, backtest)                        (synthetic wholesale book, shocks, valuation)
                └────────────────────────┬────────────────────────┘
                              Streamlit dashboard (4 pages)
```

**Run modes, all sharing the same code path:**
- `replay`: streams a timestamped, saved feed (`data/replay/feed.jsonl`) at a configurable speed, for example one market day per 60 s. It is deterministic, so it is the **default for the demo and video**.
- `live`: polls GDELT, and NewsAPI if a key is set, every N minutes.
- `batch`: processes historical data for evaluation and backtests.
- `inject`: pushes a clearly labelled synthetic headline (`source="synthetic_demo"`) into the stream to trigger Module B live during the jury demo.

**Tech stack** (versions to be pinned after the install is verified):
- **Core:** Python 3.11, pydantic v2, typer (CLI), PyYAML, python-dotenv, loguru or logging.
- **Data:** pandas, numpy, scipy, duckdb, pyarrow, requests, feedparser.
- **NLP:**
  - `transformers` and `torch` (CPU) for FinBERT and the classifier
  - `sentence-transformers` for embeddings
  - `spacy` with `en_core_web_sm` for NER
  - `rapidfuzz` for alias matching
  - `setfit` for few-shot event classification **[VERIFY compatibility]**
- **ML/stats:** scikit-learn, optionally lightgbm, statsmodels (event-study regressions).
- **Market data:** yfinance, with prices cached to `/data/prices` so demos don't depend on Yahoo uptime.
- **Serving:** FastAPI and uvicorn. Server-Sent Events (`sse-starlette` **[VERIFY]**) for subscriptions.
- **Dashboard:** Streamlit and Plotly.
- **Quality:** pytest and ruff.

---

## 4. Data sources and plan

Record every source in `data/SOURCES.md`: name, URL, licence or terms, date pulled, row count, how it is used, and whether it is committed in full, sampled, or download-script only.

| Source | Type | Role | Notes |
|---|---|---|---|
| **GDELT DOC 2.0 API** | News (live and recent) | Live feed; weak labels via GDELT themes | No key needed. **[VERIFY]** the lookback window, rate limits, and the theme filter syntax (for example `theme:ECON_BANKRUPTCY`) |
| **NewsAPI** | News (live) | Optional live feed | Free developer plan limits **[VERIFY]**. The engine must work without a key |
| **Kaggle historical financial news with tickers and timestamps**, for example "Daily Financial News for 6000+ Stocks" **[VERIFY slug, columns, licence]** | News (historical) | Impact calibration (event study); Module A backtest; replay feed | Needs per-headline ticker and date |
| **Kaggle stock tweets**, for example "Stock Tweets for Sentiment Analysis and Prediction" or the organisers' "Tweet Sentiment's Impact on Stock Returns" **[VERIFY]** | Social (historical) | Second source; replay feed | **Choose the ticker universe to maximise overlap with this dataset** |
| Reddit (PRAW) | Social (live) | Optional live social source | Needs a free app credential; optional only |
| **Twitter Financial News Sentiment** (HF, for example `zeroshot/twitter-financial-news-sentiment`) **[VERIFY]** | Labelled sentiment | **Sentiment evaluation set** | Not seen by FinBERT during training, so a fair test |
| Financial PhraseBank / Kaggle "Sentiment Analysis for Financial News" | Labelled sentiment | Optional fine-tuning only | ⚠ `ProsusAI/finbert` was fine-tuned on PhraseBank, so **do not report FinBERT accuracy on it** (leakage). Say this in the README; it is a good jury point |
| **yfinance** daily OHLCV for the universe, SPY or ^GSPC, sector ETFs, ^TNX/^IRX, HYG/LQD/IEF, CL=F, DX-Y.NYB, ^VIX | Prices | Event study, backtests, shock calibration | Cache to CSV or Parquet |
| **Kaggle "Financial Transactions Dataset"** (transactions, users, cards, MCC) **[VERIFY]**, or Salad Money | Transactions | **Seeds the Module B synthetic portfolio** (required by the brief) | See §7.2 for the mapping |
| **Human gold set** (`data/gold/`) | Labels by Arnav | Event-class test set, sentiment spot-check, impact sanity | **Labelled by Arnav himself**, never used for training |

**Ticker universe for Module A:** 20 S&P 100 stocks across sectors. Proposed list: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, BAC, GS, XOM, CVX, JNJ, PFE, UNH, WMT, KO, DIS, CAT, BA. **[VERIFY]** current S&P 100 membership, then **swap names to maximise overlap with the social dataset's tickers**. Store the list in `configs/universe.yaml` with name, sector, aliases, cashtags, and key brands or executives for entity linking.

**Repo data hygiene:**
- Commit cleaned samples, the replay feed, the gold set, synthetic portfolio CSVs, scenario calibration and cached prices.
- For large raw datasets, commit `scripts/download_*.py` plus a README instead of the raw dump.
- No file over 50 MB.

---

## 5. NLP Risk Engine: detailed design

### 5.1 Document schema (normalised)
```json
{
  "doc_id": "sha1 of source + url (or text)",
  "source": "gdelt | newsapi | kaggle_news | kaggle_tweets | reddit | synthetic_demo",
  "source_type": "news | social",
  "published_at": "2026-10-05T09:31:00Z",
  "ingested_at": "…Z",
  "title": "string or null",
  "text": "string",
  "url": "string or null",
  "author": "string or null",
  "lang": "en",
  "meta": {}
}
```
- All timestamps are stored in UTC. Market logic converts to US/Eastern, and anything published after 16:00 ET maps to the next trading day.
- **Deduplication:** exact match on URL or text hash, then near-duplicate titles (MinHash or embedding cosine above 0.95), keeping the earliest.

### 5.2 Entity linking
- Resolve mentions to tickers using an alias dictionary (company names, short names, brands, cashtags like `$AAPL`, CEO names), spaCy ORG entities, and rapidfuzz matching.
- **Disambiguation rules:** for example, "Apple" needs a financial or tech context or a cashtag. Unit tests cover the tricky cases.
- **Relevance** is between 0 and 1, from mention count, title mention (weighted ×2) and share of ORG mentions.
- Docs with no company but a macro or geopolitical topic map to the pseudo-entity **`MKT`** (market-wide), with a region tag where one can be detected. This is how Module B receives event-level signals.

### 5.3 Sentiment (field 1)
- **Model:** `ProsusAI/finbert` **[VERIFY]**. Document score = P(positive) − P(negative), in [-1, 1]. Labels: positive above +0.15, negative below −0.15, neutral otherwise (thresholds in config).
- **Entity-level sentiment (differentiator):** score only the sentences that mention the entity and aggregate them weighted by relevance. "JPM beats while BAC misses" must give opposite signs. Unit-test this.
- **Baselines:** VADER and the Loughran-McDonald lexicon.
- **Evaluation:** macro-F1 and accuracy on the Twitter Financial News Sentiment test split, plus a spot-check on Arnav's gold sample. Report FinBERT against the baselines.

### 5.4 Event classification (field 2)
**Taxonomy** (`configs/taxonomy.yaml`, each class with a definition, keywords and 5 examples):

| Class | Description |
|---|---|
| `GEOPOLITICAL` | war, sanctions, elections, trade conflict |
| `MACROECONOMIC` | inflation, GDP, jobs, central-bank and rate decisions |
| `CREDIT_EVENT` | default, downgrade, bankruptcy, restructuring, missed payment, covenant breach |
| `MERGER_ACQUISITION` | |
| `PRODUCT_LAUNCH` | |
| `EARNINGS` | results, guidance |
| `REGULATORY_LEGAL` | fines, lawsuits, investigations, approvals |
| `MANAGEMENT_CHANGE` | |
| `OPERATIONAL_ESG` | outages, cyber, accidents, recalls, ESG controversy |
| `OTHER` | |

The first five classes come from the brief and must exist.

**Approach, layered:**
1. **Naive baseline:** keyword and regex rules.
2. **Zero-shot NLI** with label descriptions, for example a DeBERTa-v3 zero-shot model or `facebook/bart-large-mnli` **[VERIFY IDs]**.
3. **Primary:** a few-shot classifier (SetFit on sentence embeddings, or logistic regression on embeddings) trained on weak labels and a small number of human seeds. Weak labels come from GDELT theme queries, high-precision keyword rules, and zero-shot predictions above 0.9 confidence.
4. **Calibration:** confidence scores are calibrated (temperature or isotonic). Below the confidence threshold, the classifier outputs `OTHER`.

**Evaluation:** macro-F1 and a confusion matrix on the **human gold test set (about 300 headlines labelled by Arnav)**, comparing the keyword baseline, zero-shot and the primary model. A small labelling helper (Streamlit page or CSV workflow) makes labelling fast. The gold set is never used for training.

### 5.5 Impact score (field 3), the centrepiece
**Feature vector** (all computable at publication time, so there is no look-ahead):

| Feature | Meaning |
|---|---|
| `T` | prior severity for the event class (config, 0 to 1). Initial values: CREDIT 0.9, GEO 0.8, MACRO 0.75, REG 0.6, M&A 0.6, EARNINGS 0.55, OPS_ESG 0.5, MGMT 0.4, PRODUCT 0.3, OTHER 0.1 |
| `S` | absolute entity sentiment |
| `V` | velocity: z-score of the entity's mention count in the last N hours against its trailing 7-day baseline, squashed with a sigmoid |
| `B` | breadth: number of distinct sources and outlets (log-scaled) |
| `N` | novelty: 1 − max cosine similarity to stories from the prior 72 h |
| `R` | entity relevance |
| `C` | source credibility (news tier above social; config) |

**v1, heuristic (must ship):** a transparent weighted formula, mapped to 1 to 10 by quantile binning on the historical batch. For example:
`raw = T · (0.35·S + 0.25·V + 0.15·B + 0.15·N + 0.10·C) · (0.5 + 0.5·R)`
Weights live in config, and every signal carries the per-driver contributions in a `drivers` field so the dashboard can explain "why 8/10."

**v2, calibrated (P1 differentiator):** an event study gives the ground truth.
- **Abnormal returns:** fit a market model on SPY over the window [−120, −20] trading days, then compute **CAR[0, +1]**. Abnormal volume = log(volume / mean volume over [−30, −5]).
- **Target:** |CAR[0, +1]| (main), plus abnormal volume (secondary).
- **Model:** fit a monotone or regularised model, either logistic or ordinal regression or LightGBM with monotone constraints, on the same features. Use a **time-based split** (train on an earlier period, test on a later one).
- **Mapping:** convert the model's predicted rank to 1 to 10 by deciles of the training distribution.

**Acceptance test for v2:**
- Spearman ρ(impact, |CAR|) on the held-out period must beat both v1 and the **naive baseline of |sentiment| alone**.
- Report a **decile chart** (mean |CAR| by impact decile) and the **top-decile hit rate**.
- If v2 doesn't beat v1, ship v1 and report the honest result.

### 5.6 Story clustering and signal aggregation
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` **[VERIFY]**. Online clustering groups items within a rolling 48 h window (cosine ≥ 0.80, configurable) into the same **story** (event), identified by `event_id`.
- **Entity signal** (for Module A): per ticker, an EWMA of entity sentiment with half-life *H* (config, for example 6 h), weighted by confidence and relevance.
- **Event signal** (for Module B): per story, the majority or confidence-weighted event class, impact = max over its documents with a breadth bonus, and the affected entities, sectors and regions.

### 5.7 Signal schema (output contract, versioned `schema_version: 1`)
```json
{
  "schema_version": 1,
  "signal_id": "sig_…",
  "signal_type": "entity | event",
  "as_of": "2026-10-05T10:15:00Z",
  "entity": {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials"},
  "event_id": "evt_…",
  "sentiment_score": -0.62,
  "sentiment_label": "negative",
  "event_class": "CREDIT_EVENT",
  "event_confidence": 0.81,
  "impact_score": 8,
  "impact_raw": 0.77,
  "confidence": 0.74,
  "n_docs": 14,
  "n_sources": 3,
  "regions": ["US"],
  "sectors": ["Financials"],
  "drivers": {"type_prior": 0.9, "sentiment": 0.62, "velocity": 0.88, "breadth": 0.55, "novelty": 0.7, "credibility": 0.8, "relevance": 0.9},
  "explanation": "Credit event; strongly negative; 14 articles from 3 sources in 2h (velocity z=3.1)",
  "evidence": [{"doc_id": "…", "source": "gdelt", "title": "…", "url": "…", "published_at": "…"}],
  "model_versions": {"sentiment": "finbert@…", "event": "setfit@…", "impact": "v2@…"}
}
```

### 5.8 Delivery: API and file
- **File:** signals are appended to `data/signals/signals.jsonl` and also stored in DuckDB.
- **FastAPI endpoints:**

| Endpoint | Purpose |
|---|---|
| `GET /health` | health check |
| `GET /signals?ticker=&event_class=&min_impact=&since=&limit=` | query signals |
| `GET /signals/latest` | latest signals |
| `GET /events?min_impact=` | event-level signals |
| `GET /entities` | the entity list |
| `GET /stream` | **SSE subscription**; this is what "subscribes" means for the modules |
| `POST /analyze` | takes `{"text": "...", "source": "..."}` and returns the full structured output **instantly**. Use it in the live demo: the jury types a headline and sees the signal |
| `POST /inject` | demo-only; adds a labelled synthetic headline to the stream |

- **Module subscription:** modules use a `SignalSubscriber` interface with two implementations, an SSE client and local DuckDB polling.

---

## 6. Module A: Tactical Index Rebalancer

**Inputs:** entity sentiment signals (SSE or the store) and cached daily prices.

**Baseline weights:** equal weight (`w0 = 1/N`). A cap-weight variant is optional.

**Target weights** at each rebalance *t*:
1. Tilt each stock by its sentiment: `w̃_i = w0_i · exp(κ · s_i · c_i)`, where:
   - `s_i` is the decay-weighted entity sentiment
   - `c_i` is its confidence
   - a deadband sets `s_i` to 0 when |s_i| < 0.10
2. Clip each weight to `[w_min, w_max]` (for example 2% to 12%) and renormalise, iterating until the constraints hold.
3. Cap turnover by moving only part of the way toward the target: `w_new = w_old + λ·(w̃ − w_old)`, where `λ = min(1, τ_max / turnover)`.
4. Charge transaction cost at `c_bps` × turnover (for example 5 bps).
5. Optional (flagged in config): scale `κ` by impact (`κ · impact/5`).

**Frequency:**
- In **replay/live mode**, rebalance on each signal batch or every M simulated minutes. This is the "high-frequency" view on the dashboard.
- The **historical backtest** is daily, close-to-close. Weights decided using information available by day *t* are applied to day *t+1* returns (**no look-ahead; test this explicitly**).

**Metrics:**
- Cumulative return, annualised volatility, Sharpe, maximum drawdown, average turnover and cost drag, measured against:
  - (a) an equal-weight buy-and-hold benchmark
  - (b) a naive sign-only rule (±x% for any positive or negative headline)
- **Information coefficient:** Spearman correlation of `s_i` with the next-day return, both per day and averaged.
- **Robustness grid** over κ and half-life. Show a stable region rather than a single best point, and make no "alpha" claims; describe it as a sentiment-tilt demonstration.

**Dashboard (Module A page):**
- stacked-area chart of weights over time, with event annotations
- current weights against baseline (bar chart)
- performance against benchmarks
- turnover
- IC chart
- clicking a weight change shows the signals and evidence that caused it

---

## 7. Module B: Strategic Portfolio Stress Testing

### 7.1 Trigger logic
- A stress test is triggered by an **event signal** with `impact ≥ threshold` (default **8**, meaning "> 7" as in the brief) **and** `event_confidence ≥ 0.6` **and** `n_sources ≥ 2`. The last two conditions stop a single tweet from firing it.
- A cooldown of 24 h per (event class, region) prevents repeated triggers.
- All thresholds live in config. Every trigger is logged with its evidence.

### 7.2 Synthetic wholesale-banking portfolio (seeded by transaction data, as the brief requires)
**Generator:** `src/riskpulse/moduleB/portfolio.py`, with a fixed random seed. Output goes to `data/portfolio/*.csv`.

**Transaction-data seeding.** Every mapping and assumption goes in ASSUMPTIONS.md.
1. Load the Kaggle Financial Transactions dataset **[VERIFY schema]**.
2. Map MCC codes to industry sectors (`configs/mcc_sector_map.yaml`).
3. Aggregate the transactions by merchant or category cluster to create **synthetic corporate obligors**:
   - Exposure size is proportional to the log of transaction volume.
   - Region comes from merchant geography.
   - A volatility measure of monthly flows tilts the obligor's probability of default (PD) and maps it to a rating bucket.

**Counterparties:** named `CP_0001`, `CP_0002` and so on, each with a sector, region and rating (AAA to CCC). No real names.

**Positions:** about 300 to 400, with total exposure around USD 10 bn. Indicative mix by exposure:

| Asset type | Approx. share | Key attributes |
|---|---|---|
| Corporate loans (term and revolving) | ~45% | EAD, PD, LGD, maturity, fixed or floating, spread |
| Bonds (corporate IG/HY, a few sovereigns) | ~30% | notional, coupon, maturity, YTM, modified duration, convexity, rating |
| Derivatives | ~15% notional-equivalent | Interest-rate swaps (pay/receive fixed; DV01), FX forwards (pair, notional), equity options or total-return swaps (delta, gamma, vega), CDS (protection bought/sold; spread DV01) |
| Equity / trading book | ~10% | sector, beta |

### 7.3 Scenario library (`configs/scenarios.yaml`)
Each event class, optionally refined by region or sector tags and sentiment direction, maps to a **risk-factor shock vector**:
- **Equity:** % change by sector.
- **Rates:** parallel shift in bp, with an optional steepener or flattener.
- **Credit spreads:** bp by rating bucket and sector.
- **FX:** % moves by currency.
- **Commodities:** oil.
- **Volatility:** Δvol.
- **PD multipliers:** by sector and region.

**Calibration from historical analogues (P1 differentiator).** Base magnitudes are not guessed. They are measured with yfinance proxies over public episodes and stored in `data/scenarios/calibration.csv` with dates and tickers. Candidate windows **[VERIFY dates]**:

| Analogue | Window | Maps to |
|---|---|---|
| Lehman | 2008-09-12 → 2008-10-10 | CREDIT_EVENT, severe |
| COVID shock | 2020-02-20 → 2020-03-23 | MACRO / OPS, severe |
| Russia-Ukraine | 2022-02-23 → 2022-03-08 | GEOPOLITICAL |
| 2022 rate shock | | MACRO, inflation/rates |
| SVB | 2023-03-08 → 2023-03-17 | CREDIT_EVENT, banking |

The proxies are SPY and sector ETFs, ^TNX (rates), HYG−IEF and LQD−IEF return spreads (spread proxy), FX, CL=F and ^VIX. The method is documented in the README.

**Severity scaling by impact:** shock = analogue shock × m(impact), where `m = {8: 0.6, 9: 0.8, 10: 1.0}` (config). For multiple concurrent events, combine per factor with **max-severity, not a sum**, to avoid double counting; a "cumulative" mode is optional.

**Sub-scenarios:** MACRO is split by keywords and sentiment into "inflation/hawkish" (rates up) and "growth scare" (rates down).

### 7.4 Valuation engine (simplified, explicit formulas, unit-tested)

| Asset | Revaluation |
|---|---|
| **Bonds** | ΔP/P ≈ −D_mod·Δy + ½·Convexity·Δy², where Δy = Δrate + Δspread(rating, sector) |
| **Loans** | Mark-to-model change ≈ −SpreadDuration·Δspread·EAD. Credit loss: PD_s = min(1, PD·mult), **ΔEL = Σ (PD_s − PD)·LGD·EAD** |
| **Interest-rate swaps** | ΔV = −DV01·Δbp for receive-fixed, +DV01·Δbp for pay-fixed (unit-test the signs) |
| **FX forwards** | ΔV = notional · Δfx% · direction |
| **Equity options / TRS** | ΔV ≈ δ·ΔS + ½·γ·ΔS² + vega·Δσ |
| **CDS** | protection buyer gains +SpreadDV01·Δspread; seller loses the same |
| **Equity book** | ΔV = exposure · sector shock |

**Capital view (P1):**
- Assume a starting CET1 ratio (for example 13%, in config) and an RWA from a simple standardised risk-weight table by rating.
- Report **stressed CET1 = (CET1 capital − losses) / stressed RWA**.
- **P2 stretch:** Basel IRB corporate risk-weight function for loans **[VERIFY formula against the BCBS text before use]**.

**Outputs per run:**
- portfolio value before and after, total P&L and % change
- P&L by asset class, sector, rating and region
- top 10 worst positions
- ΔEL
- CET1 before and after
- the triggering event with its evidence

### 7.5 Module B dashboard
- trigger timeline (events over time, coloured by class, sized by impact)
- KPI tiles: value before, value after, Δ, ΔEL, CET1 before and after
- **waterfall** of P&L by asset class
- heatmap of sector × asset class
- top-hit positions table
- "scenario explainer" card: event → shocks applied → why
- **custom scenario sliders** (what-if)
- scenario comparison
- **Validation (P1):** in historical replay, compare trigger dates with real stress markers (^VIX spikes, SPY drawdown days) and report a simple precision.

---

## 8. Dashboard (Streamlit, four pages, consulting-style light theme)

| Page | Contents |
|---|---|
| 1. **Signal Monitor** | Live document feed; signals table with filters; impact heatmap (entity × time); evidence and explanation drawer; **"Analyze a headline" box** that calls `/analyze` |
| 2. **Module A: Index Rebalancer** | As in §6 |
| 3. **Module B: Stress Testing** | As in §7.5, plus an "Inject demo event" button (labelled synthetic) |
| 4. **Model Quality** | Sentiment, event and impact metrics against baselines; confusion matrix; impact decile chart; latency and throughput |

The dashboard must render from cached outputs when the API is down. This is the `--fast` mode.

---

## 9. Evaluation and reporting (`python -m riskpulse eval all` → `reports/`)

| Area | Metrics |
|---|---|
| Sentiment | macro-F1 and accuracy against VADER and the Loughran-McDonald lexicon (no PhraseBank leakage) |
| Event classification | macro-F1 per model on the gold test set; confusion matrix |
| Impact | Spearman ρ against |CAR| for v1, v2 and \|sentiment\|; decile chart; top-decile hit rate |
| Pipeline | docs/sec, end-to-end latency p50 and p95 per doc on CPU, dedup rate |
| Module A | backtest table against benchmarks; IC; robustness grid |
| Module B | per-scenario results; trigger validation precision |

All results go to `reports/metrics.json` plus PNG charts in `reports/figures/`. **The README and deck quote only these numbers.**

---

## 10. Scope priorities

**P0, must ship:**
- Ingestion from ≥ 2 sources (news and social), with replay mode.
- Entity linking.
- Sentiment, event class and impact v1.
- Signal JSONL output and the FastAPI REST and SSE endpoints.
- Module A with the dashboard page.
- Module B with the synthetic portfolio, scenario shocks, before/after valuation and dashboard page.
- `demo --fast`.
- Mandatory README, `docs/architecture.png`, a 7-slide deck, the video, `/data` with SOURCES.md, the MIT LICENSE and tests for the core formulas.

**P1, differentiators (do all of these if P0 is green):**
- Entity-level sentiment.
- Impact v2 calibrated against the event study.
- Story clustering.
- Historical calibration of shocks.
- ΔEL and CET1.
- `/analyze` and `/inject`.
- Model Quality page.
- Module A IC and robustness grid.
- Module B trigger validation.

**P2, stretch:**
- Basel IRB RWA and rating migration.
- Optional LLM-written explanations, behind a flag and off by default.
- Docker.
- Fine-tuned tweet sentiment.
- Live Reddit source.

---

## 11. Repository layout (target)
```
iiit-dharwad-arnav-gupta-hackathon/
├── README.md  LICENSE  requirements.txt  .env.example  .gitignore
├── configs/        universe.yaml taxonomy.yaml impact.yaml scenarios.yaml
│                   moduleA.yaml moduleB.yaml mcc_sector_map.yaml app.yaml
├── src/riskpulse/
│   ├── __main__.py            # typer CLI: ingest, process, replay, serve, demo, eval, backtest, stress
│   ├── common/                # schemas (pydantic), config loader, logging, time utils
│   ├── ingestion/             # adapters: gdelt.py newsapi.py kaggle_news.py kaggle_tweets.py reddit.py; normalize.py dedupe.py replay.py
│   ├── engine/                # entities.py sentiment.py events.py impact.py clustering.py aggregate.py pipeline.py
│   ├── store/                 # duckdb store, jsonl writer
│   ├── api/                   # FastAPI app, SSE, /analyze, /inject
│   ├── subscribe/             # SignalSubscriber (SSE, store polling)
│   ├── moduleA/               # rebalancer.py backtest.py metrics.py
│   ├── moduleB/               # portfolio.py scenarios.py valuation.py capital.py trigger.py
│   ├── eval/                  # sentiment_eval.py event_eval.py impact_eval.py event_study.py pipeline_bench.py
│   └── dashboard/             # app.py + pages/, theme
├── scripts/        download_*.py, build_replay_feed.py, label_gold.py, calibrate_scenarios.py, make_architecture_diagram.py
├── data/           raw/ (samples) processed/ replay/ signals/ gold/ prices/ portfolio/ scenarios/ SOURCES.md
├── reports/        metrics.json figures/
├── tests/
├── docs/           PROJECT_SPEC.md PROGRESS.md DECISIONS.md ASSUMPTIONS.md architecture.png presentation.pdf
└── prep/           (gitignored) jury-prep notes for Arnav
```

---

## 12. Build plan: one continuous build, run phase by phase without stopping between them

Arnav is building the whole project in a single continuous push, with no day-wise schedule. Run the phases **in order, back to back**, and pause only at the **GATES** marked below. Each phase ends only when its exit criteria pass, the tests are green, the work is committed in small steps and PROGRESS.md is updated.

**Order of work:** get the whole P0 path working end-to-end first (Phases 0 to 7 in their P0 form), then go back and add P1 items in the order listed in Phase 8. A working thin system beats a half-built impressive one. If time or energy runs low, cut P2 first, then P1; never cut P0.

| # | Phase | Exit criteria |
|---|---|---|
| 0 | **Setup.** Repo scaffold, `.gitignore`, configs, CLI skeleton, PROGRESS, DECISIONS and ASSUMPTIONS files, **verification of every [VERIFY] data source** (download samples, inspect schemas) | `python -m riskpulse --help` works; SOURCES.md has a verified status for each source. **GATE A: summarise and wait for Arnav's OK** |
| 1 | **Ingestion and entity linking.** ≥ 2 sources (news and social), normalisation, dedup, replay feed builder, entity linking. **Also generate `data/gold/to_label.csv`** (about 300 stratified headlines with blank label columns) so Arnav can label in parallel | Replay feed with tickers; entity-linking tests pass. **GATE B (non-blocking): tell Arnav the labelling file is ready, then keep going** |
| 2 | **Sentiment.** Doc and entity level, baselines, evaluation | Sentiment metrics in `reports/` |
| 3 | **Event classification.** Keyword baseline, zero-shot, few-shot; story clustering. Evaluate on the gold set once Arnav returns it; until then, evaluate on a held-out slice of weak labels and mark that result provisional | Event macro-F1 reported |
| 4 | **Impact v1, aggregation, store, API.** SSE, `/analyze`, JSONL output. **ENGINE COMPLETE** | `/signals`, `/stream` and `/analyze` work |
| 5 | **Module A.** Rebalancer, backtest, metrics, dashboard page | Weights move in replay; backtest table generated |
| 6 | **Module B.** Portfolio generator (transaction-seeded), scenarios (start with config shocks), valuation engine with formula tests, triggers, dashboard page, `/inject` | Stress run produces before/after and P&L breakdown |
| 7 | **Dashboard and demo.** Signal Monitor and Model Quality pages; `demo` and `demo --fast` | **Full P0 demo runs with one command. GATE C: show Arnav, then continue** |
| 8 | **P1 upgrades, in this order:** (1) impact v2 with the event study, (2) historical calibration of shock magnitudes, (3) ΔEL and CET1, (4) Module A IC and robustness grid, (5) Module B trigger validation, (6) final gold-set evaluation of event classification | Each item lands only with tests and updated metrics |
| 9 | **Hardening.** Tests; README (mandatory template); `docs/architecture.png`; **fresh-clone test in a new venv** | Fresh clone works by following only the README |
| 10 | **Presentation materials.** Draft the 7-slide deck content and a 5-minute video script, with every number taken from `reports/metrics.json`. **GATE D: Arnav records the video, finalises the deck, makes the repo public, checks every link in an incognito window and submits** | Submitted |

---

## 13. Definition of done (pre-submission checklist)
- [ ] Repo is **public**, named `iiit-dharwad-arnav-gupta-hackathon`, with an MIT LICENSE present.
- [ ] README follows the mandatory template exactly, with the architecture diagram embedded and the video link working.
- [ ] A fresh clone, `pip install -r requirements.txt` and `python -m riskpulse demo --fast` work on a clean machine, with the OS tested stated in the README.
- [ ] `/data` contains all data used (or download scripts for large raw data) plus SOURCES.md; no file exceeds 50 MB; no proprietary or client data.
- [ ] Every number in the README and deck traces to `reports/metrics.json`.
- [ ] `docs/presentation.pdf` has at most 7 slides and follows the organisers' outline, in a light consulting style.
- [ ] The video is about 5 minutes, Unlisted, plays in incognito, and follows the organisers' flow.
- [ ] The git history shows incremental commits authored by Arnav only, with no AI co-author trailers.
- [ ] `prep/` notes exist for every module (gitignored), and Arnav can explain every formula.
- [ ] Submitted through the official form once complete, and well before 11 Oct.
