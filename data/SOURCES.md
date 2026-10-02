# Data sources

Every dataset and API RiskPulse uses, with licence, verification status and how it is used.
"Verified" means the item was actually downloaded or called on the date shown, and its schema
was printed and inspected (`scripts/profile_raw_sources.py`). Raw downloads live in
`data/raw/_downloads/` (gitignored) and are re-created with `scripts/download_*.py`.
Only cleaned samples, derived files and synthetic data are committed.

No proprietary or client data is used. Synthetic records are labelled synthetic in the data itself.

## Summary

| # | Source | Type | Licence / terms | Status (2026-10-01) | Role | In repo |
|---|---|---|---|---|---|---|
| 1 | GDELT DOC 2.0 API | News, live | Free, no key; GDELT open terms | Verified | Live feed (`live` mode) | Code only |
| 2 | GDELT GKG 2.0 raw 15-min files | News, historical | Free; GDELT open terms | Verified; pilot passed (D-017) | News for the replay window (2021-22) with themes, orgs and tone | Script + derived replay feed |
| 3 | Kaggle: equinxx/stock-tweets-for-sentiment-analysis-and-prediction | Social, historical | CC0 | Verified | Second (social) source; replay feed; Module A window | Cleaned sample for universe (Phase 1) |
| 4 | Kaggle: miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests | News, historical | CC0 | Verified | Cross-sectional event study for impact v2 | Download script + derived features |
| 5 | HF: zeroshot/twitter-financial-news-sentiment | Labelled sentiment | MIT | Verified | Sentiment evaluation (valid split = our test) | Download script |
| 6 | HF: zeroshot/twitter-financial-news-topic | Labelled topics | MIT | Verified | Weak labels / extra eval for event classes via topic map | Download script |
| 7 | yfinance (Yahoo Finance) | Prices | Yahoo terms; personal/research use | Verified | Event study, backtests, shock calibration | Cached CSV/Parquet under `data/prices/` |
| 8 | Kaggle: computingvictor/transactions-fraud-datasets | Transactions | Apache 2.0 | Verified | Seeds the Module B synthetic wholesale book | Download script + derived synthetic portfolio |
| 9 | NewsAPI (newsapi.org) Developer plan | News, live | Free dev plan, dev/testing use only | Terms verified; not called (no key) | Optional live feed | Code only |
| 10 | Wikipedia "S&P 100" constituents table | Reference | CC BY-SA | Verified | Universe membership check | Not committed |
| 11 | Human gold set (`data/gold/`) | Labels by Arnav | Own work | Template delivered (300 items); labels pending | Event-class test set, linking precision, sentiment spot check | Committed |
| 12 | S&P Global Ratings, 2024 Annual Global Corporate Default and Rating Transition Study (public PDF via S&P Maalot) | Reference | S&P publication; 7 derived numbers cited | Verified (Table 24, pp. 55-56) | One-year PD by rating for the synthetic book | Derived values only (`data/market/default_rates_by_rating.csv`) |
| 13 | FRED: ICE BofA US corporate OAS by rating; Moody's Baa/Aaa yields; 10-year Treasury | Market data | Third-party copyright (ICE, Moody's) via FRED | Verified (fetched in browser; FRED blocks scripted clients here) | Spread levels by rating; credit shocks per analogue window | Derived values only (`data/market/`, `data/scenarios/calibration.csv`) |
| 14 | Loughran-McDonald sentiment word lists (2014 release, bundled in `pysentiment2`, MIT package) | Lexicon | LM: free for academic use | Verified (2,355 negative / 354 positive words) | Sentiment baseline | Not committed (installed package) |
| 15 | Basel Framework CRE20 (bis.org) | Regulation | Public | Verified 2026-10-02 (CRE20.7, .42-.43, .57) | Risk weights for the CET1 view | Cited in DECISIONS D-024 |
| - | Kaggle: thedevastator/tweet-sentiment-s-impact-on-stock-returns | Social | CC0 | Verified, **rejected** | - | No |
| - | Kaggle: ankurzing/sentiment-analysis-for-financial-news (Financial PhraseBank) | Labelled sentiment | CC BY-NC-SA 4.0 | Metadata verified; **not used for FinBERT eval** | Optional only | No |
| - | HF: Zihan1004/FNSPID | News | CC BY-NC 4.0 | Inspected via HTTP range reads; **not used** | - | No |

## Details

### 1. GDELT DOC 2.0 API
- Endpoint: `https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=ArtList&format=json&maxrecords=N&timespan=...`
- Response: `{"articles": [{url, url_mobile, title, seendate (YYYYMMDDTHHMMSSZ), socialimage, domain, language, sourcecountry}]}`.
  There is no article body and no tone in ArtList mode; we get the title and URL only.
- Theme filter syntax `theme:ECON_BANKRUPTCY` works (returned articles). Theme matches are noisy
  (for example, a cemetery-maintenance story came back under ECON_BANKRUPTCY), so themes are weak labels only.
- **Rate limit:** over-limit requests get HTTP 429 with the message "Please limit requests to one every 5 seconds".
  In practice, requests spaced 7 to 10 s apart still got 429s after a burst, so the adapter uses a
  >= 6 s spacing plus exponential backoff on 429 (`configs/app.yaml`).
- Lookback: a query on 2026-07-15 (about 2.5 months back) returned results. Older-date probes: see the
  "GDELT DOC lookback" entry in docs/DECISIONS.md.

### 2. GDELT GKG 2.0 raw files
- URL pattern: `http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.gkg.csv.zip` (redirects to https), every 15 minutes.
- Verified file `20220224120000.gkg.csv.zip`: 6.4 MB zipped / 20.0 MB unzipped, 1,497 records, 27 tab-separated
  columns, latin-1. `<PAGE_TITLE>` is present in the Extras XML (column 26) for 100% of records in that file.
  Useful columns: 1 DATE, 3 SourceCommonName, 4 DocumentIdentifier (URL), 7 V2Themes, 13 V2Organizations, 15 V2Tone, 26 Extras.
- Sample org matches in that file included a Boeing story and a "Russia-Ukraine War Tensions ... Dow Jones Plunges" story.

### 3. Stock tweets (equinxx), CC0
- `stock_tweets.csv`: 80,793 rows; columns `Date` (UTC, e.g. `2022-09-29 23:41:16+00:00`), `Tweet`, `Stock Name`, `Company Name`; no nulls.
- Range: 2021-09-30 00:06 → 2022-09-29 23:41 UTC.
- Tickers (rows): TSLA 37,422; TSM 11,034; AAPL 5,056; MSFT 4,089; PG 4,089; AMZN 4,089; NIO 3,021; META 2,751;
  AMD 2,227; NFLX 1,727; GOOG 1,291; PYPL 843; DIS 635; BA 399; COST 393; INTC 315; KO 310; CRM 233; XPEV 225;
  ENPH 216; ZS 193; VZ 123; BX 50; F 31; NOC 31.
- **Defect:** the PG and MSFT sets (4,089 each) are exact copies of the AMZN set (mislabelled); see DECISIONS D-018.
  About 20.7% of universe tweets are cashtag lists; see D-019 for the filters and yields.
- **Final social coverage (after the original-ticker rule, D-033):** **12 of 20** universe tickers have tweets in the
  scored replay (37,467 tweets): TSLA 27,681; AAPL 2,729; AMZN 2,068; META 1,541; AMD 1,128; NFLX 1,022; GOOGL 509;
  DIS 350; COST 155; INTC 116; BA 113; KO 55. **8 have none** (news only): MSFT and PG (their sets were AMZN copies),
  JPM, BAC, GS, XOM, CVX, JNJ (not in the dataset).
- `stock_yfinance_data.csv`: 6,300 rows of daily OHLCV for the same tickers (we use our own yfinance cache instead).

### 4. Benzinga news (miguelaenlle), CC0
- `analyst_ratings_processed.csv`: 1,400,469 rows; columns `title`, `date` (UTC-4 offset, minute precision), `stock`;
  6,192 tickers; 2009-02-14 → 2020-06-11; 2,578 rows have unparseable dates or missing stock (malformed lines; dropped).
- `raw_analyst_ratings.csv`: 1,407,328 rows (headline, url, publisher, date, stock).
- `raw_partner_headlines.csv`: 1,845,559 rows; day-only dates; publishers include Seeking Alpha, Zacks and GuruFocus.
- **Coverage caveat:** per-ticker history is truncated for many mega-caps (e.g. AAPL only from 2020-03; JPM, INTC,
  DIS and BA about 10 rows each). It is therefore used cross-sectionally (many tickers) for the impact event study,
  not as a per-name history for our 20-stock universe.

### 5–6. HF Twitter financial news (zeroshot), MIT
- Sentiment: `sent_train.csv` 9,543 rows, `sent_valid.csv` 2,388 rows; columns `text`, `label`;
  label map **0 = bearish, 1 = bullish, 2 = neutral** (checked against both the README `LABEL_` map and the meta counts:
  bearish 1,789 = 1,442 + 347).
  No test split exists, so `valid` is our held-out test set.
- Topic: `topic_train.csv` 16,990 rows, `topic_valid.csv` 4,117 rows; 20 labels 0..19. **The label order is the one in
  the dataset README (`LABEL_0` Analyst Update, `LABEL_1` Fed | Central Banks, ... `LABEL_19` Stock Movement), not
  alphabetical.** An earlier note here said alphabetical; it was wrong (only label 0 coincided) and was corrected on
  2026-10-02 after the class counts disagreed with the meta file. Per-label counts match the README order exactly.
  The map lives in `configs/taxonomy.yaml` (`hf_topic_labels`).

### 7. yfinance
- All requested symbols download daily history from 2008-01-02 to 2026-09-29: SPY, ^GSPC, ^TNX, ^IRX, HYG, LQD, IEF,
  CL=F, DX-Y.NYB, ^VIX, XLF, XLK, XLE, XLV, XLY, XLP, XLI, XLU, XLB, EURUSD=X, JPY=X, GBPUSD=X.
  XLRE starts 2015-10-08 and XLC starts 2018-06-19 (so they are unavailable for the Lehman analogue).
- `^TNX` is quoted in percent (2020-03-23 close 0.764), so x100 gives bp.

### 8. Transactions (computingvictor), Apache 2.0
- `transactions_data.csv` (1.26 GB): columns id, date, client_id, card_id, amount (string like `$-77.00`), use_chip,
  merchant_id, merchant_city, merchant_state, zip, mcc, errors. In the first 500k rows: 2010-01-01 → 2010-05-31,
  19,040 merchants, 109 MCCs; merchant_state null for about 11% (online merchants).
- `mcc_codes.json`: 109 codes with descriptions → mapped in `configs/mcc_sector_map.yaml`.
- `users_data.csv` (2,000) and `cards_data.csv` (6,146) contain synthetic personal-style fields (addresses,
  card numbers, CVVs). **We do not use or commit them**; only merchant-level aggregates feed the portfolio.

### 9. NewsAPI
- Developer plan (pricing page, read 2026-10-01): free; 100 requests/day; articles have a 24 h delay; search up to
  one month old; for development and testing only. Optional adapter, off unless `NEWSAPI_KEY` is set.

### 10. S&P 100 membership
- Wikipedia constituents table (page last edited 2026-09-27): 101 rows (GOOG and GOOGL both listed).
  All 20 proposed tickers are current members.

### Rejected / not used
- **thedevastator Tweet Sentiment's Impact** (the organisers' example): the CSV is structurally broken when parsed.
  Multi-line tweets shift columns, so dates land in `STOCK` and prices in `DATE` for a large share of rows.
  Entities are company names, not tickers, and the data covers 2017–2018. Rejected in favour of the equinxx tweets.
- **Financial PhraseBank:** `ProsusAI/finbert` was fine-tuned on it, so reporting FinBERT accuracy on it would be leakage.
- **FNSPID:** CC BY-NC; the 5.7 GB / 23 GB CSVs begin with the same Benzinga data, and 2021–22 coverage could not be
  confirmed cheaply. GDELT GKG covers the replay window instead.
