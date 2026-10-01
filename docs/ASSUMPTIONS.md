# Data and modelling assumptions

Feeds the README "Dataset Used" section. Each assumption has an ID, a statement and its basis.
Items marked (proposed) await confirmation at a gate.

## Data

- **A-01 Timestamps.** All timestamps are stored in UTC. Benzinga processed dates carry an explicit UTC-4 offset, and
  tweet dates are UTC. Items published after 16:00 US/Eastern map to the next trading day for market logic.
- **A-02 Benzinga raw files.** `raw_partner_headlines.csv` has day-only dates. Per the dataset author's note, those
  items are treated as published the *next* day to avoid look-ahead. The minute-precision `analyst_ratings_processed.csv`
  is preferred.
- **A-03 Malformed rows** (unparseable date or missing ticker; 2,578 of 1,400,469 in Benzinga processed) are dropped.
- **A-04 Tweets are attributed to the scraped `Stock Name`** as a prior. Entity linking may add further tickers
  mentioned in the text (cashtags, aliases).
- **A-05 Replay window** (approved at GATE A): 2021-09-30 → 2022-09-29, the period covered by the social dataset.
- **A-08 Universe composition and tech tilt.** The 20-name universe is IT 4 (AAPL, MSFT, AMD, INTC), Communication
  Services 4 (GOOGL, META, NFLX, DIS), Consumer Discretionary 2 (AMZN, TSLA), Consumer Staples 3 (KO, PG, COST),
  Industrials 1 (BA), Financials 3 (JPM, BAC, GS), Energy 2 (XOM, CVX) and Health Care 1 (JNJ). That's 50% in
  IT + Communication + Consumer Discretionary growth names (10 of 20), because the social dataset is dominated by
  heavily-discussed tech and consumer names. Consequences: the equal-weight benchmark and the tilted index are
  growth-heavy, and 2022 (a growth drawdown year) will dominate backtest returns. Results are reported relative to the
  same-universe equal-weight benchmark, so the tilt is held constant across the comparison. Utilities, Materials and
  Real Estate are not represented.
- **A-09 Tweet coverage is uneven.** TSLA has 37,422 tweets while KO has 310, and 6 universe names have none
  (JPM, BAC, GS, XOM, CVX, JNJ). For those names, social sentiment is absent and signals come from news only.
- **A-06 GDELT GKG sampling** (proposed): about 25% of US-session news (one 15-min file per hour). Signal velocity and
  breadth computed from sampled news are therefore relative measures, not absolute article counts.
- **A-07 Sentiment labels.** HF label map: 0 = bearish → negative, 1 = bullish → positive, 2 = neutral.
- **A-10 Sentiment evaluation discipline.** The HF `valid` split is the held-out test set. Nothing is tuned on it
  (labelling thresholds, model choice, calibration); any tuning uses the HF `train` split only.

## Module B synthetic portfolio

- **A-20 Obligors** are synthetic, built by aggregating card transactions by merchant (merchant_id) into synthetic
  corporates `CP_####`. No real company names.
- **A-21 Sector** comes from the merchant's MCC via `configs/mcc_sector_map.yaml` (a judgement-based mapping of all
  109 MCCs to GICS sectors).
- **A-22 Region** comes from merchant_state (US states). Online merchants (null state) are tagged `US-ONLINE`.
  Non-US regions, if needed for geopolitical scenarios, will be assigned synthetically and labelled so.
- **A-23 Exposure size** is proportional to log(transaction volume); PD tilts with the volatility of monthly flows
  (details fixed in Phase 6).
- **A-24 Capital.** Starting CET1 ratio is 13% (config, an assumption). Risk weights follow the standardised approach for
  rated corporates, Basel Framework CRE20.42–20.43 Table 10 (verified; DECISIONS D-024). RWA covers only the synthetic
  book's credit risk (no market or operational risk RWA), so CET1 moves are illustrative of direction and scale.

## Market data

- **A-30 Prices** are yfinance adjusted closes, cached locally. `^TNX` is in percent (x100 = bp).
- **A-32 Shock calibration cutoff.** Scenario shocks are measured only on analogue windows that end before 2021-09-30.
  The 2022 episodes in the replay window are used only for out-of-sample predicted-vs-realised validation.
- **A-31 Spread proxy.** The credit spread change is proxied by the return difference HYG − IEF (HY) and LQD − IEF (IG)
  over the analogue window, converted to bp via an assumed spread duration (fixed in Phase 8 and documented there).
