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
- **A-05 Replay window** (proposed): 2021-09-30 → 2022-09-29, the period covered by the social dataset.
- **A-06 GDELT GKG sampling** (proposed): about 25% of US-session news (one 15-min file per hour). Signal velocity and
  breadth computed from sampled news are therefore relative measures, not absolute article counts.
- **A-07 Sentiment labels.** HF label map: 0 = bearish → negative, 1 = bullish → positive, 2 = neutral.

## Module B synthetic portfolio

- **A-20 Obligors** are synthetic, built by aggregating card transactions by merchant (merchant_id) into synthetic
  corporates `CP_####`. No real company names.
- **A-21 Sector** comes from the merchant's MCC via `configs/mcc_sector_map.yaml` (a judgement-based mapping of all
  109 MCCs to GICS sectors).
- **A-22 Region** comes from merchant_state (US states). Online merchants (null state) are tagged `US-ONLINE`.
  Non-US regions, if needed for geopolitical scenarios, will be assigned synthetically and labelled so.
- **A-23 Exposure size** is proportional to log(transaction volume); PD tilts with the volatility of monthly flows
  (details fixed in Phase 6).
- **A-24 Capital.** Starting CET1 ratio is 13% (config). Risk weights follow Basel II standardised corporate weights
  (to be verified against the BCBS text before use).

## Market data

- **A-30 Prices** are yfinance adjusted closes, cached locally. `^TNX` is in percent (x100 = bp).
- **A-31 Spread proxy.** The credit spread change is proxied by the return difference HYG − IEF (HY) and LQD − IEF (IG)
  over the analogue window, converted to bp via an assumed spread duration (fixed in Phase 8 and documented there).
