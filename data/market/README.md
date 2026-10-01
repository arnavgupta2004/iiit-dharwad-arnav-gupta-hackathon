# Market reference data (derived)

## credit_spread_levels.csv
Credit spread levels by rating at the portfolio as-of date (2021-09-30), used to price the synthetic book.
Retrieved from FRED on 2026-10-02 (via the browser; FRED's CSV endpoint does not answer scripted clients).

- **Shape across ratings:** median ratio of each ICE BofA US corporate OAS series to the BBB OAS series
  (`BAMLC0A4CBBB`) over the 786 days where all seven series are available on FRED (2023-10-02 → 2026-09-30).
  FRED only publishes the last three years of ICE data, so 2021 OAS levels are not available. p10/p90 show stability.
- **Level anchor:** Moody's Seasoned Baa corporate yield (`DBAA`, 3.37%) minus the 10-year Treasury (`DGS10`,
  1.52%) on 2021-09-30, giving 185 bp for BBB. Caveat: the Baa index has long maturities, so this overstates a 10-year
  OAS, which makes the book's credit spreads somewhat conservative (wide).
- `spread_bp_2021_09_30 = round(185 × median ratio)`.

Raw ICE and Moody's series are not redistributed (third-party copyright); only these derived figures are stored.

## default_rates_by_rating.csv
Long-run average one-year default rates by rating category (Y1 column), from S&P Global Ratings,
"Default, Transition, and Recovery: 2024 Annual Global Corporate Default And Rating Transition Study"
(March 27, 2025), **Table 24: Global corporate average cumulative default rates, 1981-2024** (pp. 55-56).
Public PDF hosted by S&P Global Ratings Maalot: https://maalot.co.il/Publications/FTS20250331162126.pdf
(read 2026-10-02). CCC row = "CCC/C". Used as through-the-cycle PDs for the synthetic book.
Note: the study postdates the 2021 portfolio date. Long-run averages over 1981-2024 vs 1981-2020 differ only
marginally, and these are static inputs, not signals, so this does not create look-ahead in any backtest.

## Moody's Baa/Aaa and 10-year Treasury window extracts (not committed)
`scripts/calibrate_scenarios.py` measures credit-spread changes over each analogue window as the change in
Moody's Baa (and Aaa) yield minus the 10-year Treasury yield (FRED `DBAA`, `DAAA`, `DGS10`). FRED does not answer
scripted clients from this environment, so the window extracts were fetched in a browser on 2026-10-02 and saved to
`data/raw/_downloads/fred/moodys_treasury_windows.csv` (gitignored: third-party copyright). Only the derived window
changes are committed, in `data/scenarios/calibration.csv`. To refresh: open https://fred.stlouisfed.org/series/DBAA,
download DBAA, DAAA and DGS10 as CSV for the window dates in `configs/scenarios.yaml`, and merge on date.
