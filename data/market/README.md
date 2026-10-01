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
