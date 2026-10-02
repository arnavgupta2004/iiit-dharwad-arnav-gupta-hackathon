# Gold set: human labels (test set only)

`to_label.csv` holds 300 items sampled from the replay feed (237 news headlines, 63 tweets), stratified by
source and by keyword-baseline class so that rare classes appear. **Labelled by Arnav Gupta.** This is a test
set: it is never used for training, and the training code drops any text that appears here.

## How to label
Fill the blank columns, save as `labels.csv` in this folder (same columns), and run
`python -m riskpulse eval events` (and `eval sentiment` for the spot check).

| Column | Values | Guidance |
|---|---|---|
| `label_event_class` | one of the 10 classes below | The main event the text reports. If none applies, use `OTHER`. |
| `label_sentiment` | `positive` / `negative` / `neutral` | Tone for the first ticker in `linked_tickers`, from an investor's point of view. For `MKT`, the tone for markets overall. |
| `label_entity_correct` | `y` / `n` / `partial` | Is the text actually about the company in `linked_tickers`? `partial` if only some listed tickers are right. Leave blank for `MKT`. |
| `label_impact_1_10` | 1–10 (optional) | Your gut feel for the potential market impact. Used only as a sanity check, never for training. |
| `notes` | free text | Anything ambiguous. |

## Event classes (from `configs/taxonomy.yaml`)
- **GEOPOLITICAL**: war, military conflict, sanctions, elections, coups, trade conflict and tariffs between countries.
- **MACROECONOMIC**: inflation, GDP, jobs, central-bank decisions, interest rates, recession, fiscal policy.
- **CREDIT_EVENT**: default, missed payment, downgrade, bankruptcy, restructuring, covenant breach, bank run, liquidity crisis.
- **MERGER_ACQUISITION**: mergers, acquisitions, takeover bids, buyouts, divestitures, spin-offs.
- **PRODUCT_LAUNCH**: new products, services, models or features launched or announced.
- **EARNINGS**: quarterly results, revenue, profit, guidance, beats and misses, dividends.
- **REGULATORY_LEGAL**: fines, lawsuits, investigations, regulatory approvals or rejections, antitrust.
- **MANAGEMENT_CHANGE**: CEO/CFO/board appointments, resignations, departures, succession.
- **OPERATIONAL_ESG**: outages, cyberattacks, accidents, recalls, strikes, supply disruption, environmental or governance controversy.
- **OTHER**: opinion, general market commentary, analyst notes, price moves without a clear event.

Tips: label what the text *says*, not what you know happened later. A stock-price move with no stated cause is `OTHER`.
Analyst rating changes are `OTHER` (there is no analyst class), unless the note is about a downgrade of *credit*.

## Entity-linking precision check (`entity_check.csv`)
100 random headline → ticker links from the news feed, 5 per ticker, excluding the texts above. For each row, fill
`correct` with **y** if the headline is genuinely about that company (a subsidiary or its products count, e.g.
"JPMorgan Sec. plc" → JPM, "Gillette India" → PG), and **n** if the link is wrong (wrong entity, a word collision,
or the company is only incidental). Use `notes` for borderline cases. Save in place and run
`python -m riskpulse eval linking` to write precision (overall and per ticker) to `reports/metrics.json`.
These labels are never used for training or tuning.
