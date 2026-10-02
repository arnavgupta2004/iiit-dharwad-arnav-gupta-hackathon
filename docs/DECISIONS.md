# Design decisions and deviations

Newest first within each date. Each entry: decision, rationale, and status (accepted, or proposed pending Arnav's OK at a gate).

## 2026-10-02 (P1 review by Arnav). Decisions recorded now; applied together in one final rerun

### D-041 PRE-REGISTERED before any gold label is scored: sentiment model choice on live-feed text
Recorded before `data/gold/labels.csv` exists. The fine-tuned model scored macro-F1 0.844 on the HF test split
(D-043), which is **in-domain text** (same dataset as its training data). The live feed is GDELT news headlines plus
equinxx tweets, and only the gold spot check measures that.
- **Items:** the 300 gold items with a `label_sentiment`, split by `source`: news headlines (`gdelt`, 237) and tweets
  (`kaggle_tweets`, 63). The tweet subset is small, so its CIs will be wide; this is accepted in advance.
- **Prediction, identical for both models:** the deployed rule. Entity-level score for the first ticker in
  `linked_tickers` (clause-level `entity_sentiment`; doc-level for MKT), score = P(pos) − P(neg), label with the
  configured ±0.15 band. Argmax labels are reported as a secondary view and play no part in the decision.
- **Metric:** macro-F1 over {negative, neutral, positive}. Paired bootstrap of F1(fine-tuned) − F1(FinBERT), 2,000
  resamples within each subset, seed 20261002, 95% percentile CI.
- **Decision rule:** use the fine-tuned model everywhere, **unless** on news headlines the CI of the difference is
  entirely below 0; then FinBERT for news and fine-tuned for tweets. Nothing else is decided from these labels.
- **Reporting:** 0.844 is always shown next to the gold result, labelled "in-domain (HF test split)" vs "live-feed
  text (gold)".

### D-042 Pre-specified descriptive check of market-wide impact (v1), no tuning
Market-wide (MKT) events keep impact v1 (v2 has no market-wide training target). Check, reported only:
over trading days after the burn-in (from 2022-01-03), take the maximum impact of market-wide event signals assigned
to each session (16:00 ET cutoff, as in Module A; after-close events count for the next session). Spearman of that
maximum against |SPY close-to-close return| and against |ΔVIX| (close-to-close, points) on the same session, using
days with at least one market-wide event. 95% bootstrap CI (2,000 resamples). Written to
`metrics.json → impact_market_check`. No parameter is changed from the result.

### D-043 Fine-tuned sentiment model: adopted (Arnav's item 4); hosting and licence
`metrics.json → sentiment_finetune`: FinBERT fine-tuned on 90% of the HF tweet-sentiment train split (10% of train as
dev for early stopping), 3 epochs, 23 CPU minutes. Test split touched once: **macro-F1 0.844** vs the 0.661 bar (FinBERT
argmax 0.668). Leakage check after the run: 52 of 2,388 test texts (2.2%) share a 60-character prefix with a train
text (39 exact after URL normalisation). That bounds the effect at about 2 F1 points, so it can't explain the gain. Not
re-scored without them, to avoid another look at the test split.
Adopted under the rule; the live-feed check is D-041. Hosting: Hugging Face Hub under Arnav's account (he uploads).
Pipeline order: Hub → local rebuild (`scripts/build_finetuned_sentiment.py`) → base FinBERT, logging which was used.
Licences checked 2026-10-02: training data `zeroshot/twitter-financial-news-sentiment` is **MIT** (redistribution
allowed). Base model `ProsusAI/finbert`: the HF model card declares **no licence**; the code repo
github.com/ProsusAI/finBERT is Apache-2.0. The base weights were fine-tuned on Financial PhraseBank, which is
**CC BY-NC-SA 3.0**. **Open for Arnav** before uploading publicly: the card will declare cc-by-nc-sa-3.0 as the
conservative choice for a non-commercial hackathon artefact, and state the provenance.

### D-044 GEOPOLITICAL means cross-border only (Arnav's item 3)
GEOPOLITICAL = war and military conflict, sanctions, trade conflict and tariffs between countries, international
diplomacy. Domestic politics and election chatter with no cross-border or policy-shock dimension → OTHER. The gold
labelling guide was updated *before labelling* so the gold set uses this definition. `taxonomy.yaml` (description,
zero-shot label, keywords, HF `Politics` mapping), weak labels and the classifier change in the final rerun, followed
by one evaluation on the gold set.

### D-045 Module A turnover cap 5% one-way per day, set from policy (Arnav's item 2)
A 19% average daily one-way turnover is not operationally realistic for an index product. The cap (partial rebalance
λ = min(1, τ/turnover)) goes from τ = 20% to **τ = 5%**. This is a policy choice, not a return optimisation, and
it was not chosen from the sensitivity grid. Results are reported as they come out. IC stays the headline. Returns
are secondary and shown gross, with cost drag and net.

### D-046 Final rerun protocol (Arnav's item 5)
One combined rerun once the gold labels are in, in this order: sentiment model (D-041/D-043) → event classifier with
D-044 → impact v2 retrained on Benzinga with the same split if sentiment changed, then v2 > v1 rechecked on
validation → re-score → kappa recalibrated mechanically on the same 2-month window (1.5% rule, no returns) → τ = 5%
→ triggers and 2022 predicted-vs-realised → **all** evals once → demo snapshot once. A list of every earlier look at
the 2021-22 test set is added alongside the rerun.

## 2026-10-02 (GATE C decisions by Arnav)

### D-040 2022 predicted vs realised: one hit with partial agreement, one miss (GATE A item 3)
`riskpulse eval predicted_vs_realised` → `metrics.json → moduleB_validation`, shown on the Module B page. Rule fixed
before running: the prediction is the highest-impact stress run that actually fired on the event date for the mapped
class; realised moves are measured with the calibration code from the close before the event over 10 sessions.
- **Russia–Ukraine, 2022-02-24 (GEOPOLITICAL):** fired 09:00 UTC, impact 8, 17 outlets ("EU Sanctions Hit Russian
  Defense Minister…"), scenario from Crimea 2014 / Brexit 2016 / Abqaiq 2019 at severity 0.6. Sign agreement 9 of 14
  factors (64.3%): right on European equities, financials, oil up, BBB spreads wider, EUR/GBP down, USD up and VIX up.
  Wrong on SPY, energy and tech equities (they rose), 3m rates and JPY. Magnitudes too small for oil (+4.9% vs +18.0%)
  and BBB (+6 bp vs +22 bp). Book: predicted −USD 8.1 m vs realised +USD 1.6 m. CET1 12.98% vs 12.95%; the realised
  ratio is lower despite a gain because equity and derivative exposures grow RWA. The analogues are not oil-supply
  shocks of this size.
- **FOMC 75 bp hike, 2022-06-15 (MACROECONOMIC):** **missed.** No MACROECONOMIC run fired on the date. Cooldown was not the
  cause: every MACROECONOMIC story that day peaked at impact 7, including "Fed Raises Interest Rates by
  Three-Quarters…" (20:00 UTC, confidence 0.94, 2 outlets in the 1-in-4 GKG sample), one notch below the threshold. As supplementary context only, the
  latest run in the preceding week (CPI day, 2022-06-10, impact 9, growth_scare scenario) would have predicted
  −8.9% SPY and +18.4 VIX pts, against a realised +2.2% and −4.5 pts over 06-14 → 06-29 (sign agreement 57.1%).
  The sell-off had already happened between the CPI print and the hike.

Nothing is tuned on these two episodes. They are the validation set.

### D-037 Stress trigger: macro-region cooldown, escalation, no breadth bonus (Arnav's GATE C item 1)
Changes (Arnav's choice of options (c) and (d) from GATE C):
- **Story breadth bonus removed** (`app.yaml → event_breadth_bonus: []`). Breadth already enters impact through B.
- **Cooldown key = (event class, macro-region)**, 24 h. Engine region tags map to 5 macro-regions
  (`moduleB.yaml → trigger.macro_regions`): US → Americas; EUROPE, UK, RUSSIA_UKRAINE → Europe; CHINA, JAPAN, INDIA →
  Asia-Pacific; MIDDLE_EAST → Middle East & Africa. Events that span more than one macro-region, carry no region, or
  carry only the cross-continent EM tag are keyed as Global (my implementation choice for the edge cases).
- **Escalation:** within a cooldown a new event re-runs the stress test only if its impact is strictly higher than
  the impact that fired it. An escalation resets the key's reference impact and cooldown clock.
- min_sources stays at 2; min_impact 8 and event confidence 0.6 are unchanged.

Replay result (`metrics.json → moduleB_triggers`): 4,279 high-impact candidates, **811 triggers fired** (60 of them
escalations), **459 stress runs** (previously 694). The other 352 fired triggers are classes without a market-wide scenario, e.g. EARNINGS.
Blocked: cooldown 2,111, low event confidence 1,196, a single source 161. Stress runs per month are 28-48 from Oct 2021
on (monthly table in metrics.json). Still about 1.3 runs per day; class errors remain (e.g. US political news labelled
GEOPOLITICAL). On 2022-06-15 (the FOMC hike) no MACROECONOMIC run fired; the nearest were the CPI-day runs on 06-10.
**Nothing else is tuned on the 2022 window: it is the validation set.**

### D-038 Module A: kappa from a risk budget, frozen; IC is the headline (Arnav's GATE C item 3)
kappa = 1 (a spec default) is replaced by a calibration that never uses returns (`moduleA/calibrate.py`): bisection on
kappa so the **median absolute active weight |w~ − w0| of the target weights equals 1.5%**, over all name-days of
the first 2 months (2021-09-30 → 2021-11-29), using only the signal distribution (s, c) and the existing deadband
and [2%, 12%] bounds. Result: **kappa = 7.473** (the median |s·c| outside the deadband is small, so a large kappa is
needed). It is frozen in `data/processed/moduleA_calibration.json` and the evaluation runs from 2021-11-30 to
2022-09-29 (209 days). The realised median |active| of the held weights over the evaluation is 1.23% (lower than 1.5%
because the turnover cap slows convergence to targets).

Headline (`metrics.json → moduleA.headline_information_coefficient`): mean daily IC 0.0376, t = 2.31, 56.5%
positive days. Returns are reported as secondary: tilt −26.1% vs EW rebalanced −23.3% vs EW buy-and-hold −21.5% over
the evaluation window. The cost drag is 2.0% at about 19% average daily turnover. The kappa × half-life grid
(kappa 7.47 added) is shown as a sensitivity chart only; nothing is selected from it. **Not tuned after seeing these
results.** Interpretation for the jury: the signal ranks next-day returns slightly better than chance, but a daily
tilt of this size pays more in turnover and in concentration in falling names than the IC earns.

### D-039 Impact v2 trained, evaluated once, **not adopted** (Arnav's GATE C item 2)
`reports/metrics.json → impact_v2`, `reports/figures/impact_v2_deciles.png`, `riskpulse eval impact_v2`.
LightGBM on the impact drivers plus |sentiment| (S) and log mention count, monotone-increasing constraints, target =
within-day rank of |CAR[0,+1]|. Trained on Benzinga ticker-days up to 2018-12-31 (124,436), validation 2019-01-01 →
2020-07-31 (29,079) used only for early stopping (best iteration 546). 216 tickers have prices; 84 were dropped.
Adoption rule fixed before the test: adopt only if the 95% paired-bootstrap CIs of ρ(v2)−ρ(|s|) and ρ(v2)−ρ(v1) are
both above zero on the untouched 2021-22 test set (post-burn-in ticker-days, n = 3,538).

| |CAR[0,+1]|, test | Spearman ρ | top-decile hit rate |
|---|---|---|
| impact v2 | 0.108 | 24.0% |
| impact v1 | 0.046 | 13.8% |
| abs(sentiment) | 0.104 | 25.7% |

ρ(v2)−ρ(v1) CI [0.035, 0.089] (v2 beats v1). ρ(v2)−ρ(|s|) CI [−0.020, 0.030] (includes zero). On validation v2 was
already slightly below |s| (CI [−0.023, −0.002]). **Outcome: not adopted. Per Arnav's rule, reported and stopped, with no
iteration on the test set.** The live engine keeps v1 (the spec's interpretable formula) and reports the comparison
openly.

**Revised after the test was seen (Arnav, P1 review):** v2 is adopted for company-level signals in the final rerun.
The stricter rule written before the test (beat both v1 and |sentiment|) **was not met**, and the adoption rule was
changed after the test result was known. Justification: the spec's rule ("if v2 doesn't beat v1, ship v1") is met,
v2 beats v1 on validation (CI [0.121, 0.144]) and on test (CI [0.035, 0.089]), and v1 is demonstrably worse than
|sentiment| (D-036). Market-wide events keep v1 (D-042). Framing: v2 is on par with |sentiment| for predicting market
reaction and adds explainable drivers. Gain importance is dominated by mention count and velocity; B, C and R have zero gain because they are
constant in single-publisher Benzinga. Limitations: survivorship (delisted tickers excluded), single-publisher
training data, and domain shift from Benzinga headlines to GDELT/tweets.

## 2026-10-02 (pre-GATE C review by Arnav)

### D-031 PD provenance, value by value
Every PD in `data/market/default_rates_by_rating.csv` was re-extracted from the S&P PDF itself ("Default, Transition,
and Recovery: 2024 Annual Global Corporate Default And Rating Transition Study", March 27, 2025), **Table 24 "Global
corporate average cumulative default rates, 1981-2024"**, column Y1:

| Rating | Y1 (%) | Table | Page |
|---|---|---|---|
| AAA | 0.00 | 24 | 55 |
| AA | 0.02 | 24 | 55 |
| A | 0.05 | 24 | 55 |
| BBB | 0.14 | 24 | 56 (cont.) |
| BB | 0.56 | 24 | 56 (cont.) |
| B | 2.93 | 24 | 56 (cont.) |
| CCC (CCC/C) | 26.12 | 24 | 56 (cont.) |

Caution recorded during verification: page 56 also starts **Table 25 (U.S. region)**, whose AAA/AA/A rows read
0.00/0.03/0.06. Those are *not* used. Only these seven values are stored; no table is copied into the repository.

### D-036 Impact v1 underperforms |sentiment| (Arnav's item 5; reported, not tuned away)
`reports/metrics.json → impact`, `reports/figures/impact_deciles.png`. Event study at the ticker-day level (market model
on SPY, [−120, −20]; CAR[0,+1]), post-burn-in replay window, n = 3,538: Spearman ρ(v1, |CAR|) = 0.046 (p = 0.006,
top-decile hit rate 13.8%) against ρ(|sentiment|, |CAR|) = 0.104 (p = 5e-10, hit rate 25.7%). The paired-bootstrap 95% CI of the
difference is [−0.086, −0.028]. On abnormal volume it is 0.080 vs 0.107. v1 is positively related to market reaction, but
its hand-set weights dilute the sentiment signal. This motivates impact v2 (P1 #1): weights learned on the Benzinga event
study with a time split. v2 replaces v1 only if it beats both v1 and |sentiment| on held-out data. The v1 weights are
**not** retuned on these results (that would leak the evaluation).

### D-035 Impact 1-10 as a severity scale, per population (deviation from plain deciles; for Arnav at GATE C)
With decile bins, "impact >= 8" meant the top 30% of all items: 70.7% of market-wide (MKT) mentions scored >= 8
against 5.3% of company mentions (MKT items carry relevance 1 and high class priors), and the Module B trigger fired
2,574 times in the year. Changed, still quantile binning on the burn-in as the spec requires:
- Edges at burn-in quantiles [0.15, 0.30, 0.45, 0.60, 0.72, 0.84, 0.95, 0.99, 0.998], so 8 = top 5%, 9 = top 1% and
  10 = top 0.2% (chosen on design grounds, not tuned on outcomes).
- Separate edges for company mentions and MKT items.

Result: 694 stress runs per year (from 2,150; 1,110 triggers fired, of which stress classes 694). That is still about
2 per day. Remaining drivers are spec-set values: the story breadth bonus (+1 at 5 outlets, +2 at 15) on top of the B
driver, 24 h cooldown per (class x region) (~40 keys), and n_sources >= 2, which is trivial for GDELT. Fired triggers
include the real events (2022-02-24 sanctions at impact 8-10 with 12-19 outlets; CPI 8.6% on 2022-06-10; the 75 bp hike;
Russia default fears; Kaisa and Revlon). They also include event-class errors, e.g. US election chatter labelled
GEOPOLITICAL. **Resolved at GATE C:** see D-037.

### D-033 Tweets link only to their original ticker (Arnav's item 1)
Identical tweet texts scraped under several tickers are kept once and linked only to their original ticker. Copy sets
are detected from the data: a label whose texts are ≥ 95% contained in another label's set and that has a lower
own-cashtag rate is a copy. The result is MSFT → AMZN and PG → MSFT (itself a copy), so MSFT and PG have **zero**
original tweets. Among genuine labels, the first-mentioned cashtag decides; a tweet whose original is outside the
universe (e.g. TSM) links to nothing. The rule is applied in the tweet loader (future feed builds), in the live
linker, and between stage 1 and stage 2 for the cached batch (no model re-run), using the committed map
`data/replay/tweet_original_ticker.csv` (63,686 distinct texts). News never links PG unless the headline names P&G
(context rules, D-020). PG stays in the universe. Correction to the pre-GATE C note: PG stays near baseline mainly
through confidence decay (mean c = 0.06), not the deadband (|s| < 0.10 on only 22.6% of days); its own tilt has a median
of 0.18% (ASSUMPTIONS A-09).

### D-034 Entity signals on a stream-time cadence
Batch mode called the engine once, so entity signals were emitted only at the end (20 for the whole year). Entity
signals are now emitted every 60 minutes of stream time for tickers touched since the last emission (config
`aggregation.entity_emit_every_minutes`), plus at the end of each call. Event-signal emission is unchanged (D-029).

### D-032 Stage-2 optimisation verified against the original code (Arnav's item 3)
The optimised velocity tracker and clusterer were rewritten to reproduce the original semantics exactly (velocity
baseline buckets relative to each mention's time, integer-microsecond timestamps, float64 clustering), after a first
optimisation had silently changed the baseline to aligned buckets. Verification:
- `tests/test_stage2_regression.py` (committed): the verbatim pre-optimisation code (`tests/legacy/`, from commit
  74716fb^) and the current code process the same 1,000 synthetic stage-1 documents. Story ids, impact scores, impact
  raws, all drivers (to 1e-9) and the emitted event signals are identical.
- `scripts/check_stage2_equivalence.py` on real cached stage-1 output (stage 1 not re-run), two 1,000-document slices
  (first docs; from 2022-02-24 03:00 UTC): story ids, impact scores and event signals (137 and 196) identical. In each
  slice exactly one mention differs, by one unit in the 4th stored decimal of its novelty driver (a rounding flip from
  a ~1e-16 difference in a cosine sum), changing raw impact by 1e-6 and its 1-10 score not at all.

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
max |Δp| 5e-6, with 100% argmax agreement. No result depends on the accelerator; reviewers run on CPU. Note added after the pipeline benchmark: on an idle CPU the
batch rate is 150 docs/s (`reports/metrics.json → pipeline`); the 18 docs/s seen earlier was CPU contention with a
concurrent zero-shot job, not a CPU limit.

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
