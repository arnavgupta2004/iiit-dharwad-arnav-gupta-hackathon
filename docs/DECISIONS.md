# Design decisions and deviations

Newest first within each date. Each entry: decision, rationale, and status (accepted, or proposed pending Arnav's OK at a gate).

## 2026-10-02 (event classification round 2, approved by Arnav)

### D-055 Round 2 outcome: gold-2 test (once) and the downstream rerun
**Gold-2** (`reports/events_gold2.json`, predictions in `reports/events_gold2_predictions.csv`; n = 200, 160 news / 40
tweets; evaluated once, commit 2677491). Macro-F1 over 10 classes; last column = 95% paired-bootstrap CI of
F1(C1) − F1(method), for all / news / tweets:

| Method | All | News | Tweets | Accuracy (all) | CI of C1 − method (all / news / tweets) |
|---|---|---|---|---|---|
| C1 (deployed) | 0.559 | 0.610 | 0.197 | 0.640 | - |
| Previous model | 0.577 | 0.625 | 0.207 | 0.635 | [-0.0801, 0.032] / [-0.0822, 0.0407] / [-0.0475, 0.0168] |
| Keyword | 0.595 | 0.660 | 0.216 | 0.700 | [-0.1261, 0.0487] / [-0.1487, 0.0562] / [-0.0505, 0.0043] |
| Zero-shot base | 0.578 | 0.662 | 0.035 | 0.545 | [-0.0804, 0.0983] / [-0.13, 0.0744] / [0.0428, 0.1853] |
| Zero-shot large (ref.) | 0.513 | 0.568 | 0.079 | 0.575 | [-0.0566, 0.1374] / [-0.0792, 0.1414] / [-0.0112, 0.1512] |

**Reading.** On an independent test the deployed C1 is **statistically indistinguishable** from the previous model,
the keyword baseline and both zero-shot models on all items and on news (every CI includes 0). The point estimates
put the keyword baseline highest (0.595). The only CI excluding 0 is tweets vs zero-shot base, in C1's favour.
Tweets have 40 items over 10 classes, so their macro-F1 is unstable. **Round 2 did not improve event classification.**
C1 is deployed as pre-registered. Label mix: OTHER 98, GEOPOLITICAL 33, MACRO 27, and 1 MANAGEMENT_CHANGE item.
**Reported only (D-052):** gold-2 sentiment with the deployed rule: news fine-tuned 0.383 vs
FinBERT 0.553; tweets 0.430 vs 0.429
(no CI computed; no decision taken). Entity-link precision on gold-2: 89.8%
(Wilson 95% CI 82.7-94.2%,
n = 108).
**Downstream (mechanical, D-052 step 6).** Benzinga class features recomputed with C1; v2 retrained (302 trees).
The validation check passed: ρ v2 0.192 vs v1 0.073,
CI of v2 − v1 [0.1101, 0.1285], so **v2 stays for company signals**. Test (n 3,538): v2
0.129, v1 0.062, abs(sentiment) 0.129;
v2 − v1 [0.0406, 0.0954], v2 − abs(s) [-0.0219, 0.0242]. Market-wide check: ρ
-0.015 with abs(SPY), -0.032 with abs(ΔVIX) (CIs span 0).
Module B: 2,613 candidates, 464 triggers, **349 stress runs**.
2022 check: the invasion is unchanged (06:00 UTC, 9 of 14 signs). FOMC fires at 21:00 UTC on 2022-06-15 on "Wall
Street rallies in relief after Fed's assurance on rates" (impact 9), but the macro scenario is a stress by
construction: −USD 134.1 m predicted vs +USD 36.7 m realised, 6 of 14 signs. This illustrates the direction and
surprise limitation. Committed models (D-054): `data/trained/event_clf_round2.pkl`, `data/trained/impact_v2.json`
(+ `.txt`, meta).

### D-054 Model-file policy: small trained artifacts are committed (Arnav)
Large model weights (> 20 MB) are never committed; they are hosted on the Hugging Face Hub (the fine-tuned sentiment
model, 438 MB, `arnavguptas/riskpulse-finbert-tweets`). Small trained artifacts that the reported system needs
(< 20 MB) **are committed** under `data/trained/`, so a fresh clone runs the reported system, not the fallbacks. The
fallbacks (keyword event classes, impact v1, base FinBERT) remain and log a warning when used. Each committed model
file, its training data and the licence of that data are listed in `data/SOURCES.md`.

### D-053 Round 2: cross-validation on gold-1 selects C1 (decided before gold-2 is read)
`python -m riskpulse.engine.event_round2` → `reports/event_round2_cv.json`, `metrics.json → events_round2_cv`. Run
exactly as pre-registered in D-052: StratifiedGroupKFold, 5 folds x 3 seeds = 15 folds, about 60 gold-1 items each,
grouped by story. Training set = 16,452 weak-label rows + gold-1 training folds (weight 10).

| Method | CV mean macro-F1 | SE (15 folds) |
|---|---|---|
| C1 | 0.6004 | 0.0127 |
| C2 | 0.6064 | 0.0073 |
| C3 | 0.6053 | 0.0129 |
| current_deployed | 0.5917 | 0.0126 |
| keyword_baseline | 0.6158 | 0.0096 |
| zero_shot_base | 0.6150 | 0.0109 |

**Rule applied.** Best mean: **C2** (0.6064); one-SE floor 0.5991. In the
simplicity order (C1, C3, C2), C1 (0.6004) is the first candidate at or above the floor, so it is
chosen. C1's mean is above the deployed model's (0.5917), so **C1 is selected.** The three
candidates are within about one SE of each other.
**Caveat, recorded now:** the keyword baseline (0.6158) and zero-shot
(0.6150) score higher than every candidate on gold-1. Under D-052 they were references,
not selectable. Zero-shot cannot run on the full feed on CPU. The SEs come from repeated folds that share data, so
they understate the uncertainty.
**Final model** (C1 recipe, weak labels + all 300 gold-1 rows, weight 10): `data/trained/event_clf_round2.pkl`,
66,915 bytes, SHA-256 `a9b87af858ae2bdd1e51ff67ec8f240b7d63a9a5de02510851caf1cdc6ef8adf`, 16,752 training rows.
It is committed with this entry, before `data/gold/labels_2.csv` is in the repo or has been read, so the evaluated
model is fixed blind to gold-2.
Side effect noted: excluding gold-2 texts from training (D-052) shifted the 1,200-headline zero-shot weak-label sample,
so the candidates' weak set differs slightly from the deployed model's.

### D-052 PRE-REGISTERED before gold-2 is labelled: round-2 protocol
Recorded when `data/gold/to_label_2.csv` exists with blank labels and before any candidate is trained.
1. **Gold-2 sampling** (`scripts/make_gold2.py`, seed 20261003). The pool is replay items of 25-280 characters with
   unique text. It excludes: gold-1 doc ids and texts; near-duplicates of gold-1 (rapidfuzz token-set ratio >= 90);
   items from the same story as a gold-1 item (MiniLM cosine >= 0.80); and any text in the classifier's training set.
   192,906 items remain. Selection: 160 news + 40 tweets. First, at least 5 items per class *predicted by the keyword
   baseline* and at least 5 per class *predicted by the deployed classifier* (all 10 classes reached 5 for both). The
   rest are random within source x month strata, proportional to the pool. No two picks are near-duplicates or
   same-story. Checked afterwards: max cosine to gold-1 0.797; max token-set ratio 83 (to gold-1) and 69 (within
   gold-2). Predicted classes are kept out of the labelling file to avoid anchoring; they are in
   `data/processed/gold2_sampling_meta.parquet` (gitignored).
2. **Roles.** Gold-1 (`labels.csv`, 300) is training and selection data only. Gold-2 (`labels_2.csv`, 200) is the
   final test, evaluated **once**, and is excluded from all training (`load_gold_texts`).
3. **Candidates** (all must run on the full feed on CPU):
   - **C1** - the current recipe (calibrated logistic regression on MiniLM embeddings) trained on the weak-label set
     plus gold-1 training rows with sample weight 10 (fixed in advance).
   - **C2** - the same data and weights, with features = MiniLM embedding concatenated with log(1 + keyword hits)
     per class (10 extra columns), so the model learns when keywords are reliable.
   - **C3** - C1, but when the keyword baseline hits exactly one non-OTHER class, that class is output.
   - References scored in the same folds (not selectable): the current deployed model, the keyword baseline and
     zero-shot (base). Not a candidate: `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (verified: MIT, 435M
     parameters). Base zero-shot already takes ~1.2 s per item on CPU (~70 h for the 203k-item feed; large ~3x more).
4. **Selection by cross-validation on gold-1 only.** StratifiedGroupKFold, 5 folds (stratified by label, grouped by
   story id from the replay clustering), repeated 3 times (seeds 1, 2, 3). In each fold, candidates train on weak
   labels + the 4 training folds and are scored on the held-out fold. Metric: macro-F1 over the 10 classes. Summary:
   mean and standard error over the 15 folds. **Rule:** take the highest mean. If a simpler candidate is within one
   SE of it, take the simpler (simplicity order C1, C3, C2). The chosen candidate must also have a higher CV mean
   than the current deployed model; otherwise the current model stays. The final model is the chosen recipe trained
   on weak labels + all of gold-1.
5. **Gold-2 evaluation, once.** Chosen model, current model, keyword baseline, zero-shot base, and zero-shot large
   (reference only). Macro-F1 overall, news and tweets; per-class F1; paired-bootstrap 95% CIs (2,000 resamples, seed
   20261002) of the chosen model vs each other method. **The chosen model is deployed whatever gold-2 shows**;
   gold-2 is reporting only. Gold-2 sentiment and entity labels are reported, not used for any decision.
6. **Downstream, mechanical, once.** Re-predict feed event classes. Recompute the Benzinga class prior (v2 feature
   T) with the new model and retrain v2 with the same split and parameters. Keep v2 for company signals only if the
   validation CI of rho(v2) - rho(v1) is above 0, else v1 everywhere. Then re-score, re-run the impact evals,
   triggers, Module B and the 2022 check, all logged in D-050 as another look at the 2021-22 data, plus one normal
   snapshot refresh. Repo-size rule per Arnav: no single file over 50 MB; git history under ~150 MB.

## 2026-10-02 (final rerun, D-046): results

### D-051 Final rerun results (all from `reports/metrics.json`, `riskpulse eval all` run once)
Order followed: sentiment decision (D-048) → taxonomy fix (D-044) and event retrain → v2 retrained (sentiment and events
changed) with the validation check → re-score → kappa recalibrated → τ = 5% → triggers and the 2022 check → all evals →
snapshot. Nothing was tuned on any result below.
- **Event classes, gold (n = 300, first and only gold evaluation):** macro-F1 primary 0.593, keyword baseline 0.612,
  zero-shot 0.612. CI of primary − keyword [−0.071, 0.033], primary − zero-shot [−0.073, 0.034]. **The trained
  classifier does not beat the simple baselines on live-feed text.** It was trained on weak labels whose largest
  source is HF tweets. News 0.616 / 0.657 / 0.640; tweets 0.399 / 0.389 / 0.396. Strongest classes: REGULATORY_LEGAL
  0.77, MACROECONOMIC 0.75. Weakest: MANAGEMENT_CHANGE 0.34, OTHER 0.43.
- **Entity linking:** precision 82.0% on 100 hand-checked links (Wilson 95% CI 73.3-88.3%).
- **Impact v2 (retrained: 134 trees):** validation ρ v2 0.192 vs v1 0.075 (CI of v2 − v1 [0.108, 0.126]), so the
  adoption check passes; vs |s| 0.215 (CI [−0.032, −0.014]). Test (post burn-in, n = 3,538): v2 0.129, v1 0.058,
  |s| 0.129. v2 − v1 [0.044, 0.096]; v2 − |s| [−0.022, 0.024]. Top-decile hit rate: v2 24.9%, v1 16.7%, |s| 24.9%.
  **v2 is live for company mentions; it is on par with |sentiment| and above v1.**
- **Market-wide check (D-042):** 188 sessions. ρ with |SPY return| −0.033 (CI [−0.165, 0.099]), with |ΔVIX| −0.037
  (CI [−0.170, 0.108]). **No measurable relation:** market-wide v1 impact does not rank market-move days.
- **Module A:** kappa 10.98 (median |active| 1.52% on the calibration months; the target was 1.5%, and the median is
  a step function of kappa). IC +0.0381, t = 2.39, 56.5% positive days (209 days). With τ = 5%, the tilt net return is
  −21.9% (gross −21.5%, cost drag 0.52%, turnover 5.0%/day), vs EW rebalanced −23.3%, EW buy-and-hold −21.5%, naive
  sign rule −21.8%. The realised median |active| in the evaluation is 0.76%: the 5% cap slows convergence to targets.
- **Module B:** 2,527 candidates, 470 triggers (29 escalations), **349 stress runs** (from 459). Classes: MACRO 181,
  GEOPOLITICAL 79 (from 210), CREDIT 75, OPERATIONAL_ESG 14.
- **2022 check:** Russia–Ukraine fired at 06:00 UTC on 2022-02-24 ("Global market plunges, stocks dive after
  Vladimir Putin launches military operations in Ukraine", impact 9). 9 of 14 signs right; oil +6.6% predicted vs
  +18.0% realised. The FOMC hike **now fires on the day** (00:00 UTC, "Sterling slides as US ramps up inflation
  fight", inflation_hawkish from taper_tantrum_2013). But only 6 of 14 signs are right: it predicted +36 bp on the
  10y vs −39 bp realised, and equities down vs up. Book −USD 101.0 m predicted vs +USD 36.7 m realised.
  Both misses stay as written up (supply shock vs risk-off; surprise vs consensus).
- **Pipeline (re-measured on an idle machine):** 140.4 docs/s batch; single-document latency p50 29.5 ms, p95
  33.0 ms; models take 97.7% of the time. A first run straight after `eval all` gave 107.4 docs/s on a hot machine.
- **Fresh clone (Phase 9):** clone of 88 MB; README Quickstart (`venv`, `pip install -r requirements.txt`) in 211 s;
  `pytest -q` passed; `demo --fast` served 18,210 snapshot signals and the dashboard responded (HTTP 200). This used a
  warm Hugging Face cache, so model downloads were not exercised.
- **Reporting additions (Arnav):** gold CIs per subset for trained vs keyword and vs zero-shot. The events eval was
  re-run once for this: the same models and predictions, so the macro-F1 values are unchanged. The trained classifier
  stays deployed (the a-priori choice). HF-topic results are labelled "in-domain (HF topic data)". Impact v2's JSON
  evaluator matches LightGBM exactly (`np.array_equal`) on the full validation set and on all 2021-22 test
  ticker-days (`tests/test_impact_v2_parity.py`; LightGBM runs in a child process).

### D-050 Evaluation history: every look at the 2021-22 replay window (Arnav's item 5)
The replay window (2021-09-30 → 2022-09-29) is both the demo window and the out-of-sample test period. Listed here,
in order, is every time its **outcomes** (returns, abnormal returns, VIX/SPY, realised factor moves, trigger
counts) informed or were reported by a result. Inspections of texts, links or score distributions with no outcome
attached are listed separately at the end.

| # | When | What was looked at | Outcome data | Did it change anything? |
|---|---|---|---|---|
| 1 | Phase 5 | Module A backtest, κ = 1, full window: returns, IC, κ × half-life grid | next-day returns | No parameter chosen from it; κ later replaced by a returns-free rule (D-038) |
| 2 | Phase 6 | Module B trigger replay with decile bins: 2,574 triggers/yr | trigger counts (no market outcomes) | Yes: severity scale per population (D-035); design rationale was the trigger rate, not market outcomes |
| 3 | pre-GATE C item 5 | Impact v1 vs abs(sentiment) vs abs(CAR), post burn-in (D-036) | abnormal returns/volume | No retuning of v1; motivated v2 |
| 4 | GATE C | Trigger redesign (D-037) reviewed against trigger counts per month | trigger counts | Yes: macro-region cooldown, escalation, breadth bonus removed (Arnav) |
| 5 | GATE C | Impact v2 first test (D-039) | abnormal returns/volume | Pre-registered rule not met → reported; **rule later revised** (D-039 revision) → v2 adopted |
| 6 | GATE C | Module A with frozen κ (D-038): IC and returns from month 3 | next-day returns | No; τ change (D-045) justified by turnover, not returns |
| 7 | GATE C | 2022 predicted vs realised (D-040) | realised factor moves | No; misses written up as limitations |
| 8 | Final rerun (D-046) | Gold events (first time), sentiment gold (same numbers as D-048), all evals recomputed once with the final models | all | Reported as final (D-051) |
| 9 | After D-051 | Gold events re-scored with identical predictions only to add per-subset CIs (Arnav) | gold labels | No; numbers unchanged |
| 10 | Round 2 (D-053) | Gold-1 cross-validation for selection (gold-1 = training data from here on) | gold-1 labels | Yes: selected C1, by the pre-registered rule |
| 11 | Round 2 (D-055) | Gold-2 evaluated once | gold-2 labels | No; C1 deployed as pre-registered |
| 12 | Round 2 downstream (D-055) | v2 retrained (validation check), impact evals, triggers, Module B and 2022 check rerun with C1 | abnormal returns, trigger counts, realised factor moves | No parameter changed from them |

Not outcome looks (recorded for completeness): GKG pilot and linking rules (D-017, D-020) inspected texts and links;
D-035 fitted bins on burn-in raw-impact distributions; the FOMC-day investigation in D-040 read impact values of
2022-06-15 stories; the D-042 dry run read the distribution of daily max market-wide impact (no SPY/VIX).
Other test sets: HF sentiment test split, base FinBERT evaluated in Phase 2, the fine-tuned model once (D-043);
gold set: sentiment once (D-048), events once (final rerun).

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
**CC BY-NC-SA 3.0**. **Approved by Arnav:** the weights are released under CC BY-NC-SA 3.0 and the code under MIT;
the README states both, with the reason.

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

### D-049 Impact v2 in the live engine (company mentions); engineering choices
- **Live features = the training features, causally.** v2 was trained on ticker-session aggregates (max of each
  driver over the session's mentions plus log(1 + mentions), 16:00 ET rule). In the engine, each company mention
  updates its ticker's running aggregate for the current session and is scored on it. So by the session close the
  input equals the training features, and no score uses later information. Market-wide items keep v1 (D-042).
  The 1-10 scale is unchanged in method: burn-in quantile bins per population (company bins now fitted on v2 raws).
- **v1 stays measurable.** Mentions store v1's drivers, and the evals recompute v1 from them (max deviation from the
  previously stored v1 raw: 3e-5, from 4-decimal driver rounding).
- **No lightgbm in torch processes.** On macOS, LightGBM's and PyTorch's OpenMP runtimes cannot share a process:
  lightgbm-then-torch hangs at 0% CPU, and torch-then-lightgbm segfaults (both reproduced 2026-10-02 in standalone
  scripts and in pytest). v2 is therefore fitted in a spawned subprocess and exported as a JSON tree dump. The engine,
  API and evals score it with `engine/gbm.py`, a small pure-Python/NumPy evaluator that matches
  `Booster.predict` to 1e-9 (tested in a child process, and checked again at every training run).
- **Training and evaluation are separate commands:** `riskpulse train events|impact_v2` and `riskpulse eval ...`.
  So `eval all` scores saved models and never retrains, and each test set is looked at once per evaluation.
- **Fresh clone:** model files are not committed (repo rule: no model weights in git). Without `impact_v2.json` the engine logs a warning and
  uses v1 everywhere; `demo --fast` is unaffected.

### D-048 D-041 applied: gold sentiment result and the mechanical decision
`riskpulse eval sentiment_gold` → `metrics.json → sentiment_gold`, run once with the pre-registered protocol.
Macro-F1 with the deployed rule (first-ticker entity score, ±0.15 band):

| Live-feed gold | FinBERT | Fine-tuned | F1(ft) − F1(FinBERT), 95% CI | VADER | LM |
|---|---|---|---|---|---|
| News headlines (237) | 0.592 | 0.549 | −0.043 [−0.115, 0.032] | 0.471 | 0.442 |
| Tweets (63) | 0.389 | 0.368 | −0.021 [−0.109, 0.071] | 0.297 | 0.257 |

In-domain (HF test split): fine-tuned 0.844 vs FinBERT 0.661/0.668. **The in-domain gain does not transfer to live-feed
text.** Point estimates favour FinBERT on both subsets, but neither CI lies entirely below zero, so **the rule selects
the fine-tuned model everywhere.** It is applied as written; changing it now would be a post-hoc choice. How to
report it: "fine-tuning helped in-domain; on our live feed it is statistically indistinguishable from FinBERT (point
estimate 4 F1 points lower on news)". Tweet gold has only 4 neutral items, so its macro-F1 is unstable. Both models
beat the lexicon baselines on live text.

### D-047 Stage-1 caches split by model, so the sentiment re-score can run ahead of the rerun
Before this, the feed cache was keyed by feed + all models, and the Benzinga cache only checked its row count, so a
sentiment change would have reused stale scores. Now there are three parts. **Base** = links, regions and MiniLM
embeddings, keyed by feed + universe/linking config + tweet map + embedder. **Sentiment** is keyed by a hash of the
weights files, so Hub and local copies of the same weights share a key. **Events** are predicted from the cached
embeddings, keyed by the event-model file. The existing caches were migrated without recomputation, and the new code
reproduces them **exactly** (all fields, 203,363 feed docs and 330,100 Benzinga headlines). Note: embeddings are stored
as float16. Re-predicting events from them flips 31 of 203,363 feed docs (0.015%), all within ±0.001 of the 0.5
confidence threshold. That's why events are cached rather than recomputed. After the D-044 retrain, events will come
from the float16 embeddings. Arnav asked to start the fine-tuned re-score now (compute only):
`scripts/precompute_sentiment.py --variant finetuned`, measured at about 16 min (feed) + 28 min (Benzinga) on CPU, so
the ~3 h limit that would have required a sampled Benzinga set is not reached.

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
split (7 classes; **in-domain (HF topic data)**, 0.712 after D-044) the primary embedding classifier reaches macro-F1 0.800 against 0.502 for the keyword baseline; on a
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
