# AI Finance Controller

**Live demo:** https://cashforecastor.vercel.app
**API:** https://cashforecastor.onrender.com/docs

> Note: the backend runs on Render's free tier and spins down after 15 minutes of inactivity. The first request after a period of no traffic takes 30–50 seconds to respond — this is expected, not a bug.

Two finance-ops loops for a Razorpay-style payments business, built to the same standard: real measured numbers, honest limitations, no averaging away the bad ones.

**Cash-position forecasting** (`data/generate_synthetic_ledger.py`, `src/forecast.py`) — a daily net-settled-amount forecasting pipeline for Razorpay-style merchant ledgers. Three LightGBM regressors with `objective='quantile'` at alpha=0.1, 0.5, and 0.9 produce a calibrated P10/P50/P90 band for each of the next 14 days. Two things make this different from a standard point-forecast: the quantile trajectories are independent (each feeds back its own prior predictions into the lag features, not the median), and the bands are conformally calibrated against pooled residuals from all 11 backtest calibration windows. The honest part: coverage and pinball loss are reported per merchant category and the numbers that are bad (marketplace, with a hard regime shift) are reported as-is rather than averaged away.

**Books reconciliation** (`data/generate_synthetic_recon.py`, `src/reconcile.py`) — a staged, rule-based matcher that closes the other half of the loop: reconciling Razorpay's internal transaction ledger against the bank/processor UTR settlement feed, across a synthetic 137-transaction / 113-settlement-row batch with genuinely unresolvable records planted on purpose. Every MATCHED/EXCEPTION decision is deterministic; a planted ground-truth answer key the engine never reads lets a separate scorer compute real precision, recall, and false-match rate instead of a self-reported number.

---

## accuracy

Walk-forward backtest, 11 folds (train / 14-day calibration / 14-day test, rolling forward),
2,310 merchant-fold evaluation days, November 2025 – July 2026.

| category            | coverage (raw) | coverage (global CQR) | coverage (per-category CQR) | pinball loss (P50) |
|---------------------|----------------|------------------------|------------------------------|---------------------|
| **overall**         | 77.7%          | 83.1%                  | **84.1%**                    | ₹40,020             |
| saas_subscription   | 78.2%          | 89.0%                  | **84.1%**                    | ₹3,762              |
| food_delivery       | 83.8%          | 87.2%                  | **84.9%**                    | ₹10,622             |
| d2c_ecommerce       | 76.1%          | 81.7%                  | **84.4%**                    | ₹38,245             |
| marketplace         | 74.0%          | 75.6%                  | **83.3%**                    | ₹100,103            |

Target coverage is 80%. Three calibration approaches are compared on the same 2,310 evaluation days:

**Raw** — leakage-fixed quantile trajectories with no post-hoc correction.

**Global CQR** — conformalized quantile regression with one correction applied uniformly across all categories. Residuals from all 11 calibration windows are pooled into a single score array; q_hat = empirical 80th-percentile of those scores (q_hat = −355, meaning the final model trained on full data was already slightly over-covering and needed a marginal band contraction).

**Per-category CQR** — separate q_hat per merchant category, pooled the same way. Production values: marketplace +9,023 (band expanded), d2c_ecommerce +1,058, saas_subscription −704, food_delivery −2,454 (bands contracted for the stable categories). The per-fold q_hats for marketplace ranged 6,388–103,934 across the 11 folds — the fold-1 spike aligns with the festival window — confirming the correction is responding to actual volatility rather than noise.

The direction of each change is correct: marketplace needed expansion (it was genuinely under-covered at 75.6%), while saas and food needed contraction (they were over-covered at 89% and 87%, with unnecessarily wide bands). No fold triggered the minimum-scores fallback — all four categories had calibration data in every fold.

Zero P10 > P50 or P90 ordering violations across all 2,310 evaluated days.

### what the numbers mean, and the bugs behind them

The model went through three measurable coverage improvements, each from a specific diagnosed cause:

**64.7% → 77.7%: recursive leakage fix.** All three quantile trajectories (P10/P50/P90) were built by feeding the median (P50) prediction back into lag features, regardless of which quantile was being forecast. The P90 path never accumulated a genuinely high trajectory over 14 days — it only differed from P50 in its final prediction step. Fixed by running each quantile path independently, feeding its own prior predictions back as lag fills. The band now fans open over the horizon instead of being three near-parallel shifted lines.

**77.7% → 83.1%: pooled global CQR.** Even with correct trajectories, the first calibration pass used only the last 14-day window (q_hat = ₹1,117, a low-volatility period). Switching to residuals pooled across all 11 calibration windows gave a q_hat reflecting the full volatility distribution, including the festival-spike fold where the per-fold q_hat was ₹103,252.

**83.1% → 84.1%: per-category CQR.** A single global q_hat applied uniformly was under-serving marketplace (high noise, needed band expansion) and over-correcting saas/food (stable categories, bands already wide). Separate q_hats per category fixed this: marketplace expanded (+9,023), d2c slightly expanded (+1,058), saas and food contracted (−704 and −2,454). After each fix, a systematic check verified the new implementation inherited the pooling-across-folds structure correctly rather than silently falling back to a single window.

---

## limitations

The data is synthetic, generated by `data/generate_synthetic_ledger.py`. The merchant categories, noise levels, day-of-week patterns, and settlement lags are plausible approximations, not calibrated against real Razorpay ledger statistics.

The festival spike is a clean 3.1x multiplier applied to a fixed date window. Real festival spikes vary in timing (Diwali shifts each year), ramp up and down over days, interact with inventory constraints, and are often different across sub-categories of D2C merchants. The model here has no advance signal of an upcoming festival — it only learns the pattern in-sample. A real system would want a festival/event calendar as an explicit feature.

The regime shifts (`mcht_d2c_02` at +115%, `mcht_mkt_02` at -52%) are hard step-changes on fixed dates. Real regime changes — a merchant acquiring a new distribution channel, a competitor entering the market, a logistics partner going offline — are rarely that clean. The model picks them up after a few cycles of lag history but will be wrong for approximately one lag window (14 days) after the break.

What a real deployment would need on top of this:
- Real settlement data from Razorpay's ledger system, at minimum 6–12 months per merchant before forecasting is reliable.
- A festival/event calendar feature, ideally with lead-time (days until next major event).
- Merchant onboarding metadata (category, GMV tier, payment method mix) as static features, not just derived from the time series.
- A human review threshold: the exception flags here use a simple band-width heuristic. A production system needs tuning of that threshold against actual downstream decisions (e.g., credit line sizing).
- Drift detection on the live feed. The walk-forward backtest here trains from scratch each fold; a deployed system needs to detect when the live error rate degrades and trigger retraining.
- Per-merchant minimum history enforcement. Three merchants in this dataset (`mcht_saas_05`, `mcht_d2c_05`, `mcht_food_04`) have only 32–41 days of history; they're excluded from backtest folds where training data is too thin.

---

## reconciliation

The books half closes one finance-ops loop: reconciling Razorpay's internal ledger against the bank/processor UTR settlement feed across a synthetic 137-transaction / 113-settlement-row batch. A staged, rule-based matcher decides MATCHED/EXCEPTION deterministically; a secondary 0–1 confidence score is attached to every record for human triage but never decides the label. Ground truth for the batch — which ledger row truly belongs to which settlement, and which rows are deliberately unresolvable and why — is planted alongside the data by `data/generate_synthetic_recon.py` and written to its own file (`data/recon_ground_truth.json`) that `src/reconcile.py` never opens; only `score_reconciliation.py` does, so the numbers below are measured against an answer key the engine never saw, not self-reported.

Numbers below are from an actual `python run_reconciliation.py && python score_reconciliation.py` run on the committed `data/recon_ledger.csv` / `data/recon_bank_settlement.csv` (seed 42) — every number here is what that run printed, not rounded for looks.

| metric | value | n |
|---|---|---|
| match rate | 78.8% | 108 / 137 |
| precision | 100.0% | TP=108, FP_raw=0, WA=0 |
| recall | 85.7% | TP=108, FN_raw=18, WA=0 |
| false-match rate (ledger) | **0.0%** | 0 / 11 |
| false-match rate (bank) | **0.0%** | 0 / 5 |
| false-match rate (pooled) | **0.0%** | 0 / 16 |
| wrong-attribution rate | 0.0% | 0 / 126 |
| duplicate precision / recall | 100.0% / 100.0% | 8 / 8 both |
| trap recall | 2 / 2 | both planted traps correctly left unresolved |
| reason-code accuracy (on the true-orphan/duplicate set) | 100.0% | 16 / 16 |
| throughput | ~75,000–90,000 records/sec | elapsed ~0.003s over 250 records |

Match rate and recall are lower than an earlier build of this engine reported (87.6% / 95.2%) — see the false-match bug writeup below for why, and why the lower number is the correct one to ship.

The 18 recall-gap records break into two causes, both intentional: 6 are the planted `AMBIGUOUS_SUBSET_SUM` trap (two of the eight batch settlements were deliberately constructed so their ledger members sum to the *same* target within tolerance; the engine correctly refuses to guess and leaves all 6 members plus both bank rows as `UNRESOLVED_SUBSET_SUM` rather than force-matching one arbitrarily). The other 12 are ground-truth-resolvable transactions whose bank credit happens to carry a blank `merchant_ref_code` (~10% of bank rows are generated this way) — the engine now refuses to auto-match *any* blank-ref credit on amount+date alone, full stop, because that's a coincidence, not verified attribution (see below). Neither is a shortfall the engine should try to close by guessing — an engine that "resolved" either cleanly would be hiding either a genuine ambiguity or an unverifiable attribution a real ops team needs to see, not doing a better job. Throughput is reported as a range because at ~250 records the entire matching pass runs in a few milliseconds, so `time.perf_counter()` noise is a real fraction of the number — reported honestly rather than picking the best of several runs.

The elapsed time above times Stages 0b–6 only (test-mode exclusion, duplicate quarantine, the four 1:1 passes, batch subset-sum, residual classification); CSV load, report writing, and the confidence-scoring pass that runs after classification are excluded, per the spec.

### what the numbers mean, and the bugs behind them

Five things surfaced while building this that are worth stating plainly, in the same spirit as the forecasting section above. The first three were caught while building the engine; the last two were caught afterward, by an independent adversarial review of the finished code before this was called done — which is exactly the kind of bug an author grading their own work tends to miss.

**A blank bank reference could silently produce a false match across merchants (bank-side false-match risk, closed).** ~10% of bank credits are generated with no `merchant_ref_code`, deliberately, to force a real fallback path. The first version of the matcher handled this by searching *all* open ledger rows (any merchant) for a unique amount+date coincidence and auto-accepting it as a match if exactly one candidate existed. That safeguard sounds sound — "only match when unambiguous" — but it doesn't catch the actual failure mode: a ledger row for merchant A with no real bank counterpart can coincidentally share date and amount with an unrelated blank-ref credit that really belongs to merchant B. There *is* exactly one numeric candidate, so the old code matched it — confidently, and wrongly. There is no signal available at match-decision time that can tell a correct coincidence from an incorrect one (the ledger's `utr_hint` field exists for exactly this kind of corroboration, but is reserved for the confidence score only, never the match decision, precisely so it can't be used to paper over cases like this). Given the project's own stated bar — a false reconciliation is worse than a missed one — the fix is to stop guessing entirely: blank-`merchant_ref_code` bank credits never enter the 1:1 or batch passes now, and always land as `ORPHAN_BANK_CREDIT` / `no_merchant_ref` for manual review. That closes the vulnerability by construction rather than by luck of this seed's draws (this seed's actual data never happened to trigger the collision — the bug was latent, not observed, until an adversarial reviewer constructed the scenario by hand and reproduced it against the code). The cost is real and visible in the table above: 12 of this seed's blank-ref credits were, in fact, genuine correct matches that the old code got right; the new code now leaves all 13 blank-ref rows (12 resolvable + 1 true orphan) for manual review, which is where match rate (87.6% → 78.8%) and recall (95.2% → 85.7%) went. A regression test (`test_blank_merchant_ref_never_auto_matches_even_on_unique_coincidence`) plants the exact adversarial scenario and asserts it is never force-matched.

**A timing near-miss could silently roll more than one day late, depending on the seed (latent, closed before it could surface).** `TIMING_NEAR_MISS` rows are meant to land exactly one calendar day off the expected settlement date — the engine's auto-match window (`TIMING_AUTO_DAYS`) is exactly ±1 day for this reason. The generator built this by shifting the date by ±1 day and then rolling forward over weekends/the fixed bank holiday, the same way every other settlement date in this dataset is rolled — but that second rolling step can walk the date more than one day out (e.g. a Friday expected-settlement-date shifted +1 lands on Saturday, then rolls to Tuesday across the fixed holiday: a 4-day jump, not 1). Depending on which capture dates a given seed happens to draw, that would silently turn an intended "near miss, auto-resolves" scenario into a `TIMING_OUT_OF_WINDOW` exception instead, degrading match rate for reasons that have nothing to do with the matcher's actual logic. Confirmed this specific escape did not fire for seed 42 (all 7 planted rows happened to land at offset ≤1), so it never showed up in any measured number here — but it was real and reproducible with the right dates, and the hand-built unit test fixtures for this scenario bypass the generator's date rolling entirely, so nothing would have caught it before a different seed did. Fixed by picking whichever of the two neighboring days is already a business day, with no further rolling, so the offset is provably ±1 regardless of seed (verified across four seeds by `test_timing_near_miss_rows_are_always_exactly_one_day_off`).

Three more things surfaced while building the engine itself:

**A silent fee re-draw broke the ambiguous-subset trap (0/2 → 2/2 trap recall).** The generator's `_recompute()` helper recalculates a ledger row's fee/TDS/net *and* its expected settlement date from scratch — including drawing a fresh random `razorpay_fee`, because fee is a genuine per-transaction random draw, not a deterministic function of gross amount. The `AMBIGUOUS_SUBSET_SUM` trap is built by nudging one batch member's amount so its batch's total exactly equals a second batch's total, then aligning the two batches onto the same settlement date. That date-alignment step called the same `_recompute()` — which re-rolled the fee on the just-nudged row and silently pulled the two batch totals apart again, by tens of rupees, after the equality had already been engineered. The result: both trap batches sailed through as ordinary unique-subset matches, and `UNRESOLVED_SUBSET_SUM` never fired at all (0 occurrences). Fixed by splitting the helper in two — `_recompute()` for first-time construction of a row's financial fields, and a separate `_recompute_settlement_date_only()` that only touches the date — and using the latter for any post-hoc date alignment. After the fix, both trap batches (6 ledger rows + 2 bank rows) land in `UNRESOLVED_SUBSET_SUM` as designed: trap recall 2/2.

**Duplicate-canonical selection by date instead of by ID produced a real false match (bank-side false-match rate 20% → 0%).** The spec's duplicate-quarantine rule is "keep the earliest bank_row_id ... as canonical." The first implementation read "earliest" as "earliest by `bank_credit_date`, tie-broken by ID" — plausible, but wrong: one of the three planted duplicate bank rows has a ±1-day date jitter that, this seed, landed it a day *earlier* than its true original. Sorting by date made the clone canonical and quarantined the true original as the "duplicate" instead. The clone then proceeded through ordinary 1:1 matching and — coincidentally, within tolerance — matched a real ledger row that wasn't its true counterpart. Ground truth marks that bank row `DUPLICATE`; the engine matched it anyway: a genuine false match, caught by the bank-side false-match-rate metric (1/5 = 20%) before this was fixed. Re-reading the spec's parallel ledger-side rule ("the lexicographically-first ledger_txn_id proceeds") made the intent clear: canonical selection should be pure ID order (duplicates are always created with higher sequence numbers than their source in this generator, so ID order is also chronological-creation order), not date order. After the fix: false-match rate is 0.0% on both sides, and this is explicitly asserted (not just reported) by `tests/test_score_reconciliation.py::test_false_match_rate_on_true_orphans_is_exactly_zero`.

**The fee-drift tolerance boundary doesn't land where the scenario name implies it would (10 of 11 rows, not 0).** `FEE_DRIFT_1_1` rows use a fixed ±Rs2.50–4.50 nominal drift, exactly per spec, meant to land inside the fee-adjusted tolerance (`AMOUNT_TOL_FEE_ADJ`, ±Rs5/0.5%) but outside the exact tolerance (`AMOUNT_TOL_EXACT`, ±Rs2/0.05%). That inequality only holds when the 0.05%-of-amount term stays below the Rs2 floor — i.e., for transactions under roughly Rs4,000. Most of this batch's transactions are several thousand to tens of thousands of rupees, where 0.05% already exceeds Rs2, so the exact-tolerance band is wider than the fixed drift. Measured result: 10 of the 11 fee-drift rows land inside `AMOUNT_TOL_EXACT` and match via `exact_1to1`; only 1 needs `fee_adjusted_1to1`. This isn't a correctness bug — all 11 still match correctly — but it means the `rule` column doesn't cleanly reflect "this was a fee-drift scenario" the way the scenario name suggests, and it's reported here rather than left for someone to discover by surprise.

### limitations (reconciliation)

- The fee/TDS schedule (flat 1.8–2.2% Razorpay fee, flat 1% TDS above a threshold, flat+bps bank fee) is a simplified stand-in for real Razorpay nodal-account settlement logic.
- The single fixed `BANK_HOLIDAYS` entry is a placeholder, not a real bank calendar.
- Duplicate detection collides on `(merchant_id, capture_date, gross_amount)` only — no idempotency key or webhook ID is modeled, so it would misfire on legitimately identical same-day repeat charges in a real system.
- `MAX_SUBSET_POOL=12` / `MAX_SUBSET_SIZE=6` are real scale ceilings; a real batch bundling 50+ transactions needs a DP or ILP approach, not enumeration (even meet-in-the-middle). The pool is built from candidate *density* (every open same-merchant transaction in the credit's date window), not true batch size, so an unrelated cluster of same-day transactions can push a genuinely small, resolvable batch over the cap.
- Blank-`merchant_ref_code` bank credits (~10% of the feed) are never auto-matched, by design, after the false-match bug described above — every one of them lands as `ORPHAN_BANK_CREDIT` for manual review, even when it would have coincidentally matched correctly. This is a deliberate accuracy-for-safety tradeoff, not an oversight; a real system might instead re-attempt these with a human-in-the-loop confirmation step rather than leaving them all as exceptions unconditionally.
- The confidence score's weights (0.55/0.30/0.15) are hand-chosen, not calibrated against outcome data, and never affect the MATCHED/EXCEPTION decision.
- Ground-truth blindness is enforced by file separation plus a static-source test, not process sandboxing.
- The optional LLM annotation layer, if enabled, only ever produces the human-readable `note` string and is never consulted for `reason_code` or the match decision; it is untested against a live API key in this repo's test suite.
- This seed's batch happens not to organically trigger `AMOUNT_MISMATCH` or `TIMING_OUT_OF_WINDOW` (both codes are implemented and covered by unit tests in `tests/test_reconcile.py`, just not produced by this particular planted mix) — worth knowing if you're eyeballing the reason-code counts and wondering where they went.

---

## how to run locally

**Prerequisites:** Python 3.9+, Node.js 18+ (v26.7.0 used in development).

### 1. Python environment

```bash
cd "razorpay final"
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install python-multipart uvicorn[standard]
```

### 2. Train models and run backtest

```bash
python train_and_backtest.py
```

This writes:
- `models/lgb_q10.txt`, `lgb_q50.txt`, `lgb_q90.txt`, `models/forecaster.pkl`
- `reports/backtest_results.csv` — per-fold, per-merchant rows
- `reports/backtest_summary.csv` — aggregate + per-category metrics
- `reports/exceptions.json` — low-confidence flags

Backtest takes 3–5 minutes on a laptop (11 folds × 300 LightGBM rounds × 3 quantiles, plus a second full-data training pass for the production model).

### 3. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs at http://localhost:8000/docs.

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard at http://localhost:5173. The Vite dev server proxies `/api/*` to `localhost:8000`.

### 5. Regenerate synthetic data (optional)

```bash
python data/generate_synthetic_ledger.py --seed 42 --out data/synthetic_ledger.csv
```

### 6. Reconciliation: generate the batch, run the matcher, score it

```bash
python data/generate_synthetic_recon.py --seed 42   # optional -- already committed
python run_reconciliation.py                        # writes reports/reconciliation_*
python score_reconciliation.py                       # writes reports/reconciliation_accuracy.json, prints the scored table
```

`run_reconciliation.py` writes `reports/reconciliation_matches.csv`, `reports/reconciliation_exceptions.json`, and `reports/reconciliation_summary.json`. `score_reconciliation.py` is the only script that opens `data/recon_ground_truth.json`; it compares the engine's output against it and writes `reports/reconciliation_accuracy.json`. Both run in well under a second — this is a rule-based matcher over ~250 records, not a model.

### 7. Run the test suite

```bash
pip install pytest==8.3.4   # already in requirements.txt
pytest
```

---

## architecture

```
razorpay final/
├── data/
│   ├── generate_synthetic_ledger.py   # forecasting: synthetic ledger generator
│   ├── synthetic_ledger.csv           # 4,085 rows, 18 merchants, ~265 days
│   ├── generate_synthetic_recon.py    # reconciliation: synthetic batch + ground-truth generator
│   ├── recon_ledger.csv               # 137 ledger transactions
│   ├── recon_bank_settlement.csv      # 113 bank/UTR settlement rows
│   └── recon_ground_truth.json        # planted answer key -- src/reconcile.py never reads this
├── src/
│   ├── features.py                    # lag + rolling features, build_features()
│   ├── forecast.py                    # QuantileForecaster, run_backtest()
│   ├── reconcile.py                   # staged reconciliation matching engine
│   ├── exceptions.py                  # custom error types (CashflowError + ReconciliationError families)
│   └── api/
│       ├── main.py                    # FastAPI app (forecasting + reconciliation routes)
│       └── db.py                      # SQLAlchemy, forecast_audit_log table
├── frontend/
│   └── src/
│       ├── App.jsx                    # layout, data fetching, cash/books view toggle
│       ├── api.js                     # fetch wrappers
│       └── components/
│           ├── MerchantSelector.jsx
│           ├── FanChart.jsx                        # recharts fan chart
│           ├── BacktestPanel.jsx                   # coverage + pinball table
│           ├── ExceptionsPanel.jsx                 # forecasting low-confidence flags
│           ├── ReconciliationMatchesPanel.jsx       # resolved ledger<->bank matches
│           ├── ReconciliationExceptionsPanel.jsx    # unresolved records + reason codes
│           └── ReconciliationSummaryStrip.jsx       # match rate / precision / recall chips
├── models/                            # saved LightGBM models (git-ignored)
├── reports/                           # backtest CSVs, exceptions JSON, reconciliation reports
├── tests/                             # pytest suite (reconciliation engine, generator, scorer)
├── train_and_backtest.py              # forecasting: offline training + backtest CLI
├── run_reconciliation.py              # reconciliation: offline matcher CLI
└── score_reconciliation.py            # reconciliation: honest scorer CLI (only file that reads ground truth)
```

The forecasting pipeline is intentionally flat. `features.py` builds the feature matrix; `forecast.py` trains models and runs the backtest; `api/main.py` loads the saved models at startup and serves them over HTTP. There is no streaming, no background task queue, no scheduled retraining — those would be the first things to add for a production version. The SQLite audit log (`cashflow.db`) captures every forecast request with merchant ID, timestamp, and the P10/P50/P90 values returned.

The fan chart in the dashboard renders 60 days of actual history followed by the 14-day forecast band. The backtest panel shows the honest per-category coverage numbers, color-coded: green above 75%, amber above 60%, red below 60%. The exceptions panel lists merchant-days where `(P90 - P10) / |P50| > 0.5`, which is a rough proxy for model uncertainty.

The reconciliation pipeline is equally flat and just as report-file-backed: `reconcile.py` is a pure function of two DataFrames in, a matches DataFrame + an exceptions DataFrame + a stats dict out — no database, no state. `run_reconciliation.py` is the CLI that loads the CSVs, calls it, and writes the reports the API and dashboard read directly. `score_reconciliation.py` is a separate, later step that never touches the matching logic — it only ever reads the matcher's already-written output plus the ground-truth file, which keeps the accuracy numbers honest by construction rather than by discipline. The dashboard's "books" view (toggle next to the title) mirrors the "cash" view's layout exactly: a scrollable list on the left, a summary strip + exception list stacked on the right.
