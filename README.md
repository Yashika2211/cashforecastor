# cashflow forecaster

**Live demo:** https://cashforecastor.vercel.app
**API:** https://cashforecastor.onrender.com/docs

> Note: the backend runs on Render's free tier and spins down after 15 minutes of inactivity. The first request after a period of no traffic takes 30–50 seconds to respond — this is expected, not a bug.

A daily net-settled-amount forecasting pipeline for Razorpay-style merchant ledgers. Three LightGBM regressors with `objective='quantile'` at alpha=0.1, 0.5, and 0.9 produce a calibrated P10/P50/P90 band for each of the next 14 days. Two things make this different from a standard point-forecast: the quantile trajectories are independent (each feeds back its own prior predictions into the lag features, not the median), and the bands are conformally calibrated against pooled residuals from all 11 backtest calibration windows. The honest part: coverage and pinball loss are reported per merchant category and the numbers that are bad (marketplace, with a hard regime shift) are reported as-is rather than averaged away.

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

---

## architecture

```
razorpay final/
├── data/
│   ├── generate_synthetic_ledger.py   # synthetic data generator
│   └── synthetic_ledger.csv           # 4,085 rows, 18 merchants, ~265 days
├── src/
│   ├── features.py                    # lag + rolling features, build_features()
│   ├── forecast.py                    # QuantileForecaster, run_backtest()
│   ├── exceptions.py                  # custom error types
│   └── api/
│       ├── main.py                    # FastAPI app
│       └── db.py                      # SQLAlchemy, forecast_audit_log table
├── frontend/
│   └── src/
│       ├── App.jsx                    # layout, data fetching
│       ├── api.js                     # fetch wrappers
│       └── components/
│           ├── MerchantSelector.jsx
│           ├── FanChart.jsx           # recharts fan chart
│           ├── BacktestPanel.jsx      # coverage + pinball table
│           └── ExceptionsPanel.jsx    # low-confidence flags
├── models/                            # saved LightGBM models (git-ignored)
├── reports/                           # backtest CSVs + exceptions JSON
└── train_and_backtest.py              # offline training + backtest CLI
```

The pipeline is intentionally flat. `features.py` builds the feature matrix; `forecast.py` trains models and runs the backtest; `api/main.py` loads the saved models at startup and serves them over HTTP. There is no streaming, no background task queue, no scheduled retraining — those would be the first things to add for a production version. The SQLite audit log (`cashflow.db`) captures every forecast request with merchant ID, timestamp, and the P10/P50/P90 values returned.

The fan chart in the dashboard renders 60 days of actual history followed by the 14-day forecast band. The backtest panel shows the honest per-category coverage numbers, color-coded: green above 75%, amber above 60%, red below 60%. The exceptions panel lists merchant-days where `(P90 - P10) / |P50| > 0.5`, which is a rough proxy for model uncertainty.
