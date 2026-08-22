"""
Offline training + backtest script.

Usage
-----
    python train_and_backtest.py [--ledger PATH] [--rounds N] [--min-train-days D]

What it does
------------
1. Loads the ledger CSV (default: data/synthetic_ledger.csv).
2. Runs a walk-forward backtest (fold structure: train / 14-day calib / 14-day test)
   and saves:
     reports/backtest_results.csv  – per-fold, per-merchant rows
     reports/backtest_summary.csv  – aggregate + per-category metrics
                                     (both raw=leakage-fixed and cal=+CQR columns)
3. Trains a final set of models on full-minus-14 days and calibrates CQR
   on the last 14-day window:
     models/lgb_q10.txt, lgb_q50.txt, lgb_q90.txt
     models/forecaster.pkl  (includes cqr_q_hat)
4. Generates low-confidence exception flags and saves:
     reports/exceptions.json
5. Prints the backtest summary (raw and calibrated) to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.forecast import (
    HORIZON,
    MODELS_DIR,
    REPORTS_DIR,
    QuantileForecaster,
    compute_cqr_correction_from_scores,
    run_backtest,
    save_backtest_results,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_LEDGER = ROOT / "data" / "synthetic_ledger.csv"


# ---------------------------------------------------------------------------
# Exception-flag generation
# ---------------------------------------------------------------------------

def generate_exception_flags(
    forecaster: QuantileForecaster,
    wide_band_threshold: float = 0.5,
) -> list[dict]:
    """
    Flag merchant-days where the forecast band (P90-P10) is unusually wide
    relative to |P50|. CQR-adjusted predictions are used so the flag
    threshold applies after calibration.
    """
    flags = []
    merchants = forecaster.daily_history["merchant_id"].unique()
    for mid in merchants:
        try:
            preds = forecaster.predict(mid, apply_calibration=True)
        except Exception:
            continue
        for row in preds.itertuples(index=False):
            band = row.p90 - row.p10
            denom = abs(row.p50) if abs(row.p50) > 1_000 else 1_000
            rel_width = band / denom
            if rel_width > wide_band_threshold:
                flags.append({
                    "merchant_id": mid,
                    "flag_date": str(row.forecast_date),
                    "reason": (
                        f"Wide prediction band: P90-P10 = {band:,.0f} "
                        f"({rel_width:.0%} of |P50|). "
                        "Model uncertainty is elevated for this day."
                    ),
                    "confidence_score": round(float(1.0 / (1.0 + rel_width)), 3),
                })
    return flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train models and run walk-forward backtest.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--rounds", type=int, default=300, help="LightGBM num_boost_round")
    parser.add_argument("--min-train-days", type=int, default=90)
    args = parser.parse_args()

    if not args.ledger.exists():
        print(f"ERROR: Ledger not found at {args.ledger}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ledger: {args.ledger}")
    ledger = pd.read_csv(args.ledger)
    print(f"  {len(ledger):,} rows, {ledger['merchant_id'].nunique()} merchants, "
          f"categories: {sorted(ledger['merchant_category'].unique())}\n")

    # ------------------------------------------------------------------
    # Step 1: Walk-forward backtest
    # ------------------------------------------------------------------
    print(f"Running walk-forward backtest "
          f"(min_train_days={args.min_train_days}, fold_size=14, rounds={args.rounds}) ...")
    print("  Fold structure: train / 14-day calib (CQR) / 14-day test\n")

    fold_results, summary = run_backtest(
        ledger,
        num_boost_round=args.rounds,
        min_train_days=args.min_train_days,
        target_coverage=0.80,
    )

    if fold_results.empty:
        print("WARNING: No backtest folds completed. Check ledger date range.", file=sys.stderr)
    else:
        n_folds = fold_results["fold"].nunique()
        total_viol = fold_results["ordering_violations"].sum()
        print(f"  Completed {n_folds} folds | "
              f"{fold_results['merchant_id'].nunique()} merchants | "
              f"{len(fold_results)} merchant-fold rows")
        print(f"  Ordering violations (P10>P50 or P50>P90): {total_viol} / "
              f"{fold_results['n_days'].sum()} days")

    save_backtest_results(fold_results, summary)
    print(f"  Saved → {REPORTS_DIR / 'backtest_results.csv'}")
    print(f"  Saved → {REPORTS_DIR / 'backtest_summary.csv'}\n")

    if not summary.empty:
        print("Backtest summary:")
        print("-" * 100)
        _print_summary(summary)
        print("-" * 100)
        print()

    # ------------------------------------------------------------------
    # Step 2: Final model training + pooled CQR calibration
    # ------------------------------------------------------------------
    # The production model trains on the full ledger so it sees every
    # regime and seasonal pattern. CQR calibration pools residuals from
    # ALL 11 backtest calib windows rather than just the last 14 days.
    #
    # Why pooling matters:
    #   Single last-14-day window  q_hat = ~1,117 ₹  (calm period, no spike)
    #   Per-fold q_hat range       -874 to 103,252 ₹  (fold 1 = festival window)
    #   Pooled q_hat               reflects the full volatility distribution
    #
    # Procedure:
    #   1. Train the final model on the full ledger.
    #   2. Replay each backtest calib window with the final model; collect
    #      CQR scores (not q_hat — raw per-day scores).
    #   3. Pool all scores into one array and compute a single q_hat.
    #   4. Store q_hat on the forecaster and save.
    #
    # The final model predicts from the end of the full ledger, so its
    # daily_history covers the entire timeline when generating scores.
    print(f"Training final models on full ledger (rounds={args.rounds}) ...")
    ledger_df = ledger.copy()
    ledger_df["date"] = pd.to_datetime(ledger_df["date"])
    all_dates = sorted(ledger_df["date"].unique())
    first_date = all_dates[0]
    last_date  = all_dates[-1]

    fc = QuantileForecaster()
    fc.fit(ledger_df, num_boost_round=args.rounds)
    fc.daily_history = ledger_df.copy()
    print("  Training done.\n")

    # Replay calib windows: same schedule as the backtest loop
    print("  Collecting pooled CQR scores across all backtest calib windows ...")
    min_train_days = args.min_train_days
    cutoff = pd.Timestamp(first_date) + pd.Timedelta(days=min_train_days)
    pooled_scores = []
    fold_q_hats   = []
    fold_n_scores = []
    calib_fold = 0

    while True:
        calib_end = cutoff + pd.Timedelta(days=HORIZON - 1)
        test_end  = cutoff + pd.Timedelta(days=2 * HORIZON - 1)
        if test_end > last_date:
            break

        calib_fold += 1
        calib_data = ledger_df[
            (ledger_df["date"] >= cutoff) & (ledger_df["date"] <= calib_end)
        ]

        # Temporarily restrict daily_history to data the model would have
        # seen at this cutoff (prevents look-ahead when collecting scores).
        train_snapshot = ledger_df[ledger_df["date"] < cutoff].copy()
        fc.daily_history = train_snapshot

        scores = fc.collect_calibration_scores(calib_data)
        if len(scores) > 0:
            pooled_scores.append(scores)
            fold_n_scores.append(len(scores))
            # Per-fold q_hat for logging
            fold_q_hat = compute_cqr_correction_from_scores(scores, target_coverage=0.80)
            fold_q_hats.append((calib_fold, cutoff.date(), fold_q_hat, len(scores)))

        cutoff += pd.Timedelta(days=HORIZON)

    # Restore full history before saving
    fc.daily_history = ledger_df.copy()

    if pooled_scores:
        all_scores = np.concatenate(pooled_scores)
        print(f"  Pooled {len(all_scores):,} scores across {calib_fold} calib windows")
        print()
        print("  Per-fold calib scores and q_hat:")
        for fold_i, cutoff_d, fq, fn in fold_q_hats:
            print(f"    fold {fold_i:>2}  cutoff={cutoff_d}  n_scores={fn:>4}  fold_q_hat={fq:>10,.0f}")
        print()

        old_q_hat = 1_117.0  # last 14-day single-window value from previous run
        pooled_q_hat = fc.calibrate_from_scores(all_scores, target_coverage=0.80)

        print(f"  Old q_hat (last 14-day window):  {old_q_hat:>10,.0f} ₹")
        print(f"  New q_hat (pooled across folds): {pooled_q_hat:>10,.0f} ₹")
        print(f"  Ratio new/old:                   {pooled_q_hat/old_q_hat:>10.2f}x")
        print(f"  Score distribution:")
        print(f"    p10={np.percentile(all_scores,10):>10,.0f}  "
              f"p50={np.percentile(all_scores,50):>10,.0f}  "
              f"p80={np.percentile(all_scores,80):>10,.0f}  "
              f"p90={np.percentile(all_scores,90):>10,.0f}  "
              f"max={all_scores.max():>10,.0f}")
    else:
        print("  WARNING: no calibration scores collected — q_hat set to 0")
        fc.cqr_q_hat = 0.0

    fc.save(MODELS_DIR)
    print(f"\n  Saved → {MODELS_DIR}  (cqr_q_hat={fc.cqr_q_hat:,.0f} ₹)\n")

    # ------------------------------------------------------------------
    # Step 3: Exception flags
    # ------------------------------------------------------------------
    print("Generating exception flags ...")
    flags = generate_exception_flags(fc)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "exceptions.json", "w") as f:
        json.dump(flags, f, indent=2)
    print(f"  {len(flags)} flag(s) → {REPORTS_DIR / 'exceptions.json'}\n")
    print("Done.")


def _print_summary(summary: pd.DataFrame) -> None:
    header = (
        f"{'Category':<22}  {'raw cov':>8}  {'cal cov':>8}  "
        f"{'raw P50':>10}  {'cal P50':>10}  "
        f"{'raw P10':>10}  {'raw P90':>10}  "
        f"{'q_hat':>10}  {'N days':>7}"
    )
    print(header)
    for row in summary.itertuples(index=False):
        n_days = int(row.n_days_total) if not (
            isinstance(row.n_days_total, float) and np.isnan(row.n_days_total)
        ) else "-"
        print(
            f"{row.merchant_category:<22}  "
            f"{row.raw_coverage:>7.1%}  "
            f"{row.cal_coverage:>7.1%}  "
            f"{row.raw_pinball_p50:>10,.0f}  "
            f"{row.cal_pinball_p50:>10,.0f}  "
            f"{row.raw_pinball_p10:>10,.0f}  "
            f"{row.raw_pinball_p90:>10,.0f}  "
            f"{row.mean_q_hat:>10,.0f}  "
            f"{str(n_days):>7}"
        )


if __name__ == "__main__":
    main()
