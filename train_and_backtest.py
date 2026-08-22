"""
Offline training + backtest script.

Usage
-----
    python train_and_backtest.py [--ledger PATH] [--rounds N] [--min-train-days D]

What it does
------------
1. Loads the ledger CSV (default: data/synthetic_ledger.csv).
2. Runs a walk-forward backtest (fold size = 14 days) and saves:
     reports/backtest_results.csv  – per-fold, per-merchant rows
     reports/backtest_summary.csv  – aggregate + per-category metrics
3. Trains a final set of models on the FULL ledger and saves:
     models/lgb_q10.txt, lgb_q50.txt, lgb_q90.txt
     models/forecaster.pkl
4. Generates low-confidence exception flags and saves:
     reports/exceptions.json
5. Prints the backtest summary to stdout so you can check the numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.forecast import (
    MODELS_DIR,
    REPORTS_DIR,
    QuantileForecaster,
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
    relative to P50 — a proxy for low model confidence.

    Threshold: if (P90 - P10) / |P50| > wide_band_threshold, flag it.
    Falls back to an absolute threshold (100_000) when P50 is near zero.
    """
    flags = []
    merchants = forecaster.daily_history["merchant_id"].unique()
    for mid in merchants:
        try:
            preds = forecaster.predict(mid)
        except Exception:
            continue
        for row in preds.itertuples(index=False):
            band = row.p90 - row.p10
            denom = abs(row.p50) if abs(row.p50) > 1_000 else 1_000
            rel_width = band / denom
            if rel_width > wide_band_threshold:
                flags.append(
                    {
                        "merchant_id": mid,
                        "flag_date": str(row.forecast_date),
                        "reason": (
                            f"Wide prediction band: P90-P10 = {band:,.0f} "
                            f"({rel_width:.0%} of |P50|). "
                            "Model uncertainty is elevated for this day."
                        ),
                        "confidence_score": round(float(1.0 / (1.0 + rel_width)), 3),
                    }
                )
    return flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train models and run walk-forward backtest.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="Path to ledger CSV")
    parser.add_argument("--rounds", type=int, default=300, help="LightGBM num_boost_round")
    parser.add_argument("--min-train-days", type=int, default=90, help="Min training days before first fold")
    args = parser.parse_args()

    if not args.ledger.exists():
        print(f"ERROR: Ledger not found at {args.ledger}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ledger: {args.ledger}")
    ledger = pd.read_csv(args.ledger)
    print(f"  {len(ledger):,} rows, {ledger['merchant_id'].nunique()} merchants, "
          f"categories: {sorted(ledger['merchant_category'].unique())}")

    # ------------------------------------------------------------------
    # Step 1: Walk-forward backtest
    # ------------------------------------------------------------------
    print(f"\nRunning walk-forward backtest (min_train_days={args.min_train_days}, "
          f"fold_size=14, rounds={args.rounds}) ...")
    fold_results, summary = run_backtest(
        ledger,
        num_boost_round=args.rounds,
        min_train_days=args.min_train_days,
    )

    if fold_results.empty:
        print("WARNING: No backtest folds completed. Check ledger date range.", file=sys.stderr)
    else:
        n_folds = fold_results["fold"].nunique()
        print(f"  Completed {n_folds} folds across "
              f"{fold_results['merchant_id'].nunique()} merchants.")

    save_backtest_results(fold_results, summary)
    print(f"  Saved → {REPORTS_DIR / 'backtest_results.csv'}")
    print(f"  Saved → {REPORTS_DIR / 'backtest_summary.csv'}")

    # Print summary table
    if not summary.empty:
        print("\nBacktest summary:")
        print("-" * 80)
        _print_summary(summary)
        print("-" * 80)

    # ------------------------------------------------------------------
    # Step 2: Final model training on full ledger
    # ------------------------------------------------------------------
    print(f"\nTraining final models on full ledger (rounds={args.rounds}) ...")
    fc = QuantileForecaster()
    fc.fit(ledger, num_boost_round=args.rounds)
    fc.save(MODELS_DIR)
    print(f"  Saved → {MODELS_DIR}")

    # ------------------------------------------------------------------
    # Step 3: Generate exception flags
    # ------------------------------------------------------------------
    print("\nGenerating exception flags ...")
    flags = generate_exception_flags(fc)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "exceptions.json", "w") as f:
        json.dump(flags, f, indent=2)
    print(f"  {len(flags)} flag(s) → {REPORTS_DIR / 'exceptions.json'}")

    print("\nDone.")


def _print_summary(summary: pd.DataFrame) -> None:
    header = f"{'Category':<22}  {'P10 pinball':>12}  {'P50 pinball':>12}  {'P90 pinball':>12}  {'Coverage':>10}  {'N days':>8}"
    print(header)
    for row in summary.itertuples(index=False):
        n_days = int(row.n_days_total) if hasattr(row, "n_days_total") and not (
            isinstance(row.n_days_total, float) and np.isnan(row.n_days_total)
        ) else "-"
        print(
            f"{row.merchant_category:<22}  "
            f"{row.pinball_p10:>12.2f}  "
            f"{row.pinball_p50:>12.2f}  "
            f"{row.pinball_p90:>12.2f}  "
            f"{row.coverage_p10_p90:>9.1%}  "
            f"{str(n_days):>8}"
        )


if __name__ == "__main__":
    main()
