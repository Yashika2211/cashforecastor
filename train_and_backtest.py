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
    apply_cqr,
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
        print("Backtest summary (raw / global-CQR / per-category-CQR):")
        print("-" * 110)
        _print_summary(summary)
        print("-" * 110)
        print()

    # Per-fold per-category q_hats — stability check
    if not fold_results.empty and "q_hat_percat" in fold_results.columns:
        print("Per-fold per-category q_hat (watch for wild swings):")
        cats = sorted(fold_results["merchant_category"].unique())
        folds = sorted(fold_results["fold"].unique())
        # header
        hdr = f"  {'fold':>5}  {'cutoff':>12}" + "".join(f"  {c[:12]:>13}" for c in cats)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for fold_i in folds:
            fold_slice = fold_results[fold_results["fold"] == fold_i]
            cutoff_d = fold_slice["cutoff_date"].iloc[0]
            row_str = f"  {fold_i:>5}  {str(cutoff_d):>12}"
            for cat in cats:
                cat_rows = fold_slice[fold_slice["merchant_category"] == cat]
                if cat_rows.empty:
                    row_str += f"  {'—':>13}"
                else:
                    q = cat_rows["q_hat_percat"].iloc[0]
                    fb = cat_rows["percat_fallback"].iloc[0]
                    marker = "*" if fb else " "
                    row_str += f"  {q:>12,.0f}{marker}"
            print(row_str)
        print("  (* = fell back to global q_hat, category had < 10 calib scores in that fold)")
        print()

    # ------------------------------------------------------------------
    # Step 2: Final model training + per-category CQR calibration
    # ------------------------------------------------------------------
    # We train on the full ledger, then replay all 11 backtest calib windows
    # with the final model to collect CQR scores. Scores are grouped by
    # merchant_category so we can compute a separate q_hat per category.
    #
    # A global q_hat is also computed for comparison and used as a fallback
    # for any category not present in the calibration set.
    #
    # Minimum scores guard: if any category has < 50 calibration scores,
    # the per-category estimate is flagged as unreliable and the global
    # q_hat is used as fallback for that category.
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

    # Replay calib windows with the final model, collecting scores by category
    print("  Collecting per-category CQR scores across all backtest calib windows ...")
    min_train_days = args.min_train_days
    cutoff = pd.Timestamp(first_date) + pd.Timedelta(days=min_train_days)

    # pooled_scores_global: flat list for global q_hat
    # pooled_scores_by_cat: {category: list of score arrays}
    pooled_global: list[np.ndarray] = []
    pooled_by_cat: dict[str, list[np.ndarray]] = {}
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

        # Restrict history to what the model would have seen at this cutoff
        fc.daily_history = ledger_df[ledger_df["date"] < cutoff].copy()

        scores_by_cat = fc.collect_calibration_scores_by_category(calib_data)
        for cat, arr in scores_by_cat.items():
            if len(arr) > 0:
                pooled_by_cat.setdefault(cat, []).append(arr)
                pooled_global.append(arr)

        cutoff += pd.Timedelta(days=HORIZON)

    # Restore full history
    fc.daily_history = ledger_df.copy()

    if not pooled_global:
        print("  WARNING: no calibration scores collected — skipping CQR")
        fc.cqr_q_hat = 0.0
    else:
        all_scores_global = np.concatenate(pooled_global)
        all_scores_by_cat = {cat: np.concatenate(arrs) for cat, arrs in pooled_by_cat.items()}

        # Compute global q_hat first (used as fallback)
        global_q_hat = fc.calibrate_from_scores(all_scores_global, target_coverage=0.80)

        # Compute per-category q_hats (flags categories with < 50 scores)
        print()
        cat_q_hats = fc.calibrate_per_category(
            all_scores_by_cat,
            target_coverage=0.80,
            min_scores=50,
        )

        # Print comparison table
        print()
        print(f"  {'category':<22}  {'n_scores':>9}  {'global q̂':>10}  {'per-cat q̂':>11}  {'delta':>8}")
        print(f"  {'-'*70}")
        for cat in sorted(cat_q_hats.keys()):
            n = len(all_scores_by_cat.get(cat, []))
            pcat_q = cat_q_hats[cat]
            delta = pcat_q - global_q_hat
            flag = "  ← fallback" if n < 50 else ""
            print(f"  {cat:<22}  {n:>9,}  {global_q_hat:>10,.0f}  {pcat_q:>11,.0f}  {delta:>+8,.0f}{flag}")
        print(f"  {'global (all categories)':<22}  {len(all_scores_global):>9,}  {global_q_hat:>10,.0f}  {'—':>11}  {'':>8}")
        print()

        # Score distribution per category
        print("  Score distribution by category (₹):")
        print(f"  {'category':<22}  {'p10':>10}  {'p50':>10}  {'p80':>10}  {'p90':>10}  {'max':>10}")
        print(f"  {'-'*75}")
        for cat in sorted(all_scores_by_cat.keys()):
            s = all_scores_by_cat[cat]
            print(f"  {cat:<22}  "
                  f"{np.percentile(s,10):>10,.0f}  "
                  f"{np.percentile(s,50):>10,.0f}  "
                  f"{np.percentile(s,80):>10,.0f}  "
                  f"{np.percentile(s,90):>10,.0f}  "
                  f"{s.max():>10,.0f}")
        print()

        # Validate per-category calibration on a quick held-out check
        print("  Quick validation — per-category vs global CQR on last 14-day hold-out:")
        _validate_calibration(ledger_df, fc, cat_q_hats, global_q_hat)
        print()

    fc.save(MODELS_DIR)
    print(f"  Saved → {MODELS_DIR}")
    print(f"  cqr_q_hat (global fallback): {fc.cqr_q_hat:,.0f} ₹")
    print(f"  cqr_q_hat_by_category: { {k: round(v) for k, v in fc.cqr_q_hat_by_category.items()} }\n")

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


def _validate_calibration(
    ledger_df: pd.DataFrame,
    fc: QuantileForecaster,
    cat_q_hats: dict,
    global_q_hat: float,
) -> None:
    """
    Hold out the last 14 days per merchant, compare coverage under
    global vs per-category CQR. Prints a compact per-category table.
    """
    from src.forecast import _coverage, apply_cqr, HORIZON, MIN_HISTORY_DAYS
    from src.features import TARGET_COLUMN

    all_dates = sorted(ledger_df["date"].unique())
    test_start = pd.Timestamp(all_dates[-HORIZON])
    train_snap = ledger_df[ledger_df["date"] < test_start].copy()
    test_snap  = ledger_df[ledger_df["date"] >= test_start].copy()

    results: dict[str, dict] = {}
    for mid in sorted(ledger_df["merchant_id"].unique()):
        hist = train_snap[train_snap["merchant_id"] == mid]
        test = test_snap[test_snap["merchant_id"] == mid].sort_values("date")
        if len(hist) < MIN_HISTORY_DAYS or test.empty:
            continue
        cat = hist["merchant_category"].iloc[-1]
        try:
            preds = fc._recursive_predict(hist, mid, horizon=len(test))
        except Exception:
            continue

        y = test[TARGET_COLUMN].values
        raw_p10, raw_p90 = preds["p10"].values, preds["p90"].values

        p10_global, p90_global = apply_cqr(raw_p10, raw_p90, global_q_hat)
        pcat_q = cat_q_hats.get(cat, global_q_hat)
        p10_pcat, p90_pcat = apply_cqr(raw_p10, raw_p90, pcat_q)

        entry = results.setdefault(cat, {"raw": [], "global": [], "percat": [], "n": 0})
        entry["raw"].extend(((y >= raw_p10) & (y <= raw_p90)).tolist())
        entry["global"].extend(((y >= p10_global) & (y <= p90_global)).tolist())
        entry["percat"].extend(((y >= p10_pcat) & (y <= p90_pcat)).tolist())
        entry["n"] += len(y)

    print(f"  {'category':<22}  {'raw':>7}  {'global CQR':>10}  {'per-cat CQR':>12}  {'q̂ used':>10}  {'n_days':>7}")
    print(f"  {'-'*78}")
    for cat in sorted(results.keys()):
        r = results[cat]
        raw_cov    = np.mean(r["raw"])
        global_cov = np.mean(r["global"])
        pcat_cov   = np.mean(r["percat"])
        q_used     = cat_q_hats.get(cat, global_q_hat)
        print(
            f"  {cat:<22}  {raw_cov:>6.1%}  {global_cov:>9.1%}  {pcat_cov:>11.1%}  "
            f"{q_used:>10,.0f}  {r['n']:>7}"
        )


def _print_summary(summary: pd.DataFrame) -> None:
    has_percat = "percat_coverage" in summary.columns
    header = (
        f"{'Category':<22}  {'raw':>7}  {'global':>7}  "
        + (f"{'per-cat':>8}  " if has_percat else "")
        + f"{'pb P50':>10}  {'pb P10':>10}  {'pb P90':>10}  {'q_hat':>10}  {'N days':>7}"
    )
    print(header)
    for row in summary.itertuples(index=False):
        n_days = int(row.n_days_total) if not (
            isinstance(row.n_days_total, float) and np.isnan(row.n_days_total)
        ) else "-"
        percat_str = ""
        if has_percat:
            pc = getattr(row, "percat_coverage", float("nan"))
            percat_str = f"{pc:>7.1%}  " if not np.isnan(pc) else f"{'—':>7}  "
        print(
            f"{row.merchant_category:<22}  "
            f"{row.raw_coverage:>6.1%}  "
            f"{row.cal_coverage:>6.1%}  "
            + percat_str +
            f"{row.cal_pinball_p50:>10,.0f}  "
            f"{row.cal_pinball_p10:>10,.0f}  "
            f"{row.cal_pinball_p90:>10,.0f}  "
            f"{row.mean_q_hat:>10,.0f}  "
            f"{str(n_days):>7}"
        )


if __name__ == "__main__":
    main()
