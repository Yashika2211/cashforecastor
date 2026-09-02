"""
Offline reconciliation runner.

Usage
-----
    python run_reconciliation.py [--ledger PATH] [--bank PATH] [--out-dir DIR] [--annotate]

What it does
------------
1. Loads the internal ledger CSV and the bank/processor settlement CSV.
2. Runs the staged matching engine (src/reconcile.run_reconciliation).
3. Optionally annotates exceptions with a human-readable note (templated
   offline; only calls out to an LLM if ANTHROPIC_API_KEY is set).
4. Writes:
     reports/reconciliation_matches.csv
     reports/reconciliation_exceptions.json
     reports/reconciliation_summary.json
5. Prints a stdout summary table (rule counts, reason-code counts, match
   rate, throughput) in train_and_backtest.py's tone.

This script never opens the planted ground-truth answer-key file -- only
score_reconciliation.py does. Real precision/recall is computed there.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.exceptions import ReconciliationError
from src.reconcile import annotate_exceptions, run_reconciliation

DEFAULT_LEDGER = ROOT / "data" / "recon_ledger.csv"
DEFAULT_BANK = ROOT / "data" / "recon_bank_settlement.csv"
DEFAULT_OUT_DIR = ROOT / "reports"

RULE_ORDER = [
    "exact_1to1", "fee_adjusted_1to1", "timing_near_miss_exact",
    "timing_near_miss_fee_adjusted", "batch_subset_sum_hinted", "batch_subset_sum",
]
REASON_ORDER = [
    "AMOUNT_MISMATCH", "ORPHAN_BANK_CREDIT", "ORPHAN_LEDGER_TXN",
    "DUPLICATE_SUSPECTED", "TIMING_OUT_OF_WINDOW", "UNRESOLVED_SUBSET_SUM",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reconciliation matcher end to end.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--annotate", action="store_true", help="Attach human-readable notes to exceptions.")
    args = parser.parse_args()

    if not args.ledger.exists():
        print(f"ERROR: ledger not found at {args.ledger}", file=sys.stderr)
        sys.exit(1)
    if not args.bank.exists():
        print(f"ERROR: bank settlement file not found at {args.bank}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ledger: {args.ledger}")
    ledger = pd.read_csv(args.ledger)
    print(f"  {len(ledger):,} rows")
    print(f"Loading bank settlement feed: {args.bank}")
    bank = pd.read_csv(args.bank)
    print(f"  {len(bank):,} rows\n")

    print("Running reconciliation engine ...")
    try:
        result = run_reconciliation(ledger, bank)
    except ReconciliationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.annotate:
        print("Annotating exceptions ...")
        result.exceptions = annotate_exceptions(result.exceptions)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    matches_out = result.matches.copy()
    if not matches_out.empty:
        matches_out["ledger_txn_ids"] = matches_out["ledger_txn_ids"].apply(lambda ids: "|".join(ids))
    matches_out.to_csv(args.out_dir / "reconciliation_matches.csv", index=False)

    exceptions_records = result.exceptions.to_dict(orient="records")
    with open(args.out_dir / "reconciliation_exceptions.json", "w") as f:
        json.dump(exceptions_records, f, indent=2, default=lambda x: None if pd.isna(x) else x)

    n_ledger = result.stats["n_ledger_rows"]
    n_bank = result.stats["n_bank_rows"]
    n_ledger_matched = sum(result.matches["member_count"]) if not result.matches.empty else 0
    n_ledger_exception = int((result.exceptions["record_type"] == "ledger").sum()) if not result.exceptions.empty else 0
    match_rate = n_ledger_matched / n_ledger if n_ledger else 0.0

    rule_counts = {r: result.stats["rule_counts"].get(r, 0) for r in RULE_ORDER}
    reason_code_counts = {r: result.stats["reason_code_counts"].get(r, 0) for r in REASON_ORDER}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger_path": str(args.ledger),
        "bank_path": str(args.bank),
        "n_ledger_rows": n_ledger,
        "n_bank_rows": n_bank,
        "n_ledger_matched": int(n_ledger_matched),
        "n_ledger_exception": n_ledger_exception,
        "match_rate": round(match_rate, 4),
        "rule_counts": rule_counts,
        "reason_code_counts": reason_code_counts,
        "elapsed_seconds": result.stats["elapsed_seconds"],
        "records_per_second": result.stats["records_per_second"],
    }
    with open(args.out_dir / "reconciliation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {len(matches_out)} match rows -> {args.out_dir / 'reconciliation_matches.csv'}")
    print(f"Wrote {len(exceptions_records)} exception rows -> {args.out_dir / 'reconciliation_exceptions.json'}")
    print(f"Wrote summary -> {args.out_dir / 'reconciliation_summary.json'}\n")

    print("Rule counts:")
    for rule, n in rule_counts.items():
        print(f"  {rule:<32} {n:>4}")
    print("\nReason-code counts:")
    for code, n in reason_code_counts.items():
        print(f"  {code:<24} {n:>4}")
    print(f"\nledger: {n_ledger} rows | bank: {n_bank} rows")
    print(f"ledger matched: {n_ledger_matched} ({match_rate:.1%}) | ledger exceptions: {n_ledger_exception}")
    print(f"elapsed: {result.stats['elapsed_seconds']:.4f}s | throughput: {result.stats['records_per_second']:,.0f} records/sec")


if __name__ == "__main__":
    main()
