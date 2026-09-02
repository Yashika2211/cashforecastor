"""
Honest scorer: compares the engine's output against the planted ground-truth
answer key to compute real precision/recall/false-match-rate.

This is the ONLY script that ever opens data/recon_ground_truth.json.
src/reconcile.py and run_reconciliation.py never see it.

Usage
-----
    python score_reconciliation.py [--matches PATH] [--exceptions PATH]
                                    [--ground-truth PATH] [--out PATH]

Writes reports/reconciliation_accuracy.json and prints a table with the raw
numerator/denominator behind every percentage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_MATCHES = ROOT / "reports" / "reconciliation_matches.csv"
DEFAULT_EXCEPTIONS = ROOT / "reports" / "reconciliation_exceptions.json"
DEFAULT_GROUND_TRUTH = ROOT / "data" / "recon_ground_truth.json"
DEFAULT_OUT = ROOT / "reports" / "reconciliation_accuracy.json"
DEFAULT_SUMMARY = ROOT / "reports" / "reconciliation_summary.json"

EXPECTED_REASON_CODE = {
    "test_mode_capture": "ORPHAN_LEDGER_TXN",
    "pending_payout_censored": "ORPHAN_LEDGER_TXN",
    "refund_reversed_internally": "ORPHAN_LEDGER_TXN",
    "prior_period_adjustment": "ORPHAN_BANK_CREDIT",
    "unrelated_interest_credit": "ORPHAN_BANK_CREDIT",
}


def _load_matches(path: Path) -> dict:
    """Returns {record_id: claimed_group_set} for ledger and bank sides."""
    import csv
    ledger_claimed_group: dict[str, frozenset] = {}
    bank_claimed_group: dict[str, frozenset] = {}
    ledger_engine_matched: dict[str, str] = {}  # ledger_txn_id -> bank_row_id
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids = frozenset(row["ledger_txn_ids"].split("|")) if row["ledger_txn_ids"] else frozenset()
            for lid in ids:
                ledger_claimed_group[lid] = ids
                ledger_engine_matched[lid] = row["bank_row_id"]
            bank_claimed_group[row["bank_row_id"]] = ids
    return {
        "ledger_claimed_group": ledger_claimed_group,
        "bank_claimed_group": bank_claimed_group,
        "ledger_engine_matched": ledger_engine_matched,
    }


def _load_exceptions(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _score_side(
    truth_map: dict, claimed_group_map: dict, exceptions: list[dict], record_type: str,
    group_lookup_key: str,
) -> dict:
    """Compute TP/WA/FP_raw/FN_raw/TN for one side (ledger or bank)."""
    exc_by_id = {e["record_id"]: e for e in exceptions if e["record_type"] == record_type}

    TP = WA = FP_raw = FN_raw = TN = 0
    tn_confusion: dict[str, dict[str, int]] = {}
    tn_expected_denominator = 0

    for rec_id, truth in truth_map.items():
        true_label = truth["true_label"]
        engine_matched = rec_id in claimed_group_map
        if true_label == "MATCHED":
            true_group = set(truth[group_lookup_key])
            if engine_matched:
                claimed = set(claimed_group_map[rec_id])
                if claimed == true_group:
                    TP += 1
                else:
                    WA += 1
            else:
                FN_raw += 1
        else:  # ORPHAN or DUPLICATE
            if engine_matched:
                FP_raw += 1
            else:
                TN += 1
                exc = exc_by_id.get(rec_id)
                engine_code = exc["reason_code"] if exc else "MISSING_EXCEPTION_RECORD"
                if true_label == "DUPLICATE":
                    expected = "DUPLICATE_SUSPECTED"
                else:
                    expected = EXPECTED_REASON_CODE.get(truth.get("orphan_reason"), "UNKNOWN")
                tn_confusion.setdefault(expected, {}).setdefault(engine_code, 0)
                tn_confusion[expected][engine_code] += 1
                tn_expected_denominator += 1

    return {
        "TP": TP, "WA": WA, "FP_raw": FP_raw, "FN_raw": FN_raw, "TN": TN,
        "tn_confusion": tn_confusion, "tn_total": tn_expected_denominator,
        "n_orphan_or_dup": FP_raw + TN,
        "n_matched_true": TP + WA + FN_raw,
    }


def score(matches_path: Path, exceptions_path: Path, ground_truth_path: Path) -> dict:
    gt = json.load(open(ground_truth_path))
    loaded = _load_matches(matches_path)
    exceptions = _load_exceptions(exceptions_path)

    ledger_score = _score_side(
        gt["ledger_truth"], loaded["ledger_claimed_group"], exceptions, "ledger", "true_group_id",
    )
    # ledger true_group_id needs resolving through match_groups to a ledger_txn_id set
    ledger_truth_expanded = {}
    for lid, t in gt["ledger_truth"].items():
        t2 = dict(t)
        if t["true_label"] == "MATCHED":
            t2["true_group_id"] = gt["match_groups"][t["true_group_id"]]["ledger_txn_ids"]
        ledger_score_key = "true_group_id"
        ledger_truth_expanded[lid] = t2
    ledger_score = _score_side(ledger_truth_expanded, loaded["ledger_claimed_group"], exceptions, "ledger", "true_group_id")

    bank_score = _score_side(
        gt["bank_truth"], loaded["bank_claimed_group"], exceptions, "bank", "true_ledger_txn_ids",
    )

    def precision_recall(s):
        denom_p = s["TP"] + s["FP_raw"] + s["WA"]
        denom_r = s["TP"] + s["FN_raw"] + s["WA"]
        precision = s["TP"] / denom_p if denom_p else 0.0
        recall = s["TP"] / denom_r if denom_r else 0.0
        return precision, recall

    ledger_precision, ledger_recall = precision_recall(ledger_score)

    n_ledger = len(gt["ledger_truth"])
    n_ledger_matched_by_engine = len(loaded["ledger_claimed_group"])
    match_rate = n_ledger_matched_by_engine / n_ledger if n_ledger else 0.0

    false_match_rate_ledger = ledger_score["FP_raw"] / ledger_score["n_orphan_or_dup"] if ledger_score["n_orphan_or_dup"] else 0.0
    false_match_rate_bank = bank_score["FP_raw"] / bank_score["n_orphan_or_dup"] if bank_score["n_orphan_or_dup"] else 0.0
    pooled_num = ledger_score["FP_raw"] + bank_score["FP_raw"]
    pooled_den = ledger_score["n_orphan_or_dup"] + bank_score["n_orphan_or_dup"]
    false_match_rate_pooled = pooled_num / pooled_den if pooled_den else 0.0

    wrong_attribution_rate = ledger_score["WA"] / ledger_score["n_matched_true"] if ledger_score["n_matched_true"] else 0.0

    # Duplicate precision/recall (ledger-side reason codes)
    dup_flagged = [e for e in exceptions if e["reason_code"] == "DUPLICATE_SUSPECTED"]
    dup_flagged_correct = sum(
        1 for e in dup_flagged
        if (gt["ledger_truth"].get(e["record_id"], {}).get("true_label") == "DUPLICATE"
            if e["record_type"] == "ledger"
            else gt["bank_truth"].get(e["record_id"], {}).get("true_label") == "DUPLICATE")
    )
    duplicate_precision = dup_flagged_correct / len(dup_flagged) if dup_flagged else 0.0
    n_true_duplicates = (
        sum(1 for v in gt["ledger_truth"].values() if v["true_label"] == "DUPLICATE")
        + sum(1 for v in gt["bank_truth"].values() if v["true_label"] == "DUPLICATE")
    )
    duplicate_recall = dup_flagged_correct / n_true_duplicates if n_true_duplicates else 0.0

    # Trap recall
    exc_ids_ledger = {e["record_id"] for e in exceptions if e["record_type"] == "ledger"}
    exc_ids_bank = {e["record_id"] for e in exceptions if e["record_type"] == "bank"}
    trap_hits = 0
    for trap in gt["planted_traps"]:
        all_exception = (
            all(lid in exc_ids_ledger for lid in trap["ledger_txn_ids"])
            and all(bid in exc_ids_bank for bid in trap["bank_row_ids"])
        )
        if all_exception:
            trap_hits += 1
    trap_recall = trap_hits / len(gt["planted_traps"]) if gt["planted_traps"] else 0.0

    # Reason-code accuracy + confusion (pooled ledger + bank TN sets)
    pooled_confusion: dict[str, dict[str, int]] = {}
    for side_score in (ledger_score, bank_score):
        for expected, engine_map in side_score["tn_confusion"].items():
            for engine_code, n in engine_map.items():
                pooled_confusion.setdefault(expected, {}).setdefault(engine_code, 0)
                pooled_confusion[expected][engine_code] += n
    tn_total = ledger_score["tn_total"] + bank_score["tn_total"]
    tn_correct = sum(
        engine_map.get(expected, 0) for expected, engine_map in pooled_confusion.items()
    )
    reason_code_accuracy = tn_correct / tn_total if tn_total else 0.0

    summary_rps = None
    if DEFAULT_SUMMARY.exists():
        summary_rps = json.load(open(DEFAULT_SUMMARY))["records_per_second"]

    warning = None
    if match_rate >= 0.98 or ledger_recall > 0.97:
        warning = ("match_rate/recall suspiciously high -- verify the exception-injection "
                   "counts in recon_ground_truth.json's summary_counts weren't accidentally zeroed")

    result = {
        "precision": round(ledger_precision, 4),
        "precision_counts": {"TP": ledger_score["TP"], "FP_raw": ledger_score["FP_raw"], "WA": ledger_score["WA"]},
        "recall": round(ledger_recall, 4),
        "recall_counts": {"TP": ledger_score["TP"], "FN_raw": ledger_score["FN_raw"], "WA": ledger_score["WA"]},
        "match_rate": round(match_rate, 4),
        "match_rate_n": [n_ledger_matched_by_engine, n_ledger],
        "false_match_rate_ledger": round(false_match_rate_ledger, 4),
        "false_match_rate_ledger_counts": [ledger_score["FP_raw"], ledger_score["n_orphan_or_dup"]],
        "false_match_rate_bank": round(false_match_rate_bank, 4),
        "false_match_rate_bank_counts": [bank_score["FP_raw"], bank_score["n_orphan_or_dup"]],
        "false_match_rate_pooled": round(false_match_rate_pooled, 4),
        "false_match_rate_pooled_counts": [pooled_num, pooled_den],
        "wrong_attribution_rate": round(wrong_attribution_rate, 4),
        "wrong_attribution_counts": [ledger_score["WA"], ledger_score["n_matched_true"]],
        "duplicate_precision": round(duplicate_precision, 4),
        "duplicate_precision_n": [dup_flagged_correct, len(dup_flagged)],
        "duplicate_recall": round(duplicate_recall, 4),
        "duplicate_recall_n": [dup_flagged_correct, n_true_duplicates],
        "trap_recall": round(trap_recall, 4),
        "trap_counts": [trap_hits, len(gt["planted_traps"])],
        "reason_code_accuracy": round(reason_code_accuracy, 4),
        "reason_code_accuracy_n": [tn_correct, tn_total],
        "confusion": pooled_confusion,
        "records_per_second": summary_rps,
        "warning": warning,
        "bank_precision_recall": {
            "precision": round(precision_recall(bank_score)[0], 4),
            "recall": round(precision_recall(bank_score)[1], 4),
            "TP": bank_score["TP"], "FP_raw": bank_score["FP_raw"],
            "FN_raw": bank_score["FN_raw"], "WA": bank_score["WA"],
        },
    }
    return result


def _print_table(result: dict) -> None:
    def pct(key, n_key=None):
        val = result[key]
        n = result.get(n_key) if n_key else None
        n_str = f" ({n[0]}/{n[1]})" if n else ""
        return f"{val:.1%}{n_str}"

    print(f"{'metric':<32} {'value':>10}   n")
    print("-" * 70)
    print(f"{'match rate':<32} {pct('match_rate', 'match_rate_n')}")
    print(f"{'precision (ledger)':<32} {result['precision']:.1%}   TP={result['precision_counts']['TP']} FP_raw={result['precision_counts']['FP_raw']} WA={result['precision_counts']['WA']}")
    print(f"{'recall (ledger)':<32} {result['recall']:.1%}   TP={result['recall_counts']['TP']} FN_raw={result['recall_counts']['FN_raw']} WA={result['recall_counts']['WA']}")
    print(f"{'false-match rate (ledger)':<32} {pct('false_match_rate_ledger', 'false_match_rate_ledger_counts')}")
    print(f"{'false-match rate (bank)':<32} {pct('false_match_rate_bank', 'false_match_rate_bank_counts')}")
    print(f"{'false-match rate (pooled)':<32} {pct('false_match_rate_pooled', 'false_match_rate_pooled_counts')}")
    print(f"{'wrong-attribution rate':<32} {pct('wrong_attribution_rate', 'wrong_attribution_counts')}")
    print(f"{'duplicate precision':<32} {pct('duplicate_precision', 'duplicate_precision_n')}")
    print(f"{'duplicate recall':<32} {pct('duplicate_recall', 'duplicate_recall_n')}")
    print(f"{'trap recall':<32} {pct('trap_recall', 'trap_counts')}")
    print(f"{'reason-code accuracy (TN set)':<32} {pct('reason_code_accuracy', 'reason_code_accuracy_n')}")
    if result["records_per_second"]:
        print(f"{'throughput':<32} {result['records_per_second']:,.0f} records/sec")
    print()
    print("confusion (expected reason code -> engine reason code, on the true-orphan/duplicate set):")
    for expected, engine_map in result["confusion"].items():
        print(f"  {expected}: {engine_map}")
    if result["warning"]:
        print(f"\nWARNING: {result['warning']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score reconciliation engine output against planted ground truth.")
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for p in (args.matches, args.exceptions, args.ground_truth):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    result = score(args.matches, args.exceptions, args.ground_truth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    _print_table(result)
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
