from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from src.reconcile import run_reconciliation

ROOT = Path(__file__).resolve().parents[1]

LEDGER_COLS = [
    "ledger_txn_id", "merchant_id", "merchant_category", "capture_date",
    "gross_amount", "razorpay_fee", "tds_amount", "expected_net_amount",
    "expected_settlement_date", "txn_status", "utr_hint",
]
BANK_COLS = [
    "bank_row_id", "utr", "merchant_ref_code", "bank_credit_date",
    "credited_amount", "bank_fee", "bank_reported_txn_count", "narration",
]


def ledger_row(txn_id, merchant_id="mcht_test", category="saas_subscription",
               capture_date="2026-02-01", gross=1000.0, fee=20.0, tds=0.0,
               net=None, exp_settle="2026-02-03", status="captured", utr_hint=""):
    if net is None:
        net = round(gross - fee - tds, 2)
    return {
        "ledger_txn_id": txn_id, "merchant_id": merchant_id, "merchant_category": category,
        "capture_date": capture_date, "gross_amount": gross, "razorpay_fee": fee,
        "tds_amount": tds, "expected_net_amount": net, "expected_settlement_date": exp_settle,
        "txn_status": status, "utr_hint": utr_hint,
    }


def bank_row(row_id, utr, merchant_ref="mcht_test", credit_date="2026-02-03",
             credited=980.0, fee=5.0, count=None, narration="NEFT CR TEST"):
    return {
        "bank_row_id": row_id, "utr": utr, "merchant_ref_code": merchant_ref,
        "bank_credit_date": credit_date, "credited_amount": credited, "bank_fee": fee,
        "bank_reported_txn_count": count, "narration": narration,
    }


def ledger_df(rows):
    return pd.DataFrame(rows, columns=LEDGER_COLS)


def bank_df(rows):
    return pd.DataFrame(rows, columns=BANK_COLS)


def exc_by_id(result, record_id):
    df = result.exceptions
    match = df[df["record_id"] == record_id]
    assert len(match) == 1, f"expected exactly one exception for {record_id}, found {len(match)}"
    return match.iloc[0]


def match_for_bank(result, bank_row_id):
    df = result.matches
    match = df[df["bank_row_id"] == bank_row_id]
    assert len(match) == 1
    return match.iloc[0]


# ---------------------------------------------------------------------------
# Duplicate quarantine
# ---------------------------------------------------------------------------

def test_ledger_duplicate_quarantine():
    ledger = ledger_df([
        ledger_row("LTX000001", gross=1000.0, net=980.0),
        ledger_row("LTX000002", gross=1000.0, net=980.0),  # same merchant/date/gross -> duplicate
        ledger_row("LTX000003", merchant_id="mcht_other", gross=1000.0, net=980.0),
    ])
    bank = bank_df([])
    result = run_reconciliation(ledger, bank)
    e = exc_by_id(result, "LTX000002")
    assert e["reason_code"] == "DUPLICATE_SUSPECTED"
    assert e["detail"] == "ledger_duplicate"
    assert e["candidate_id"] == "LTX000001"
    # canonical (LTX000001) should NOT be quarantined -- it proceeds to matching
    # and ends up an ORPHAN_LEDGER_TXN (no bank data at all here), not a duplicate.
    e1 = exc_by_id(result, "LTX000001")
    assert e1["reason_code"] != "DUPLICATE_SUSPECTED"


def test_bank_duplicate_quarantine():
    ledger = ledger_df([ledger_row("LTX000001", gross=1000.0, net=980.0)])
    bank = bank_df([
        bank_row("BNK000001", utr="UTR1", credited=980.0, fee=0.0),
        bank_row("BNK000002", utr="UTR1", credited=980.30, fee=0.0, credit_date="2026-02-04"),
    ])
    result = run_reconciliation(ledger, bank)
    e = exc_by_id(result, "BNK000002")
    assert e["reason_code"] == "DUPLICATE_SUSPECTED"
    assert e["detail"] == "bank_duplicate_retry"
    assert e["candidate_id"] == "BNK000001"
    # the canonical bank row should have proceeded to a clean match
    m = match_for_bank(result, "BNK000001")
    assert m["rule"] == "exact_1to1"


# ---------------------------------------------------------------------------
# Exact 1:1 match
# ---------------------------------------------------------------------------

def test_exact_1to1_match():
    ledger = ledger_df([ledger_row("LTX000001", gross=1000.0, fee=20.0, net=980.0)])
    bank = bank_df([bank_row("BNK000001", utr="UTR1", credited=980.0, fee=0.0)])
    result = run_reconciliation(ledger, bank)
    assert result.exceptions.empty
    m = match_for_bank(result, "BNK000001")
    assert m["rule"] == "exact_1to1"
    assert m["ledger_txn_ids"] == ["LTX000001"]


# ---------------------------------------------------------------------------
# Fee-adjusted tolerance boundary
# ---------------------------------------------------------------------------

def test_fee_adjusted_boundary_matches_inside_tolerance():
    # net=50,000.00 -> target_paise = 5,000,000; tol_exact = max(200, 2500) = 2500 (Rs 25)
    # tol_fee_adj = max(500, 25000) = 25000 (Rs 250). Delta of Rs 100 is inside fee-adj, outside exact.
    ledger = ledger_df([ledger_row("LTX000001", gross=51000.0, fee=1000.0, net=50000.0)])
    bank = bank_df([bank_row("BNK000001", utr="UTR1", credited=49900.0, fee=0.0)])
    result = run_reconciliation(ledger, bank)
    assert result.exceptions.empty
    m = match_for_bank(result, "BNK000001")
    assert m["rule"] == "fee_adjusted_1to1"


def test_amount_beyond_fee_tolerance_becomes_amount_mismatch():
    # Same net as above, but delta of Rs 1000 is outside fee-adj tolerance (Rs 250)
    # and within 5% of 50,000 (Rs 2,500) -> AMOUNT_MISMATCH, not a match.
    ledger = ledger_df([ledger_row("LTX000001", gross=51000.0, fee=1000.0, net=50000.0)])
    bank = bank_df([bank_row("BNK000001", utr="UTR1", credited=49000.0, fee=0.0)])
    result = run_reconciliation(ledger, bank)
    assert result.matches.empty
    e = exc_by_id(result, "LTX000001")
    assert e["reason_code"] == "AMOUNT_MISMATCH"
    assert e["detail"] == "nearest_candidate_outside_fee_tolerance"


# ---------------------------------------------------------------------------
# Timing near-miss
# ---------------------------------------------------------------------------

def test_timing_near_miss_one_day_auto_matches():
    ledger = ledger_df([ledger_row("LTX000001", gross=1000.0, fee=20.0, net=980.0,
                                    exp_settle="2026-02-03")])
    bank = bank_df([bank_row("BNK000001", utr="UTR1", credited=980.0, fee=0.0,
                              credit_date="2026-02-04")])  # +1 day
    result = run_reconciliation(ledger, bank)
    assert result.exceptions.empty
    m = match_for_bank(result, "BNK000001")
    assert m["rule"].startswith("timing_near_miss")


def test_timing_three_days_late_is_out_of_window_not_matched():
    ledger = ledger_df([ledger_row("LTX000001", gross=1000.0, fee=20.0, net=980.0,
                                    exp_settle="2026-02-03")])
    bank = bank_df([bank_row("BNK000001", utr="UTR1", credited=980.0, fee=0.0,
                              credit_date="2026-02-06")])  # +3 days
    result = run_reconciliation(ledger, bank)
    assert result.matches.empty
    e = exc_by_id(result, "LTX000001")
    assert e["reason_code"] == "TIMING_OUT_OF_WINDOW"
    assert e["detail"] == "settled_3_days_late"


# ---------------------------------------------------------------------------
# Batch subset-sum
# ---------------------------------------------------------------------------

def _batch_ledger_rows():
    return [
        ledger_row("LTX000001", gross=1030.0, fee=30.0, net=1000.0, exp_settle="2026-02-05"),
        ledger_row("LTX000002", gross=2060.0, fee=60.0, net=2000.0, exp_settle="2026-02-05"),
        ledger_row("LTX000003", gross=3090.0, fee=90.0, net=3000.0, exp_settle="2026-02-05"),
        ledger_row("LTX000004", gross=515.0, fee=15.0, net=500.0, exp_settle="2026-02-05"),
    ]


def test_batch_subset_sum_with_hint():
    # true batch = LTX1+2+3 = 6000.00; hint says 3 members.
    ledger = ledger_df(_batch_ledger_rows())
    bank = bank_df([bank_row("BNK000001", utr="UTR1", merchant_ref="mcht_test",
                              credit_date="2026-02-05", credited=6000.0, fee=0.0, count=3)])
    result = run_reconciliation(ledger, bank)
    m = match_for_bank(result, "BNK000001")
    assert m["rule"] == "batch_subset_sum_hinted"
    assert set(m["ledger_txn_ids"]) == {"LTX000001", "LTX000002", "LTX000003"}


def test_batch_subset_sum_without_hint():
    ledger = ledger_df(_batch_ledger_rows())
    bank = bank_df([bank_row("BNK000001", utr="UTR1", merchant_ref="mcht_test",
                              credit_date="2026-02-05", credited=6000.0, fee=0.0, count=None)])
    result = run_reconciliation(ledger, bank)
    m = match_for_bank(result, "BNK000001")
    assert m["rule"] == "batch_subset_sum"
    assert set(m["ledger_txn_ids"]) == {"LTX000001", "LTX000002", "LTX000003"}


def test_max_subset_pool_cap_skips_and_returns_quickly():
    rows = [
        ledger_row(f"LTX{i:06d}", gross=103.0 + i, fee=3.0, net=100.0 + i,
                   capture_date=f"2026-02-{i:02d}", exp_settle="2026-02-05")
        for i in range(1, 14)  # 13 candidates > MAX_SUBSET_POOL (12), distinct amounts/dates
    ]
    ledger = ledger_df(rows)
    bank = bank_df([bank_row("BNK000001", utr="UTR1", merchant_ref="mcht_test",
                              credit_date="2026-02-05", credited=500.0, fee=0.0, count=None)])
    start = time.perf_counter()
    result = run_reconciliation(ledger, bank)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, "13-candidate pool should be skipped, not enumerated"
    e = exc_by_id(result, "BNK000001")
    assert e["reason_code"] == "ORPHAN_BANK_CREDIT"
    assert e["detail"] == "pool_too_large"


def test_ambiguous_subset_sum_leaves_all_unresolved():
    # Two disjoint 2-member subsets both sum to 3000.00 (distinct amounts/dates
    # so Stage 0b's duplicate quarantine doesn't collide LTX3/LTX4 first).
    ledger = ledger_df([
        ledger_row("LTX000001", gross=1030.0, fee=30.0, net=1000.0,
                   capture_date="2026-02-01", exp_settle="2026-02-05"),
        ledger_row("LTX000002", gross=2060.0, fee=60.0, net=2000.0,
                   capture_date="2026-02-02", exp_settle="2026-02-05"),
        ledger_row("LTX000003", gross=1442.0, fee=42.0, net=1400.0,
                   capture_date="2026-02-03", exp_settle="2026-02-05"),
        ledger_row("LTX000004", gross=1648.0, fee=48.0, net=1600.0,
                   capture_date="2026-02-04", exp_settle="2026-02-05"),
    ])
    bank = bank_df([bank_row("BNK000001", utr="UTR1", merchant_ref="mcht_test",
                              credit_date="2026-02-05", credited=3000.0, fee=0.0, count=None)])
    result = run_reconciliation(ledger, bank)
    assert result.matches.empty
    for rid in ["LTX000001", "LTX000002", "LTX000003", "LTX000004", "BNK000001"]:
        e = exc_by_id(result, rid)
        assert e["reason_code"] == "UNRESOLVED_SUBSET_SUM"


# ---------------------------------------------------------------------------
# Blank merchant_ref -- never auto-matched, even on a unique coincidence
# ---------------------------------------------------------------------------

def test_blank_merchant_ref_never_auto_matches_even_on_unique_coincidence():
    # A blank-ref bank credit that happens to share date+amount with some
    # OTHER merchant's ledger row is not evidence it belongs to that
    # merchant -- it's an unverifiable coincidence, and it is (by
    # construction here) the only numeric candidate, which is exactly the
    # case a naive "exactly one candidate" safeguard would wrongly accept.
    # The engine must never force-match on amount+date alone when there is
    # no merchant reference to verify it against.
    ledger = ledger_df([
        ledger_row("LTX000001", merchant_id="mcht_a", gross=1030.0, fee=30.0,
                   net=1000.0, exp_settle="2026-02-05"),
    ])
    bank = bank_df([
        bank_row("BNK000001", utr="UTR1", merchant_ref="", credit_date="2026-02-05",
                  credited=1000.0, fee=0.0),
    ])
    result = run_reconciliation(ledger, bank)
    assert result.matches.empty
    e_bank = exc_by_id(result, "BNK000001")
    assert e_bank["reason_code"] == "ORPHAN_BANK_CREDIT"
    assert e_bank["detail"] == "no_merchant_ref"
    e_ledger = exc_by_id(result, "LTX000001")
    assert e_ledger["reason_code"] == "ORPHAN_LEDGER_TXN"


# ---------------------------------------------------------------------------
# Paise rounding
# ---------------------------------------------------------------------------

def test_paise_rounding_avoids_float_error():
    # 100.10 + 200.20 = 300.30 in exact decimal, but float addition can drift.
    ledger = ledger_df([ledger_row("LTX000001", gross=None, fee=0.0, net=300.30,
                                    exp_settle="2026-02-05")])
    ledger.loc[0, "gross_amount"] = 300.30
    bank = bank_df([bank_row("BNK000001", utr="UTR1", credited=100.10 + 200.20, fee=0.0,
                              credit_date="2026-02-05")])
    result = run_reconciliation(ledger, bank)
    assert result.exceptions.empty
    m = match_for_bank(result, "BNK000001")
    assert m["rule"] == "exact_1to1"


# ---------------------------------------------------------------------------
# Static leak-check
# ---------------------------------------------------------------------------

def test_reconcile_never_references_ground_truth_file():
    src = (ROOT / "src" / "reconcile.py").read_text()
    assert "recon_ground_truth" not in src


def test_run_reconciliation_script_never_references_ground_truth_file():
    src = (ROOT / "run_reconciliation.py").read_text()
    assert "recon_ground_truth" not in src
