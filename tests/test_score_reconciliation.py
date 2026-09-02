from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from score_reconciliation import score
from src.reconcile import REASON_CODES, run_reconciliation

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Hand-built 8-row ground truth + engine-output fixture with a known
# TP / FP / FN / TN / WA composition, computed by hand below.
#
# Ledger side (6 rows):
#   LTX1: true MATCHED with BNK1 (grp_1)         -> engine matches [LTX1]->BNK1        => TP
#   LTX2: true MATCHED with BNK2 (grp_2)         -> engine leaves as exception          => FN_raw
#   LTX3: true MATCHED with BNK3+LTX4 (grp_3)    -> engine matches only [LTX3]->BNK3    => WA (wrong group)
#   LTX4: true MATCHED with BNK3+LTX3 (grp_3)    -> engine leaves as exception          => (part of the WA group's other member; scored independently as FN_raw here since it never appears in any matches row)
#   LTX5: true ORPHAN                            -> engine leaves as exception          => TN
#   LTX6: true DUPLICATE (of LTX1)               -> engine force-matches to BNK4        => FP_raw
#
# Bank side (4 rows): BNK1 (matched, TP), BNK2 (matched, FN_raw), BNK3 (WA),
# BNK4 (true ORPHAN, force-matched -> FP_raw).
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture_dir(tmp_path):
    ground_truth = {
        "meta": {"seed": 1, "window_start": "2026-01-01", "window_end": "2026-01-31", "generated_at": "x"},
        "ledger_truth": {
            "LTX1": {"true_label": "MATCHED", "true_group_id": "grp_1", "true_utr": "UTR1", "duplicate_of": None, "orphan_reason": None},
            "LTX2": {"true_label": "MATCHED", "true_group_id": "grp_2", "true_utr": "UTR2", "duplicate_of": None, "orphan_reason": None},
            "LTX3": {"true_label": "MATCHED", "true_group_id": "grp_3", "true_utr": "UTR3", "duplicate_of": None, "orphan_reason": None},
            "LTX4": {"true_label": "MATCHED", "true_group_id": "grp_3", "true_utr": "UTR3", "duplicate_of": None, "orphan_reason": None},
            "LTX5": {"true_label": "ORPHAN", "true_group_id": None, "true_utr": None, "duplicate_of": None, "orphan_reason": "no_candidate_in_window"},
            "LTX6": {"true_label": "DUPLICATE", "true_group_id": None, "true_utr": None, "duplicate_of": "LTX1", "orphan_reason": None},
        },
        "bank_truth": {
            "BNK1": {"true_label": "MATCHED", "true_group_id": "grp_1", "true_ledger_txn_ids": ["LTX1"], "duplicate_of": None, "orphan_reason": None},
            "BNK2": {"true_label": "MATCHED", "true_group_id": "grp_2", "true_ledger_txn_ids": ["LTX2"], "duplicate_of": None, "orphan_reason": None},
            "BNK3": {"true_label": "MATCHED", "true_group_id": "grp_3", "true_ledger_txn_ids": ["LTX3", "LTX4"], "duplicate_of": None, "orphan_reason": None},
            "BNK4": {"true_label": "ORPHAN", "true_group_id": None, "true_ledger_txn_ids": [], "duplicate_of": None, "orphan_reason": "prior_period_adjustment"},
        },
        "match_groups": {
            "grp_1": {"utr": "UTR1", "bank_row_id": "BNK1", "ledger_txn_ids": ["LTX1"], "match_type": "CLEAN_1_1"},
            "grp_2": {"utr": "UTR2", "bank_row_id": "BNK2", "ledger_txn_ids": ["LTX2"], "match_type": "CLEAN_1_1"},
            "grp_3": {"utr": "UTR3", "bank_row_id": "BNK3", "ledger_txn_ids": ["LTX3", "LTX4"], "match_type": "BATCH_NET"},
        },
        "planted_traps": [],
        "summary_counts": {},
    }

    matches_csv = (
        "match_group_id,ledger_txn_ids,bank_row_id,utr,rule,member_count,matched_amount,expected_amount,delta,confidence\n"
        "grp_00001,LTX1,BNK1,UTR1,exact_1to1,1,100.0,100.0,0.0,1.0\n"
        "grp_00002,LTX3,BNK3,UTR3,exact_1to1,1,50.0,50.0,0.0,0.9\n"          # WA: claimed {LTX3} != true {LTX3,LTX4}
        "grp_00003,LTX6,BNK4,UTR4,exact_1to1,1,10.0,10.0,0.0,0.5\n"          # FP_raw: LTX6 is a true DUPLICATE
    )

    exceptions_json = [
        {"record_type": "ledger", "record_id": "LTX2", "merchant_id": "m", "reason_code": "AMOUNT_MISMATCH",
         "detail": "nearest_candidate_outside_fee_tolerance", "candidate_id": "BNK2", "candidate_amount": 90.0,
         "delta": 20.0, "confidence": 0.3, "competing_candidates": None},
        {"record_type": "ledger", "record_id": "LTX4", "merchant_id": "m", "reason_code": "ORPHAN_LEDGER_TXN",
         "detail": "no_candidate_in_window", "candidate_id": None, "candidate_amount": None,
         "delta": None, "confidence": None, "competing_candidates": None},
        {"record_type": "ledger", "record_id": "LTX5", "merchant_id": "m", "reason_code": "ORPHAN_LEDGER_TXN",
         "detail": "no_candidate_in_window", "candidate_id": None, "candidate_amount": None,
         "delta": None, "confidence": None, "competing_candidates": None},
        {"record_type": "bank", "record_id": "BNK2", "merchant_id": "m", "reason_code": "AMOUNT_MISMATCH",
         "detail": "nearest_candidate_outside_fee_tolerance", "candidate_id": "LTX2", "candidate_amount": 100.0,
         "delta": 20.0, "confidence": 0.3, "competing_candidates": None},
    ]

    matches_path = tmp_path / "matches.csv"
    exceptions_path = tmp_path / "exceptions.json"
    truth_path = tmp_path / "truth.json"
    matches_path.write_text(matches_csv)
    exceptions_path.write_text(json.dumps(exceptions_json))
    truth_path.write_text(json.dumps(ground_truth))
    return matches_path, exceptions_path, truth_path


def test_precision_recall_hand_computed(fixture_dir):
    matches_path, exceptions_path, truth_path = fixture_dir
    result = score(matches_path, exceptions_path, truth_path)

    # TP = LTX1 only (exact group match). WA = LTX3 (claimed {LTX3} != true {LTX3,LTX4}).
    # FP_raw = LTX6 (true DUPLICATE, force-matched). FN_raw = LTX2, LTX4 (true MATCHED, left as exception).
    assert result["precision_counts"] == {"TP": 1, "FP_raw": 1, "WA": 1}
    assert result["recall_counts"] == {"TP": 1, "FN_raw": 2, "WA": 1}
    assert result["precision"] == pytest.approx(1 / 3, abs=1e-3)
    assert result["recall"] == pytest.approx(1 / 4, abs=1e-3)


def test_false_match_rate_ledger(fixture_dir):
    matches_path, exceptions_path, truth_path = fixture_dir
    result = score(matches_path, exceptions_path, truth_path)
    # Ledger-side orphan/duplicate set = {LTX5 (orphan), LTX6 (duplicate)} = 2 records.
    # LTX6 was force-matched -> FP_raw = 1. false_match_rate_ledger = 1/2.
    assert result["false_match_rate_ledger_counts"] == [1, 2]
    assert result["false_match_rate_ledger"] == pytest.approx(0.5)


def test_false_match_rate_bank(fixture_dir):
    matches_path, exceptions_path, truth_path = fixture_dir
    result = score(matches_path, exceptions_path, truth_path)
    # Bank-side orphan/duplicate set = {BNK4} = 1 record, force-matched -> FP_raw=1.
    assert result["false_match_rate_bank_counts"] == [1, 1]
    assert result["false_match_rate_bank"] == pytest.approx(1.0)


def test_wrong_attribution_rate(fixture_dir):
    matches_path, exceptions_path, truth_path = fixture_dir
    result = score(matches_path, exceptions_path, truth_path)
    # WA = 1 (LTX3). n_matched_true = TP+WA+FN_raw = 1+1+2 = 4.
    assert result["wrong_attribution_counts"] == [1, 4]
    assert result["wrong_attribution_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# End-to-end pipeline requirements (deliverables checklist item 4):
#   - real match rate against ground truth clears a stated threshold
#   - every exception carries one of the six specific reason codes, never a
#     generic catch-all
#   - the engine does NOT force-match any record ground truth marks as a
#     deliberate ORPHAN -- false_match_rate on true orphans must be exactly 0,
#     and this test explicitly fails on any non-zero value.
# ---------------------------------------------------------------------------

# Lowered from 0.80 after closing a false-match vulnerability: blank
# merchant_ref bank credits (~10% of bank rows) no longer auto-match on
# amount+date coincidence alone, since that coincidence isn't verifiable
# evidence of the right merchant -- see src/reconcile.py's candidates_for()
# comment. That trades match rate (87.6% -> 78.8% on seed 42) for a
# false-match rate that is 0% by construction rather than 0% by luck of the
# draw. 0.75 stays a meaningful regression floor without re-permitting the
# old behavior.
MATCH_RATE_THRESHOLD = 0.75


@pytest.fixture(scope="module")
def full_pipeline_run(tmp_path_factory):
    """Regenerate the batch fresh, run the real engine, and score it for real
    (no mocking) -- this exercises the actual generator + engine + scorer
    together, not just isolated unit fixtures."""
    tmp = tmp_path_factory.mktemp("full_pipeline")
    out_ledger, out_bank, out_truth = tmp / "recon_ledger.csv", tmp / "recon_bank_settlement.csv", tmp / "recon_ground_truth.json"
    subprocess.run(
        [sys.executable, str(ROOT / "data" / "generate_synthetic_recon.py"),
         "--seed", "42", "--out-ledger", str(out_ledger), "--out-bank", str(out_bank), "--out-truth", str(out_truth)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    ledger, bank = pd.read_csv(out_ledger), pd.read_csv(out_bank)
    result = run_reconciliation(ledger, bank)

    matches_path, exceptions_path = tmp / "matches.csv", tmp / "exceptions.json"
    matches_out = result.matches.copy()
    if not matches_out.empty:
        matches_out["ledger_txn_ids"] = matches_out["ledger_txn_ids"].apply(lambda ids: "|".join(ids))
    matches_out.to_csv(matches_path, index=False)
    with open(exceptions_path, "w") as f:
        json.dump(result.exceptions.to_dict(orient="records"), f, default=lambda x: None if pd.isna(x) else x)

    accuracy = score(matches_path, exceptions_path, out_truth)
    return result, accuracy, out_truth


def test_match_rate_clears_threshold(full_pipeline_run):
    _, accuracy, _ = full_pipeline_run
    assert accuracy["match_rate"] >= MATCH_RATE_THRESHOLD, (
        f"match rate {accuracy['match_rate']:.1%} did not clear the {MATCH_RATE_THRESHOLD:.0%} threshold"
    )


def test_every_exception_has_a_specific_reason_code(full_pipeline_run):
    result, _, _ = full_pipeline_run
    assert not result.exceptions.empty, "expected genuinely unresolvable records to be planted"
    codes_seen = set(result.exceptions["reason_code"].unique())
    assert codes_seen.issubset(set(REASON_CODES)), f"unexpected reason codes: {codes_seen - set(REASON_CODES)}"
    assert result.exceptions["reason_code"].notna().all()
    assert result.exceptions["detail"].notna().all(), "every exception must carry a specific detail, not a bare code"


def test_false_match_rate_on_true_orphans_is_exactly_zero(full_pipeline_run):
    """Hard requirement: the engine must never force-match a record the ground
    truth plants as a deliberate orphan. This must fail loudly on any non-zero
    value -- it is not acceptable to silently let this pass."""
    result, _, out_truth = full_pipeline_run
    gt = json.loads(out_truth.read_text())

    # Recompute directly from the fresh run's ground truth + matches to isolate
    # ORPHAN-only false matches (score()'s false_match_rate pools ORPHAN+DUPLICATE).
    ledger_claimed = {lid for ids in result.matches["ledger_txn_ids"] for lid in ids}
    bank_claimed = set(result.matches["bank_row_id"])

    orphan_fp_ledger = [lid for lid, t in gt["ledger_truth"].items() if t["true_label"] == "ORPHAN" and lid in ledger_claimed]
    orphan_fp_bank = [bid for bid, t in gt["bank_truth"].items() if t["true_label"] == "ORPHAN" and bid in bank_claimed]

    assert orphan_fp_ledger == [], f"engine force-matched true orphan ledger rows: {orphan_fp_ledger}"
    assert orphan_fp_bank == [], f"engine force-matched true orphan bank rows: {orphan_fp_bank}"
