from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

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


def _run_generator(tmp_path: Path, seed: int = 42) -> tuple[Path, Path, Path]:
    out_ledger = tmp_path / "recon_ledger.csv"
    out_bank = tmp_path / "recon_bank_settlement.csv"
    out_truth = tmp_path / "recon_ground_truth.json"
    subprocess.run(
        [sys.executable, str(ROOT / "data" / "generate_synthetic_recon.py"),
         "--seed", str(seed),
         "--out-ledger", str(out_ledger), "--out-bank", str(out_bank), "--out-truth", str(out_truth)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return out_ledger, out_bank, out_truth


def test_same_seed_determinism(tmp_path):
    l1, b1, t1 = _run_generator(tmp_path / "run1", seed=42)
    l2, b2, t2 = _run_generator(tmp_path / "run2", seed=42)
    # All three outputs -- including the ground-truth JSON's metadata -- must
    # be byte-identical across reruns of the same seed. There is no wall-clock
    # or other non-deterministic field left in any of them.
    assert l1.read_bytes() == l2.read_bytes()
    assert b1.read_bytes() == b2.read_bytes()
    assert t1.read_bytes() == t2.read_bytes()


def test_schema_exact(tmp_path):
    l, b, t = _run_generator(tmp_path, seed=42)
    ledger = pd.read_csv(l)
    bank = pd.read_csv(b)
    assert list(ledger.columns) == LEDGER_COLS
    assert list(bank.columns) == BANK_COLS


def test_timing_near_miss_rows_are_always_exactly_one_day_off(tmp_path):
    # Regression test: a naive `expected_settlement_date +- 1 day` shift that
    # gets rolled forward over a weekend/holiday can land more than 1 day out,
    # silently breaking the intended near-miss (src/reconcile.py's auto-match
    # window is exactly TIMING_AUTO_DAYS=1). Every planted TIMING_NEAR_MISS
    # row must land at exactly a 1-day offset, for any seed.
    for seed in (42, 7, 123, 2026):
        l, b, t = _run_generator(tmp_path / f"seed{seed}", seed=seed)
        ledger = pd.read_csv(l, parse_dates=["expected_settlement_date"]).set_index("ledger_txn_id")
        bank = pd.read_csv(b, parse_dates=["bank_credit_date"]).set_index("bank_row_id")
        gt = json.loads(t.read_text())
        near_miss_groups = [g for g in gt["match_groups"].values() if g["match_type"] == "TIMING_NEAR_MISS"]
        assert near_miss_groups, f"seed {seed}: no TIMING_NEAR_MISS rows planted"
        for g in near_miss_groups:
            (ltx,) = g["ledger_txn_ids"]
            expected = ledger.loc[ltx, "expected_settlement_date"]
            credited = bank.loc[g["bank_row_id"], "bank_credit_date"]
            offset = (credited - expected).days
            assert abs(offset) == 1, f"seed {seed}: {g['bank_row_id']} is {offset} days off {ltx}, not 1"


def test_summary_counts_match_realized_rows(tmp_path):
    l, b, t = _run_generator(tmp_path, seed=42)
    ledger = pd.read_csv(l)
    bank = pd.read_csv(b)
    gt = json.loads(t.read_text())
    sc = gt["summary_counts"]

    assert sc["total_ledger_rows"] == len(ledger) == len(gt["ledger_truth"])
    assert sc["total_bank_rows"] == len(bank) == len(gt["bank_truth"])

    n_matched_ledger = sum(1 for v in gt["ledger_truth"].values() if v["true_label"] == "MATCHED")
    n_dup_ledger = sum(1 for v in gt["ledger_truth"].values() if v["true_label"] == "DUPLICATE")
    n_orphan_ledger = sum(1 for v in gt["ledger_truth"].values() if v["true_label"] == "ORPHAN")
    assert sc["duplicate_ledger"] == n_dup_ledger
    assert sc["orphan_ledger"] == n_orphan_ledger
    assert n_matched_ledger + n_dup_ledger + n_orphan_ledger == sc["total_ledger_rows"]

    n_dup_bank = sum(1 for v in gt["bank_truth"].values() if v["true_label"] == "DUPLICATE")
    n_orphan_bank = sum(1 for v in gt["bank_truth"].values() if v["true_label"] == "ORPHAN")
    assert sc["duplicate_bank"] == n_dup_bank
    assert sc["orphan_bank"] == n_orphan_bank

    batch_groups = [g for g in gt["match_groups"].values() if g["match_type"] == "BATCH_NET"]
    assert sc["batch_net_groups"] == len(batch_groups)
    assert sc["batch_net_members"] == sum(len(g["ledger_txn_ids"]) for g in batch_groups)

    clean_groups = [g for g in gt["match_groups"].values() if g["match_type"] == "CLEAN_1_1"]
    fee_drift_groups = [g for g in gt["match_groups"].values() if g["match_type"] == "FEE_DRIFT_1_1"]
    timing_groups = [g for g in gt["match_groups"].values() if g["match_type"] == "TIMING_NEAR_MISS"]
    assert sc["clean_1_1"] == len(clean_groups)
    assert sc["fee_drift_1_1"] == len(fee_drift_groups)
    assert sc["timing_near_miss"] == len(timing_groups)

    assert sc["traps"] == len(gt["planted_traps"]) == 2


def test_scale_and_mix_requirements(tmp_path):
    """At least 50 records, ~100-150 ledger txns, genuinely unresolvable records planted."""
    l, b, t = _run_generator(tmp_path, seed=42)
    ledger = pd.read_csv(l)
    gt = json.loads(t.read_text())
    assert len(ledger) >= 50
    assert 100 <= len(ledger) <= 150
    assert gt["summary_counts"]["orphan_ledger"] > 0
    assert gt["summary_counts"]["orphan_bank"] > 0
    assert gt["summary_counts"]["duplicate_ledger"] > 0
    assert gt["summary_counts"]["duplicate_bank"] > 0
    assert gt["summary_counts"]["batch_net_groups"] > 0
