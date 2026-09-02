"""
Synthetic reconciliation batch generator.

Writes two engine-readable CSVs (the internal ledger and the bank/processor
settlement feed) plus a ground-truth JSON answer key that only
`score_reconciliation.py` ever reads — `src/reconcile.py` must never see it.

Mirrors the style of `generate_synthetic_ledger.py`: argparse CLI, a single
`np.random.default_rng(seed)` for full determinism, pandas builders, and a
`_print_summary()` at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_synthetic_ledger import CATEGORY_SPEC, MERCHANTS  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed universe
# ---------------------------------------------------------------------------

RECON_MERCHANT_IDS = [
    "mcht_saas_02", "mcht_d2c_02", "mcht_d2c_04",
    "mcht_mkt_01", "mcht_mkt_03", "mcht_food_01",
]
WINDOW_START = pd.Timestamp("2026-02-01")
WINDOW_END = pd.Timestamp("2026-03-02")  # 30 days inclusive
BANK_HOLIDAYS = [pd.Timestamp("2026-02-16")]  # single fixed synthetic holiday (a Monday)
TDS_FLAT_RATE = 0.01
TDS_GROSS_THRESHOLD = 50_000.0

MERCHANT_LOOKUP = {m["merchant_id"]: m for m in MERCHANTS}
RECON_MERCHANTS = [MERCHANT_LOOKUP[mid] for mid in RECON_MERCHANT_IDS]

# 8 batches summing to 26 members, weighted toward the two marketplace merchants.
# mcht_mkt_01's two batches are equal-sized (3, 3) on purpose — they are the
# AMBIGUOUS_SUBSET_SUM trap pair (§1.6.2), and equal sizes keep the two batch
# totals in the same ballpark so a small per-member nudge can create a real
# cross-batch collision.
BATCH_PLAN = [
    {"merchant_id": "mcht_mkt_01", "size": 3, "trap_pair": True},
    {"merchant_id": "mcht_mkt_01", "size": 3, "trap_pair": True},
    {"merchant_id": "mcht_mkt_03", "size": 5, "trap_pair": False},
    {"merchant_id": "mcht_mkt_03", "size": 4, "trap_pair": False},
    {"merchant_id": "mcht_saas_02", "size": 4, "trap_pair": False},
    {"merchant_id": "mcht_d2c_02", "size": 3, "trap_pair": False},
    {"merchant_id": "mcht_d2c_04", "size": 2, "trap_pair": False},
    {"merchant_id": "mcht_food_01", "size": 2, "trap_pair": False},
]
assert sum(b["size"] for b in BATCH_PLAN) == 26
assert len(BATCH_PLAN) == 8


def roll_to_business_day(date: pd.Timestamp) -> pd.Timestamp:
    while date.dayofweek >= 5 or date in BANK_HOLIDAYS:
        date = date + pd.Timedelta(days=1)
    return date


def _gross_fee_tds_net(merchant: dict, gross: float) -> tuple[float, float, float]:
    fee = round(gross * _rng.uniform(0.018, 0.022), 2)
    if merchant["merchant_category"] in {"marketplace", "d2c_ecommerce"} and gross > TDS_GROSS_THRESHOLD:
        tds = round(TDS_FLAT_RATE * gross, 2)
    else:
        tds = 0.0
    net = round(gross - fee - tds, 2)
    return fee, tds, net


def _recompute(row: dict, merchant: dict) -> None:
    """Recompute fee/tds/net/expected_settlement_date from gross_amount + capture_date.

    Draws a fresh fee (fee is a per-transaction random draw, not a deterministic
    function of gross_amount) -- only safe to call while a row's financial
    fields are still being established for the first time, never afterwards
    (a second call would silently perturb amounts already relied on elsewhere,
    e.g. a nudge computed to make two batch totals collide).
    """
    fee, tds, net = _gross_fee_tds_net(merchant, row["gross_amount"])
    row["razorpay_fee"] = fee
    row["tds_amount"] = tds
    row["expected_net_amount"] = net
    _recompute_settlement_date_only(row, merchant)


def _recompute_settlement_date_only(row: dict, merchant: dict) -> None:
    """Recompute expected_settlement_date from capture_date, leaving amounts untouched."""
    lag = CATEGORY_SPEC[merchant["merchant_category"]]["settlement_lag"]
    row["expected_settlement_date"] = roll_to_business_day(row["capture_date"] + pd.Timedelta(days=lag))


# ---------------------------------------------------------------------------
# Step 1: base ledger rows (132 = 22 per merchant x 6 merchants)
# ---------------------------------------------------------------------------

def _base_rows(rng: np.random.Generator) -> list[dict]:
    global _rng
    _rng = rng
    rows = []
    all_dates = pd.date_range(WINDOW_START, WINDOW_END)
    for merchant in RECON_MERCHANTS:
        lag = CATEGORY_SPEC[merchant["merchant_category"]]["settlement_lag"]
        capture_dates = sorted(rng.choice(all_dates, size=22, replace=True))
        for capture_date in capture_dates:
            capture_date = pd.Timestamp(capture_date)
            base = merchant["base"]
            gross = round(
                float(np.clip(rng.lognormal(mean=np.log(base / 15), sigma=0.55), 500, base * 3)), 2
            )
            fee, tds, net = _gross_fee_tds_net(merchant, gross)
            row = {
                "merchant_id": merchant["merchant_id"],
                "merchant_category": merchant["merchant_category"],
                "capture_date": capture_date,
                "gross_amount": gross,
                "razorpay_fee": fee,
                "tds_amount": tds,
                "expected_net_amount": net,
                "expected_settlement_date": roll_to_business_day(capture_date + pd.Timedelta(days=lag)),
                "txn_status": "captured",
                "utr_hint": "",
                "scenario": None,
                "orphan_reason": None,
                "batch_group": None,
            }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Step 2: scenario allocation
# ---------------------------------------------------------------------------

def _allocate_scenarios(rows: list[dict], rng: np.random.Generator) -> dict:
    """Mutates `rows` in place, assigning a `scenario` to every row.

    Returns metadata needed downstream: the realized batch plan (with member
    row indices) and the indices chosen for each generic scenario.
    """
    by_merchant: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_merchant.setdefault(row["merchant_id"], []).append(i)

    # --- BATCH_NET: per-merchant quotas, forced same-day clusters ---
    realized_batches = []
    for batch in BATCH_PLAN:
        mid = batch["merchant_id"]
        pool = by_merchant[mid]
        size = batch["size"]
        chosen = list(rng.choice(pool, size=size, replace=False))
        for idx in chosen:
            pool.remove(idx)
        # Force the cluster: every member takes the first member's capture_date.
        anchor_date = rows[chosen[0]]["capture_date"]
        merchant = MERCHANT_LOOKUP[mid]
        for idx in chosen:
            rows[idx]["capture_date"] = anchor_date
            _recompute(rows[idx], merchant)
            rows[idx]["scenario"] = "BATCH_NET"
        realized_batches.append({"merchant_id": mid, "member_idx": chosen, "trap_pair": batch["trap_pair"]})

    # --- Ambiguous-subset-sum trap: nudge one member of the first trap batch
    #     so its total lands within tolerance of the second trap batch's total.
    trap_batches = [b for b in realized_batches if b["trap_pair"]]
    assert len(trap_batches) == 2
    batch_i, batch_j = trap_batches
    merchant = MERCHANT_LOOKUP[batch_i["merchant_id"]]
    sum_i = sum(rows[idx]["expected_net_amount"] for idx in batch_i["member_idx"])
    sum_j = sum(rows[idx]["expected_net_amount"] for idx in batch_j["member_idx"])
    delta_needed = sum_j - sum_i
    # Push the whole delta onto one member of batch_i by adjusting its gross amount.
    nudge_idx = batch_i["member_idx"][0]
    row = rows[nudge_idx]
    new_net = round(row["expected_net_amount"] + delta_needed, 2)
    new_gross = round(new_net + row["razorpay_fee"] + row["tds_amount"], 2)
    row["gross_amount"] = new_gross
    row["expected_net_amount"] = new_net
    # Force both trap batches onto the *same* expected_settlement_date. The
    # matching engine's Stage-5 candidate window is backward-looking
    # ([bank_credit_date - 1, bank_credit_date]), so equal dates are required
    # (not just "within 1 day") for each trap bank row's pool to see *both*
    # batches' members and genuinely collide in both directions.
    date_i = rows[batch_i["member_idx"][0]]["expected_settlement_date"]
    date_j = rows[batch_j["member_idx"][0]]["expected_settlement_date"]
    if date_i != date_j:
        for idx in batch_j["member_idx"]:
            rows[idx]["capture_date"] = rows[idx]["capture_date"] + (date_i - date_j)
            _recompute_settlement_date_only(rows[idx], MERCHANT_LOOKUP[batch_j["merchant_id"]])

    # --- Remaining rows: generic pool (any merchant) ---
    remaining = [i for pool in by_merchant.values() for i in pool]
    perm = rng.permutation(remaining)
    orphan_idx = list(perm[:6])
    timing_idx = list(perm[6:13])
    fee_drift_idx = list(perm[13:24])
    clean_idx = list(perm[24:])
    assert len(clean_idx) == 82, len(clean_idx)

    # ORPHAN_LEDGER_TXN: split 2/2/2
    orphan_reasons = ["test_mode_capture", "test_mode_capture",
                       "pending_payout_censored", "pending_payout_censored",
                       "refund_reversed_internally", "refund_reversed_internally"]
    for idx, reason in zip(orphan_idx, orphan_reasons):
        row = rows[idx]
        row["scenario"] = "ORPHAN_LEDGER_TXN"
        row["orphan_reason"] = reason
        row["utr_hint"] = ""
        if reason == "test_mode_capture":
            row["txn_status"] = "captured_test_mode"
        elif reason == "pending_payout_censored":
            merchant = MERCHANT_LOOKUP[row["merchant_id"]]
            lag = CATEGORY_SPEC[merchant["merchant_category"]]["settlement_lag"]
            max_offset = min(2, lag - 1)
            offset = int(rng.integers(0, max_offset + 1))
            row["capture_date"] = WINDOW_END - pd.Timedelta(days=offset)
            _recompute(row, merchant)
            assert row["expected_settlement_date"] > WINDOW_END
            row["txn_status"] = "captured_pending_payout"
        # refund_reversed_internally: leave txn_status="captured", ground truth only

    for idx in timing_idx:
        rows[idx]["scenario"] = "TIMING_NEAR_MISS"
    for idx in fee_drift_idx:
        rows[idx]["scenario"] = "FEE_DRIFT_1_1"
    for idx in clean_idx:
        rows[idx]["scenario"] = "CLEAN_1_1"

    return {
        "batches": realized_batches,
        "trap_batches": (batch_i, batch_j),
        "orphan_idx": orphan_idx,
        "timing_idx": timing_idx,
        "fee_drift_idx": fee_drift_idx,
        "clean_idx": clean_idx,
        "amount_collision_ledger_idx": orphan_idx[4],  # first refund_reversed_internally row
    }


# ---------------------------------------------------------------------------
# Step 3: bank settlement rows
# ---------------------------------------------------------------------------

def _bank_fee(net_total: float) -> float:
    return round(5.0 + 0.0001 * net_total, 2)


def _build_bank_rows(rows: list[dict], plan: dict, rng: np.random.Generator) -> tuple[list[dict], dict]:
    bank_rows: list[dict] = []
    match_groups: dict[str, dict] = {}
    ledger_true_utr: dict[int, str] = {}
    ledger_true_group: dict[int, str] = {}
    group_counter = 0

    def new_utr() -> str:
        return f"UTR{rng.integers(10**11, 10**12)}"

    def merchant_ref(mid: str) -> str:
        return mid if rng.random() < 0.90 else ""

    def bnk_id(n: int) -> str:
        return f"BNK{n:06d}"

    bnk_counter = 0

    def next_bnk_id() -> str:
        nonlocal bnk_counter
        bnk_counter += 1
        return bnk_id(bnk_counter)

    # --- CLEAN_1_1 ---
    for idx in plan["clean_idx"]:
        row = rows[idx]
        utr = new_utr()
        fee = _bank_fee(row["expected_net_amount"])
        credited = round(row["expected_net_amount"] - fee, 2)
        bnkid = next_bnk_id()
        count = 1 if rng.random() < 0.70 else None
        bank_rows.append({
            "bank_row_id": bnkid, "utr": utr, "merchant_ref_code": merchant_ref(row["merchant_id"]),
            "bank_credit_date": row["expected_settlement_date"], "credited_amount": credited,
            "bank_fee": fee, "bank_reported_txn_count": count,
            "narration": f"NEFT CR RAZORPAY SETTLEMENT {utr}",
            "_scenario": "CLEAN_1_1",
        })
        group_counter += 1
        gid = f"grp_{group_counter:05d}"
        match_groups[gid] = {"utr": utr, "bank_row_id": bnkid, "ledger_txn_ids": [idx], "match_type": "CLEAN_1_1"}
        ledger_true_utr[idx] = utr
        ledger_true_group[idx] = gid

    clean_bank_rows = [b for b in bank_rows if b["_scenario"] == "CLEAN_1_1"]

    # --- BATCH_NET ---
    for bi, batch in enumerate(plan["batches"]):
        members = batch["member_idx"]
        net_total = sum(rows[i]["expected_net_amount"] for i in members)
        fee = _bank_fee(net_total)
        credited = round(net_total - fee, 2)
        utr = new_utr()
        bnkid = next_bnk_id()
        settle_date = rows[members[0]]["expected_settlement_date"]
        true_count = len(members)
        bank_rows.append({
            "bank_row_id": bnkid, "utr": utr, "merchant_ref_code": merchant_ref(batch["merchant_id"]),
            "bank_credit_date": settle_date, "credited_amount": credited,
            "bank_fee": fee, "bank_reported_txn_count": true_count,  # placeholder, overwritten below
            "narration": f"NEFT CR RAZORPAY SETTLEMENT {utr}",
            "_scenario": "BATCH_NET",
        })
        group_counter += 1
        gid = f"grp_{group_counter:05d}"
        match_groups[gid] = {"utr": utr, "bank_row_id": bnkid, "ledger_txn_ids": list(members), "match_type": "BATCH_NET"}
        for i in members:
            ledger_true_utr[i] = utr
            ledger_true_group[i] = gid
        batch["bank_row_id"] = bnkid
        batch["true_count"] = true_count

    # bank_reported_txn_count hints across the 8 batches: the two trap batches
    # get None / deliberately-wrong (both force a full subset sweep per §3.6's
    # fallback rule); of the remaining 6, choose which get true vs None/wrong
    # per spec (true on 6, None on 1, wrong on 1 -- here folded into the trap
    # pair so all 6 non-trap batches carry the true count).
    trap_bank_ids = {b["bank_row_id"] for b in plan["trap_batches"]}
    trap_i_bnk = plan["trap_batches"][0]["bank_row_id"]
    trap_j_bnk = plan["trap_batches"][1]["bank_row_id"]
    for b in bank_rows:
        if b["_scenario"] != "BATCH_NET":
            continue
        if b["bank_row_id"] == trap_i_bnk:
            b["bank_reported_txn_count"] = None
        elif b["bank_row_id"] == trap_j_bnk:
            b["bank_reported_txn_count"] = b["bank_reported_txn_count"] + 1  # deliberately wrong
        # else: keep true count

    # --- FEE_DRIFT_1_1 ---
    for idx in plan["fee_drift_idx"]:
        row = rows[idx]
        fee = _bank_fee(row["expected_net_amount"])
        sign = rng.choice([1, -1])
        drift = sign * round(rng.uniform(2.50, 4.50), 2)
        credited = round(row["expected_net_amount"] - fee + drift, 2)
        utr = new_utr()
        bnkid = next_bnk_id()
        count = 1 if rng.random() < 0.70 else None
        bank_rows.append({
            "bank_row_id": bnkid, "utr": utr, "merchant_ref_code": merchant_ref(row["merchant_id"]),
            "bank_credit_date": row["expected_settlement_date"], "credited_amount": credited,
            "bank_fee": fee, "bank_reported_txn_count": count,
            "narration": f"NEFT CR RAZORPAY SETTLEMENT {utr}",
            "_scenario": "FEE_DRIFT_1_1",
        })
        group_counter += 1
        gid = f"grp_{group_counter:05d}"
        match_groups[gid] = {"utr": utr, "bank_row_id": bnkid, "ledger_txn_ids": [idx], "match_type": "FEE_DRIFT_1_1"}
        ledger_true_utr[idx] = utr
        ledger_true_group[idx] = gid

    # --- TIMING_NEAR_MISS ---
    # A near-miss must land exactly 1 calendar day off expected_settlement_date
    # (matching src/reconcile.py's TIMING_AUTO_DAYS=1 auto-match window). Naively
    # shifting by 1 day and then rolling forward over a weekend/holiday can walk
    # the date multiple days out, silently turning an intended "near miss" into
    # a TIMING_OUT_OF_WINDOW exception depending on which capture dates the RNG
    # draws -- so pick whichever sign lands on an already-open business day
    # with no further rolling.
    for idx in plan["timing_idx"]:
        row = rows[idx]
        fee = _bank_fee(row["expected_net_amount"])
        credited = round(row["expected_net_amount"] - fee, 2)
        preferred_sign = int(rng.choice([1, -1]))
        credit_date = None
        for sign in (preferred_sign, -preferred_sign):
            candidate = row["expected_settlement_date"] + pd.Timedelta(days=sign)
            if candidate.dayofweek < 5 and candidate not in BANK_HOLIDAYS:
                credit_date = candidate
                break
        if credit_date is None:
            # Not reachable with the single fixed BANK_HOLIDAYS entry this
            # generator uses today -- fail loudly instead of silently rolling
            # multiple days if the holiday calendar is ever widened.
            raise ValueError(
                f"TIMING_NEAR_MISS row {idx}: neither neighbouring day of "
                f"{row['expected_settlement_date'].date()} is a business day"
            )
        utr = new_utr()
        bnkid = next_bnk_id()
        count = 1 if rng.random() < 0.70 else None
        bank_rows.append({
            "bank_row_id": bnkid, "utr": utr, "merchant_ref_code": merchant_ref(row["merchant_id"]),
            "bank_credit_date": credit_date, "credited_amount": credited,
            "bank_fee": fee, "bank_reported_txn_count": count,
            "narration": f"NEFT CR RAZORPAY SETTLEMENT {utr}",
            "_scenario": "TIMING_NEAR_MISS",
        })
        group_counter += 1
        gid = f"grp_{group_counter:05d}"
        match_groups[gid] = {"utr": utr, "bank_row_id": bnkid, "ledger_txn_ids": [idx], "match_type": "TIMING_NEAR_MISS"}
        ledger_true_utr[idx] = utr
        ledger_true_group[idx] = gid

    # --- +2 pure ORPHAN_BANK_CREDIT rows ---
    collision_ledger_idx = plan["amount_collision_ledger_idx"]
    collision_row = rows[collision_ledger_idx]
    other_merchants = [m for m in RECON_MERCHANT_IDS if m != collision_row["merchant_id"]]
    collision_ref_merchant = rng.choice(other_merchants)

    prior_period_bnkid = next_bnk_id()
    bank_rows.append({
        "bank_row_id": prior_period_bnkid, "utr": new_utr(), "merchant_ref_code": str(collision_ref_merchant),
        "bank_credit_date": WINDOW_START + pd.Timedelta(days=12), "credited_amount": 4500.00,
        "bank_fee": 0.0, "bank_reported_txn_count": None,
        "narration": "NEFT CR PRIOR PERIOD ADJUSTMENT",
        "_scenario": "ORPHAN_BANK_CREDIT", "_orphan_reason": "prior_period_adjustment",
    })
    # Adjust the collision ledger row so its net amount lands within Rs.3 of 4500.00,
    # while merchant differs from collision_ref_merchant (guaranteed by construction).
    collision_row["gross_amount"] = 4600.0
    collision_row["razorpay_fee"] = 100.0
    collision_row["tds_amount"] = 0.0
    collision_row["expected_net_amount"] = 4500.0

    interest_bnkid = next_bnk_id()
    interest_amount = round(float(150.0 + rng.uniform(-2.0, 2.0)), 2)
    bank_rows.append({
        "bank_row_id": interest_bnkid, "utr": new_utr(), "merchant_ref_code": "",
        "bank_credit_date": WINDOW_START + pd.Timedelta(days=19), "credited_amount": interest_amount,
        "bank_fee": 0.0, "bank_reported_txn_count": None,
        "narration": "NEFT CR INTEREST ON DEPOSIT",
        "_scenario": "ORPHAN_BANK_CREDIT", "_orphan_reason": "unrelated_interest_credit",
    })

    # --- +3 DUPLICATE bank rows (clone from 3 of the 82 CLEAN_1_1 rows) ---
    dup_source_positions = rng.choice(len(clean_bank_rows), size=3, replace=False)
    dup_bank_meta = []
    for pos in dup_source_positions:
        src = clean_bank_rows[pos]
        bnkid = next_bnk_id()
        perturb = round(float(rng.uniform(-0.50, 0.50)), 2)
        day_shift = int(rng.choice([-1, 0, 1]))
        clone = {
            "bank_row_id": bnkid, "utr": src["utr"], "merchant_ref_code": src["merchant_ref_code"],
            "bank_credit_date": src["bank_credit_date"] + pd.Timedelta(days=day_shift),
            "credited_amount": round(src["credited_amount"] + perturb, 2),
            "bank_fee": src["bank_fee"], "bank_reported_txn_count": src["bank_reported_txn_count"],
            "narration": src["narration"],
            "_scenario": "DUPLICATE", "_duplicate_of": src["bank_row_id"],
        }
        bank_rows.append(clone)
        dup_bank_meta.append(clone)

    meta = {
        "match_groups": match_groups,
        "ledger_true_utr": ledger_true_utr,
        "ledger_true_group": ledger_true_group,
        "prior_period_bnkid": prior_period_bnkid,
        "interest_bnkid": interest_bnkid,
        "collision_bank_row_id": prior_period_bnkid,
        "dup_bank_meta": dup_bank_meta,
    }
    return bank_rows, meta


# ---------------------------------------------------------------------------
# Step 4: additive duplicate ledger rows
# ---------------------------------------------------------------------------

def _add_duplicate_ledger_rows(rows: list[dict], plan: dict, rng: np.random.Generator) -> list[dict]:
    dup_source_positions = rng.choice(len(plan["clean_idx"]), size=5, replace=False)
    dup_rows = []
    for pos in dup_source_positions:
        src_idx = plan["clean_idx"][pos]
        src = rows[src_idx]
        clone = dict(src)
        clone["scenario"] = "DUPLICATE"
        clone["utr_hint"] = ""
        clone["_duplicate_of_idx"] = src_idx
        dup_rows.append(clone)
    return dup_rows


# ---------------------------------------------------------------------------
# Assemble everything
# ---------------------------------------------------------------------------

def generate(seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = _base_rows(rng)
    plan = _allocate_scenarios(rows, rng)

    # utr_hint: ~85% of rows, drawn once per row, never re-drawn; blank for orphans.
    for row in rows:
        if row["scenario"] == "ORPHAN_LEDGER_TXN":
            row["utr_hint"] = ""
        else:
            row["_hint_draw"] = rng.random() < 0.85

    bank_rows, bank_meta = _build_bank_rows(rows, plan, rng)

    for i, row in enumerate(rows):
        if row.get("_hint_draw"):
            row["utr_hint"] = bank_meta["ledger_true_utr"].get(i, "")

    dup_ledger_rows = _add_duplicate_ledger_rows(rows, plan, rng)

    # Assign final sequential ledger_txn_ids: base rows sorted by (merchant_id, capture_date)
    # were already built in that order; ids follow build order (§1.2/1.4).
    for i, row in enumerate(rows):
        row["ledger_txn_id"] = f"LTX{i+1:06d}"
    base_id_by_pos = {i: rows[i]["ledger_txn_id"] for i in range(len(rows))}

    for j, row in enumerate(dup_ledger_rows):
        row["ledger_txn_id"] = f"LTX{len(rows)+j+1:06d}"
        row["duplicate_of_id"] = base_id_by_pos[row["_duplicate_of_idx"]]

    all_ledger_rows = rows + dup_ledger_rows

    ledger_df = pd.DataFrame([
        {
            "ledger_txn_id": r["ledger_txn_id"],
            "merchant_id": r["merchant_id"],
            "merchant_category": r["merchant_category"],
            "capture_date": r["capture_date"].strftime("%Y-%m-%d"),
            "gross_amount": r["gross_amount"],
            "razorpay_fee": r["razorpay_fee"],
            "tds_amount": r["tds_amount"],
            "expected_net_amount": r["expected_net_amount"],
            "expected_settlement_date": r["expected_settlement_date"].strftime("%Y-%m-%d"),
            "txn_status": r["txn_status"],
            "utr_hint": r["utr_hint"],
        }
        for r in all_ledger_rows
    ])

    bank_df = pd.DataFrame([
        {
            "bank_row_id": b["bank_row_id"],
            "utr": b["utr"],
            "merchant_ref_code": b["merchant_ref_code"],
            "bank_credit_date": b["bank_credit_date"].strftime("%Y-%m-%d"),
            "credited_amount": b["credited_amount"],
            "bank_fee": b["bank_fee"],
            "bank_reported_txn_count": b["bank_reported_txn_count"],
            "narration": b["narration"],
        }
        for b in bank_rows
    ])

    return rows, dup_ledger_rows, all_ledger_rows, bank_rows, plan, bank_meta, ledger_df, bank_df


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _build_ground_truth(seed, rows, dup_ledger_rows, all_ledger_rows, bank_rows, plan, bank_meta) -> dict:
    ledger_truth = {}
    for i, row in enumerate(rows):
        lid = row["ledger_txn_id"]
        if row["scenario"] == "ORPHAN_LEDGER_TXN":
            ledger_truth[lid] = {
                "true_label": "ORPHAN", "true_group_id": None, "true_utr": None,
                "duplicate_of": None, "orphan_reason": row["orphan_reason"],
            }
        else:
            gid = bank_meta["ledger_true_group"][i]
            ledger_truth[lid] = {
                "true_label": "MATCHED", "true_group_id": gid,
                "true_utr": bank_meta["ledger_true_utr"][i],
                "duplicate_of": None, "orphan_reason": None,
            }
    for row in dup_ledger_rows:
        ledger_truth[row["ledger_txn_id"]] = {
            "true_label": "DUPLICATE", "true_group_id": None, "true_utr": None,
            "duplicate_of": row["duplicate_of_id"], "orphan_reason": None,
        }

    bank_truth = {}
    for b in bank_rows:
        bid = b["bank_row_id"]
        scenario = b["_scenario"]
        if scenario == "DUPLICATE":
            bank_truth[bid] = {
                "true_label": "DUPLICATE", "true_group_id": None, "true_ledger_txn_ids": [],
                "duplicate_of": b["_duplicate_of"], "orphan_reason": None,
            }
        elif scenario == "ORPHAN_BANK_CREDIT":
            bank_truth[bid] = {
                "true_label": "ORPHAN", "true_group_id": None, "true_ledger_txn_ids": [],
                "duplicate_of": None, "orphan_reason": b["_orphan_reason"],
            }
        else:
            gid = next(g for g, v in bank_meta["match_groups"].items() if v["bank_row_id"] == bid)
            bank_truth[bid] = {
                "true_label": "MATCHED", "true_group_id": gid,
                "true_ledger_txn_ids": [rows[i]["ledger_txn_id"] for i in bank_meta["match_groups"][gid]["ledger_txn_ids"]],
                "duplicate_of": None, "orphan_reason": None,
            }

    match_groups_out = {}
    for gid, g in bank_meta["match_groups"].items():
        match_groups_out[gid] = {
            "utr": g["utr"], "bank_row_id": g["bank_row_id"],
            "ledger_txn_ids": [rows[i]["ledger_txn_id"] for i in g["ledger_txn_ids"]],
            "match_type": g["match_type"],
        }

    collision_idx = plan["amount_collision_ledger_idx"]
    trap_i, trap_j = plan["trap_batches"]
    planted_traps = [
        {
            "trap_type": "AMOUNT_COLLISION_NO_RELATION",
            "ledger_txn_ids": [rows[collision_idx]["ledger_txn_id"]],
            "bank_row_ids": [bank_meta["prior_period_bnkid"]],
            "note": "orphan ledger txn and orphan bank credit land within Rs.3 of each other; different merchants, must not be matched",
        },
        {
            "trap_type": "AMBIGUOUS_SUBSET_SUM",
            "ledger_txn_ids": [rows[i]["ledger_txn_id"] for i in trap_i["member_idx"] + trap_j["member_idx"]],
            "bank_row_ids": [trap_i["bank_row_id"], trap_j["bank_row_id"]],
            "note": "two disjoint subsets both sum within tolerance of the other batch's credited_amount",
        },
    ]

    summary_counts = {
        "total_ledger_rows": len(all_ledger_rows),
        "total_bank_rows": len(bank_rows),
        "clean_1_1": len(plan["clean_idx"]),
        "batch_net_members": sum(len(b["member_idx"]) for b in plan["batches"]),
        "batch_net_groups": len(plan["batches"]),
        "fee_drift_1_1": len(plan["fee_drift_idx"]),
        "timing_near_miss": len(plan["timing_idx"]),
        "orphan_ledger": len(plan["orphan_idx"]),
        "orphan_bank": 2,
        "duplicate_ledger": len(dup_ledger_rows),
        "duplicate_bank": len(bank_meta["dup_bank_meta"]),
        "traps": len(planted_traps),
    }

    return {
        "meta": {
            "seed": seed,
            "window_start": WINDOW_START.strftime("%Y-%m-%d"),
            "window_end": WINDOW_END.strftime("%Y-%m-%d"),
            # Deliberately NOT a wall-clock timestamp: the ledger/bank CSVs are
            # byte-identical across reruns of the same --seed, and this file
            # is no exception -- a live datetime.now() here would make the one
            # claimed-deterministic output silently not reproduce byte-for-byte.
            "generated_by_seed": seed,
        },
        "ledger_truth": ledger_truth,
        "bank_truth": bank_truth,
        "match_groups": match_groups_out,
        "planted_traps": planted_traps,
        "summary_counts": summary_counts,
    }


def _print_summary(ground_truth: dict) -> None:
    sc = ground_truth["summary_counts"]
    meta = ground_truth["meta"]
    print(f"Wrote {sc['total_ledger_rows']} ledger rows, {sc['total_bank_rows']} bank rows")
    print(f"Window: {meta['window_start']} -> {meta['window_end']} (seed={meta['seed']})")
    print("Scenario counts:")
    for k, v in sc.items():
        if k in ("total_ledger_rows", "total_bank_rows"):
            continue
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic reconciliation batch (ledger + bank feed + ground truth).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-ledger", type=Path, default=ROOT / "data" / "recon_ledger.csv")
    parser.add_argument("--out-bank", type=Path, default=ROOT / "data" / "recon_bank_settlement.csv")
    parser.add_argument("--out-truth", type=Path, default=ROOT / "data" / "recon_ground_truth.json")
    args = parser.parse_args()

    rows, dup_ledger_rows, all_ledger_rows, bank_rows, plan, bank_meta, ledger_df, bank_df = generate(seed=args.seed)
    ground_truth = _build_ground_truth(args.seed, rows, dup_ledger_rows, all_ledger_rows, bank_rows, plan, bank_meta)

    for p in (args.out_ledger, args.out_bank, args.out_truth):
        p.parent.mkdir(parents=True, exist_ok=True)

    ledger_df.to_csv(args.out_ledger, index=False)
    bank_df.to_csv(args.out_bank, index=False)
    with open(args.out_truth, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote {len(ledger_df)} rows -> {args.out_ledger}")
    print(f"Wrote {len(bank_df)} rows -> {args.out_bank}")
    print(f"Wrote ground truth -> {args.out_truth}")
    _print_summary(ground_truth)


if __name__ == "__main__":
    main()
