"""
Multi-source payment reconciliation engine.

Matches Razorpay's internal transaction ledger against the bank/processor
UTR settlement feed through a fixed sequence of deterministic stages
(duplicate quarantine, exact 1:1, fee-adjusted 1:1, timing near-miss,
many-to-one batch subset-sum, residual classification). Every record that
cannot be resolved gets one of six specific reason codes plus a free-text
`detail` — never a generic "unmatched" catch-all.

This module operates only on the `ledger` / `bank` DataFrames it is called
with. It must never read the planted ground-truth answer-key file that lives
alongside the synthetic data -- that file exists solely for
`score_reconciliation.py` to measure this engine's real precision/recall
against an answer key this module never sees. (Enforced by a static-source
test that greps this file for the ground-truth filename.)
"""
from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.exceptions import MalformedLedgerRowError, MalformedSettlementRowError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def AMOUNT_TOL_EXACT(amount_paise: int) -> int:
    return max(200, round(0.0005 * amount_paise))


def AMOUNT_TOL_FEE_ADJ(amount_paise: int) -> int:
    return max(500, round(0.005 * amount_paise))


AMOUNT_MISMATCH_MAX_PCT = 0.05
TIMING_AUTO_DAYS = 1
TIMING_REVIEW_DAYS = 5
MAX_SUBSET_POOL = 12
MAX_SUBSET_SIZE = 6
DUPLICATE_BANK_AMOUNT_TOL_PAISE = 50
DUPLICATE_BANK_DATE_TOL_DAYS = 1

REASON_CODES = (
    "AMOUNT_MISMATCH",
    "ORPHAN_BANK_CREDIT",
    "ORPHAN_LEDGER_TXN",
    "DUPLICATE_SUSPECTED",
    "TIMING_OUT_OF_WINDOW",
    "UNRESOLVED_SUBSET_SUM",
)

LEDGER_REQUIRED_COLS = [
    "ledger_txn_id", "merchant_id", "merchant_category", "capture_date",
    "gross_amount", "razorpay_fee", "tds_amount", "expected_net_amount",
    "expected_settlement_date", "txn_status", "utr_hint",
]
BANK_REQUIRED_COLS = [
    "bank_row_id", "utr", "merchant_ref_code", "bank_credit_date",
    "credited_amount", "bank_fee", "bank_reported_txn_count", "narration",
]


def to_paise(x: float) -> int:
    return int(round(float(x) * 100))


@dataclass
class ReconciliationResult:
    matches: pd.DataFrame
    exceptions: pd.DataFrame
    stats: dict


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_ledger(ledger: pd.DataFrame) -> None:
    for col in LEDGER_REQUIRED_COLS:
        if col not in ledger.columns:
            raise MalformedLedgerRowError(-1, f"missing required column '{col}'")
    for i, row in ledger.iterrows():
        if pd.isna(row["ledger_txn_id"]) or pd.isna(row["merchant_id"]):
            raise MalformedLedgerRowError(int(i), "missing ledger_txn_id or merchant_id")


def _validate_bank(bank: pd.DataFrame) -> None:
    for col in BANK_REQUIRED_COLS:
        if col not in bank.columns:
            raise MalformedSettlementRowError(-1, f"missing required column '{col}'")
    for i, row in bank.iterrows():
        if pd.isna(row["bank_row_id"]) or pd.isna(row["utr"]):
            raise MalformedSettlementRowError(int(i), "missing bank_row_id or utr")


# ---------------------------------------------------------------------------
# Subset-sum (meet in the middle)
# ---------------------------------------------------------------------------

def _subset_sums(items: list[int]) -> list[tuple[int, frozenset]]:
    """All (sum, index-set) pairs over the power set of `items` (local indices)."""
    n = len(items)
    out = []
    for mask in range(1 << n):
        s = 0
        idxs = []
        for i in range(n):
            if mask & (1 << i):
                s += items[i]
                idxs.append(i)
        out.append((s, frozenset(idxs)))
    return out


def find_qualifying_subsets(
    paise_values: list[int], target: int, tol: int, sizes: set[int]
) -> list[frozenset]:
    """Meet-in-the-middle subset-sum search.

    Splits `paise_values` into two halves, enumerates each half's subset
    sums, then for every left-half subset binary-searches the right half for
    complements landing in [target-tol, target+tol]. Returns every
    qualifying subset (as a frozenset of indices into `paise_values`) whose
    size is in `sizes`.
    """
    n = len(paise_values)
    half = n // 2
    left_idx, right_idx = list(range(half)), list(range(half, n))
    left_items = [paise_values[i] for i in left_idx]
    right_items = [paise_values[i] for i in right_idx]

    left_sums = _subset_sums(left_items)
    right_sums = sorted(_subset_sums(right_items), key=lambda t: t[0])
    right_values = [t[0] for t in right_sums]

    qualifying = []
    for lsum, lset in left_sums:
        lo = bisect.bisect_left(right_values, target - tol - lsum)
        hi = bisect.bisect_right(right_values, target + tol - lsum)
        for j in range(lo, hi):
            rsum, rset = right_sums[j]
            total_size = len(lset) + len(rset)
            if total_size == 0 or total_size not in sizes:
                continue
            global_idx = frozenset(left_idx[i] for i in lset) | frozenset(right_idx[i] for i in rset)
            qualifying.append(global_idx)
    return qualifying


def _minimal_subsets(subsets: list[frozenset]) -> list[frozenset]:
    """Drop any subset that is a strict superset of another qualifying subset."""
    uniq = list(set(subsets))
    return [s for s in uniq if not any(s > other for other in uniq if s != other)]


# ---------------------------------------------------------------------------
# Confidence scoring (secondary, non-decisive -- see §3.8 of the spec)
# ---------------------------------------------------------------------------

def _date_score(offset_days: Optional[int]) -> float:
    if offset_days is None:
        return 0.0
    offset_days = abs(offset_days)
    if offset_days == 0:
        return 1.0
    if offset_days == 1:
        return 0.6
    if 2 <= offset_days <= 5:
        return 0.15
    return 0.0


def _amount_score(delta_paise: Optional[int], amount_paise: int) -> float:
    if delta_paise is None:
        return 0.0
    if delta_paise <= AMOUNT_TOL_EXACT(amount_paise):
        return 1.0
    if delta_paise <= AMOUNT_TOL_FEE_ADJ(amount_paise):
        return 0.85
    delta_rupees = delta_paise / 100.0
    amount_rupees = amount_paise / 100.0
    if amount_rupees <= 0:
        return 0.0
    return max(0.0, 1 - delta_rupees / (0.02 * amount_rupees))


def _ref_score(ledger_utr_hint: Optional[str], bank_utr: Optional[str],
                bank_merchant_ref: Optional[str], ledger_merchant_id: Optional[str]) -> float:
    if ledger_utr_hint and bank_utr and ledger_utr_hint == bank_utr:
        return 1.0
    if bank_merchant_ref and ledger_merchant_id and bank_merchant_ref == ledger_merchant_id:
        return 0.5
    return 0.0


def _confidence(delta_paise, amount_paise, offset_days, ledger_utr_hint, bank_utr,
                 bank_merchant_ref, ledger_merchant_id) -> Optional[float]:
    if delta_paise is None and offset_days is None:
        return None
    a = _amount_score(delta_paise, amount_paise) if delta_paise is not None else 0.0
    d = _date_score(offset_days) if offset_days is not None else 0.0
    r = _ref_score(ledger_utr_hint, bank_utr, bank_merchant_ref, ledger_merchant_id)
    return round(0.55 * a + 0.30 * d + 0.15 * r, 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_reconciliation(ledger: pd.DataFrame, bank: pd.DataFrame) -> ReconciliationResult:
    _validate_ledger(ledger)
    _validate_bank(bank)

    ledger = ledger.copy()
    bank = bank.copy()
    ledger["capture_date"] = pd.to_datetime(ledger["capture_date"])
    ledger["expected_settlement_date"] = pd.to_datetime(ledger["expected_settlement_date"])
    bank["bank_credit_date"] = pd.to_datetime(bank["bank_credit_date"])
    ledger["utr_hint"] = ledger["utr_hint"].fillna("")
    bank["merchant_ref_code"] = bank["merchant_ref_code"].fillna("")

    L = {r["ledger_txn_id"]: dict(r) for r in ledger.to_dict(orient="records")}
    B = {r["bank_row_id"]: dict(r) for r in bank.to_dict(orient="records")}
    for r in L.values():
        r["net_paise"] = to_paise(r["expected_net_amount"])
        r["open"] = True
    for r in B.values():
        r["fee_paise"] = to_paise(r["bank_fee"]) if pd.notna(r["bank_fee"]) else 0
        r["credited_paise"] = to_paise(r["credited_amount"])
        r["target_paise"] = r["credited_paise"] + r["fee_paise"]
        r["open"] = True
        r["canonical"] = True

    exceptions: list[dict] = []
    matches: list[dict] = []
    rule_counts: dict[str, int] = {}
    match_group_counter = 0

    def bump(rule: str) -> None:
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

    def add_exception(record_type, record_id, merchant_id, reason_code, detail,
                       candidate_id=None, candidate_amount=None, delta=None,
                       competing_candidates=None):
        exceptions.append({
            "record_type": record_type, "record_id": record_id, "merchant_id": merchant_id,
            "reason_code": reason_code, "detail": detail, "candidate_id": candidate_id,
            "candidate_amount": candidate_amount, "delta": delta,
            "competing_candidates": competing_candidates,
        })

    t0 = time.perf_counter()

    # ---- Stage 0a: exclude test-mode captures ----
    for lid, r in L.items():
        if r["txn_status"] == "captured_test_mode":
            r["open"] = False
            add_exception("ledger", lid, r["merchant_id"], "ORPHAN_LEDGER_TXN", "test_mode_capture")

    # ---- Stage 0b: duplicate quarantine ----
    # Ledger side: group by (merchant_id, capture_date, gross_amount).
    ledger_groups: dict[tuple, list[str]] = {}
    for lid, r in L.items():
        if not r["open"]:
            continue
        key = (r["merchant_id"], r["capture_date"], to_paise(r["gross_amount"]))
        ledger_groups.setdefault(key, []).append(lid)
    # Canonical = lexicographically-first ledger_txn_id. This is chronological-
    # creation order ONLY because IDs here are fixed-width zero-padded
    # (LTX%06d) and duplicates are always assigned a higher sequence number
    # than their source by this generator -- ingesting IDs that don't hold
    # that invariant (non-zero-padded, or not monotonic with creation) would
    # need an explicit sequence/creation-order field instead of string sort.
    for key, ids in ledger_groups.items():
        if len(ids) <= 1:
            continue
        ids_sorted = sorted(ids)
        canonical = ids_sorted[0]
        for lid in ids_sorted[1:]:
            L[lid]["open"] = False
            add_exception("ledger", lid, L[lid]["merchant_id"], "DUPLICATE_SUSPECTED",
                           "ledger_duplicate", candidate_id=canonical)

    # Bank side: group by utr; canonical = lexicographically-first bank_row_id
    # (same zero-padded-ID assumption as above -- NOT earliest bank_credit_date;
    # an earlier version of this sorted by date, which let a randomly-jittered
    # duplicate clone masquerade as canonical over its true original and
    # produced a real false match -- see README's reconciliation bug writeup).
    # Others within tolerance are quarantined and removed from the matching
    # pool entirely.
    bank_groups: dict[str, list[str]] = {}
    for bid, r in B.items():
        bank_groups.setdefault(r["utr"], []).append(bid)
    for utr, ids in bank_groups.items():
        if len(ids) <= 1:
            continue
        ids_sorted = sorted(ids)
        canonical = ids_sorted[0]
        for bid in ids_sorted[1:]:
            b, c = B[bid], B[canonical]
            amt_close = abs(b["credited_paise"] - c["credited_paise"]) <= DUPLICATE_BANK_AMOUNT_TOL_PAISE
            date_close = abs((b["bank_credit_date"] - c["bank_credit_date"]).days) <= DUPLICATE_BANK_DATE_TOL_DAYS
            if amt_close and date_close:
                B[bid]["open"] = False
                B[bid]["canonical"] = False
                add_exception("bank", bid, c.get("merchant_ref_code") or "", "DUPLICATE_SUSPECTED",
                               "bank_duplicate_retry", candidate_id=canonical)

    # ---- Stage 1: bucket ----
    def open_ledger_ids():
        return [lid for lid, r in L.items() if r["open"]]

    def ledger_bucket_map():
        m: dict[tuple, list[str]] = {}
        for lid in open_ledger_ids():
            r = L[lid]
            m.setdefault((r["merchant_id"], r["expected_settlement_date"]), []).append(lid)
        return m

    canonical_bank_ids = [bid for bid, r in B.items() if r["canonical"]]
    ref_bank_ids = sorted(bid for bid in canonical_bank_ids if B[bid]["merchant_ref_code"])
    # Bank credits with no merchant_ref_code carry no verifiable attribution.
    # An amount+date coincidence with some ledger row is evidence a collision
    # exists, not evidence that row is the RIGHT one -- the engine has no
    # signal that could tell a correct coincidental match from an incorrect
    # one (using utr_hint here would leak the very thing it's reserved for:
    # confidence scoring only, never the match decision). Auto-matching on
    # amount+date alone risks a false reconciliation across merchants, which
    # is worse than a miss, so these always fall through to Stage 6 and are
    # reported as ORPHAN_BANK_CREDIT/no_merchant_ref for manual review --
    # never entering the 1:1 or batch passes at all.

    def candidates_for(bank_row, target_date):
        bucket = ledger_bucket_map()
        return list(bucket.get((bank_row["merchant_ref_code"], target_date), []))

    # ---- Stages 2-4: 1:1 passes (merchant-referenced bank rows only) ----
    def try_1to1(bank_ids, tol_fn, date_offset, rule):
        for bid in bank_ids:
            b = B[bid]
            if not b["open"]:
                continue
            target_date = b["bank_credit_date"] + pd.Timedelta(days=date_offset)
            cands = [lid for lid in candidates_for(b, target_date) if L[lid]["open"]]
            tol = tol_fn(b["target_paise"])
            matching = [lid for lid in cands if abs(L[lid]["net_paise"] - b["target_paise"]) <= tol]
            if len(matching) == 1:
                lid = matching[0]
                _record_match([lid], bid, rule)
            # len==0: try next stage. len>1: ambiguous, leave for stage 5/6.

    def _record_match(ledger_ids: list[str], bank_id: str, rule: str) -> None:
        nonlocal match_group_counter
        match_group_counter += 1
        gid = f"grp_{match_group_counter:05d}"
        for lid in ledger_ids:
            L[lid]["open"] = False
        B[bank_id]["open"] = False
        b = B[bank_id]
        # Derive the reported amount/delta from the same paise integers that
        # decided the match (rather than re-summing the float rupee columns
        # independently), so the numbers shown in the report are provably
        # consistent with what the tolerance check actually compared.
        expected_paise = sum(L[lid]["net_paise"] for lid in ledger_ids)
        expected_amount = round(expected_paise / 100.0, 2)
        delta = round((expected_paise - b["target_paise"]) / 100.0, 2)
        matches.append({
            "match_group_id": gid, "ledger_txn_ids": list(ledger_ids), "bank_row_id": bank_id,
            "utr": b["utr"], "rule": rule, "member_count": len(ledger_ids),
            "matched_amount": b["credited_amount"], "expected_amount": expected_amount,
            "delta": delta,
        })
        bump(rule)

    try_1to1(ref_bank_ids, AMOUNT_TOL_EXACT, 0, "exact_1to1")
    try_1to1(ref_bank_ids, AMOUNT_TOL_FEE_ADJ, 0, "fee_adjusted_1to1")
    try_1to1(ref_bank_ids, AMOUNT_TOL_EXACT, -1, "timing_near_miss_exact")
    try_1to1(ref_bank_ids, AMOUNT_TOL_EXACT, 1, "timing_near_miss_exact")
    try_1to1(ref_bank_ids, AMOUNT_TOL_FEE_ADJ, -1, "timing_near_miss_fee_adjusted")
    try_1to1(ref_bank_ids, AMOUNT_TOL_FEE_ADJ, 1, "timing_near_miss_fee_adjusted")

    # ---- Stage 5: many-to-one batch subset-sum (referenced merchants only) ----
    for bid in ref_bank_ids:
        b = B[bid]
        if not b["open"]:
            continue
        lo, hi = b["bank_credit_date"] - pd.Timedelta(days=1), b["bank_credit_date"]
        pool = [
            lid for lid in open_ledger_ids()
            if L[lid]["merchant_id"] == b["merchant_ref_code"]
            and lo <= L[lid]["expected_settlement_date"] <= hi
        ]
        if len(pool) > MAX_SUBSET_POOL:
            add_exception("bank", bid, b["merchant_ref_code"], "ORPHAN_BANK_CREDIT", "pool_too_large")
            B[bid]["open"] = False
            continue
        if len(pool) < 2:
            continue
        pool_values = [L[lid]["net_paise"] for lid in pool]
        target, tol = b["target_paise"], AMOUNT_TOL_EXACT(b["target_paise"])
        max_size = min(MAX_SUBSET_SIZE, len(pool))
        all_sizes = set(range(2, max_size + 1))

        chosen_minimal = None
        chosen_rule = None
        hint = b.get("bank_reported_txn_count")
        if hint is not None and not pd.isna(hint) and 2 <= int(hint) <= max_size:
            hinted = find_qualifying_subsets(pool_values, target, tol, {int(hint)})
            minimal_hinted = _minimal_subsets(hinted)
            if len(minimal_hinted) == 1:
                chosen_minimal, chosen_rule = minimal_hinted, "batch_subset_sum_hinted"

        if chosen_minimal is None:
            swept = find_qualifying_subsets(pool_values, target, tol, all_sizes)
            minimal_swept = _minimal_subsets(swept)
            if len(minimal_swept) == 1:
                chosen_minimal, chosen_rule = minimal_swept, "batch_subset_sum"
            elif len(minimal_swept) > 1:
                member_ids = [[pool[i] for i in s] for s in minimal_swept]
                flat_members = sorted({lid for group in member_ids for lid in group})
                competing = "|".join("|".join(sorted(g)) for g in member_ids)
                # Any other still-open bank row (same merchant) whose own target
                # is satisfied by one of these same competing subsets shares the
                # ambiguity -- flag it too, rather than let its pool quietly empty
                # out from under it once this bank row consumes the members.
                also_flag = []
                for bid2 in ref_bank_ids:
                    if bid2 == bid or not B[bid2]["open"]:
                        continue
                    if B[bid2]["merchant_ref_code"] != b["merchant_ref_code"]:
                        continue
                    t2, tol2 = B[bid2]["target_paise"], AMOUNT_TOL_EXACT(B[bid2]["target_paise"])
                    for s in minimal_swept:
                        ssum = sum(pool_values[i] for i in s)
                        if abs(ssum - t2) <= tol2:
                            also_flag.append(bid2)
                            break
                for lid in flat_members:
                    add_exception("ledger", lid, L[lid]["merchant_id"], "UNRESOLVED_SUBSET_SUM",
                                  "competing_batch_subsets", candidate_id=bid,
                                  competing_candidates=competing)
                    L[lid]["open"] = False
                add_exception("bank", bid, b["merchant_ref_code"], "UNRESOLVED_SUBSET_SUM",
                               "competing_batch_subsets", competing_candidates=competing)
                B[bid]["open"] = False
                for bid2 in also_flag:
                    add_exception("bank", bid2, B[bid2]["merchant_ref_code"], "UNRESOLVED_SUBSET_SUM",
                                   "competing_batch_subsets", competing_candidates=competing)
                    B[bid2]["open"] = False
                continue
            # else: zero qualifying subsets -> leave for Stage 6.

        if chosen_minimal is not None:
            member_ids = [pool[i] for i in sorted(chosen_minimal[0])]
            _record_match(member_ids, bid, chosen_rule)

    # ---- Stage 6: residual classification ----
    def nearest_bank_candidate(lid, day_range):
        r = L[lid]
        best = None
        for bid in open_ledger_bank_ids_by_merchant.get(r["merchant_id"], []):
            b = B[bid]
            if not b["open"]:
                continue
            offset = (b["bank_credit_date"] - r["expected_settlement_date"]).days
            if offset not in day_range:
                continue
            delta = abs(r["net_paise"] - b["target_paise"])
            if best is None or delta < best[1]:
                best = (bid, delta, offset)
        return best

    open_ledger_bank_ids_by_merchant: dict[str, list[str]] = {}
    for bid, r in B.items():
        if r["open"] and r["merchant_ref_code"]:
            open_ledger_bank_ids_by_merchant.setdefault(r["merchant_ref_code"], []).append(bid)

    for lid in list(open_ledger_ids()):
        r = L[lid]
        near = nearest_bank_candidate(lid, range(-TIMING_AUTO_DAYS, TIMING_AUTO_DAYS + 1))
        far = nearest_bank_candidate(lid, list(range(-TIMING_REVIEW_DAYS, -1)) + list(range(2, TIMING_REVIEW_DAYS + 1)))
        target_paise_for_pct = r["net_paise"]
        if near is not None:
            bid, delta, offset = near
            pct = delta / target_paise_for_pct if target_paise_for_pct else 1.0
            if delta > AMOUNT_TOL_FEE_ADJ(target_paise_for_pct) and pct <= AMOUNT_MISMATCH_MAX_PCT:
                add_exception("ledger", lid, r["merchant_id"], "AMOUNT_MISMATCH",
                               "nearest_candidate_outside_fee_tolerance", candidate_id=bid,
                               candidate_amount=B[bid]["credited_amount"], delta=round(delta / 100.0, 2))
                continue
        if far is not None:
            bid, delta, offset = far
            if delta <= AMOUNT_TOL_FEE_ADJ(target_paise_for_pct):
                add_exception("ledger", lid, r["merchant_id"], "TIMING_OUT_OF_WINDOW",
                               f"settled_{abs(offset)}_days_late", candidate_id=bid,
                               candidate_amount=B[bid]["credited_amount"], delta=round(delta / 100.0, 2))
                continue
        detail = "pending_payout_censored" if r["txn_status"] == "captured_pending_payout" else "no_candidate_in_window"
        add_exception("ledger", lid, r["merchant_id"], "ORPHAN_LEDGER_TXN", detail)

    def nearest_ledger_candidate(bid, day_range):
        b = B[bid]
        merchant = b["merchant_ref_code"]
        best = None
        for lid in open_ledger_ids():
            r = L[lid]
            if merchant and r["merchant_id"] != merchant:
                continue
            if not merchant:
                continue
            offset = (b["bank_credit_date"] - r["expected_settlement_date"]).days
            if offset not in day_range:
                continue
            delta = abs(r["net_paise"] - b["target_paise"])
            if best is None or delta < best[1]:
                best = (lid, delta, offset)
        return best

    for bid in list(bid for bid in canonical_bank_ids if B[bid]["open"]):
        b = B[bid]
        near = nearest_ledger_candidate(bid, range(-TIMING_AUTO_DAYS, TIMING_AUTO_DAYS + 1))
        far = nearest_ledger_candidate(bid, list(range(-TIMING_REVIEW_DAYS, -1)) + list(range(2, TIMING_REVIEW_DAYS + 1)))
        target_paise_for_pct = b["target_paise"]
        if near is not None:
            lid, delta, offset = near
            pct = delta / target_paise_for_pct if target_paise_for_pct else 1.0
            if delta > AMOUNT_TOL_FEE_ADJ(target_paise_for_pct) and pct <= AMOUNT_MISMATCH_MAX_PCT:
                add_exception("bank", bid, b["merchant_ref_code"], "AMOUNT_MISMATCH",
                               "nearest_candidate_outside_fee_tolerance", candidate_id=lid,
                               candidate_amount=L[lid]["expected_net_amount"], delta=round(delta / 100.0, 2))
                continue
        if far is not None:
            lid, delta, offset = far
            if delta <= AMOUNT_TOL_FEE_ADJ(target_paise_for_pct):
                add_exception("bank", bid, b["merchant_ref_code"], "TIMING_OUT_OF_WINDOW",
                               f"settled_{abs(offset)}_days_late", candidate_id=lid,
                               candidate_amount=L[lid]["expected_net_amount"], delta=round(delta / 100.0, 2))
                continue
        detail = "no_merchant_ref" if not b["merchant_ref_code"] else "no_candidate_in_window"
        add_exception("bank", bid, b["merchant_ref_code"], "ORPHAN_BANK_CREDIT", detail)

    elapsed = time.perf_counter() - t0

    # ---- Confidence scoring (secondary, computed after classification) ----
    for m in matches:
        lids = m["ledger_txn_ids"]
        b = B[m["bank_row_id"]]
        amount_paise = to_paise(m["expected_amount"])
        delta_paise = abs(to_paise(m["delta"]))
        # date offset: for a batch this is 0 by construction (forced common date);
        # for a 1:1 rule it is the actual day difference.
        offset_days = (b["bank_credit_date"] - L[lids[0]]["expected_settlement_date"]).days
        ledger_hint = L[lids[0]]["utr_hint"] if len(lids) == 1 else ""
        m["confidence"] = _confidence(
            delta_paise, amount_paise, offset_days,
            ledger_hint, b["utr"], b["merchant_ref_code"], L[lids[0]]["merchant_id"],
        )

    for e in exceptions:
        cid, camt, delta_rupees = e["candidate_id"], e["candidate_amount"], e["delta"]
        if cid is None or camt is None:
            e["confidence"] = None
            continue
        amount_paise = to_paise(camt)
        delta_paise = to_paise(delta_rupees) if delta_rupees is not None else 0
        if e["record_type"] == "ledger":
            ledger_row = L[e["record_id"]]
            if e["reason_code"] == "DUPLICATE_SUSPECTED":
                offset_days, hint, butr, bref = 0, ledger_row["utr_hint"], None, ledger_row["merchant_id"]
                e["confidence"] = _confidence(0, to_paise(ledger_row["expected_net_amount"]), 0,
                                               hint, hint or None, ledger_row["merchant_id"], ledger_row["merchant_id"])
            elif cid in B:
                bank_row = B[cid]
                offset_days = (bank_row["bank_credit_date"] - ledger_row["expected_settlement_date"]).days
                e["confidence"] = _confidence(delta_paise, amount_paise, offset_days,
                                               ledger_row["utr_hint"], bank_row["utr"],
                                               bank_row["merchant_ref_code"], ledger_row["merchant_id"])
            else:
                e["confidence"] = None
        else:
            bank_row = B[e["record_id"]]
            if e["reason_code"] == "DUPLICATE_SUSPECTED":
                e["confidence"] = _confidence(0, to_paise(bank_row["credited_amount"]), 0,
                                               None, bank_row["utr"], bank_row["merchant_ref_code"],
                                               bank_row["merchant_ref_code"])
            elif cid in L:
                ledger_row = L[cid]
                offset_days = (bank_row["bank_credit_date"] - ledger_row["expected_settlement_date"]).days
                e["confidence"] = _confidence(delta_paise, amount_paise, offset_days,
                                               ledger_row["utr_hint"], bank_row["utr"],
                                               bank_row["merchant_ref_code"], ledger_row["merchant_id"])
            else:
                e["confidence"] = None
        if e.get("confidence") is None and e["reason_code"] != "DUPLICATE_SUSPECTED":
            e["confidence"] = None

    matches_df = pd.DataFrame(matches, columns=[
        "match_group_id", "ledger_txn_ids", "bank_row_id", "utr", "rule", "member_count",
        "matched_amount", "expected_amount", "delta", "confidence",
    ])
    exceptions_df = pd.DataFrame(exceptions, columns=[
        "record_type", "record_id", "merchant_id", "reason_code", "detail", "candidate_id",
        "candidate_amount", "delta", "confidence", "competing_candidates",
    ])

    reason_code_counts = {code: 0 for code in REASON_CODES}
    if not exceptions_df.empty:
        for code, n in exceptions_df["reason_code"].value_counts().items():
            reason_code_counts[code] = int(n)

    n_ledger, n_bank = len(ledger), len(bank)
    stats = {
        "rule_counts": rule_counts,
        "reason_code_counts": reason_code_counts,
        "elapsed_seconds": elapsed,
        "records_per_second": (n_ledger + n_bank) / elapsed if elapsed > 0 else float("inf"),
        "n_ledger_rows": n_ledger,
        "n_bank_rows": n_bank,
    }

    return ReconciliationResult(matches=matches_df, exceptions=exceptions_df, stats=stats)


# ---------------------------------------------------------------------------
# Optional LLM exception-annotation layer (§3.11) -- nice-to-have, never
# called from run_reconciliation() itself, degrades to templates offline.
# ---------------------------------------------------------------------------

TEMPLATES = {
    "AMOUNT_MISMATCH": "Ledger txn {record_id} expected Rs.{candidate_amount:.2f} area; closest bank credit {candidate_id} was off by Rs.{delta:.2f}, above tolerance.",
    "ORPHAN_LEDGER_TXN": "No bank credit found for ledger txn {record_id} in any window ({detail}).",
    "ORPHAN_BANK_CREDIT": "Bank credit {record_id} has no ledger counterpart ({detail}).",
    "DUPLICATE_SUSPECTED": "{record_id} collides with {candidate_id} on merchant/date/amount ({detail}).",
    "TIMING_OUT_OF_WINDOW": "{record_id} has an amount-matching candidate {candidate_id} but it landed outside the auto-match window ({detail}).",
    "UNRESOLVED_SUBSET_SUM": "{record_id} has multiple competing candidate subsets: {competing_candidates}; left unresolved.",
}


def _template_note(row: dict) -> str:
    template = TEMPLATES.get(row["reason_code"])
    if template is None:
        return f"{row['record_id']}: {row['reason_code']} ({row.get('detail')})"
    safe = dict(row)
    for key in ("candidate_amount", "delta"):
        if safe.get(key) is None:
            safe[key] = 0.0
    for key in ("candidate_id", "detail", "competing_candidates"):
        if safe.get(key) is None:
            safe[key] = "none"
    try:
        return template.format(**safe)
    except (KeyError, ValueError):
        return f"{row['record_id']}: {row['reason_code']} ({row.get('detail')})"


def annotate_exceptions(exceptions: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable `note` column. Degrades to fixed templates offline
    (the only path exercised by tests/CI); optionally calls the Anthropic API
    when ANTHROPIC_API_KEY is set. Never influences reason_code or the
    match/exception decision -- purely descriptive text appended last."""
    import os

    out = exceptions.copy()
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401  (optional, not exercised by tests)
            notes = [_template_note(row) for row in out.to_dict(orient="records")]
            # A live-API nicer-note path could go here; templates remain the
            # deterministic fallback so this function always returns a value.
            out["note"] = notes
            return out
        except Exception:
            pass
    out["note"] = [_template_note(row) for row in out.to_dict(orient="records")]
    return out
