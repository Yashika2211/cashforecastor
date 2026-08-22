from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_DAYS = 265
START = pd.Timestamp("2025-11-01")
END = START + pd.Timedelta(days=N_DAYS - 1)
FESTIVAL_START = pd.Timestamp("2026-01-20")
FESTIVAL_END = pd.Timestamp("2026-01-29")
TDS_YTD_THRESHOLD = 500_000.0

# Settlement lag (days after capture). Food is T+1; marketplace is T+5; others T+2.
CATEGORY_SPEC = {
    "saas_subscription": {
        "settlement_lag": 2,
        "refund_rate": 0.003,
        "refund_lag": (0, 2),
        "chargeback_rate": 0.0004,
        "chargeback_lag": (10, 25),
        "noise": 0.028,
        "dow": np.array([1.07, 1.04, 1.02, 1.00, 0.98, 0.93, 0.90]),
        "festival_mult": 1.0,
    },
    "d2c_ecommerce": {
        "settlement_lag": 2,
        "refund_rate": 0.10,
        "refund_lag": (3, 10),
        "chargeback_rate": 0.0035,
        "chargeback_lag": (8, 21),
        "noise": 0.11,
        "dow": np.array([0.88, 0.90, 0.94, 0.98, 1.12, 1.48, 1.42]),
        "festival_mult": 3.1,
    },
    "marketplace": {
        "settlement_lag": 5,
        "refund_rate": 0.055,
        "refund_lag": (1, 6),
        "chargeback_rate": 0.022,
        "chargeback_lag": (7, 35),
        "noise": 0.32,
        "dow": np.array([1.02, 0.99, 1.01, 1.04, 1.08, 1.12, 0.96]),
        "festival_mult": 1.15,
    },
    "food_delivery": {
        "settlement_lag": 1,
        "refund_rate": 0.035,
        "refund_lag": (0, 1),
        "chargeback_rate": 0.0025,
        "chargeback_lag": (5, 18),
        "noise": 0.09,
        "dow": np.array([0.78, 0.84, 0.92, 1.02, 1.28, 1.82, 1.68]),
        "festival_mult": 1.05,
    },
}

MERCHANTS = [
    {"merchant_id": "mcht_saas_01", "merchant_category": "saas_subscription", "base": 95_000},
    {"merchant_id": "mcht_saas_02", "merchant_category": "saas_subscription", "base": 180_000},
    {"merchant_id": "mcht_saas_03", "merchant_category": "saas_subscription", "base": 310_000},
    {"merchant_id": "mcht_saas_04", "merchant_category": "saas_subscription", "base": 140_000},
    {"merchant_id": "mcht_saas_05", "merchant_category": "saas_subscription", "base": 72_000},
    {"merchant_id": "mcht_d2c_01", "merchant_category": "d2c_ecommerce", "base": 160_000},
    {"merchant_id": "mcht_d2c_02", "merchant_category": "d2c_ecommerce", "base": 240_000},
    {"merchant_id": "mcht_d2c_03", "merchant_category": "d2c_ecommerce", "base": 410_000},
    {"merchant_id": "mcht_d2c_04", "merchant_category": "d2c_ecommerce", "base": 125_000},
    {"merchant_id": "mcht_d2c_05", "merchant_category": "d2c_ecommerce", "base": 88_000},
    {"merchant_id": "mcht_mkt_01", "merchant_category": "marketplace", "base": 520_000},
    {"merchant_id": "mcht_mkt_02", "merchant_category": "marketplace", "base": 890_000},
    {"merchant_id": "mcht_mkt_03", "merchant_category": "marketplace", "base": 1_150_000},
    {"merchant_id": "mcht_mkt_04", "merchant_category": "marketplace", "base": 430_000},
    {"merchant_id": "mcht_food_01", "merchant_category": "food_delivery", "base": 110_000},
    {"merchant_id": "mcht_food_02", "merchant_category": "food_delivery", "base": 195_000},
    {"merchant_id": "mcht_food_03", "merchant_category": "food_delivery", "base": 265_000},
    {"merchant_id": "mcht_food_04", "merchant_category": "food_delivery", "base": 78_000},
]

# Recently onboarded: short series ending on the panel end date.
THIN_HISTORY = {
    "mcht_saas_05": 32,
    "mcht_d2c_05": 41,
    "mcht_food_04": 37,
}

# Hard step changes — not ramped.
REGIME_SHIFTS = {
    "mcht_d2c_02": {"on": pd.Timestamp("2026-04-01"), "mult": 2.15},
    "mcht_mkt_02": {"on": pd.Timestamp("2026-03-20"), "mult": 0.48},
}


def _gmv_series(merchant: dict, txn_dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    spec = CATEGORY_SPEC[merchant["merchant_category"]]
    dow_mult = spec["dow"][txn_dates.dayofweek]
    festival = np.where(
        (txn_dates >= FESTIVAL_START)
        & (txn_dates <= FESTIVAL_END)
        & (merchant["merchant_category"] == "d2c_ecommerce"),
        spec["festival_mult"],
        1.0,
    )
    # SaaS: small 1st-of-month billing bump on top of the weekly cycle.
    month_start = np.where(
        (merchant["merchant_category"] == "saas_subscription") & (txn_dates.day <= 2),
        1.16,
        1.0,
    )
    regime = np.ones(len(txn_dates))
    shift = REGIME_SHIFTS.get(merchant["merchant_id"])
    if shift is not None:
        regime = np.where(txn_dates >= shift["on"], shift["mult"], 1.0)

    noise = rng.normal(0.0, spec["noise"], size=len(txn_dates))
    gmv = merchant["base"] * dow_mult * festival * month_start * regime * np.clip(1.0 + noise, 0.15, None)
    return pd.Series(np.round(gmv, 2), index=txn_dates)


def _allocate_lagged(gmv: pd.Series, rate: float, lag_range: tuple[int, int], rng: np.random.Generator) -> pd.Series:
    """Book a fraction of each day's GMV onto a later date (refunds / chargebacks)."""
    out = pd.Series(0.0, index=gmv.index)
    lo, hi = lag_range
    for txn_date, amount in gmv.items():
        if amount <= 0:
            continue
        booked = float(amount * rate * rng.lognormal(0.0, 0.12))
        lag = int(rng.integers(lo, hi + 1))
        land = txn_date + pd.Timedelta(days=lag)
        if land in out.index:
            out.loc[land] += booked
    return out.round(2)


def _merchant_ledger(merchant: dict, rng: np.random.Generator) -> pd.DataFrame:
    spec = CATEGORY_SPEC[merchant["merchant_category"]]
    lag = spec["settlement_lag"]
    n_hist = THIN_HISTORY.get(merchant["merchant_id"], N_DAYS)
    settle_start = END - pd.Timedelta(days=n_hist - 1)
    settle_dates = pd.date_range(settle_start, END, freq="D")
    txn_start = settle_start - pd.Timedelta(days=lag)
    txn_dates = pd.date_range(txn_start, END, freq="D")

    gmv = _gmv_series(merchant, txn_dates, rng)
    refunds = _allocate_lagged(gmv, spec["refund_rate"], spec["refund_lag"], rng)
    chargebacks = _allocate_lagged(gmv, spec["chargeback_rate"], spec["chargeback_lag"], rng)

    rows = []
    ytd_gross = 0.0
    ytd_year = None
    for settle_date in settle_dates:
        capture_date = settle_date - pd.Timedelta(days=lag)
        gross = float(gmv.get(capture_date, 0.0))
        refund = float(refunds.get(settle_date, 0.0))
        chargeback = float(chargebacks.get(settle_date, 0.0))
        fee = round(gross * rng.uniform(0.018, 0.022), 2)

        if ytd_year != settle_date.year:
            ytd_year = settle_date.year
            ytd_gross = 0.0
        ytd_gross += gross
        tds = round(0.01 * gross, 2) if ytd_gross > TDS_YTD_THRESHOLD else 0.0

        net = round(gross - refund - chargeback - fee - tds, 2)
        rows.append(
            {
                "merchant_id": merchant["merchant_id"],
                "merchant_category": merchant["merchant_category"],
                "date": settle_date.strftime("%Y-%m-%d"),
                "gross_transaction_amount": round(gross, 2),
                "refund_amount": round(refund, 2),
                "chargeback_amount": round(chargeback, 2),
                "razorpay_fee": fee,
                "tds_hold": tds,
                "net_settled_amount": net,
            }
        )
    return pd.DataFrame(rows)


def generate_ledger(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [_merchant_ledger(merchant, rng) for merchant in MERCHANTS]
    return pd.concat(frames, ignore_index=True).sort_values(["date", "merchant_id"]).reset_index(drop=True)


def _print_summary(ledger: pd.DataFrame, path: Path) -> None:
    dates = pd.to_datetime(ledger["date"])
    counts = ledger.groupby("merchant_category")["merchant_id"].nunique()
    print(f"Wrote {len(ledger)} rows to {path}")
    print(f"Date range: {dates.min().date()} -> {dates.max().date()}")
    print("Merchant count per category:")
    for category, n in counts.items():
        print(f"  {category}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic daily merchant settlement ledger.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_ledger.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ledger = generate_ledger(seed=args.seed)
    ledger.to_csv(args.out, index=False)
    _print_summary(ledger, args.out)


if __name__ == "__main__":
    main()
