from __future__ import annotations

import pandas as pd

from src.exceptions import InvalidLedgerError

LEDGER_COLUMNS = (
    "merchant_id",
    "merchant_category",
    "date",
    "gross_transaction_amount",
    "refund_amount",
    "chargeback_amount",
    "razorpay_fee",
    "tds_hold",
    "net_settled_amount",
)
LAGS = (1, 7, 14)
ROLL_WINDOWS = (7, 28)
MIN_HISTORY_DAYS = max(LAGS) + 1
TARGET_COLUMN = "net_settled_amount"

LAG_COLUMNS = [f"net_settled_amount_lag_{lag}" for lag in LAGS]
ROLL_COLUMNS = [
    *(f"net_settled_amount_roll_mean_{w}" for w in ROLL_WINDOWS),
    *(f"net_settled_amount_roll_std_{w}" for w in ROLL_WINDOWS),
]
FEATURE_COLUMNS = [
    *LAG_COLUMNS,
    *ROLL_COLUMNS,
    "day_of_week",
    "is_weekend",
    "is_month_end",
    "days_since_first_transaction",
    "merchant_category",
]


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in LEDGER_COLUMNS if c not in df.columns]
    if missing:
        raise InvalidLedgerError(f"Ledger missing columns: {missing}")


def build_features(df: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    _require_columns(df)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["merchant_id", "date"]).reset_index(drop=True)

    grouped = out.groupby("merchant_id", sort=False)
    net = grouped["net_settled_amount"]

    for lag in LAGS:
        out[f"net_settled_amount_lag_{lag}"] = net.shift(lag)

    # Shift first so the rolling window never includes the same-day target.
    prior_net = net.shift(1)
    rolled = prior_net.groupby(out["merchant_id"], sort=False)
    for window in ROLL_WINDOWS:
        # 28-day window uses min_periods=7 so recently onboarded merchants still get a row
        # after the 14-day lag drop, instead of being wiped out by a full 28-day requirement.
        min_periods = 7 if window == 28 else window
        rolling = rolled.rolling(window, min_periods=min_periods)
        out[f"net_settled_amount_roll_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        out[f"net_settled_amount_roll_std_{window}"] = rolling.std().reset_index(level=0, drop=True)

    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    month_end = out["date"] + pd.offsets.MonthEnd(0)
    out["is_month_end"] = ((month_end - out["date"]).dt.days <= 2).astype(int)
    out["days_since_first_transaction"] = grouped["date"].transform(lambda s: (s - s.min()).dt.days)
    out["merchant_category"] = out["merchant_category"].astype("category")

    if drop_incomplete:
        out = out.dropna(subset=LAG_COLUMNS)
    return out.reset_index(drop=True)
