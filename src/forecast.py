"""
Quantile cash-flow forecaster.

Three LightGBM models (objective='quantile') at alpha=0.1, 0.5, 0.9 predict
net_settled_amount 14 days ahead using features from features.py.

Walk-forward backtest:
  - Start from the first date that has enough training history.
  - For each fold, train on [start, cutoff), forecast the next 14 days, compare
    to actuals, then roll the cutoff forward by 14 days.
  - Repeat until fewer than 14 actual days remain.

Metrics per fold (and aggregated):
  - Pinball loss at each quantile.
  - Coverage: fraction of actuals inside [P10, P90] (target ~80%).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.exceptions import (
    InsufficientHistoryError,
    MerchantNotFoundError,
    ModelNotTrainedError,
)
from src.features import FEATURE_COLUMNS, MIN_HISTORY_DAYS, TARGET_COLUMN, build_features

QUANTILES = (0.1, 0.5, 0.9)
HORIZON = 14
# Minimum training rows (across all merchants) before the first backtest fold.
MIN_TRAIN_ROWS = 500
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

_LGB_BASE = {
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}


def _lgb_params(alpha: float) -> dict:
    return {**_LGB_BASE, "objective": "quantile", "alpha": alpha}


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss."""
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def _coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    """Fraction of actuals that fall inside [P10, P90]."""
    inside = (y_true >= p10) & (y_true <= p90)
    return float(inside.mean())


# ---------------------------------------------------------------------------
# Core quantile models wrapper
# ---------------------------------------------------------------------------

@dataclass
class QuantileForecaster:
    """Holds three trained LightGBM quantile models."""

    models: Dict[float, lgb.Booster] = field(default_factory=dict)
    daily_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        ledger: pd.DataFrame,
        num_boost_round: int = 300,
        cutoff_date: pd.Timestamp | None = None,
    ) -> "QuantileForecaster":
        """
        Train all three quantile models on the ledger data.

        If cutoff_date is provided, training uses only rows up to that date
        (inclusive). The full ledger is stored as daily_history for inference.
        """
        self.daily_history = ledger.copy()
        self.daily_history["date"] = pd.to_datetime(self.daily_history["date"])

        train_ledger = ledger.copy()
        if cutoff_date is not None:
            train_ledger = train_ledger[
                pd.to_datetime(train_ledger["date"]) <= cutoff_date
            ]

        featured = build_features(train_ledger, drop_incomplete=True)
        if featured.empty or len(featured) < MIN_TRAIN_ROWS:
            raise InsufficientHistoryError(
                "global", len(featured), MIN_TRAIN_ROWS
            )

        X = featured[self.feature_names]
        y = featured[TARGET_COLUMN].values

        dataset = lgb.Dataset(
            X, label=y, categorical_feature=["merchant_category"], free_raw_data=False
        )
        for alpha in QUANTILES:
            self.models[alpha] = lgb.train(
                _lgb_params(alpha),
                dataset,
                num_boost_round=num_boost_round,
            )
        return self

    def save(self, directory: Path = MODELS_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for alpha, model in self.models.items():
            label = int(alpha * 100)
            model.save_model(str(directory / f"lgb_q{label:02d}.txt"))
        # Also pickle the whole object for easy reload (without history for size).
        stub = QuantileForecaster(
            models=self.models,
            daily_history=self.daily_history,
        )
        with open(directory / "forecaster.pkl", "wb") as f:
            pickle.dump(stub, f)

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "QuantileForecaster":
        pkl_path = directory / "forecaster.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        # Fallback: load individual models (no history).
        obj = cls()
        for alpha in QUANTILES:
            label = int(alpha * 100)
            p = directory / f"lgb_q{label:02d}.txt"
            if not p.exists():
                raise ModelNotTrainedError()
            obj.models[alpha] = lgb.Booster(model_file=str(p))
        return obj

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, merchant_id: str, horizon: int = HORIZON
    ) -> pd.DataFrame:
        """
        Recursive 14-day forward forecast for a single merchant.
        Returns a DataFrame with columns: forecast_date, p10, p50, p90.
        """
        self._check_ready()

        history = self.daily_history[
            self.daily_history["merchant_id"] == merchant_id
        ].copy()
        if history.empty:
            raise MerchantNotFoundError(merchant_id)
        if len(history) < MIN_HISTORY_DAYS:
            raise InsufficientHistoryError(
                merchant_id, len(history), MIN_HISTORY_DAYS
            )

        return self._recursive_predict(history, merchant_id, horizon)

    def _recursive_predict(
        self,
        history: pd.DataFrame,
        merchant_id: str,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Append synthetic stubs one day at a time so lags stay consistent.
        Each new stub uses the P50 prediction as the placeholder net_settled_amount
        (so subsequent lag features are grounded in P50, not NaN).
        """
        working = history.sort_values("date").copy()
        category = working["merchant_category"].iloc[0]
        last_date = pd.to_datetime(working["date"].max())
        rows = []

        for step in range(1, horizon + 1):
            target_date = last_date + timedelta(days=step)
            stub = _make_stub(merchant_id, category, target_date)
            featured = build_features(
                pd.concat([working, pd.DataFrame([stub])], ignore_index=True),
                drop_incomplete=False,
            )
            x = featured.iloc[[-1]][self.feature_names]

            preds = {alpha: float(m.predict(x)[0]) for alpha, m in self.models.items()}

            # Use P50 to fill in the rolling history so next step's lags are sensible.
            stub["net_settled_amount"] = preds[0.5]
            working = pd.concat(
                [working, pd.DataFrame([stub])], ignore_index=True
            )
            rows.append(
                {
                    "merchant_id": merchant_id,
                    "forecast_date": target_date.date(),
                    "horizon_day": step,
                    "p10": round(preds[0.1], 2),
                    "p50": round(preds[0.5], 2),
                    "p90": round(preds[0.9], 2),
                }
            )
        return pd.DataFrame(rows)

    def _check_ready(self) -> None:
        if not self.models or self.daily_history.empty:
            raise ModelNotTrainedError()


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(
    ledger: pd.DataFrame,
    num_boost_round: int = 300,
    min_train_days: int = 90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest across the ledger.

    Strategy
    --------
    * Sort dates. The first cutoff is start_date + min_train_days.
    * Each fold: train on [earliest_date, cutoff), predict next HORIZON days
      per merchant, compare to actuals that exist in the ledger.
    * Roll cutoff forward by HORIZON days until fewer than HORIZON actual
      days remain after the cutoff.

    Returns
    -------
    fold_results : per-fold, per-merchant-category metric rows
    summary      : overall + per-category aggregate
    """
    ledger = ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])
    all_dates = sorted(ledger["date"].unique())
    first_date = all_dates[0]
    last_date = all_dates[-1]

    cutoff = first_date + pd.Timedelta(days=min_train_days)
    fold_rows: List[dict] = []

    fold_idx = 0
    while True:
        window_end = cutoff + pd.Timedelta(days=HORIZON - 1)
        if window_end > last_date:
            break  # not enough actuals left for a full fold

        fold_idx += 1
        train_data = ledger[ledger["date"] < cutoff]
        test_data = ledger[
            (ledger["date"] >= cutoff) & (ledger["date"] <= window_end)
        ]

        try:
            fc = QuantileForecaster()
            fc.fit(train_data, num_boost_round=num_boost_round)
            # Give the forecaster the full ledger up to cutoff for recursive predict.
            fc.daily_history = train_data.copy()
        except InsufficientHistoryError:
            cutoff += pd.Timedelta(days=HORIZON)
            continue

        merchants = test_data["merchant_id"].unique()
        for mid in merchants:
            m_test = test_data[test_data["merchant_id"] == mid].sort_values("date")
            if m_test.empty:
                continue
            m_history = train_data[train_data["merchant_id"] == mid]
            if len(m_history) < MIN_HISTORY_DAYS:
                continue

            category = m_history["merchant_category"].iloc[-1]

            try:
                preds_df = fc._recursive_predict(m_history, mid, horizon=len(m_test))
            except (InsufficientHistoryError, MerchantNotFoundError):
                continue

            y_true = m_test[TARGET_COLUMN].values
            p10 = preds_df["p10"].values
            p50 = preds_df["p50"].values
            p90 = preds_df["p90"].values

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "cutoff_date": cutoff.date(),
                    "merchant_id": mid,
                    "merchant_category": category,
                    "n_days": len(y_true),
                    "pinball_p10": _pinball(y_true, p10, 0.1),
                    "pinball_p50": _pinball(y_true, p50, 0.5),
                    "pinball_p90": _pinball(y_true, p90, 0.9),
                    "coverage_p10_p90": _coverage(y_true, p10, p90),
                }
            )

        cutoff += pd.Timedelta(days=HORIZON)

    fold_results = pd.DataFrame(fold_rows)

    if fold_results.empty:
        return fold_results, pd.DataFrame()

    # Aggregate: weight by n_days so merchants with more backtest days count more.
    def _wavg(grp: pd.DataFrame) -> pd.Series:
        w = grp["n_days"]
        return pd.Series(
            {
                "pinball_p10": np.average(grp["pinball_p10"], weights=w),
                "pinball_p50": np.average(grp["pinball_p50"], weights=w),
                "pinball_p90": np.average(grp["pinball_p90"], weights=w),
                "coverage_p10_p90": np.average(grp["coverage_p10_p90"], weights=w),
                "n_merchant_folds": len(grp),
                "n_days_total": int(w.sum()),
            }
        )

    overall = _wavg(fold_results).to_frame().T
    overall.insert(0, "merchant_category", "overall")

    per_cat = (
        fold_results.groupby("merchant_category", group_keys=False)
        .apply(_wavg)
        .reset_index()
    )
    per_cat.columns = ["merchant_category"] + list(per_cat.columns[1:])

    summary = pd.concat([overall, per_cat], ignore_index=True)
    for col in ["pinball_p10", "pinball_p50", "pinball_p90", "coverage_p10_p90"]:
        summary[col] = summary[col].round(4)

    return fold_results, summary


def save_backtest_results(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    directory: Path = REPORTS_DIR,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(directory / "backtest_results.csv", index=False)
    summary.to_csv(directory / "backtest_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub(merchant_id: str, category: str, date: pd.Timestamp) -> dict:
    return {
        "merchant_id": merchant_id,
        "merchant_category": category,
        "date": date,
        "gross_transaction_amount": np.nan,
        "refund_amount": np.nan,
        "chargeback_amount": np.nan,
        "razorpay_fee": np.nan,
        "tds_hold": np.nan,
        "net_settled_amount": np.nan,
    }


# ---------------------------------------------------------------------------
# Legacy compatibility shim (old CashflowForecaster name used in old main.py)
# ---------------------------------------------------------------------------

CashflowForecaster = QuantileForecaster
