"""
Quantile cash-flow forecaster — v2 (recursive-leakage fix + CQR calibration).

Bug fixed in v2
---------------
v1 ran all three quantile trajectories (P10, P50, P90) through the SAME
recursive path: every day's lag features were populated with the P50 prediction
regardless of which quantile was being forecast. This caused the P90 trajectory
to understate upside uncertainty (it was effectively a P50 path + a one-step
P90 offset) and P10 to overstate downside breadth in the opposite direction.

Fix: _recursive_predict_single runs one trajectory at a time, feeding each
quantile's own prior predictions back as the lag fill. The published p10/p50/p90
values are then the endpoints of three independent autoregressive paths.

CQR calibration (step 3)
------------------------
Even with correct trajectories, quantile regression models are often
systematically miscalibrated — the empirical coverage drifts from the nominal
level, especially under distribution shift. Conformalized Quantile Regression
(CQR, Angelopoulos & Bates 2021 / Romano et al. 2019) corrects this without
retraining:

  1. Reserve a calibration fold from the walk-forward sequence.
  2. Collect calibration scores: s_i = max(p10_i - y_i, y_i - p90_i).
     s_i > 0 means y_i fell outside the band; s_i <= 0 means it was inside.
  3. Compute q_hat = empirical (1 - alpha) quantile of {s_i}, where
     alpha = 0.2 for an 80% target interval. Add a finite-sample correction:
     q_hat = quantile at level ceil((n+1)(1-alpha))/n.
  4. At test time: adjusted_p10 = p10 - q_hat, adjusted_p90 = p90 + q_hat.
     This symmetrically expands (or contracts) the band by q_hat.

q_hat is reported alongside the backtest summary so you can see how large
the correction was relative to the raw band width.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def _coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    return float(((y_true >= p10) & (y_true <= p90)).mean())


def _count_ordering_violations(preds: pd.DataFrame) -> int:
    """Return number of rows where P10 > P50 or P50 > P90."""
    return int(((preds["p10"] > preds["p50"]) | (preds["p50"] > preds["p90"])).sum())


# ---------------------------------------------------------------------------
# CQR calibration helpers
# ---------------------------------------------------------------------------

def compute_cqr_correction(
    y_true: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """
    Compute the CQR correction q_hat from calibration data.

    Scores: s_i = max(p10_i - y_i, y_i - p90_i)
      s_i <= 0  => y_i is inside the band
      s_i >  0  => y_i is outside; s_i is how far outside

    q_hat is the ceil((n+1)(1-alpha))/n empirical quantile of scores.
    """
    scores = np.maximum(p10 - y_true, y_true - p90)
    return compute_cqr_correction_from_scores(scores, target_coverage)


def compute_cqr_correction_from_scores(
    scores: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """
    Compute q_hat directly from a pre-computed array of CQR scores.

    Separating score collection from q_hat computation lets callers
    pool scores across multiple calibration windows before computing
    the single corrective offset that governs production band width.
    """
    alpha = 1.0 - target_coverage
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def apply_cqr(
    p10: np.ndarray,
    p90: np.ndarray,
    q_hat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand (or contract if q_hat < 0) the band symmetrically.
    Returns (adjusted_p10, adjusted_p90) with P10 <= P90 guaranteed.
    """
    adj_p10 = p10 - q_hat
    adj_p90 = p90 + q_hat
    # Guarantee ordering even when q_hat is very negative
    lo = np.minimum(adj_p10, adj_p90)
    hi = np.maximum(adj_p10, adj_p90)
    return lo, hi


# ---------------------------------------------------------------------------
# Core quantile forecaster
# ---------------------------------------------------------------------------

@dataclass
class QuantileForecaster:
    """
    Three LightGBM quantile models (P10, P50, P90) with:
    - Per-quantile recursive trajectories (leakage fix)
    - Optional CQR calibration offset (q_hat)
    """

    models: Dict[float, lgb.Booster] = field(default_factory=dict)
    daily_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    # CQR correction — set after calibration; None means no correction applied
    cqr_q_hat: Optional[float] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        ledger: pd.DataFrame,
        num_boost_round: int = 300,
        cutoff_date: Optional[pd.Timestamp] = None,
    ) -> "QuantileForecaster":
        self.daily_history = ledger.copy()
        self.daily_history["date"] = pd.to_datetime(self.daily_history["date"])

        train_ledger = ledger.copy()
        if cutoff_date is not None:
            train_ledger = train_ledger[
                pd.to_datetime(train_ledger["date"]) <= cutoff_date
            ]

        featured = build_features(train_ledger, drop_incomplete=True)
        if featured.empty or len(featured) < MIN_TRAIN_ROWS:
            raise InsufficientHistoryError("global", len(featured), MIN_TRAIN_ROWS)

        X = featured[self.feature_names]
        y = featured[TARGET_COLUMN].values
        dataset = lgb.Dataset(
            X, label=y, categorical_feature=["merchant_category"], free_raw_data=False
        )
        for alpha in QUANTILES:
            self.models[alpha] = lgb.train(
                _lgb_params(alpha), dataset, num_boost_round=num_boost_round
            )
        return self

    def calibrate(
        self,
        cal_ledger: pd.DataFrame,
        target_coverage: float = 0.80,
    ) -> float:
        """
        Run CQR calibration against a held-out calibration ledger.

        For every merchant in cal_ledger, generate predictions over the
        calibration window, collect CQR scores, compute q_hat, and store it
        on self.cqr_q_hat.

        Returns q_hat so callers can log it.
        """
        scores = self.collect_calibration_scores(cal_ledger)
        if len(scores) < 10:
            self.cqr_q_hat = 0.0
            return 0.0
        q_hat = compute_cqr_correction_from_scores(scores, target_coverage)
        self.cqr_q_hat = q_hat
        return q_hat

    def calibrate_from_scores(
        self,
        scores: np.ndarray,
        target_coverage: float = 0.80,
    ) -> float:
        """
        Set cqr_q_hat from a pre-pooled array of CQR scores.

        This is the entry point for pooled-across-folds calibration:
        the caller collects scores from multiple calib windows using
        collect_calibration_scores(), concatenates them, then calls this
        method once. The result is a single q_hat that reflects the
        volatility distribution seen across the entire backtest period,
        not just the last 14-day window.
        """
        if len(scores) < 10:
            self.cqr_q_hat = 0.0
            return 0.0
        q_hat = compute_cqr_correction_from_scores(scores, target_coverage)
        self.cqr_q_hat = q_hat
        return q_hat

    def collect_calibration_scores(
        self,
        cal_ledger: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate predictions over cal_ledger and return raw CQR scores.

        Score definition: s_i = max(p10_i - y_i, y_i - p90_i)
          s_i <= 0  => y_i was inside the band
          s_i >  0  => y_i was outside by s_i units

        Returns a 1-D float array of scores (one per merchant-day in the
        calibration window). Does NOT set cqr_q_hat — use calibrate() or
        calibrate_from_scores() for that.
        """
        cal_ledger = cal_ledger.copy()
        cal_ledger["date"] = pd.to_datetime(cal_ledger["date"])
        cal_dates = sorted(cal_ledger["date"].unique())
        if not cal_dates:
            return np.array([])
        cal_start = cal_dates[0]

        all_y: List[float] = []
        all_p10: List[float] = []
        all_p90: List[float] = []

        for mid in cal_ledger["merchant_id"].unique():
            m_cal = cal_ledger[cal_ledger["merchant_id"] == mid].sort_values("date")
            m_hist = self.daily_history[
                (self.daily_history["merchant_id"] == mid)
                & (self.daily_history["date"] < cal_start)
            ]
            if len(m_hist) < MIN_HISTORY_DAYS or m_cal.empty:
                continue
            try:
                preds = self._recursive_predict(m_hist, mid, horizon=len(m_cal))
            except (InsufficientHistoryError, MerchantNotFoundError):
                continue

            y_true = m_cal[TARGET_COLUMN].values
            n = min(len(y_true), len(preds))
            all_y.extend(y_true[:n].tolist())
            all_p10.extend(preds["p10"].values[:n].tolist())
            all_p90.extend(preds["p90"].values[:n].tolist())

        if not all_y:
            return np.array([])

        y = np.array(all_y)
        p10 = np.array(all_p10)
        p90 = np.array(all_p90)
        return np.maximum(p10 - y, y - p90)

    def save(self, directory: Path = MODELS_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for alpha, model in self.models.items():
            label = int(alpha * 100)
            model.save_model(str(directory / f"lgb_q{label:02d}.txt"))
        with open(directory / "forecaster.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, directory: Path = MODELS_DIR) -> "QuantileForecaster":
        pkl_path = directory / "forecaster.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
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
        self,
        merchant_id: str,
        horizon: int = HORIZON,
        apply_calibration: bool = True,
    ) -> pd.DataFrame:
        self._check_ready()
        history = self.daily_history[
            self.daily_history["merchant_id"] == merchant_id
        ].copy()
        if history.empty:
            raise MerchantNotFoundError(merchant_id)
        if len(history) < MIN_HISTORY_DAYS:
            raise InsufficientHistoryError(merchant_id, len(history), MIN_HISTORY_DAYS)

        preds = self._recursive_predict(history, merchant_id, horizon)
        if apply_calibration and self.cqr_q_hat is not None and self.cqr_q_hat != 0.0:
            adj_p10, adj_p90 = apply_cqr(
                preds["p10"].values, preds["p90"].values, self.cqr_q_hat
            )
            preds = preds.copy()
            preds["p10"] = np.round(adj_p10, 2)
            preds["p90"] = np.round(adj_p90, 2)
        return preds

    def _recursive_predict(
        self,
        history: pd.DataFrame,
        merchant_id: str,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Run three SEPARATE recursive trajectories — one per quantile.

        v1 bug: all three quantiles shared the same working DataFrame, which
        was updated with the P50 prediction each step. So P90 for day N+1
        was computed from a lag vector that reflected the P50 path, not the
        P90 path. The upside uncertainty never accumulated through the lags.

        v2 fix: run _recursive_predict_single for each alpha independently.
        The P10 trajectory feeds its own pessimistic predictions back into
        the lag features; P90 feeds its own optimistic ones. This means:
        - The P90 path sees higher lags => higher forecast => wider upper band
        - The P10 path sees lower lags  => lower forecast => wider lower band

        The result is a fan that actually opens up over the horizon rather
        than three parallel lines with a fixed offset.
        """
        p10_traj = self._recursive_predict_single(history, merchant_id, horizon, alpha=0.1)
        p50_traj = self._recursive_predict_single(history, merchant_id, horizon, alpha=0.5)
        p90_traj = self._recursive_predict_single(history, merchant_id, horizon, alpha=0.9)

        # Sort each trajectory to maintain weak ordering at each step.
        # Crossing can still happen at extreme outlier steps; sort enforces
        # P10 <= P50 <= P90 without widening or narrowing the total band.
        p10_vals = p10_traj["pred"].values
        p50_vals = p50_traj["pred"].values
        p90_vals = p90_traj["pred"].values

        # Per-row sort so ordering holds without biasing any single trajectory
        stacked = np.sort(np.stack([p10_vals, p50_vals, p90_vals], axis=1), axis=1)

        rows = []
        for i, step in enumerate(range(1, horizon + 1)):
            rows.append({
                "merchant_id": merchant_id,
                "forecast_date": p50_traj.iloc[i]["forecast_date"],
                "horizon_day": step,
                "p10": round(float(stacked[i, 0]), 2),
                "p50": round(float(stacked[i, 1]), 2),
                "p90": round(float(stacked[i, 2]), 2),
            })
        return pd.DataFrame(rows)

    def _recursive_predict_single(
        self,
        history: pd.DataFrame,
        merchant_id: str,
        horizon: int,
        alpha: float,
    ) -> pd.DataFrame:
        """
        Single-quantile autoregressive forecast.

        Each step appends a stub row with THIS quantile's prediction as the
        net_settled_amount fill. That means the lag-1, lag-7, lag-14 features
        for day N+k genuinely reflect the quantile's own trajectory, not P50.
        """
        working = history.sort_values("date").copy()
        category = working["merchant_category"].iloc[0]
        last_date = pd.to_datetime(working["date"].max())
        model = self.models[alpha]
        rows = []

        for step in range(1, horizon + 1):
            target_date = last_date + timedelta(days=step)
            stub = _make_stub(merchant_id, category, target_date)

            featured = build_features(
                pd.concat([working, pd.DataFrame([stub])], ignore_index=True),
                drop_incomplete=False,
            )
            x = featured.iloc[[-1]][self.feature_names]
            pred = float(model.predict(x)[0])

            # Feed THIS quantile's prediction back into the working history
            stub["net_settled_amount"] = pred
            working = pd.concat([working, pd.DataFrame([stub])], ignore_index=True)

            rows.append({
                "forecast_date": target_date.date(),
                "horizon_day": step,
                "pred": pred,
            })
        return pd.DataFrame(rows)

    def _check_ready(self) -> None:
        if not self.models or self.daily_history.empty:
            raise ModelNotTrainedError()


# ---------------------------------------------------------------------------
# Walk-forward backtest with CQR
# ---------------------------------------------------------------------------

def run_backtest(
    ledger: pd.DataFrame,
    num_boost_round: int = 300,
    min_train_days: int = 90,
    target_coverage: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest with CQR calibration.

    Fold structure
    --------------
    Each iteration has three windows:
      train   : [start, cutoff)              — model training
      calib   : [cutoff, cutoff+HORIZON)     — CQR score collection
      test    : [cutoff+HORIZON, cutoff+2*HORIZON)  — evaluation

    The correction q_hat is computed on the calib window and applied to
    test-window predictions. This ensures the held-out test numbers are
    never used to compute q_hat (no information leakage into calibration).

    We need 2*HORIZON days after cutoff, so the loop ends when
    cutoff + 2*HORIZON - 1 > last_date.

    Both raw (uncalibrated) and calibrated metrics are reported so you can
    see the delta from the leakage fix alone vs. leakage fix + CQR.
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
        calib_end = cutoff + pd.Timedelta(days=HORIZON - 1)
        test_end  = cutoff + pd.Timedelta(days=2 * HORIZON - 1)
        if test_end > last_date:
            break  # not enough data for a full calib+test pair

        fold_idx += 1
        train_data = ledger[ledger["date"] < cutoff]
        calib_data = ledger[
            (ledger["date"] >= cutoff) & (ledger["date"] <= calib_end)
        ]
        test_data = ledger[
            (ledger["date"] > calib_end) & (ledger["date"] <= test_end)
        ]

        # ---- Train ----
        try:
            fc = QuantileForecaster()
            fc.fit(train_data, num_boost_round=num_boost_round)
            fc.daily_history = train_data.copy()
        except InsufficientHistoryError:
            cutoff += pd.Timedelta(days=HORIZON)
            continue

        # ---- Calibrate (CQR) ----
        # Give the forecaster history up through train window for calib predictions
        q_hat = fc.calibrate(calib_data, target_coverage=target_coverage)

        # ---- Evaluate on test window ----
        merchants = test_data["merchant_id"].unique()
        for mid in merchants:
            m_test = test_data[test_data["merchant_id"] == mid].sort_values("date")
            if m_test.empty:
                continue
            m_history = train_data[train_data["merchant_id"] == mid]
            if len(m_history) < MIN_HISTORY_DAYS:
                continue

            category = m_history["merchant_category"].iloc[-1]

            # We predict from end of calib window — give forecaster calib data
            # as additional history so its lags reflect recent values
            m_calib = calib_data[calib_data["merchant_id"] == mid].sort_values("date")
            m_hist_through_calib = pd.concat(
                [m_history, m_calib], ignore_index=True
            ).sort_values("date")

            try:
                # Raw predictions (no CQR)
                raw_preds = fc._recursive_predict(
                    m_hist_through_calib, mid, horizon=len(m_test)
                )
            except (InsufficientHistoryError, MerchantNotFoundError):
                continue

            y_true = m_test[TARGET_COLUMN].values
            raw_p10 = raw_preds["p10"].values
            raw_p50 = raw_preds["p50"].values
            raw_p90 = raw_preds["p90"].values

            # CQR-adjusted predictions
            cal_p10, cal_p90 = apply_cqr(raw_p10, raw_p90, q_hat)

            # Ordering violations (diagnostic)
            n_viol = _count_ordering_violations(raw_preds)

            fold_rows.append({
                "fold": fold_idx,
                "cutoff_date": cutoff.date(),
                "merchant_id": mid,
                "merchant_category": category,
                "n_days": len(y_true),
                "q_hat": round(q_hat, 2),
                # Raw metrics (leakage-fixed trajectories, no CQR)
                "raw_pinball_p10": _pinball(y_true, raw_p10, 0.1),
                "raw_pinball_p50": _pinball(y_true, raw_p50, 0.5),
                "raw_pinball_p90": _pinball(y_true, raw_p90, 0.9),
                "raw_coverage": _coverage(y_true, raw_p10, raw_p90),
                # Calibrated metrics (leakage fix + CQR)
                "cal_pinball_p10": _pinball(y_true, cal_p10, 0.1),
                "cal_pinball_p50": _pinball(y_true, raw_p50, 0.5),
                "cal_pinball_p90": _pinball(y_true, cal_p90, 0.9),
                "cal_coverage": _coverage(y_true, cal_p10, cal_p90),
                # Diagnostics
                "ordering_violations": n_viol,
            })

        cutoff += pd.Timedelta(days=HORIZON)

    fold_results = pd.DataFrame(fold_rows)
    if fold_results.empty:
        return fold_results, pd.DataFrame()

    summary = _aggregate_summary(fold_results)
    return fold_results, summary


def _aggregate_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Weighted average by n_days, overall and per category."""

    def _wavg(grp: pd.DataFrame) -> pd.Series:
        w = grp["n_days"]
        return pd.Series({
            "raw_pinball_p10":  np.average(grp["raw_pinball_p10"],  weights=w),
            "raw_pinball_p50":  np.average(grp["raw_pinball_p50"],  weights=w),
            "raw_pinball_p90":  np.average(grp["raw_pinball_p90"],  weights=w),
            "raw_coverage":     np.average(grp["raw_coverage"],     weights=w),
            "cal_pinball_p10":  np.average(grp["cal_pinball_p10"],  weights=w),
            "cal_pinball_p50":  np.average(grp["cal_pinball_p50"],  weights=w),
            "cal_pinball_p90":  np.average(grp["cal_pinball_p90"],  weights=w),
            "cal_coverage":     np.average(grp["cal_coverage"],     weights=w),
            "mean_q_hat":       float(grp["q_hat"].mean()),
            "ordering_violations": int(grp["ordering_violations"].sum()),
            "n_merchant_folds": len(grp),
            "n_days_total":     int(w.sum()),
        })

    overall = _wavg(fold_results).to_frame().T
    overall.insert(0, "merchant_category", "overall")

    per_cat = (
        fold_results.groupby("merchant_category", group_keys=False)
        .apply(_wavg, include_groups=False)
        .reset_index()
    )

    summary = pd.concat([overall, per_cat], ignore_index=True)
    rnd_cols = [c for c in summary.columns if "pinball" in c or "coverage" in c or "q_hat" in c]
    for col in rnd_cols:
        summary[col] = summary[col].round(4)
    return summary


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
# Legacy alias
# ---------------------------------------------------------------------------
CashflowForecaster = QuantileForecaster
