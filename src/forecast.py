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
    - Optional CQR calibration: global q_hat and/or per-category q_hat
    """

    models: Dict[float, lgb.Booster] = field(default_factory=dict)
    daily_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    # Global CQR correction — pooled across all categories, used as fallback
    cqr_q_hat: Optional[float] = None
    # Per-category CQR corrections — takes precedence over global when available
    cqr_q_hat_by_category: Dict[str, float] = field(default_factory=dict)

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
        scored = self._score_cal_ledger(cal_ledger)
        if not scored:
            return np.array([])
        return np.concatenate([v for v in scored.values()])

    def collect_calibration_scores_by_category(
        self,
        cal_ledger: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Like collect_calibration_scores but groups scores by merchant_category.

        Returns {category: 1-D score array}. Used by the per-category
        calibration path in train_and_backtest.py.
        """
        return self._score_cal_ledger(cal_ledger)

    def calibrate_per_category(
        self,
        scores_by_category: Dict[str, np.ndarray],
        target_coverage: float = 0.80,
        min_scores: int = 50,
    ) -> Dict[str, float]:
        """
        Compute a separate q_hat for each category and store on
        self.cqr_q_hat_by_category.

        Categories with fewer than min_scores calibration points are flagged
        and fall back to self.cqr_q_hat (the global value). This is printed
        as a warning so the caller knows the estimate is unreliable.

        Returns the full {category: q_hat} dict (including fallbacks) so
        callers can log it.
        """
        result: Dict[str, float] = {}
        for cat, scores in scores_by_category.items():
            n = len(scores)
            if n < min_scores:
                fallback = self.cqr_q_hat if self.cqr_q_hat is not None else 0.0
                print(
                    f"  WARNING: category '{cat}' has only {n} calibration scores "
                    f"(< {min_scores} minimum). Per-category q_hat is unreliable — "
                    f"falling back to global q_hat={fallback:,.0f}"
                )
                result[cat] = fallback
            else:
                result[cat] = compute_cqr_correction_from_scores(scores, target_coverage)
        self.cqr_q_hat_by_category = result
        return result

    def _score_cal_ledger(
        self,
        cal_ledger: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Internal: run predictions over cal_ledger, return
        {category: score_array}. Shared by both collect methods.
        """
        cal_ledger = cal_ledger.copy()
        cal_ledger["date"] = pd.to_datetime(cal_ledger["date"])
        cal_dates = sorted(cal_ledger["date"].unique())
        if not cal_dates:
            return {}
        cal_start = cal_dates[0]

        by_cat: Dict[str, List[float]] = {}

        for mid in cal_ledger["merchant_id"].unique():
            m_cal = cal_ledger[cal_ledger["merchant_id"] == mid].sort_values("date")
            m_hist = self.daily_history[
                (self.daily_history["merchant_id"] == mid)
                & (self.daily_history["date"] < cal_start)
            ]
            if len(m_hist) < MIN_HISTORY_DAYS or m_cal.empty:
                continue

            category = m_hist["merchant_category"].iloc[-1]

            try:
                preds = self._recursive_predict(m_hist, mid, horizon=len(m_cal))
            except (InsufficientHistoryError, MerchantNotFoundError):
                continue

            y_true = m_cal[TARGET_COLUMN].values
            n = min(len(y_true), len(preds))
            scores = np.maximum(
                preds["p10"].values[:n] - y_true[:n],
                y_true[:n] - preds["p90"].values[:n],
            )
            if category not in by_cat:
                by_cat[category] = []
            by_cat[category].extend(scores.tolist())

        return {cat: np.array(v) for cat, v in by_cat.items()}

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

        if apply_calibration:
            # Prefer per-category q_hat; fall back to global if category not
            # present in calibration set (e.g. new category added after training)
            category = history["merchant_category"].iloc[-1]
            q_hat = self._resolve_q_hat(category)
            if q_hat is not None and q_hat != 0.0:
                adj_p10, adj_p90 = apply_cqr(
                    preds["p10"].values, preds["p90"].values, q_hat
                )
                preds = preds.copy()
                preds["p10"] = np.round(adj_p10, 2)
                preds["p90"] = np.round(adj_p90, 2)
        return preds

    def _resolve_q_hat(self, category: str) -> Optional[float]:
        """
        Return the q_hat to use for a given category.

        Priority:
          1. cqr_q_hat_by_category[category]  — per-category (most specific)
          2. cqr_q_hat                         — global pooled fallback
          3. None                              — no calibration applied
        """
        if self.cqr_q_hat_by_category and category in self.cqr_q_hat_by_category:
            return self.cqr_q_hat_by_category[category]
        return self.cqr_q_hat

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
    min_cat_scores: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest with both global and per-category CQR calibration.

    Fold structure
    --------------
    Each iteration:
      train   : [start, cutoff)
      calib   : [cutoff, cutoff+HORIZON)     — scores collected here
      test    : [cutoff+HORIZON, cutoff+2*HORIZON)  — evaluated here

    For each fold, calib scores are grouped by merchant_category.
    A per-category q_hat is computed from those category scores (with a
    min_cat_scores guard — if a category has fewer than that many scores in
    the fold's calib window, it falls back to the global q_hat for that fold).

    Fold rows carry both cal_coverage (global q_hat) and percat_coverage
    (per-category q_hat) so the aggregate comparison is fair: same merchants,
    same test windows, same total day count.

    Also records percat_q_hat_used (the actual q_hat applied for that
    merchant's category in that fold) and a percat_q_hat_fallback flag so
    you can see which folds fell back to global.
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
            break

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

        # ---- Calibrate: global q_hat + per-category q_hats ----
        # Global: one number across all categories in this fold's calib window
        scores_by_cat = fc._score_cal_ledger(calib_data)
        all_scores_flat = (
            np.concatenate(list(scores_by_cat.values()))
            if scores_by_cat else np.array([])
        )
        global_q_hat = (
            compute_cqr_correction_from_scores(all_scores_flat, target_coverage)
            if len(all_scores_flat) >= 10 else 0.0
        )

        # Per-category: one q_hat per category, with fallback to global
        # if the category has fewer than min_cat_scores in this fold.
        cat_q_hats: Dict[str, float] = {}
        cat_n_scores: Dict[str, int] = {}
        cat_fallback: Dict[str, bool] = {}
        for cat, scores in scores_by_cat.items():
            n = len(scores)
            cat_n_scores[cat] = n
            if n < min_cat_scores:
                cat_q_hats[cat] = global_q_hat
                cat_fallback[cat] = True
            else:
                cat_q_hats[cat] = compute_cqr_correction_from_scores(scores, target_coverage)
                cat_fallback[cat] = False

        # ---- Evaluate on test window ----
        for mid in test_data["merchant_id"].unique():
            m_test = test_data[test_data["merchant_id"] == mid].sort_values("date")
            if m_test.empty:
                continue
            m_history = train_data[train_data["merchant_id"] == mid]
            if len(m_history) < MIN_HISTORY_DAYS:
                continue

            category = m_history["merchant_category"].iloc[-1]

            m_calib = calib_data[calib_data["merchant_id"] == mid].sort_values("date")
            m_hist_through_calib = pd.concat(
                [m_history, m_calib], ignore_index=True
            ).sort_values("date")

            try:
                raw_preds = fc._recursive_predict(
                    m_hist_through_calib, mid, horizon=len(m_test)
                )
            except (InsufficientHistoryError, MerchantNotFoundError):
                continue

            y_true = m_test[TARGET_COLUMN].values
            raw_p10 = raw_preds["p10"].values
            raw_p50 = raw_preds["p50"].values
            raw_p90 = raw_preds["p90"].values

            # Global CQR
            cal_p10, cal_p90 = apply_cqr(raw_p10, raw_p90, global_q_hat)

            # Per-category CQR — use category-specific q_hat, fall back to
            # global if category wasn't present in this fold's calib window
            pcat_q = cat_q_hats.get(category, global_q_hat)
            used_fallback = cat_fallback.get(category, True)
            percat_p10, percat_p90 = apply_cqr(raw_p10, raw_p90, pcat_q)

            n_viol = _count_ordering_violations(raw_preds)

            fold_rows.append({
                "fold":             fold_idx,
                "cutoff_date":      cutoff.date(),
                "merchant_id":      mid,
                "merchant_category": category,
                "n_days":           len(y_true),
                # q_hats
                "q_hat_global":     round(global_q_hat, 2),
                "q_hat_percat":     round(pcat_q, 2),
                "percat_fallback":  used_fallback,
                "cat_n_cal_scores": cat_n_scores.get(category, 0),
                # kept as q_hat for backward compat with existing summary code
                "q_hat":            round(global_q_hat, 2),
                # Raw
                "raw_pinball_p10":  _pinball(y_true, raw_p10, 0.1),
                "raw_pinball_p50":  _pinball(y_true, raw_p50, 0.5),
                "raw_pinball_p90":  _pinball(y_true, raw_p90, 0.9),
                "raw_coverage":     _coverage(y_true, raw_p10, raw_p90),
                # Global CQR
                "cal_pinball_p10":  _pinball(y_true, cal_p10, 0.1),
                "cal_pinball_p50":  _pinball(y_true, raw_p50, 0.5),
                "cal_pinball_p90":  _pinball(y_true, cal_p90, 0.9),
                "cal_coverage":     _coverage(y_true, cal_p10, cal_p90),
                # Per-category CQR
                "percat_pinball_p10": _pinball(y_true, percat_p10, 0.1),
                "percat_pinball_p50": _pinball(y_true, raw_p50, 0.5),
                "percat_pinball_p90": _pinball(y_true, percat_p90, 0.9),
                "percat_coverage":    _coverage(y_true, percat_p10, percat_p90),
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
        d: dict = {
            "raw_pinball_p10":     np.average(grp["raw_pinball_p10"],  weights=w),
            "raw_pinball_p50":     np.average(grp["raw_pinball_p50"],  weights=w),
            "raw_pinball_p90":     np.average(grp["raw_pinball_p90"],  weights=w),
            "raw_coverage":        np.average(grp["raw_coverage"],     weights=w),
            "cal_pinball_p10":     np.average(grp["cal_pinball_p10"],  weights=w),
            "cal_pinball_p50":     np.average(grp["cal_pinball_p50"],  weights=w),
            "cal_pinball_p90":     np.average(grp["cal_pinball_p90"],  weights=w),
            "cal_coverage":        np.average(grp["cal_coverage"],     weights=w),
            "mean_q_hat":          float(grp["q_hat"].mean()),
        }
        # per-category CQR columns (added in updated run_backtest)
        if "percat_coverage" in grp.columns:
            d["percat_pinball_p10"] = np.average(grp["percat_pinball_p10"], weights=w)
            d["percat_pinball_p50"] = np.average(grp["percat_pinball_p50"], weights=w)
            d["percat_pinball_p90"] = np.average(grp["percat_pinball_p90"], weights=w)
            d["percat_coverage"]    = np.average(grp["percat_coverage"],    weights=w)
            d["percat_fallback_pct"] = float(grp["percat_fallback"].mean()) if "percat_fallback" in grp.columns else float("nan")
        d["ordering_violations"] = int(grp["ordering_violations"].sum())
        d["n_merchant_folds"]    = len(grp)
        d["n_days_total"]        = int(w.sum())
        return pd.Series(d)

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
