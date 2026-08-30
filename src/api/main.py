"""
FastAPI application for the merchant cash-flow forecaster.

Endpoints
---------
GET  /health                   – liveness check + model status
GET  /forecast/{merchant_id}   – 14-day P10/P50/P90 forecast; logged to audit table
GET  /backtest-metrics         – saved backtest summary (coverage + pinball per category)
GET  /exceptions               – current low-confidence flags from exceptions.py
POST /train                    – (utility) re-train models from the default ledger
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.db import ForecastAuditLog, get_session, init_db
from src.exceptions import (
    CashflowError,
    InsufficientHistoryError,
    MerchantNotFoundError,
    ModelNotTrainedError,
)
from src.forecast import MODELS_DIR, REPORTS_DIR, QuantileForecaster

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "data" / "synthetic_ledger.csv"

# ---------------------------------------------------------------------------
# Global model state — loaded once at startup.
# ---------------------------------------------------------------------------
_forecaster: Optional[QuantileForecaster] = None


def _get_forecaster() -> QuantileForecaster:
    if _forecaster is None:
        raise ModelNotTrainedError()
    return _forecaster


# ---------------------------------------------------------------------------
# Lifespan: init DB + try to load pre-trained models.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _forecaster
    init_db()
    if (MODELS_DIR / "forecaster.pkl").exists():
        try:
            _forecaster = QuantileForecaster.load(MODELS_DIR)
        except Exception:
            pass  # model file corrupt / missing — will surface as 503 on forecast calls
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Merchant cash-flow forecast", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ForecastDay(BaseModel):
    forecast_date: str
    horizon_day: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    merchant_id: str
    forecast_origin: str   # last date with actual data
    forecast: List[ForecastDay]


class BacktestRow(BaseModel):
    merchant_category: str
    # Raw = leakage-fixed trajectories, no CQR
    raw_coverage: Optional[float] = None
    # Calibrated = leakage-fixed + CQR; this is the primary coverage number
    coverage_p10_p90: float          # alias for cal_coverage — what the frontend shows
    pinball_p10: float               # calibrated pinball (cal_pinball_p10)
    pinball_p50: float               # calibrated pinball (cal_pinball_p50)
    pinball_p90: float               # calibrated pinball (cal_pinball_p90)
    mean_q_hat: Optional[float] = None
    n_days_total: Optional[int] = None


class ExceptionFlag(BaseModel):
    merchant_id: str
    flag_date: str
    reason: str
    confidence_score: Optional[float] = None


class TrainResponse(BaseModel):
    status: str
    merchants: int
    daily_rows: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _forecaster is not None,
        "backtest_available": (REPORTS_DIR / "backtest_summary.csv").exists(),
    }


@app.get("/forecast/{merchant_id}", response_model=ForecastResponse)
def get_forecast(
    merchant_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> ForecastResponse:
    """
    Return a 14-day forward forecast (P10 / P50 / P90 per day) for the merchant.
    Every response is written to the forecast_audit_log table.
    """
    fc = _require_model()

    try:
        preds = fc.predict(merchant_id)
    except MerchantNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Merchant '{merchant_id}' not found in ledger history.",
        )
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ModelNotTrainedError:
        raise HTTPException(
            status_code=503,
            detail="Models not yet trained. Run POST /train first.",
        )
    except CashflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Derive forecast origin from the last actual date in history.
    history = fc.daily_history[fc.daily_history["merchant_id"] == merchant_id]
    forecast_origin = str(pd.to_datetime(history["date"]).max().date())

    now = datetime.utcnow()
    audit_rows = [
        ForecastAuditLog(
            merchant_id=merchant_id,
            request_timestamp=now,
            forecast_date=row.forecast_date,
            horizon_day=int(row.horizon_day),
            p10=float(row.p10),
            p50=float(row.p50),
            p90=float(row.p90),
        )
        for row in preds.itertuples(index=False)
    ]
    session.add_all(audit_rows)
    session.commit()

    return ForecastResponse(
        merchant_id=merchant_id,
        forecast_origin=forecast_origin,
        forecast=[
            ForecastDay(
                forecast_date=str(row.forecast_date),
                horizon_day=int(row.horizon_day),
                p10=float(row.p10),
                p50=float(row.p50),
                p90=float(row.p90),
            )
            for row in preds.itertuples(index=False)
        ],
    )


@app.get("/backtest-metrics", response_model=List[BacktestRow])
def get_backtest_metrics() -> List[BacktestRow]:
    """
    Return backtest summary: pinball loss and coverage broken out by merchant category.
    Reads the CSV written by the training script; returns 404 if not yet generated.
    """
    summary_path = REPORTS_DIR / "backtest_summary.csv"
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Backtest results not found. Run the training script first.",
        )
    df = pd.read_csv(summary_path)

    # The summary CSV uses prefixed column names (raw_/cal_) from the v2 backtest.
    # Map them to the stable BacktestRow schema so the frontend needs no changes.
    def _col(df: pd.DataFrame, *candidates: str) -> pd.Series:
        """Return the first column that exists, or a series of NaN."""
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([float("nan")] * len(df))

    return [
        BacktestRow(
            merchant_category=str(row.merchant_category),
            raw_coverage=float(_col(df, "raw_coverage").iloc[i])
                if "raw_coverage" in df.columns else None,
            coverage_p10_p90=float(
                _col(df, "cal_coverage", "coverage_p10_p90").iloc[i]
            ),
            pinball_p10=float(_col(df, "cal_pinball_p10", "pinball_p10").iloc[i]),
            pinball_p50=float(_col(df, "cal_pinball_p50", "pinball_p50").iloc[i]),
            pinball_p90=float(_col(df, "cal_pinball_p90", "pinball_p90").iloc[i]),
            mean_q_hat=float(_col(df, "mean_q_hat").iloc[i])
                if "mean_q_hat" in df.columns else None,
            n_days_total=int(row.n_days_total)
                if pd.notna(row.n_days_total) else None,
        )
        for i, row in enumerate(df.itertuples(index=False))
    ]


@app.get("/exceptions", response_model=List[ExceptionFlag])
def get_exceptions() -> List[ExceptionFlag]:
    """
    Return the current list of low-confidence forecast flags.
    Reads the JSON file written by the training script; returns empty list if none.
    """
    flags_path = REPORTS_DIR / "exceptions.json"
    if not flags_path.exists():
        return []
    with open(flags_path) as f:
        raw = json.load(f)
    return [
        ExceptionFlag(
            merchant_id=item["merchant_id"],
            flag_date=item["flag_date"],
            reason=item["reason"],
            confidence_score=item.get("confidence_score"),
        )
        for item in raw
    ]


@app.get("/merchants", response_model=List[str])
def list_merchants() -> List[str]:
    """List all merchant IDs available in loaded ledger."""
    fc = _require_model()
    return sorted(fc.daily_history["merchant_id"].unique().tolist())


@app.get("/history/{merchant_id}")
def get_history(
    merchant_id: str,
    days: int = Query(default=60, ge=7, le=365),
) -> Dict[str, Any]:
    """
    Return the last N days of actual net_settled_amount for a merchant.
    Used by the frontend to draw the historical line leading up to the forecast.
    """
    fc = _require_model()
    history = fc.daily_history[fc.daily_history["merchant_id"] == merchant_id]
    if history.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Merchant '{merchant_id}' not found in ledger history.",
        )
    recent = (
        history.sort_values("date")
        .tail(days)[["date", "net_settled_amount"]]
        .assign(date=lambda d: d["date"].astype(str))
    )
    return {
        "merchant_id": merchant_id,
        "history": recent.to_dict(orient="records"),
    }


@app.get("/backtest-folds")
def get_backtest_folds() -> List[Dict[str, Any]]:
    """Per-fold coverage and q_hat for the fold reliability strip."""
    results_path = REPORTS_DIR / "backtest_results.csv"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Backtest results not found.")
    df = pd.read_csv(results_path)
    fold_summary = (
        df.groupby(["fold", "cutoff_date"])
        .agg(cal_coverage=("cal_coverage", "mean"), q_hat=("q_hat", "mean"), n_days=("n_days", "sum"))
        .reset_index()
    )
    return [
        {
            "fold": int(r.fold),
            "cutoff_date": str(r.cutoff_date),
            "coverage": round(float(r.cal_coverage), 4),
            "q_hat": round(float(r.q_hat), 0),
            "n_days": int(r.n_days),
        }
        for r in fold_summary.itertuples(index=False)
    ]



def train_models(ledger_path: Optional[str] = None) -> TrainResponse:
    """
    Re-train models from the default (or supplied) ledger CSV.
    Intended for development use; production training runs via the CLI script.
    """
    global _forecaster
    path = Path(ledger_path) if ledger_path else DEFAULT_LEDGER
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Ledger not found at {path}")
    ledger = pd.read_csv(path)
    try:
        fc = QuantileForecaster()
        fc.fit(ledger, num_boost_round=300)
        fc.save(MODELS_DIR)
        _forecaster = fc
    except CashflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TrainResponse(
        status="trained",
        merchants=int(fc.daily_history["merchant_id"].nunique()),
        daily_rows=int(len(fc.daily_history)),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_model() -> QuantileForecaster:
    if _forecaster is None:
        raise HTTPException(
            status_code=503,
            detail="Models not yet trained. Run POST /train or the training script first.",
        )
    return _forecaster
