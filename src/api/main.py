"""
FastAPI application for the AI Finance Controller (cash-position forecasting
+ books reconciliation).

Endpoints
---------
GET  /health                       – liveness check + model status
GET  /forecast/{merchant_id}       – 14-day P10/P50/P90 forecast; logged to audit table
GET  /backtest-metrics             – saved backtest summary (coverage + pinball per category)
GET  /exceptions                   – current low-confidence flags from exceptions.py
POST /train                        – (utility) re-train models from the default ledger
GET  /reconciliation-summary       – matcher run summary + accuracy (when scored)
GET  /reconciliation-matches       – resolved ledger<->bank match groups
GET  /reconciliation-exceptions    – unresolved records with reason codes
GET  /reconciliation-accuracy      – scored precision/recall/false-match rates
POST /reconcile                    – (utility) re-run the matcher on the default batch
"""
from __future__ import annotations

import json
import math
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
    ReconciliationError,
)
from src.forecast import MODELS_DIR, REPORTS_DIR, QuantileForecaster
from src.reconcile import run_reconciliation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECON_LEDGER = ROOT / "data" / "recon_ledger.csv"
DEFAULT_RECON_BANK = ROOT / "data" / "recon_bank_settlement.csv"


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
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
    # Global CQR = one q_hat pooled across all categories
    global_coverage: Optional[float] = None
    # Per-category CQR = separate q_hat per category; primary number shown in dashboard
    coverage_p10_p90: float          # alias for percat_coverage (frontend reads this)
    pinball_p10: float               # per-category calibrated pinball
    pinball_p50: float
    pinball_p90: float
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


class ReconciliationMatch(BaseModel):
    match_group_id: str
    ledger_txn_ids: List[str]
    bank_row_id: str
    utr: str
    rule: str
    member_count: int
    matched_amount: float
    expected_amount: float
    delta: float
    confidence: Optional[float] = None


class ReconciliationException(BaseModel):
    record_type: str            # "ledger" | "bank"
    record_id: str
    merchant_id: str
    reason_code: str
    detail: Optional[str] = None
    candidate_id: Optional[str] = None
    candidate_amount: Optional[float] = None
    delta: Optional[float] = None
    confidence: Optional[float] = None
    competing_candidates: Optional[List[str]] = None
    note: Optional[str] = None


class ReconciliationSummary(BaseModel):
    n_ledger_rows: int
    n_bank_rows: int
    n_ledger_matched: int
    n_ledger_exception: int
    match_rate: float
    records_per_second: float
    rule_counts: Dict[str, int]
    reason_code_counts: Dict[str, int]
    # populated only when reconciliation_accuracy.json also exists, else all None:
    precision: Optional[float] = None
    recall: Optional[float] = None
    false_match_rate_pooled: Optional[float] = None
    wrong_attribution_rate: Optional[float] = None
    trap_recall: Optional[float] = None


class ReconciliationAccuracy(BaseModel):
    precision: float
    recall: float
    match_rate: float
    false_match_rate_ledger: float
    false_match_rate_bank: float
    false_match_rate_pooled: float
    wrong_attribution_rate: float
    duplicate_precision: float
    duplicate_recall: float
    trap_recall: float
    reason_code_accuracy: float
    records_per_second: float
    warning: Optional[str] = None


class ReconcileRunResponse(BaseModel):
    status: str
    n_ledger_rows: int
    n_bank_rows: int
    match_rate: float
    records_per_second: float


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
            global_coverage=float(_col(df, "cal_coverage").iloc[i])
                if "cal_coverage" in df.columns else None,
            # Per-category CQR is the primary number; fall back through global then raw
            coverage_p10_p90=float(
                _col(df, "percat_coverage", "cal_coverage", "coverage_p10_p90").iloc[i]
            ),
            pinball_p10=float(_col(df, "percat_pinball_p10", "cal_pinball_p10", "pinball_p10").iloc[i]),
            pinball_p50=float(_col(df, "percat_pinball_p50", "cal_pinball_p50", "pinball_p50").iloc[i]),
            pinball_p90=float(_col(df, "percat_pinball_p90", "cal_pinball_p90", "pinball_p90").iloc[i]),
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


# ---------------------------------------------------------------------------
# Reconciliation ("books half") routes
# ---------------------------------------------------------------------------

@app.get("/reconciliation-summary", response_model=ReconciliationSummary)
def get_reconciliation_summary() -> ReconciliationSummary:
    """
    Reads reports/reconciliation_summary.json, merging in
    reports/reconciliation_accuracy.json when present (precision/recall/
    false_match_rate/wrong_attribution/trap_recall stay None otherwise).
    """
    summary_path = REPORTS_DIR / "reconciliation_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Reconciliation not run. Run run_reconciliation.py first.")
    with open(summary_path) as f:
        summary = json.load(f)

    accuracy: Dict[str, Any] = {}
    accuracy_path = REPORTS_DIR / "reconciliation_accuracy.json"
    if accuracy_path.exists():
        with open(accuracy_path) as f:
            accuracy = json.load(f)

    return ReconciliationSummary(
        n_ledger_rows=summary["n_ledger_rows"],
        n_bank_rows=summary["n_bank_rows"],
        n_ledger_matched=summary["n_ledger_matched"],
        n_ledger_exception=summary["n_ledger_exception"],
        match_rate=summary["match_rate"],
        records_per_second=summary["records_per_second"],
        rule_counts=summary["rule_counts"],
        reason_code_counts=summary["reason_code_counts"],
        precision=accuracy.get("precision"),
        recall=accuracy.get("recall"),
        false_match_rate_pooled=accuracy.get("false_match_rate_pooled"),
        wrong_attribution_rate=accuracy.get("wrong_attribution_rate"),
        trap_recall=accuracy.get("trap_recall"),
    )


@app.get("/reconciliation-matches", response_model=List[ReconciliationMatch])
def get_reconciliation_matches() -> List[ReconciliationMatch]:
    """Resolved ledger<->bank match groups from the last matcher run."""
    matches_path = REPORTS_DIR / "reconciliation_matches.csv"
    if not matches_path.exists():
        raise HTTPException(status_code=404, detail="Reconciliation not run. Run run_reconciliation.py first.")
    df = pd.read_csv(matches_path)
    return [
        ReconciliationMatch(
            match_group_id=str(row.match_group_id),
            ledger_txn_ids=str(row.ledger_txn_ids).split("|") if row.ledger_txn_ids else [],
            bank_row_id=str(row.bank_row_id),
            utr=str(row.utr),
            rule=str(row.rule),
            member_count=int(row.member_count),
            matched_amount=float(row.matched_amount),
            expected_amount=float(row.expected_amount),
            delta=float(row.delta),
            confidence=_none_if_nan(row.confidence),
        )
        for row in df.itertuples(index=False)
    ]


@app.get("/reconciliation-exceptions", response_model=List[ReconciliationException])
def get_reconciliation_exceptions() -> List[ReconciliationException]:
    """Unresolved records from the last matcher run, each with a specific reason code."""
    exceptions_path = REPORTS_DIR / "reconciliation_exceptions.json"
    if not exceptions_path.exists():
        raise HTTPException(status_code=404, detail="Reconciliation not run. Run run_reconciliation.py first.")
    with open(exceptions_path) as f:
        raw = json.load(f)
    return [
        ReconciliationException(
            record_type=item["record_type"],
            record_id=item["record_id"],
            merchant_id=item.get("merchant_id") or "",
            reason_code=item["reason_code"],
            detail=item.get("detail"),
            candidate_id=item.get("candidate_id"),
            candidate_amount=_none_if_nan(item.get("candidate_amount")),
            delta=_none_if_nan(item.get("delta")),
            confidence=_none_if_nan(item.get("confidence")),
            competing_candidates=item["competing_candidates"].split("|")
                if item.get("competing_candidates") else None,
            note=item.get("note"),
        )
        for item in raw
    ]


@app.get("/reconciliation-accuracy", response_model=ReconciliationAccuracy)
def get_reconciliation_accuracy() -> ReconciliationAccuracy:
    """Scored precision/recall/false-match rates from the last score_reconciliation.py run."""
    accuracy_path = REPORTS_DIR / "reconciliation_accuracy.json"
    if not accuracy_path.exists():
        raise HTTPException(status_code=404, detail="Reconciliation not scored. Run score_reconciliation.py first.")
    with open(accuracy_path) as f:
        acc = json.load(f)
    return ReconciliationAccuracy(
        precision=acc["precision"],
        recall=acc["recall"],
        match_rate=acc["match_rate"],
        false_match_rate_ledger=acc["false_match_rate_ledger"],
        false_match_rate_bank=acc["false_match_rate_bank"],
        false_match_rate_pooled=acc["false_match_rate_pooled"],
        wrong_attribution_rate=acc["wrong_attribution_rate"],
        duplicate_precision=acc["duplicate_precision"],
        duplicate_recall=acc["duplicate_recall"],
        trap_recall=acc["trap_recall"],
        reason_code_accuracy=acc["reason_code_accuracy"],
        records_per_second=acc["records_per_second"] or 0.0,
        warning=acc.get("warning"),
    )


@app.post("/reconcile", response_model=ReconcileRunResponse)
def reconcile_now() -> ReconcileRunResponse:
    """
    Utility mirroring POST /train: runs the matcher in-process on the default
    data/recon_ledger.csv + data/recon_bank_settlement.csv and writes the same
    report files run_reconciliation.py would. Dev convenience only.
    """
    if not DEFAULT_RECON_LEDGER.exists() or not DEFAULT_RECON_BANK.exists():
        raise HTTPException(status_code=404, detail="Reconciliation data not found. Run data/generate_synthetic_recon.py first.")
    ledger = pd.read_csv(DEFAULT_RECON_LEDGER)
    bank = pd.read_csv(DEFAULT_RECON_BANK)
    try:
        result = run_reconciliation(ledger, bank)
    except ReconciliationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matches_out = result.matches.copy()
    if not matches_out.empty:
        matches_out["ledger_txn_ids"] = matches_out["ledger_txn_ids"].apply(lambda ids: "|".join(ids))
    matches_out.to_csv(REPORTS_DIR / "reconciliation_matches.csv", index=False)

    exceptions_records = result.exceptions.to_dict(orient="records")
    with open(REPORTS_DIR / "reconciliation_exceptions.json", "w") as f:
        json.dump(exceptions_records, f, indent=2, default=lambda x: None if pd.isna(x) else x)

    n_ledger, n_bank = result.stats["n_ledger_rows"], result.stats["n_bank_rows"]
    n_ledger_matched = int(sum(result.matches["member_count"])) if not result.matches.empty else 0
    n_ledger_exception = int((result.exceptions["record_type"] == "ledger").sum()) if not result.exceptions.empty else 0
    match_rate = n_ledger_matched / n_ledger if n_ledger else 0.0

    rule_counts = result.stats["rule_counts"]
    reason_code_counts = result.stats["reason_code_counts"]
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "ledger_path": str(DEFAULT_RECON_LEDGER),
        "bank_path": str(DEFAULT_RECON_BANK),
        "n_ledger_rows": n_ledger,
        "n_bank_rows": n_bank,
        "n_ledger_matched": n_ledger_matched,
        "n_ledger_exception": n_ledger_exception,
        "match_rate": round(match_rate, 4),
        "rule_counts": rule_counts,
        "reason_code_counts": reason_code_counts,
        "elapsed_seconds": result.stats["elapsed_seconds"],
        "records_per_second": result.stats["records_per_second"],
    }
    with open(REPORTS_DIR / "reconciliation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return ReconcileRunResponse(
        status="reconciled", n_ledger_rows=n_ledger, n_bank_rows=n_bank,
        match_rate=round(match_rate, 4), records_per_second=result.stats["records_per_second"],
    )


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
