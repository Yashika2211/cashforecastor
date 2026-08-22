"""
SQLAlchemy models and session wiring.

Tables
------
forecast_audit_log  – one row per forecast point returned by GET /forecast/{merchant_id}.
                      Provides an audit trail: who asked, when, what was returned.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = "sqlite:///./cashflow.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class ForecastAuditLog(Base):
    """
    Every call to GET /forecast/{merchant_id} writes one row per forecast day here.

    Fields
    ------
    merchant_id       – the merchant whose forecast was requested
    request_timestamp – UTC time the request was received
    forecast_date     – calendar date the prediction covers
    horizon_day       – 1-based day offset from the forecast origin
    p10               – 10th-percentile predicted net_settled_amount
    p50               – 50th-percentile (median) predicted net_settled_amount
    p90               – 90th-percentile predicted net_settled_amount
    """

    __tablename__ = "forecast_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    merchant_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    request_timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False)
    horizon_day: Mapped[int] = mapped_column(Integer, nullable=False)
    p10: Mapped[float] = mapped_column(Float, nullable=False)
    p50: Mapped[float] = mapped_column(Float, nullable=False)
    p90: Mapped[float] = mapped_column(Float, nullable=False)


# Keep the old name around so any stale imports don't break.
ForecastRecord = ForecastAuditLog


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
