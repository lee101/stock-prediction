from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import math

import pytz
from sqlalchemy import DateTime, Float, Integer, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DEFAULT_MIN_RISK_THRESHOLD = 0.01
MAX_RISK_THRESHOLD = 1.5


def _resolve_database_path() -> Path:
    configured = os.getenv("PORTFOLIO_DB_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "stock.db"


DB_PATH = _resolve_database_path()
DATABASE_URL = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    risk_threshold: Mapped[float] = mapped_column(Float, nullable=False)


@dataclass(frozen=True)
class PortfolioSnapshotRecord:
    observed_at: datetime
    portfolio_value: float
    risk_threshold: float


_engine: Engine | None = None
_initialized = False
_current_risk_threshold: Optional[float] = None


def _get_engine():
    global _engine
    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            DATABASE_URL,
            future=True,
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engine


def _ensure_initialized() -> None:
    global _initialized
    if not _initialized:
        Base.metadata.create_all(_get_engine())
        _initialized = True


def _coerce_to_utc(observed_at: Optional[datetime]) -> datetime:
    if observed_at is None:
        observed_at = datetime.now(timezone.utc)
    elif observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    else:
        observed_at = observed_at.astimezone(timezone.utc)
    return observed_at


def _select_latest_snapshot(session: Session) -> Optional[PortfolioSnapshot]:
    stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.observed_at.desc()).limit(1)
    return session.execute(stmt).scalars().first()


def _select_reference_snapshot(session: Session, observed_at: datetime) -> Optional[PortfolioSnapshot]:
    est = pytz.timezone("US/Eastern")
    local_date = observed_at.astimezone(est).date()
    local_start = datetime.combine(local_date, time.min, tzinfo=est)
    local_start_utc = local_start.astimezone(timezone.utc)

    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.observed_at < local_start_utc)
        .order_by(PortfolioSnapshot.observed_at.desc())
        .limit(1)
    )
    reference = session.execute(stmt).scalars().first()
    if reference is not None:
        return reference
    return _select_latest_snapshot(session)


def record_portfolio_snapshot(
    portfolio_value: float,
    observed_at: Optional[datetime] = None,
    day_pl: Optional[float] = None,
) -> PortfolioSnapshotRecord:
    """Persist a portfolio snapshot and update the global risk threshold.

    Args:
        portfolio_value: Current portfolio or exposure value being tracked.
        observed_at: Optional timestamp for the snapshot. Defaults to now in UTC.
        day_pl: Optional realised or unrealised day P&L. When provided, the risk threshold
            will be set to MAX_RISK_THRESHOLD when the value is non-negative and
            DEFAULT_MIN_RISK_THRESHOLD when the value is negative. If omitted or invalid,
            the threshold falls back to comparing the portfolio value against the
            reference snapshot.
    """
    global _current_risk_threshold

    _ensure_initialized()
    observed_at = _coerce_to_utc(observed_at)

    with Session(_get_engine()) as session:
        reference = _select_reference_snapshot(session, observed_at)
        effective_day_pl: Optional[float]
        if day_pl is None:
            effective_day_pl = None
        else:
            try:
                effective_day_pl = float(day_pl)
            except (TypeError, ValueError):
                effective_day_pl = None
            else:
                if not math.isfinite(effective_day_pl):
                    effective_day_pl = None

        if effective_day_pl is not None:
            risk_threshold = MAX_RISK_THRESHOLD if effective_day_pl >= 0 else DEFAULT_MIN_RISK_THRESHOLD
        elif reference is None:
            risk_threshold = DEFAULT_MIN_RISK_THRESHOLD
        else:
            risk_threshold = MAX_RISK_THRESHOLD if portfolio_value >= reference.portfolio_value else DEFAULT_MIN_RISK_THRESHOLD

        snapshot = PortfolioSnapshot(
            observed_at=observed_at,
            portfolio_value=float(portfolio_value),
            risk_threshold=float(risk_threshold),
        )
        session.add(snapshot)
        session.commit()
        session.refresh(snapshot)

    _current_risk_threshold = float(snapshot.risk_threshold)
    return PortfolioSnapshotRecord(
        observed_at=snapshot.observed_at,
        portfolio_value=snapshot.portfolio_value,
        risk_threshold=snapshot.risk_threshold,
    )


def get_global_risk_threshold() -> float:
    """Return the most recently calculated global risk threshold."""
    global _current_risk_threshold
    if _current_risk_threshold is not None:
        return _current_risk_threshold

    _ensure_initialized()
    with Session(_get_engine()) as session:
        latest = _select_latest_snapshot(session)
        if latest is None:
            _current_risk_threshold = DEFAULT_MIN_RISK_THRESHOLD
        else:
            _current_risk_threshold = float(latest.risk_threshold)
    return _current_risk_threshold


def fetch_snapshots(limit: Optional[int] = None) -> List[PortfolioSnapshotRecord]:
    """Return ordered portfolio snapshots for analytics/visualisation."""
    _ensure_initialized()
    stmt = select(PortfolioSnapshot).order_by(PortfolioSnapshot.observed_at.asc())
    if limit is not None:
        stmt = stmt.limit(limit)
    with Session(_get_engine()) as session:
        rows: Iterable[PortfolioSnapshot] = session.execute(stmt).scalars().all()
    return [
        PortfolioSnapshotRecord(
            observed_at=row.observed_at,
            portfolio_value=row.portfolio_value,
            risk_threshold=row.risk_threshold,
        )
        for row in rows
    ]


def fetch_latest_snapshot() -> Optional[PortfolioSnapshotRecord]:
    """Return the most recent snapshot or None if no data."""
    _ensure_initialized()
    with Session(_get_engine()) as session:
        latest = _select_latest_snapshot(session)
        if latest is None:
            return None
        return PortfolioSnapshotRecord(
            observed_at=latest.observed_at,
            portfolio_value=latest.portfolio_value,
            risk_threshold=latest.risk_threshold,
        )


def reset_cached_threshold() -> None:
    """Testing helper to reset the in-memory risk threshold cache."""
    global _current_risk_threshold
    _current_risk_threshold = None
