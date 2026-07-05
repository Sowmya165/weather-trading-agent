"""
database/db.py

Persistence layer using SQLAlchemy 2.0 (async) over SQLite.

Why SQLAlchemy over raw sqlite3:
  - async-native via aiosqlite driver — non-blocking inside our asyncio
    event loop, so DB writes don't stall the trading pipeline
  - Pydantic models map to SQLAlchemy rows via a single to_row() helper;
    no ORM inheritance required, keeping models clean
  - Schema migrations are handled by create_all() on startup — fine for an
    assessment; a production system would use Alembic

Three tables mirror our three core Pydantic types:
  predictions    — one row per LLM call (city, prob, reasoning, edge, EV)
  risk_decisions — one row per RiskManager.evaluate() call, FK to prediction
  trades         — one row per paper trade (primary + hedge both stored here)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean, Column, Float, Integer, String, Text,
    create_engine, text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column

from models.market import Prediction, RiskDecision, Trade, TradeStatus

logger = logging.getLogger("database.db")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ── ORM table definitions ──────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: MappedColumn[int]            = mapped_column(Integer, primary_key=True, autoincrement=True)
    condition_id: MappedColumn[str]  = mapped_column(String, nullable=True, index=True)
    city: MappedColumn[str]          = mapped_column(String, nullable=False, index=True)
    question: MappedColumn[str]      = mapped_column(Text,   nullable=True)
    predicted_probability: MappedColumn[float] = mapped_column(Float, nullable=False)
    market_implied_probability: MappedColumn[float | None] = mapped_column(Float, nullable=True)
    edge: MappedColumn[float | None] = mapped_column(Float,  nullable=True)
    expected_value: MappedColumn[float | None] = mapped_column(Float, nullable=True)
    confidence: MappedColumn[float]  = mapped_column(Float,  nullable=False, default=0.5)
    reasoning: MappedColumn[str]     = mapped_column(Text,   nullable=False)
    generated_at: MappedColumn[str]  = mapped_column(String, nullable=False)  # ISO-8601


class RiskDecisionRow(Base):
    __tablename__ = "risk_decisions"

    id: MappedColumn[int]                   = mapped_column(Integer, primary_key=True, autoincrement=True)
    condition_id: MappedColumn[str]         = mapped_column(String,  nullable=False, index=True)
    kelly_fraction_applied: MappedColumn[float] = mapped_column(Float, nullable=False)
    raw_kelly_stake_usd: MappedColumn[float]    = mapped_column(Float, nullable=False)
    capped_stake_usd: MappedColumn[float]       = mapped_column(Float, nullable=False)
    approved: MappedColumn[bool]            = mapped_column(Boolean, nullable=False)
    reasons_json: MappedColumn[str]         = mapped_column(Text,    nullable=False)  # JSON list
    hedge_condition_id: MappedColumn[str | None] = mapped_column(String, nullable=True)
    hedge_stake_usd: MappedColumn[float | None]  = mapped_column(Float,  nullable=True)
    recorded_at: MappedColumn[str]          = mapped_column(String,  nullable=False)


class TradeRow(Base):
    __tablename__ = "trades"

    id: MappedColumn[str]            = mapped_column(String,  primary_key=True)
    condition_id: MappedColumn[str]  = mapped_column(String,  nullable=False, index=True)
    city: MappedColumn[str]          = mapped_column(String,  nullable=False, index=True)
    side: MappedColumn[str]          = mapped_column(String,  nullable=False)
    outcome_name: MappedColumn[str]  = mapped_column(String,  nullable=False)
    outcome_token_id: MappedColumn[str] = mapped_column(String, nullable=False)
    stake_usd: MappedColumn[float]   = mapped_column(Float,   nullable=False)
    size_shares: MappedColumn[float] = mapped_column(Float,   nullable=False)
    entry_price: MappedColumn[float] = mapped_column(Float,   nullable=False)
    status: MappedColumn[str]        = mapped_column(String,  nullable=False, index=True)
    is_hedge: MappedColumn[bool]     = mapped_column(Boolean, nullable=False, default=False)
    linked_trade_id: MappedColumn[str | None] = mapped_column(String, nullable=True)
    placed_at: MappedColumn[str]     = mapped_column(String,  nullable=False)
    reasoning_summary: MappedColumn[str] = mapped_column(Text, nullable=False)
    order_payload_json: MappedColumn[str] = mapped_column(Text, nullable=False)
    market_question: MappedColumn[str]   = mapped_column(Text, nullable=True, default="")
    resolved_at: MappedColumn[str | None] = mapped_column(String, nullable=True)
    resolution: MappedColumn[str | None]  = mapped_column(String, nullable=True)
    pnl_usd: MappedColumn[float | None]   = mapped_column(Float,  nullable=True)


# ── DatabaseManager ────────────────────────────────────────────────────────


class DatabaseManager:
    """
    Async-safe DB access layer. Callers never write SQL directly — every
    operation goes through a typed method here, keeping SQL isolated from
    business logic and making it trivially mockable in tests.

    Usage:
        db = DatabaseManager("sqlite+aiosqlite:///database/app.db")
        await db.init()                     # creates tables if not exist
        await db.insert_cycle(p, r, t)      # one full pipeline result
        preds = await db.fetch_predictions(limit=50)
    """

    def __init__(self, database_url: str):
        self._url = database_url
        self._engine = create_async_engine(database_url, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init(self) -> None:
        """Create all tables. Idempotent — safe to call on every startup."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialised at %s", self._url)

    async def insert_cycle(
        self,
        prediction: Prediction,
        risk_decision: RiskDecision,
        trade: Trade,
    ) -> None:
        """
        Atomically persist one complete pipeline cycle (prediction + risk
        decision + trade). All three rows are committed together so the DB
        never contains a partial cycle from a crash mid-write.
        """
        async with self._session_factory() as session:
            async with session.begin():
                now_iso = datetime.now(timezone.utc).isoformat()

                session.add(PredictionRow(
                    condition_id=prediction.condition_id,
                    city=prediction.city,
                    question=prediction.question,
                    predicted_probability=prediction.predicted_probability,
                    market_implied_probability=prediction.market_implied_probability,
                    edge=prediction.edge,
                    expected_value=prediction.expected_value,
                    confidence=prediction.confidence,
                    reasoning=prediction.reasoning,
                    generated_at=(prediction.generated_at or datetime.now(timezone.utc)).isoformat(),
                ))

                session.add(RiskDecisionRow(
                    condition_id=risk_decision.condition_id,
                    kelly_fraction_applied=risk_decision.kelly_fraction_applied,
                    raw_kelly_stake_usd=risk_decision.raw_kelly_stake_usd,
                    capped_stake_usd=risk_decision.capped_stake_usd,
                    approved=risk_decision.approved,
                    reasons_json=json.dumps(risk_decision.reasons),
                    hedge_condition_id=risk_decision.hedge_condition_id,
                    hedge_stake_usd=risk_decision.hedge_stake_usd,
                    recorded_at=now_iso,
                ))

                session.add(_trade_to_row(trade))

        logger.info(
            "Cycle persisted | city=%s condition=%s trade_id=%s",
            prediction.city, prediction.condition_id, trade.id,
        )

    async def insert_trade(self, trade: Trade) -> None:
        """Insert a standalone trade row (e.g. a hedge leg persisted separately)."""
        async with self._session_factory() as session:
            async with session.begin():
                session.add(_trade_to_row(trade))

    # ── Fetch methods (consumed by FastAPI endpoints) ──────────────────

    async def fetch_predictions(self, limit: int = 100, city: str | None = None) -> list[dict]:
        async with self._session_factory() as session:
            q = "SELECT * FROM predictions"
            params: dict[str, Any] = {}
            if city:
                q += " WHERE city = :city"
                params["city"] = city
            q += " ORDER BY id DESC LIMIT :limit"
            params["limit"] = limit
            result = await session.execute(text(q), params)
            return [dict(row._mapping) for row in result.fetchall()]

    async def fetch_risk_decisions(self, limit: int = 100) -> list[dict]:
        async with self._session_factory() as session:
            result = await session.execute(
                text("SELECT * FROM risk_decisions ORDER BY id DESC LIMIT :limit"),
                {"limit": limit},
            )
            rows = []
            for row in result.fetchall():
                d = dict(row._mapping)
                d["reasons"] = json.loads(d.pop("reasons_json", "[]"))
                d["approved"] = bool(d["approved"])  # SQLite stores bool as 0/1 integer
                rows.append(d)
            return rows

    async def fetch_trades(
        self,
        limit: int = 100,
        city: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        async with self._session_factory() as session:
            conditions, params = [], {}
            if city:
                conditions.append("city = :city")
                params["city"] = city
            if status:
                conditions.append("status = :status")
                params["status"] = status
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params["limit"] = limit
            result = await session.execute(
                text(f"SELECT * FROM trades {where} ORDER BY placed_at DESC LIMIT :limit"),
                params,
            )
            return [dict(row._mapping) for row in result.fetchall()]

    async def fetch_status(self) -> dict:
        """Aggregate stats for the /api/status endpoint."""
        async with self._session_factory() as session:
            total_trades = (await session.execute(
                text("SELECT COUNT(*) FROM trades WHERE status != 'rejected'")
            )).scalar() or 0

            filled_trades = (await session.execute(
                text("SELECT COUNT(*) FROM trades WHERE status = 'filled'")
            )).scalar() or 0

            total_staked = (await session.execute(
                text("SELECT COALESCE(SUM(stake_usd), 0) FROM trades WHERE status = 'filled'")
            )).scalar() or 0.0

            realized_pnl = (await session.execute(
                text("SELECT COALESCE(SUM(pnl_usd), 0) FROM trades WHERE pnl_usd IS NOT NULL")
            )).scalar() or 0.0

            last_prediction = (await session.execute(
                text("SELECT generated_at FROM predictions ORDER BY id DESC LIMIT 1")
            )).scalar()

            total_predictions = (await session.execute(
                text("SELECT COUNT(*) FROM predictions")
            )).scalar() or 0

        return {
            "status": "operational",
            "total_predictions": total_predictions,
            "total_trades_placed": total_trades,
            "filled_trades": filled_trades,
            "total_staked_usd": round(float(total_staked), 2),
            "realized_pnl_usd": round(float(realized_pnl), 2),
            "last_run_at": last_prediction,
        }

    async def close(self) -> None:
        await self._engine.dispose()
        logger.info("Database connection closed.")


# ── Helpers ────────────────────────────────────────────────────────────────


def _trade_to_row(trade: Trade) -> TradeRow:
    return TradeRow(
        id=trade.id,
        condition_id=trade.condition_id,
        city=trade.city,
        side=trade.side.value,
        outcome_name=trade.outcome_name,
        outcome_token_id=trade.outcome_token_id,
        stake_usd=trade.stake_usd,
        size_shares=trade.size_shares,
        entry_price=trade.entry_price,
        status=trade.status.value,
        is_hedge=trade.is_hedge,
        linked_trade_id=trade.linked_trade_id,
        placed_at=trade.placed_at.isoformat(),
        reasoning_summary=trade.reasoning_summary,
        order_payload_json=trade.order_payload_json,
        market_question=trade.market_question,
        resolved_at=trade.resolved_at.isoformat() if trade.resolved_at else None,
        resolution=trade.resolution,
        pnl_usd=trade.pnl_usd,
    )
