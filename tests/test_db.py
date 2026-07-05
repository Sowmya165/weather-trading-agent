"""
Tests for DatabaseManager using an in-memory SQLite database so nothing
hits the filesystem and every test starts from a clean state.
"""
import pytest
from datetime import datetime, timezone

from database.db import DatabaseManager
from models.market import (
    Prediction, RiskDecision, Trade, TradeSide, TradeStatus,
)


@pytest.fixture
async def db():
    """Fresh in-memory DB per test — no teardown needed."""
    manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    await manager.init()
    yield manager
    await manager.close()


def _prediction(city: str = "London", condition_id: str = "cond-1") -> Prediction:
    return Prediction(
        city=city,
        condition_id=condition_id,
        question="Will it rain tomorrow?",
        predicted_probability=0.72,
        market_implied_probability=0.50,
        edge=0.22,
        expected_value=0.11,
        confidence=0.88,
        reasoning="Strong low-pressure system approaching from the west.",
        generated_at=datetime.now(timezone.utc),
    )


def _risk_decision(condition_id: str = "cond-1") -> RiskDecision:
    return RiskDecision(
        condition_id=condition_id,
        kelly_fraction_applied=0.055,
        raw_kelly_stake_usd=55.0,
        capped_stake_usd=50.0,
        approved=True,
        reasons=["Capped to max per-position limit ($50.00)."],
    )


def _trade(trade_id: str = "trade-abc", condition_id: str = "cond-1", city: str = "London") -> Trade:
    return Trade(
        id=trade_id,
        condition_id=condition_id,
        city=city,
        side=TradeSide.BUY,
        outcome_name="Yes",
        outcome_token_id="tok-yes",
        stake_usd=50.0,
        size_shares=100.0,
        entry_price=0.50,
        status=TradeStatus.FILLED,
        placed_at=datetime.now(timezone.utc),
        reasoning_summary="Edge detected; Kelly approved.",
        order_payload_json='{"orderType":"FOK"}',
        market_question="Will it rain tomorrow?",
    )


async def test_insert_cycle_and_fetch_predictions(db: DatabaseManager):
    await db.insert_cycle(_prediction(), _risk_decision(), _trade())
    rows = await db.fetch_predictions(limit=10)
    assert len(rows) == 1
    assert rows[0]["city"] == "London"
    assert rows[0]["predicted_probability"] == 0.72
    assert rows[0]["reasoning"] == "Strong low-pressure system approaching from the west."


async def test_insert_cycle_and_fetch_risk_decisions(db: DatabaseManager):
    await db.insert_cycle(_prediction(), _risk_decision(), _trade())
    rows = await db.fetch_risk_decisions(limit=10)
    assert len(rows) == 1
    assert rows[0]["approved"] is True
    assert rows[0]["capped_stake_usd"] == 50.0
    # reasons_json must be deserialized back to a list
    assert isinstance(rows[0]["reasons"], list)
    assert "Capped" in rows[0]["reasons"][0]


async def test_insert_cycle_and_fetch_trades(db: DatabaseManager):
    await db.insert_cycle(_prediction(), _risk_decision(), _trade())
    rows = await db.fetch_trades(limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == "trade-abc"
    assert rows[0]["status"] == "filled"
    assert rows[0]["stake_usd"] == 50.0


async def test_fetch_trades_filtered_by_city(db: DatabaseManager):
    await db.insert_cycle(_prediction(city="London"), _risk_decision("c1"), _trade("t1", "c1", "London"))
    await db.insert_cycle(_prediction(city="Tokyo"), _risk_decision("c2"), _trade("t2", "c2", "Tokyo"))
    rows = await db.fetch_trades(city="Tokyo")
    assert len(rows) == 1
    assert rows[0]["city"] == "Tokyo"


async def test_fetch_trades_filtered_by_status(db: DatabaseManager):
    await db.insert_cycle(_prediction(), _risk_decision(), _trade())
    hedge = _trade(trade_id="hedge-xyz")
    hedge.status = TradeStatus.HEDGED
    hedge.is_hedge = True
    await db.insert_trade(hedge)
    filled = await db.fetch_trades(status="filled")
    hedged = await db.fetch_trades(status="hedged")
    assert all(r["status"] == "filled" for r in filled)
    assert all(r["status"] == "hedged" for r in hedged)


async def test_status_aggregates_correctly(db: DatabaseManager):
    await db.insert_cycle(_prediction(), _risk_decision(), _trade())
    status = await db.fetch_status()
    assert status["status"] == "operational"
    assert status["total_predictions"] == 1
    assert status["filled_trades"] == 1
    assert status["total_staked_usd"] == 50.0


async def test_multiple_cycles_accumulate(db: DatabaseManager):
    for i in range(3):
        await db.insert_cycle(
            _prediction(city="Paris", condition_id=f"cond-{i}"),
            _risk_decision(condition_id=f"cond-{i}"),
            _trade(trade_id=f"trade-{i}", condition_id=f"cond-{i}"),
        )
    preds = await db.fetch_predictions(limit=10)
    trades = await db.fetch_trades(limit=10)
    assert len(preds) == 3
    assert len(trades) == 3


async def test_init_is_idempotent(db: DatabaseManager):
    """Calling init() twice must not raise or duplicate tables."""
    await db.init()
    rows = await db.fetch_predictions()
    assert rows == []
