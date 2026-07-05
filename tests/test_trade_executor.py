import json
from datetime import datetime, timezone

import pytest

from models.market import (
    MarketOutcome, PolymarketMarket, Prediction,
    RiskDecision, TradeStatus, TradeSide,
)
from services.trade_executor import PolymarketPaperTrader


def _market(condition_id: str = "cond-1") -> PolymarketMarket:
    return PolymarketMarket(
        condition_id=condition_id,
        question="Will it rain in New York tomorrow?",
        city="New York",
        closes_at=datetime.now(timezone.utc),
        outcomes=[
            MarketOutcome(name="Yes", token_id="tok-yes", price=0.45),
            MarketOutcome(name="No",  token_id="tok-no",  price=0.55),
        ],
    )


def _prediction() -> Prediction:
    return Prediction(
        city="New York",
        predicted_probability=0.70,
        reasoning="High humidity and low pressure system approaching.",
        condition_id="cond-1",
        question="Will it rain in New York tomorrow?",
        market_implied_probability=0.45,
        edge=0.25,
        confidence=0.85,
        generated_at=datetime.now(timezone.utc),
    )


def _approved(stake: float = 40.0) -> RiskDecision:
    return RiskDecision(
        condition_id="cond-1",
        kelly_fraction_applied=0.08,
        raw_kelly_stake_usd=50.0,
        capped_stake_usd=stake,
        approved=True,
        reasons=[],
    )


def _rejected() -> RiskDecision:
    return RiskDecision(
        condition_id="cond-1",
        kelly_fraction_applied=0.0,
        raw_kelly_stake_usd=0.0,
        capped_stake_usd=0.0,
        approved=False,
        reasons=["No edge."],
    )


def test_filled_trade_has_correct_fields():
    result = PolymarketPaperTrader().execute_trade(_approved(), _prediction(), _market())
    assert result.primary.status == TradeStatus.FILLED
    assert result.primary.city == "New York"
    assert result.primary.condition_id == "cond-1"
    assert result.primary.side == TradeSide.BUY
    assert result.primary.is_hedge is False


def test_shares_calculated_from_stake_and_price():
    result = PolymarketPaperTrader().execute_trade(_approved(stake=45.0), _prediction(), _market())
    assert abs(result.primary.size_shares - 100.0) < 0.001  # 45 / 0.45


def test_order_payload_is_valid_json_with_required_fields():
    result = PolymarketPaperTrader().execute_trade(_approved(), _prediction(), _market())
    payload = json.loads(result.primary.order_payload_json)
    for field in ("orderType", "tokenID", "side", "price", "size", "conditionId"):
        assert field in payload, f"Missing CLOB field: {field}"
    assert payload["side"] == "BUY"
    assert payload["orderType"] == "FOK"


def test_rejected_decision_produces_rejected_trade():
    result = PolymarketPaperTrader().execute_trade(_rejected(), _prediction(), _market())
    assert result.primary.status == TradeStatus.REJECTED
    assert result.primary.stake_usd == 0.0
    assert result.hedge is None


def test_hedge_trade_placed_on_no_outcome():
    decision = RiskDecision(
        condition_id="cond-1", kelly_fraction_applied=0.08,
        raw_kelly_stake_usd=50.0, capped_stake_usd=40.0, approved=True,
        reasons=[], hedge_condition_id="cond-1", hedge_stake_usd=10.0,
    )
    result = PolymarketPaperTrader().execute_trade(decision, _prediction(), _market())
    assert result.hedge is not None
    assert result.hedge.outcome_name == "No"
    assert result.hedge.is_hedge is True
    assert result.hedge.stake_usd == 10.0


def test_execute_with_hedge_links_trade_ids():
    decision = RiskDecision(
        condition_id="cond-1", kelly_fraction_applied=0.08,
        raw_kelly_stake_usd=50.0, capped_stake_usd=40.0, approved=True,
        reasons=[], hedge_condition_id="cond-1", hedge_stake_usd=10.0,
    )
    result = PolymarketPaperTrader().execute_trade(decision, _prediction(), _market())
    assert result.hedge is not None
    assert result.hedge.linked_trade_id == result.primary.id


def test_nonstandard_outcome_names_fall_back_to_first_outcome():
    market = PolymarketMarket(
        condition_id="cond-x", question="Will NYC exceed 90F?", city="New York",
        closes_at=datetime.now(timezone.utc),
        outcomes=[
            MarketOutcome(name="Over",  token_id="tok-over",  price=0.40),
            MarketOutcome(name="Under", token_id="tok-under", price=0.60),
        ],
    )
    result = PolymarketPaperTrader().execute_trade(_approved(), _prediction(), market)
    # No "Yes" outcome — should fall back to outcomes[0] = "Over"
    assert result.primary.outcome_name == "Over"


def test_ledger_accumulates_across_trades():
    trader = PolymarketPaperTrader()
    trader.execute_trade(_approved(), _prediction(), _market())
    trader.execute_trade(_approved(), _prediction(), _market())
    assert len(trader.ledger) == 2


def test_summary_reflects_staked_amount():
    trader = PolymarketPaperTrader()
    trader.execute_trade(_approved(stake=40.0), _prediction(), _market())
    summary = trader.summary()
    assert summary["total_trades"] == 1
    assert summary["total_staked_usd"] == 40.0
