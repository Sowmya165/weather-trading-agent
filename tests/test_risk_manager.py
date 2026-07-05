from datetime import datetime, timezone

import pytest

from config.settings import Settings
from models.market import Prediction
from services.risk_manager import PortfolioState, RiskManager


def _settings(**overrides) -> Settings:
    base = dict(
        kelly_fraction=0.25,
        max_position_pct_of_bankroll=0.05,
        max_daily_loss_pct=0.10,
        max_portfolio_exposure_pct=0.40,
        starting_bankroll_usd=1000.0,
    )
    base.update(overrides)
    return Settings(**base)


def _prediction(probability: float, market_price: float, confidence: float = 0.85, condition_id="c1") -> Prediction:
    return Prediction(
        city="New York",
        condition_id=condition_id,
        question="Will it rain?",
        predicted_probability=probability,
        market_implied_probability=market_price,
        edge=probability - market_price,
        confidence=confidence,
        reasoning="test",
        expected_value=0.0,
        generated_at=datetime.now(timezone.utc),
    )


def _portfolio(bankroll=1000.0, exposure=0.0, pnl_today=0.0) -> PortfolioState:
    return PortfolioState(
        bankroll_usd=bankroll,
        open_exposure_usd=exposure,
        realized_pnl_today_usd=pnl_today,
        open_positions_by_condition_id={},
    )


def test_positive_edge_produces_positive_kelly_stake():
    rm = RiskManager(_settings())
    pred = _prediction(probability=0.7, market_price=0.5)  # strong edge
    decision = rm.evaluate(pred, _portfolio())
    assert decision.approved
    assert decision.capped_stake_usd > 0


def test_no_edge_rejects_trade():
    rm = RiskManager(_settings())
    pred = _prediction(probability=0.5, market_price=0.5)  # zero edge
    decision = rm.evaluate(pred, _portfolio())
    assert not decision.approved
    assert decision.capped_stake_usd == 0.0


def test_negative_edge_rejects_trade():
    rm = RiskManager(_settings())
    pred = _prediction(probability=0.3, market_price=0.6)
    decision = rm.evaluate(pred, _portfolio())
    assert not decision.approved


def test_stake_never_exceeds_max_position_limit():
    rm = RiskManager(_settings(max_position_pct_of_bankroll=0.05, kelly_fraction=1.0))
    pred = _prediction(probability=0.95, market_price=0.2)  # huge raw Kelly edge
    decision = rm.evaluate(pred, _portfolio(bankroll=1000.0))
    assert decision.capped_stake_usd <= 0.05 * 1000.0 + 1e-6


def test_daily_loss_limit_blocks_new_trades():
    rm = RiskManager(_settings(max_daily_loss_pct=0.10))
    pred = _prediction(probability=0.7, market_price=0.5)
    portfolio = _portfolio(bankroll=1000.0, pnl_today=-150.0)  # already breached -10%
    decision = rm.evaluate(pred, portfolio)
    assert not decision.approved
    assert decision.capped_stake_usd == 0.0


def test_portfolio_exposure_cap_limits_stake():
    rm = RiskManager(_settings(max_portfolio_exposure_pct=0.40, max_position_pct_of_bankroll=1.0, kelly_fraction=1.0))
    pred = _prediction(probability=0.9, market_price=0.3)
    portfolio = _portfolio(bankroll=1000.0, exposure=390.0)  # only $10 of room left
    decision = rm.evaluate(pred, portfolio)
    assert decision.capped_stake_usd <= 10.0 + 1e-6


def test_low_confidence_halves_stake():
    rm = RiskManager(_settings())
    pred_high_conf = _prediction(probability=0.7, market_price=0.5, confidence=0.9, condition_id="hi")
    pred_low_conf = _prediction(probability=0.7, market_price=0.5, confidence=0.3, condition_id="lo")
    portfolio = _portfolio()
    hi = rm.evaluate(pred_high_conf, portfolio)
    lo = rm.evaluate(pred_low_conf, portfolio)
    assert lo.capped_stake_usd < hi.capped_stake_usd


def test_invalid_market_price_is_handled_gracefully():
    rm = RiskManager(_settings())
    pred = _prediction(probability=0.7, market_price=0.0)  # invalid, would div-by-zero
    decision = rm.evaluate(pred, _portfolio())
    assert not decision.approved
    assert "error" in decision.reasons[0].lower()


def test_hedge_sized_when_correlated_markets_disagree():
    rm = RiskManager(_settings())
    primary = _prediction(probability=0.7, market_price=0.5, confidence=0.6, condition_id="primary")
    # Correlated market where our edge points the opposite way -> conflicting signal
    correlated = _prediction(probability=0.3, market_price=0.5, confidence=0.6, condition_id="correlated")
    decision = rm.evaluate(primary, _portfolio(), correlated_condition_id="correlated", correlated_prediction=correlated)
    assert decision.hedge_condition_id == "correlated"
    assert decision.hedge_stake_usd is not None
    assert decision.hedge_stake_usd > 0
