"""Typed models spanning market data, predictions, risk sizing, and trades."""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MarketOutcome(BaseModel):
    """One side of a binary Polymarket market (e.g. 'Yes' / 'No')."""

    name: str
    token_id: str
    price: float = Field(ge=0.0, le=1.0)  # implied probability


class PolymarketMarket(BaseModel):
    condition_id: str
    question: str
    city: str
    closes_at: datetime
    outcomes: list[MarketOutcome]
    volume_24h_usd: float | None = None


class Prediction(BaseModel):
    """
    Output of the prediction pipeline for one market.

    `predicted_probability` and `reasoning` are the two fields the LLM is
    forced to emit directly (see agents/prediction_agent.py's JSON schema) —
    the model should never be asked to invent market_implied_probability,
    edge, or expected_value itself, since those depend on the *live* market
    price fetched separately via PolymarketDataService. The orchestrator
    populates those fields after the LLM call returns, by combining
    predicted_probability with the actual market quote.

    `probability` is kept as a read-only alias so existing consumers
    (e.g. RiskManager's Kelly calculation) don't need to know about the
    LLM-vs-derived field split.
    """

    city: str
    predicted_probability: float = Field(ge=0.0, le=1.0, description="LLM's raw probability estimate for 'Yes'.")
    reasoning: str = Field(description="Brief explanation the LLM gives for its probability estimate.")

    # Populated by the orchestrator after combining predicted_probability
    # with a live PolymarketMarket quote — not set directly by the LLM call.
    condition_id: str | None = None
    question: str | None = None
    market_implied_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    edge: float | None = None  # predicted_probability - market_implied_probability
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="From AggregatedWeather.confidence_score, not the LLM.")
    expected_value: float | None = None
    generated_at: datetime | None = None

    @property
    def probability(self) -> float:
        """Backward-compatible alias used by RiskManager and other downstream consumers."""
        return self.predicted_probability


class RiskDecision(BaseModel):
    """Output of the risk engine: how much (if anything) to stake."""

    condition_id: str
    kelly_fraction_applied: float
    raw_kelly_stake_usd: float
    capped_stake_usd: float
    reasons: list[str] = Field(default_factory=list)
    approved: bool
    hedge_condition_id: str | None = None
    hedge_stake_usd: float | None = None


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    FILLED = "filled"
    REJECTED = "rejected"
    HEDGED = "hedged"


class Trade(BaseModel):
    """
    A single paper-trade execution record, persisted for the dashboard.

    `order_payload_json` stores the exact JSON that would be sent to the
    Polymarket CLOB API in a live execution — logged so a reviewer can see
    the full order structure without this ever touching real funds.
    `simulated_execution_price` is the market's mid-price at time of
    logging, used as the fill price for PnL tracking.
    """

    id: str
    condition_id: str
    city: str
    side: TradeSide
    outcome_name: str           # "Yes" or "No"
    outcome_token_id: str
    stake_usd: float
    size_shares: float          # stake / entry_price
    entry_price: float          # simulated execution price (market mid at log time)
    status: TradeStatus
    is_hedge: bool = False
    linked_trade_id: str | None = None
    placed_at: datetime
    reasoning_summary: str
    order_payload_json: str     # exact payload that would be sent to CLOB API
    market_question: str = ""

    # Resolved later when the market closes
    resolved_at: datetime | None = None
    resolution: str | None = None   # "YES" | "NO" | None (pending)
    pnl_usd: float | None = None


class TradeResult(BaseModel):
    """Wraps a Trade plus any hedge trade placed alongside it."""

    primary: Trade
    hedge: Trade | None = None
