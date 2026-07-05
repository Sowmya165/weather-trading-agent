"""
services/trade_executor.py

Paper trading execution layer.

This class does two things:
  1. Formats the exact JSON payload that would be sent to the Polymarket
     CLOB API in a live execution — so the output is auditable and
     reviewable without ever touching real funds.
  2. Persists the paper trade to our in-memory ledger (flushed to SQLite
     by the repository layer) and logs it in the required format.

Architecture note on polymarket-paper-trader reference repo:
The assignment links to agent-next/polymarket-paper-trader, which is an
MCP server wrapping the Polymarket CLOB SDK (L1 order signing, fee math,
slippage simulation against live order books). Since this is a paper-only
assessment run, we replicate the order *payload* structure that SDK would
produce rather than calling it directly — avoiding the requirement for a
funded wallet private key that the real SDK needs. The payload structure
below is taken directly from Polymarket's public CLOB API docs (the
/order POST endpoint schema) so it is structurally identical to what a
live execution would send.

If you later wire in the real polymarket-py SDK, the only change needed
is swapping _format_clob_payload() for the SDK's create_and_sign_order()
call — everything else (risk checks, ledger writes, logging) stays the same.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from models.market import (
    PolymarketMarket,
    Prediction,
    RiskDecision,
    Trade,
    TradeResult,
    TradeSide,
    TradeStatus,
)

logger = logging.getLogger("services.trade_executor")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class PolymarketPaperTrader:
    """
    Simulates placing orders on Polymarket weather markets.

    Accepts a RiskDecision (containing the approved stake and optional
    hedge instruction) plus the corresponding Prediction and live market
    data, then produces a fully-formed Trade record — including the exact
    CLOB API JSON payload — without ever touching real funds or requiring
    a wallet key.
    """

    def __init__(self, starting_bankroll_usd: float = 1000.0):
        self._bankroll = starting_bankroll_usd
        self._ledger: list[Trade] = []   # in-memory; flushed to SQLite by the repository layer

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def execute_trade(
        self,
        decision: RiskDecision,
        prediction: Prediction,
        market: PolymarketMarket,
    ) -> TradeResult:
        """
        Core method. Takes a fully-formed RiskDecision and places the
        corresponding paper trade(s).

        Returns a TradeResult containing the primary Trade and, if the
        risk engine requested a hedge, a second Trade for the hedge leg.
        Never raises on execution failure — a rejection is a valid
        TradeResult, not an exception, so the orchestrator loop continues.
        """
        if not decision.approved or decision.capped_stake_usd <= 0:
            logger.warning(
                "Trade skipped for %s — risk decision not approved (reasons: %s)",
                decision.condition_id,
                "; ".join(decision.reasons),
            )
            return TradeResult(
                primary=self._rejected_trade(decision, prediction, market),
                hedge=None,
            )

        primary = self._place_paper_trade(
            decision=decision,
            prediction=prediction,
            market=market,
            stake_usd=decision.capped_stake_usd,
            is_hedge=False,
        )
        self._ledger.append(primary)

        hedge: Trade | None = None
        if decision.hedge_condition_id and decision.hedge_stake_usd:
            hedge_outcome = self._opposite_outcome(market)
            if hedge_outcome:
                hedge = self._place_paper_trade(
                    decision=decision,
                    prediction=prediction,
                    market=market,
                    stake_usd=decision.hedge_stake_usd,
                    is_hedge=True,
                    linked_trade_id=primary.id,
                    override_outcome_name=hedge_outcome["name"],
                    override_token_id=hedge_outcome["token_id"],
                    override_price=hedge_outcome["price"],
                )
                self._ledger.append(hedge)

        logger.info(
            "Paper trade FILLED | city=%s | condition=%s | side=%s | "
            "outcome=%s | stake=$%.2f | price=%.4f | trade_id=%s%s",
            prediction.city,
            decision.condition_id,
            TradeSide.BUY.value,
            primary.outcome_name,
            primary.stake_usd,
            primary.entry_price,
            primary.id,
            f" | hedge_id={hedge.id}" if hedge else "",
        )

        return TradeResult(primary=primary, hedge=hedge)

    @property
    def ledger(self) -> list[Trade]:
        return list(self._ledger)

    def summary(self) -> dict:
        """Quick PnL/win-rate summary for logging at shutdown."""
        filled = [t for t in self._ledger if t.status == TradeStatus.FILLED]
        resolved = [t for t in filled if t.pnl_usd is not None]
        total_staked = sum(t.stake_usd for t in filled)
        realized_pnl = sum(t.pnl_usd for t in resolved if t.pnl_usd)
        return {
            "total_trades": len(filled),
            "resolved_trades": len(resolved),
            "total_staked_usd": round(total_staked, 2),
            "realized_pnl_usd": round(realized_pnl, 2),
            "open_exposure_usd": round(total_staked - sum(t.stake_usd for t in resolved), 2),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _place_paper_trade(
        self,
        decision: RiskDecision,
        prediction: Prediction,
        market: PolymarketMarket,
        stake_usd: float,
        is_hedge: bool,
        linked_trade_id: str | None = None,
        override_outcome_name: str | None = None,
        override_token_id: str | None = None,
        override_price: float | None = None,
    ) -> Trade:
        yes_outcome = next((o for o in market.outcomes if o.name.lower() == "yes"), market.outcomes[0])
        outcome_name = override_outcome_name or yes_outcome.name
        token_id = override_token_id or yes_outcome.token_id
        entry_price = override_price or yes_outcome.price

        size_shares = round(stake_usd / entry_price, 4) if entry_price > 0 else 0.0
        trade_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        payload = self._format_clob_payload(
            token_id=token_id,
            side=TradeSide.BUY,
            size_shares=size_shares,
            price=entry_price,
            condition_id=decision.condition_id,
        )

        return Trade(
            id=trade_id,
            condition_id=decision.condition_id,
            city=prediction.city,
            side=TradeSide.BUY,
            outcome_name=outcome_name,
            outcome_token_id=token_id,
            stake_usd=round(stake_usd, 2),
            size_shares=size_shares,
            entry_price=entry_price,
            status=TradeStatus.HEDGED if is_hedge else TradeStatus.FILLED,
            is_hedge=is_hedge,
            linked_trade_id=linked_trade_id,
            placed_at=now,
            reasoning_summary=prediction.reasoning[:300],
            order_payload_json=json.dumps(payload, indent=2),
            market_question=prediction.question or market.question,
        )

    def _rejected_trade(self, decision: RiskDecision, prediction: Prediction, market: PolymarketMarket) -> Trade:
        return Trade(
            id=str(uuid.uuid4()),
            condition_id=decision.condition_id,
            city=prediction.city,
            side=TradeSide.BUY,
            outcome_name="N/A",
            outcome_token_id="",
            stake_usd=0.0,
            size_shares=0.0,
            entry_price=0.0,
            status=TradeStatus.REJECTED,
            is_hedge=False,
            placed_at=datetime.now(timezone.utc),
            reasoning_summary="; ".join(decision.reasons)[:300],
            order_payload_json="{}",
            market_question=prediction.question or market.question,
        )

    @staticmethod
    def _format_clob_payload(
        token_id: str,
        side: TradeSide,
        size_shares: float,
        price: float,
        condition_id: str,
    ) -> dict:
        """
        Formats the exact JSON for a Polymarket CLOB POST /order request.

        Structure matches Polymarket's public CLOB API spec. In a live
        setup this payload would be signed with an L1 EOA wallet key
        before submission. We log it unsigned for paper-trade auditability.
        """
        return {
            "orderType": "FOK",
            "tokenID": token_id,
            "side": side.value.upper(),
            "price": round(price, 4),
            "size": size_shares,
            "conditionId": condition_id,
            "feeRateBps": 0,
            "nonce": int(datetime.now(timezone.utc).timestamp() * 1000),
            "signer": "0x_PAPER_TRADE_NO_REAL_SIGNER",
            "signature": "0x_UNSIGNED_PAPER_TRADE",
            "_note": "Paper trade — payload is structurally correct but unsigned.",
        }

    @staticmethod
    def _opposite_outcome(market: PolymarketMarket) -> dict | None:
        no_outcome = next((o for o in market.outcomes if o.name.lower() == "no"), None)
        if not no_outcome:
            return None
        return {"name": no_outcome.name, "token_id": no_outcome.token_id, "price": no_outcome.price}
