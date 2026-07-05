"""
risk_manager.py

Mathematical core of position sizing and hedging for the weather trading
agent. Deliberately has zero LLM calls and zero network I/O — every function
here is a pure, deterministic calculation over typed inputs, which makes it
trivially unit-testable and auditable (a reviewer or a risk dashboard can
reproduce any stake decision by hand from logged inputs).

Kelly Criterion recap (binary market form):
    f* = (b*p - q) / b
where:
    p = our estimated probability the outcome occurs
    q = 1 - p
    b = net odds received on a win = (1 - market_price) / market_price
    f* = fraction of bankroll to stake

We never stake full Kelly — it's mathematically optimal for long-run growth
under perfect probability estimates, but real forecasts are noisy, so full
Kelly produces large, account-threatening swings on bad estimates. We apply
a fractional Kelly multiplier (KELLY_FRACTION, default 0.25) and then clamp
against hard exposure limits regardless of what Kelly says.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import Settings
from models.market import Prediction, RiskDecision

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


@dataclass(frozen=True)
class PortfolioState:
    """
    Snapshot of current exposure, supplied by the caller (read from the
    trade ledger) rather than tracked internally — keeps this module
    side-effect-free and easy to test with synthetic states.
    """

    bankroll_usd: float
    open_exposure_usd: float
    realized_pnl_today_usd: float
    open_positions_by_condition_id: dict[str, float]  # condition_id -> staked_usd


class RiskManager:
    """Computes Kelly-sized, limit-checked, hedge-aware stake decisions."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def evaluate(
        self,
        prediction: Prediction,
        portfolio: PortfolioState,
        correlated_condition_id: str | None = None,
        correlated_prediction: Prediction | None = None,
    ) -> RiskDecision:
        """
        Main entry point: turns a Prediction into a final stake decision.

        Args:
            prediction: agent's probability estimate + market price for one market.
            portfolio: current bankroll/exposure snapshot.
            correlated_condition_id: condition_id of a market correlated with
                this one (e.g. "NYC > 80F" vs "NYC > 85F" same day), if any.
            correlated_prediction: the agent's Prediction for that correlated
                market, used to size a hedge if our edge is weak/uncertain.

        Returns:
            RiskDecision with approved/capped stake and, if applicable, a
            hedge instruction.
        """
        try:
            reasons: list[str] = []

            raw_kelly_fraction = self._kelly_fraction(prediction.probability, prediction.market_implied_probability)
            applied_fraction = raw_kelly_fraction * self._settings.kelly_fraction
            raw_stake = max(0.0, applied_fraction) * portfolio.bankroll_usd

            if raw_kelly_fraction <= 0:
                reasons.append("Kelly fraction is non-positive — no edge or negative edge, skipping trade.")
                return RiskDecision(
                    condition_id=prediction.condition_id,
                    kelly_fraction_applied=0.0,
                    raw_kelly_stake_usd=0.0,
                    capped_stake_usd=0.0,
                    reasons=reasons,
                    approved=False,
                )

            capped_stake, cap_reasons = self._apply_limits(raw_stake, prediction, portfolio)
            reasons.extend(cap_reasons)

            hedge_condition_id = None
            hedge_stake_usd = None
            if correlated_condition_id and correlated_prediction is not None:
                hedge_stake_usd = self._hedge_size(prediction, correlated_prediction, capped_stake)
                if hedge_stake_usd > 0:
                    hedge_condition_id = correlated_condition_id
                    reasons.append(
                        f"Low-confidence/correlated exposure detected — sizing hedge of "
                        f"${hedge_stake_usd:.2f} against {correlated_condition_id}."
                    )

            approved = capped_stake > 0
            if not approved and not any("edge" in r.lower() for r in reasons):
                reasons.append("Stake capped to zero by risk limits.")

            logger.info(
                "Risk decision for %s | raw_kelly=%.4f applied_fraction=%.4f "
                "raw_stake=$%.2f capped_stake=$%.2f approved=%s",
                prediction.condition_id,
                raw_kelly_fraction,
                applied_fraction,
                raw_stake,
                capped_stake,
                approved,
            )

            return RiskDecision(
                condition_id=prediction.condition_id,
                kelly_fraction_applied=round(applied_fraction, 4),
                raw_kelly_stake_usd=round(raw_stake, 2),
                capped_stake_usd=round(capped_stake, 2),
                reasons=reasons,
                approved=approved,
                hedge_condition_id=hedge_condition_id,
                hedge_stake_usd=round(hedge_stake_usd, 2) if hedge_stake_usd else None,
            )

        except (ValueError, ZeroDivisionError, ArithmeticError) as e:
            # A malformed prediction (e.g. market_price of 0 or 1) should never
            # crash the pipeline — it should produce a logged, rejected decision.
            logger.error("Risk evaluation failed for %s: %s", prediction.condition_id, e, exc_info=True)
            return RiskDecision(
                condition_id=prediction.condition_id,
                kelly_fraction_applied=0.0,
                raw_kelly_stake_usd=0.0,
                capped_stake_usd=0.0,
                reasons=[f"Risk evaluation error: {e}"],
                approved=False,
            )

    def _kelly_fraction(self, p_estimate: float, market_price: float) -> float:
        """
        f* = (b*p - q) / b, where b = (1 - market_price) / market_price.

        market_price is the cost of $1 of payoff on a "Yes" share — Polymarket's
        own implied probability. Guard against price of exactly 0 or 1, which
        would make b undefined/infinite.
        """
        if market_price <= 0.0 or market_price >= 1.0:
            raise ValueError(f"market_price must be in (0, 1), got {market_price}")

        p = p_estimate
        q = 1.0 - p
        b = (1.0 - market_price) / market_price

        f_star = (b * p - q) / b
        return f_star

    def _apply_limits(
        self, raw_stake: float, prediction: Prediction, portfolio: PortfolioState
    ) -> tuple[float, list[str]]:
        """Clamp the raw Kelly stake against per-position, daily-loss, and portfolio caps."""
        reasons: list[str] = []
        stake = raw_stake

        max_position = self._settings.max_position_pct_of_bankroll * portfolio.bankroll_usd
        if stake > max_position:
            reasons.append(f"Capped to max per-position limit (${max_position:.2f}).")
            stake = max_position

        daily_loss_limit = self._settings.max_daily_loss_pct * portfolio.bankroll_usd
        if portfolio.realized_pnl_today_usd <= -daily_loss_limit:
            reasons.append("Daily loss limit already breached — blocking new trades today.")
            return 0.0, reasons

        max_portfolio_exposure = self._settings.max_portfolio_exposure_pct * portfolio.bankroll_usd
        projected_exposure = portfolio.open_exposure_usd + stake
        if projected_exposure > max_portfolio_exposure:
            available = max(0.0, max_portfolio_exposure - portfolio.open_exposure_usd)
            reasons.append(
                f"Capped to remaining portfolio exposure room (${available:.2f} of "
                f"${max_portfolio_exposure:.2f} total limit)."
            )
            stake = available

        if prediction.confidence < 0.5:
            reasons.append(f"Low forecast confidence ({prediction.confidence:.2f}) — halving stake.")
            stake *= 0.5

        return max(0.0, stake), reasons

    def _hedge_size(self, primary: Prediction, correlated: Prediction, primary_stake: float) -> float:
        """
        Sizes an offsetting position in a correlated market.

        Heuristic: hedge proportionally to (a) how much the two predictions
        disagree in direction/confidence and (b) overall confidence in the
        primary trade. A confident, well-aligned pair needs little hedge;
        a primary position taken under low confidence or against a
        correlated market the agent is unsure about gets a larger hedge.
        """
        if primary_stake <= 0:
            return 0.0

        # Disagreement in implied edge direction signals the two markets
        # aren't moving as one would expect from a shared weather driver.
        edge_alignment = primary.edge * correlated.edge
        uncertainty = 1.0 - min(primary.confidence, correlated.confidence)

        if edge_alignment < 0:
            # Edges point opposite directions — markets are giving conflicting
            # signals about a correlated event. Hedge meaningfully.
            hedge_pct = 0.5 + 0.3 * uncertainty
        else:
            # Aligned signals — light hedge scaled only by uncertainty.
            hedge_pct = 0.15 * uncertainty

        return round(primary_stake * min(hedge_pct, 0.75), 2)
