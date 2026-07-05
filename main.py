"""
main.py — Orchestrator entry point.

Runs the full pipeline for all 5 cities in one pass:
  1. ApifyWeatherCollector  → raw observations per city
  2. confidence_scorer      → AggregatedWeather + confidence score
  3. PolymarketDataService  → live market questions + implied prices
  4. HermesPredictionAgent  → predicted_probability + reasoning (LLM)
  5. RiskManager            → Kelly-sized stake + hedge decision
  6. PolymarketPaperTrader  → paper trade execution + CLOB payload log
  7. DatabaseManager        → persist full cycle to SQLite

Run:
    python main.py                 # single pass
    python main.py --loop 300      # repeat every 300 s (5 min)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

from config.settings import get_settings
from database.db import DatabaseManager
from services.confidence_scorer import score_and_aggregate
from services.polymarket_data import PolymarketDataService
from services.risk_manager import PortfolioState, RiskManager
from services.trade_executor import PolymarketPaperTrader
from services.weather_collector import ApifyWeatherCollector, WeatherCollector
from agents.prediction_agent import HermesPredictionAgent

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# ── Portfolio state (in-memory; loaded from DB on startup in a full prod system)
_portfolio = PortfolioState(
    bankroll_usd=1000.0,
    open_exposure_usd=0.0,
    realized_pnl_today_usd=0.0,
    open_positions_by_condition_id={},
)


async def run_pipeline(
    weather_collector: WeatherCollector,
    apify_collector: ApifyWeatherCollector,
    market_service: PolymarketDataService,
    prediction_agent: HermesPredictionAgent,
    risk_manager: RiskManager,
    trader: PolymarketPaperTrader,
    db: DatabaseManager,
) -> None:
    settings = get_settings()
    cities = list(settings.cities.keys())
    logger.info("=== Pipeline run started | %s | cities: %s ===",
                datetime.now(timezone.utc).isoformat(), cities)

    # ── Step 1: Collect weather from all sources concurrently ──────────
    logger.info("Step 1 — Fetching weather data ...")
    global_obs, apify_obs = await asyncio.gather(
        weather_collector.collect_all(),
        apify_collector.collect_all(),
    )

    # ── Step 2: Score & aggregate per city ────────────────────────────
    logger.info("Step 2 — Scoring and aggregating observations ...")
    aggregated = {}
    for city in cities:
        all_obs = global_obs.get(city, []) + apify_obs.get(city, [])
        if not all_obs:
            logger.warning("No observations for %s — skipping city.", city)
            continue
        try:
            aggregated[city] = score_and_aggregate(city, all_obs)
            logger.info("  %s → temp=%.1f°C  precip=%.1fmm  confidence=%.2f",
                        city,
                        aggregated[city].temperature_c,
                        aggregated[city].precipitation_mm,
                        aggregated[city].confidence_score)
        except Exception as e:
            logger.error("Aggregation failed for %s: %s", city, e)

    # ── Step 3: Fetch live Polymarket weather markets ──────────────────
    logger.info("Step 3 — Fetching Polymarket markets ...")
    all_markets = await market_service.get_all_weather_markets()
    for city, markets in all_markets.items():
        logger.info("  %s → %d market(s) found", city, len(markets))

    # ── Steps 4-7: Predict → Size → Execute → Persist — per city/market
    # Steps 4-7 — Predict → Size → Execute → Persist (batch)
    logger.info("Steps 4-7 — Batch predict / Size / Execute / Persist ...")

# Build top-2-markets-per-city maps to pass to batch predictor
    markets_slice = {city: mkts[:2] for city, mkts in all_markets.items() if mkts}
    predictions_by_city = await prediction_agent.generate_predictions_batch(aggregated, markets_slice)

    for city, predictions in predictions_by_city.items():
        for prediction, market in zip(predictions, markets_slice.get(city, [])):
            yes_outcome = next((o for o in market.outcomes if o.name.lower() == "yes"), market.outcomes[0] if market.outcomes else None)
            if yes_outcome:
                prediction.condition_id = market.condition_id
                prediction.market_implied_probability = yes_outcome.price
                prediction.edge = round(prediction.predicted_probability - yes_outcome.price, 4)
                prediction.expected_value = round(
                    prediction.predicted_probability * (1 - yes_outcome.price)
                    - (1 - prediction.predicted_probability) * yes_outcome.price, 4)

            logger.info("  [%s] prob=%.2f  market=%.2f  edge=%.3f", city,
                        prediction.predicted_probability,
                        prediction.market_implied_probability or 0,
                        prediction.edge or 0)

            decision = risk_manager.evaluate(prediction, _portfolio)
            result = trader.execute_trade(decision, prediction, market)
            logger.info("  [%s] Trade: status=%s  id=%s", city, result.primary.status.value, result.primary.id)

            try:
                await db.insert_cycle(prediction, decision, result.primary)
                if result.hedge:
                    await db.insert_trade(result.hedge)
            except Exception as e:
                logger.error("  [%s] DB write failed (non-fatal): %s", city, e)
            
    summary = trader.summary()
    logger.info(
        "=== Run complete | trades=%d  staked=$%.2f  realized_pnl=$%.2f  open_exposure=$%.2f ===",
        summary["total_trades"],
        summary["total_staked_usd"],
        summary["realized_pnl_usd"],
        summary["open_exposure_usd"],
    )


async def main(loop_interval_secs: int | None = None) -> None:
    settings = get_settings()
    cities = list(settings.cities.keys())

    # Initialise all services
    db                = DatabaseManager(settings.database_url)
    weather_collector = WeatherCollector(settings)
    apify_collector   = ApifyWeatherCollector(settings, cities=cities)
    market_service    = PolymarketDataService(settings)
    prediction_agent  = HermesPredictionAgent(settings)
    risk_manager      = RiskManager(settings)
    trader            = PolymarketPaperTrader(settings.starting_bankroll_usd)

    await db.init()

    try:
        if loop_interval_secs:
            logger.info("Running in loop mode — interval=%ds. Ctrl+C to stop.", loop_interval_secs)
            while True:
                await run_pipeline(
                    weather_collector, apify_collector,
                    market_service, prediction_agent,
                    risk_manager, trader, db,
                )
                logger.info("Sleeping %ds until next run ...", loop_interval_secs)
                await asyncio.sleep(loop_interval_secs)
        else:
            await run_pipeline(
                weather_collector, apify_collector,
                market_service, prediction_agent,
                risk_manager, trader, db,
            )

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received — shutting down gracefully.")
        summary = trader.summary()
        logger.info(
            "Final session summary | trades=%d  staked=$%.2f  pnl=$%.2f",
            summary["total_trades"],
            summary["total_staked_usd"],
            summary["realized_pnl_usd"],
        )
    finally:
        logger.info("Closing all connections ...")
        await db.close()
        await weather_collector.aclose()
        await market_service.aclose()
        await prediction_agent.aclose()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Trading Agent")
    parser.add_argument(
        "--loop", type=int, default=None,
        metavar="SECONDS",
        help="Re-run the pipeline every N seconds. Omit for a single pass.",
    )
    args = parser.parse_args()
    asyncio.run(main(loop_interval_secs=args.loop))
