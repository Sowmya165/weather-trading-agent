"""
Polymarket data service.

Gamma API (gamma-api.polymarket.com) is public, read-only, no auth — used
here purely for market discovery and indicative pricing ("use Gamma to find
what to trade"). Order placement itself happens through polymarket-paper-
trader in services/trade_executor.py, which talks to the CLOB layer.

We don't hit the CLOB for prices in the research step: Gamma's outcomePrices
are good enough for computing edge before deciding whether a trade is worth
sizing at all. Only once the risk engine approves a trade do we need
live order-book depth, and that happens inside the paper trader.
"""
import json
from datetime import datetime, timezone

import httpx
from loguru import logger

from config.settings import Settings
from models.market import MarketOutcome, PolymarketMarket

GAMMA_EVENTS_ENDPOINT = "/events"

# Polymarket doesn't expose free-text search on Gamma, so we filter by tag
# and keyword-match the question text client-side instead of relying on a
# query param (a documented gotcha — ?search= silently no-ops).
WEATHER_TAG_SLUG = "weather"


class PolymarketDataService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=10.0, base_url=settings.polymarket_gamma_host)

    async def get_weather_markets_for_city(self, city: str) -> list[PolymarketMarket]:
        """
        Fetch active events tagged/matching weather, then filter to ones
        whose question text mentions this city. Polymarket doesn't have a
        per-city tag, so question-text matching is the practical approach.
        """
        params = {
            "tag_slug": WEATHER_TAG_SLUG,
            "active": "true",
            "closed": "false",
            "limit": 100,
        }
        try:
            resp = await self._client.get(GAMMA_EVENTS_ENDPOINT, params=params)
            resp.raise_for_status()
            events = resp.json()
        except httpx.HTTPError as e:
            logger.warning(f"Polymarket Gamma fetch failed: {e}")
            return []

        markets: list[PolymarketMarket] = []
        for event in events:
            for raw_market in event.get("markets", []):
                question = raw_market.get("question", "")
                if city.lower() not in question.lower():
                    continue
                parsed = _parse_market(raw_market, city)
                if parsed:
                    markets.append(parsed)
        return markets

    async def get_all_weather_markets(self) -> dict[str, list[PolymarketMarket]]:
        import asyncio

        cities = list(self._settings.cities.keys())
        results = await asyncio.gather(*(self.get_weather_markets_for_city(c) for c in cities))
        return dict(zip(cities, results))

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_market(raw_market: dict, city: str) -> PolymarketMarket | None:
    """
    Gamma returns outcomes/outcomePrices as JSON-encoded string arrays that
    map 1:1 by index, and clobTokenIds similarly — this normalizes all three
    into typed MarketOutcome objects in one place so nothing downstream has
    to know about Gamma's string-encoding quirk.
    """
    try:
        outcome_names = json.loads(raw_market.get("outcomes", "[]"))
        outcome_prices = json.loads(raw_market.get("outcomePrices", "[]"))
        token_ids = json.loads(raw_market.get("clobTokenIds", "[]"))
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse market outcomes for {raw_market.get('question')}: {e}")
        return None

    if not (len(outcome_names) == len(outcome_prices) == len(token_ids)):
        logger.warning(f"Mismatched outcome arrays for {raw_market.get('question')} — skipping.")
        return None

    outcomes = [
        MarketOutcome(name=name, token_id=token_id, price=float(price))
        for name, price, token_id in zip(outcome_names, outcome_prices, token_ids)
    ]

    end_date = raw_market.get("endDate")
    try:
        closes_at = datetime.fromisoformat(end_date.replace("Z", "+00:00")) if end_date else datetime.now(timezone.utc)
    except ValueError:
        closes_at = datetime.now(timezone.utc)

    return PolymarketMarket(
        condition_id=raw_market.get("conditionId", raw_market.get("id", "")),
        question=raw_market.get("question", ""),
        city=city,
        closes_at=closes_at,
        outcomes=outcomes,
        volume_24h_usd=raw_market.get("volume24hr"),
    )
