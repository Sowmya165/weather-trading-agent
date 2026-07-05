"""
Custom Hermes Agent tools.

Hermes Agent is extended via plugins (functions registered as tools), not
by subclassing or reimplementing its orchestration loop. This module wraps
our three data services as plain async-safe callables the agent can invoke
mid-reasoning, and is registered with AIAgent in prediction_agent.py.

Keeping these as thin wrappers (collect -> score -> return plain dict) means
the agent only ever sees clean, already-validated JSON-able data — it never
parses raw API responses itself, which is where hallucination risk creeps in.
"""
import asyncio
from datetime import datetime

from config.settings import Settings, get_settings
from services.confidence_scorer import score_and_aggregate
from services.local_research import LocalResearchService
from services.polymarket_data import PolymarketDataService
from services.weather_collector import WeatherCollector


def _run_async(coro):
    """Hermes tool functions are called synchronously; bridge to our async services."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Defensive: shouldn't normally happen since the agent's own loop calls
        # tools synchronously, but avoids a silent deadlock if it ever does.
        raise RuntimeError("get_weather_snapshot called from within a running event loop")
    return asyncio.run(coro)


def get_weather_snapshot(city: str) -> dict:
    """
    Tool: fetch and reconcile weather data for a city from all configured
    sources, returning the aggregated reading plus a confidence score and
    any source disagreements. Always call this before estimating a weather
    market's probability — never guess a forecast from general knowledge.
    """
    settings = get_settings()
    if city not in settings.cities:
        return {"error": f"Unsupported city '{city}'. Supported: {list(settings.cities.keys())}"}

    async def _collect():
        collector = WeatherCollector(settings)
        try:
            observations = await collector.collect_for_city(city)
        finally:
            await collector.aclose()
        if not observations:
            return {"error": f"No weather observations available for {city}."}
        aggregated = score_and_aggregate(city, observations)
        return aggregated.model_dump(mode="json")

    return _run_async(_collect())


def get_local_research(city: str) -> dict:
    """
    Tool: fetch storm alerts, government warnings, and local weather news
    for a city via Apify. Returns a list of snippets with inferred severity.
    Use this to check for anything a pure numeric forecast would miss.
    """
    settings = get_settings()

    async def _research():
        service = LocalResearchService(settings)
        snippets = await service.research_city(city)
        return {"city": city, "snippets": [s.model_dump(mode="json") for s in snippets]}

    return _run_async(_research())


def get_weather_markets(city: str) -> dict:
    """
    Tool: fetch currently open Polymarket weather markets for a city, with
    each outcome's name, token ID, and current implied probability (price).
    Always check this before forming a prediction — the market's current
    price is what 'edge' is measured against.
    """
    settings = get_settings()

    async def _fetch():
        service = PolymarketDataService(settings)
        try:
            markets = await service.get_weather_markets_for_city(city)
        finally:
            await service.aclose()
        return {"city": city, "markets": [m.model_dump(mode="json") for m in markets]}

    return _run_async(_fetch())


# Hermes plugin tool registry — the shape expected by AIAgent's custom tool
# loading (name -> callable, with the docstring used as the tool description).
WEATHER_TRADING_TOOLS = {
    "get_weather_snapshot": get_weather_snapshot,
    "get_local_research": get_local_research,
    "get_weather_markets": get_weather_markets,
}
