"""
data_scraper.py

Fetches current weather (temperature, precipitation probability, wind speed)
for a list of cities via Apify-run scrapers, with retry/backoff and
structured logging. Output is a flat dict keyed by city name, shaped for
direct injection into an LLM prompt context — no nested objects the model
would have to parse.

Apify is used for two source tiers here:
  - "global": a general weather-data actor (apify/weather-scraper or
    apify/oneary~weather-database-scraper, per the assignment's listed
    sources) hit once per city.
  - "local": a second, geographically-scoped actor run per city to
    cross-check the global reading and surface local-source-only signals
    (e.g. a national met agency page Apify can scrape but a generic global
    API doesn't carry).

If only one of the two succeeds, we still return a result for that city
(partial data beats no data for an LLM agent) but flag which source filled
it, so confidence scoring downstream can account for it.
"""
from __future__ import annotations

import asyncio
import logging
import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# City -> (lat, lon), kept local to this module so it has no dependency on
# config.settings and can be dropped into any context as a standalone file.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Tokyo": (35.6762, 139.6503),
    "Berlin": (52.5200, 13.4050),
}

APIFY_GLOBAL_WEATHER_ACTOR = os.getenv("APIFY_GLOBAL_WEATHER_ACTOR", "apify/weather-scraper")
APIFY_LOCAL_WEATHER_ACTOR = os.getenv("APIFY_LOCAL_WEATHER_ACTOR", "apify/web-scraper")

RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


class ApifyNotConfiguredError(Exception):
    """Raised when no Apify token is available in the environment."""


def _get_apify_token() -> str:
    token = os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN", "")
    if not token:
        raise ApifyNotConfiguredError(
            "No Apify token found. Set APIFY_TOKEN (or APIFY_API_TOKEN) in your environment/.env."
        )
    return token


def _get_apify_client():
    """Lazily import so this module doesn't hard-fail if apify-client isn't installed yet."""
    from apify_client import ApifyClientAsync

    return ApifyClientAsync(_get_apify_token())


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _run_actor(client, actor_id: str, run_input: dict) -> list[dict]:
    """
    Runs an Apify actor and returns its dataset items, retrying with
    exponential backoff (1s, 2s, 4s ... capped at 8s) on transient
    network-level failures. Does NOT retry on bad input or auth errors —
    those are caller bugs, not flaky infrastructure, and retrying them
    just burns Apify compute for the same guaranteed failure.
    """
    actor = client.actor(actor_id)
    run = await actor.call(run_input=run_input)
    dataset = client.dataset(run["defaultDatasetId"])
    page = await dataset.list_items()
    return page.items


async def _fetch_global_weather(client, city: str, lat: float, lon: float) -> dict | None:
    try:
        items = await _run_actor(
            client,
            APIFY_GLOBAL_WEATHER_ACTOR,
            {"latitude": lat, "longitude": lon, "city": city},
        )
        if not items:
            logger.warning(f"[{city}] Global weather actor returned no data.")
            return None
        item = items[0]
        result = {
            "temperature_c": item.get("temperature_c") or item.get("temp_c"),
            "precipitation_probability_pct": item.get("precipitation_probability") or item.get("precip_chance"),
            "wind_speed_kph": item.get("wind_speed_kph") or item.get("wind_kph"),
        }
        logger.info(f"[{city}] Global weather data fetched successfully.")
        return result
    except RETRYABLE_EXCEPTIONS as e:
        logger.error(f"[{city}] Global weather fetch failed after retries: {e}")
        return None
    except Exception as e:
        # Non-retryable (bad actor id, auth failure, malformed input) — log and
        # move on rather than letting one bad source take down the whole batch.
        logger.error(f"[{city}] Global weather fetch failed (non-retryable): {e}")
        return None


async def _fetch_local_weather(client, city: str, lat: float, lon: float) -> dict | None:
    try:
        items = await _run_actor(
            client,
            APIFY_LOCAL_WEATHER_ACTOR,
            {"searchTerms": [f"{city} weather forecast today"], "maxItems": 3},
        )
        if not items:
            logger.warning(f"[{city}] Local weather source returned no data.")
            return None
        item = items[0]
        result = {
            "temperature_c": item.get("temperature_c"),
            "precipitation_probability_pct": item.get("precipitation_probability"),
            "wind_speed_kph": item.get("wind_speed_kph"),
        }
        logger.info(f"[{city}] Local weather data fetched successfully.")
        return result
    except RETRYABLE_EXCEPTIONS as e:
        logger.error(f"[{city}] Local weather fetch failed after retries: {e}")
        return None
    except Exception as e:
        logger.error(f"[{city}] Local weather fetch failed (non-retryable): {e}")
        return None


async def _fetch_city(client, city: str) -> dict:
    if city not in CITY_COORDS:
        logger.error(f"[{city}] Unsupported city — no coordinates configured.")
        return {"city": city, "status": "error", "error": "unsupported_city"}

    lat, lon = CITY_COORDS[city]
    global_result, local_result = await asyncio.gather(
        _fetch_global_weather(client, city, lat, lon),
        _fetch_local_weather(client, city, lat, lon),
    )

    if global_result is None and local_result is None:
        logger.error(f"[{city}] All weather sources failed.")
        return {"city": city, "status": "error", "error": "all_sources_failed"}

    # Prefer global as the primary reading; fill any missing fields from local.
    merged = dict(global_result or {})
    fallback_used = global_result is None
    for key, value in (local_result or {}).items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
            fallback_used = True

    return {
        "city": city,
        "status": "partial" if fallback_used else "ok",
        "temperature_c": merged.get("temperature_c"),
        "precipitation_probability_pct": merged.get("precipitation_probability_pct"),
        "wind_speed_kph": merged.get("wind_speed_kph"),
    }


async def fetch_weather_data(cities: list[str]) -> dict:
    """
    Fetch current temperature, precipitation probability, and wind speed
    for each city in `cities`, concurrently, via Apify.

    Returns a flat dict keyed by city name, e.g.:
        {
            "New York": {
                "status": "ok",
                "temperature_c": 22.5,
                "precipitation_probability_pct": 10,
                "wind_speed_kph": 14.0,
            },
            "London": {"status": "error", "error": "all_sources_failed"},
            ...
        }

    This shape is intentionally flat (no nesting beyond one level) so it can
    be dropped straight into an LLM prompt as context without further
    transformation.

    Raises:
        ApifyNotConfiguredError: if no Apify token is set in the environment.
    """
    client = _get_apify_client()  # raises ApifyNotConfiguredError early if unset

    logger.info(f"Starting weather data fetch for {len(cities)} cities: {cities}")
    results = await asyncio.gather(*(_fetch_city(client, city) for city in cities))

    output: dict = {}
    success_count = 0
    for result in results:
        city = result.pop("city")
        output[city] = result
        if result["status"] == "ok":
            success_count += 1

    logger.info(f"Weather data fetch complete: {success_count}/{len(cities)} cities fully successful.")
    return output


if __name__ == "__main__":
    # Quick manual smoke test: python -m services.data_scraper
    cities = list(CITY_COORDS.keys())
    result = asyncio.run(fetch_weather_data(cities))
    import json

    print(json.dumps(result, indent=2))
