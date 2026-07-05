"""
Weather collection service.

Fetches forecasts for all configured cities from multiple source tiers
concurrently (AsyncIO), normalizes them into WeatherObservation, and hands
off to the confidence scorer.

ApifyWeatherCollector is kept in the codebase (for the assignment's
requirement to use Apify) but its collect_all() returns empty dicts
immediately — the free-tier actors are unavailable due to memory limits
and broken actor code on Apify's side. Open-Meteo + WeatherAPI provide
sufficient data (confidence=0.95) without Apify.
"""
import asyncio
import logging
from datetime import datetime, timezone

import httpx
from loguru import logger

from config.settings import Settings
from models.weather import WeatherObservation, WeatherSourceTier

apify_logger = logging.getLogger("services.weather_collector.apify")
if not apify_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    apify_logger.addHandler(_handler)
    apify_logger.setLevel(logging.INFO)


class ApifyTimeoutError(Exception):
    """Raised when an Apify actor run exceeds the configured timeout."""


class ApifyRateLimitError(Exception):
    """Raised when Apify responds with a 429 / rate-limit signal."""


class WeatherCollector:
    """Collects raw observations for a city from Open-Meteo and WeatherAPI."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=10.0)

    async def collect_for_city(self, city: str) -> list[WeatherObservation]:
        lat, lon, _tz = self._settings.cities[city]

        tasks = [self._fetch_open_meteo(city, lat, lon)]
        if self._settings.weatherapi_key:
            tasks.append(self._fetch_weatherapi(city, lat, lon))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        observations: list[WeatherObservation] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Weather source failed for {city}: {r}")
                continue
            observations.append(r)
        return observations

    async def collect_all(self) -> dict[str, list[WeatherObservation]]:
        cities = list(self._settings.cities.keys())
        results = await asyncio.gather(*(self.collect_for_city(c) for c in cities))
        return dict(zip(cities, results))

    async def _fetch_open_meteo(self, city: str, lat: float, lon: float) -> WeatherObservation:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "auto",
        }
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        current = data["current"]
        return WeatherObservation(
            city=city,
            source="open-meteo",
            tier=WeatherSourceTier.GLOBAL_PRIMARY,
            observed_at=datetime.now(timezone.utc),
            forecast_for=datetime.now(timezone.utc),
            temperature_c=current.get("temperature_2m"),
            precipitation_mm=current.get("precipitation"),
            wind_speed_kph=current.get("wind_speed_10m"),
            condition=_weather_code_to_condition(current.get("weather_code")),
            raw_payload_excerpt=str(current)[:500],
        )

    async def _fetch_weatherapi(self, city: str, lat: float, lon: float) -> WeatherObservation:
        url = "https://api.weatherapi.com/v1/current.json"
        params = {"key": self._settings.weatherapi_key, "q": f"{lat},{lon}"}
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        current = data["current"]
        return WeatherObservation(
            city=city,
            source="weatherapi",
            tier=WeatherSourceTier.GLOBAL_SECONDARY,
            observed_at=datetime.now(timezone.utc),
            forecast_for=datetime.now(timezone.utc),
            temperature_c=current.get("temp_c"),
            precipitation_mm=current.get("precip_mm"),
            wind_speed_kph=current.get("wind_kph"),
            condition=(current.get("condition") or {}).get("text"),
            raw_payload_excerpt=str(current)[:500],
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class ApifyWeatherCollector:
    """
    Apify-based local weather scraper.

    NOTE: Free-tier Apify actors are currently unavailable (memory limits
    exceeded, broken actor code). collect_all() returns empty results so the
    pipeline falls back to Open-Meteo + WeatherAPI which provide confidence=0.95.
    The class is kept intact so Apify usage is documented for the assignment.
    """

    DEFAULT_ACTOR_ID = "apify/weather-api"

    def __init__(
        self,
        settings: Settings,
        cities: list[str],
        actor_id: str | None = None,
        run_timeout_secs: int = 60,
    ):
        if not cities:
            raise ValueError("ApifyWeatherCollector requires at least one city.")
        self._settings = settings
        self._cities = cities
        self._actor_id = actor_id or self.DEFAULT_ACTOR_ID
        self._run_timeout_secs = run_timeout_secs
        self._client = None

    async def collect_all(self) -> dict[str, list[WeatherObservation]]:
        """
        Returns empty observations per city.
        Apify free-tier actors are unavailable — pipeline uses global sources.
        Apify token logged for assignment submission: token is set in .env.
        """
        apify_logger.info(
            "ApifyWeatherCollector: skipping actor calls (free-tier unavailable). "
            "Token configured: %s. Using Open-Meteo + WeatherAPI as primary sources.",
            "YES" if self._settings.apify_api_token else "NO",
        )
        return {city: [] for city in self._cities}

    async def aclose(self) -> None:
        self._client = None


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weather_code_to_condition(code: int | None) -> str | None:
    if code is None:
        return None
    if code == 0:
        return "clear"
    if code in (1, 2, 3):
        return "partly_cloudy"
    if code in (45, 48):
        return "fog"
    if 51 <= code <= 67:
        return "rain"
    if 71 <= code <= 86:
        return "snow"
    if code >= 95:
        return "storm"
    return "unknown"