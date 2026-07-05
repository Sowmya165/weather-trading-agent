"""
Typed models for raw and aggregated weather data.

Why this layer exists: every source (Open-Meteo, a secondary global API,
an Apify local scraper) returns a different shape. Normalizing into
WeatherObservation immediately means every downstream service only ever
deals with one schema, not three — and Pydantic's validation means a
malformed scrape result fails loudly at the model boundary instead of
silently propagating a bad float into a trading decision three layers down.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class WeatherSourceTier(str, Enum):
    GLOBAL_PRIMARY = "global_primary"      # e.g. Open-Meteo
    GLOBAL_SECONDARY = "global_secondary"  # e.g. WeatherAPI, used to cross-check
    LOCAL_SCRAPED = "local_scraped"        # Apify actor pulling a local met agency / station


class WeatherObservation(BaseModel):
    """
    A single source's reading for one city, normalized to common units
    (Celsius, mm, kph). Each scraped/fetched data point becomes exactly
    one of these — never a raw dict — before it touches any other module.
    """

    city: str
    source: str = Field(description="Identifier of the originating source, e.g. 'apify:weather-database-scraper'")
    tier: WeatherSourceTier
    observed_at: datetime
    forecast_for: datetime

    temperature_c: float | None = Field(default=None, ge=-90.0, le=60.0)
    humidity_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    precipitation_mm: float | None = Field(default=None, ge=0.0)
    wind_speed_kph: float | None = Field(default=None, ge=0.0)
    condition: str | None = None  # e.g. "clear", "rain", "snow"

    raw_payload_excerpt: str | None = Field(
        default=None, description="Truncated raw response for debugging/audit, not for parsing."
    )

    @field_validator("city")
    @classmethod
    def _city_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("city must not be blank")
        return v


class ForecastDisagreement(BaseModel):
    """Flag raised when sources diverge beyond tolerance for a given field."""

    field: str
    values_by_source: dict[str, float]
    spread: float
    flagged: bool


class AggregatedWeather(BaseModel):
    """
    Reconciled view of all observations collected for one city/forecast
    window, plus the disagreement detail that produced it. This — never
    a raw WeatherObservation list — is what the prediction agent consumes.
    """

    city: str
    forecast_for: datetime
    temperature_c: float
    humidity_pct: float | None = None
    precipitation_mm: float
    wind_speed_kph: float

    confidence_score: float = Field(ge=0.0, le=1.0)
    sources_used: list[str]
    sources_rejected: list[str] = Field(default_factory=list)
    disagreements: list[ForecastDisagreement] = Field(default_factory=list)
    observation_count: int = Field(default=0, ge=0)

    @classmethod
    def from_observations(cls, city: str, observations: list[WeatherObservation], **aggregated_fields):
        """Convenience constructor so callers don't have to pass observation_count manually."""
        return cls(city=city, observation_count=len(observations), **aggregated_fields)
