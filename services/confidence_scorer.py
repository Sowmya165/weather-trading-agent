"""
Confidence scoring service.

Takes raw observations from multiple sources for one city and reconciles
them into a single AggregatedWeather record. This is the "compare sources,
compute confidence, detect disagreements, reject noisy data" requirement
from the brief, isolated into one auditable, unit-testable function rather
than being buried inside agent prompt logic.

Design choice: confidence is NOT an LLM judgment call. It's a deterministic
statistical function of source agreement, so it's reproducible and auditable
in the dashboard. The LLM (in the agent layer) consumes this score; it doesn't
invent it.
"""
from datetime import datetime, timezone
from statistics import mean, pstdev

from models.weather import AggregatedWeather, ForecastDisagreement, WeatherObservation

# Tolerances: spread above this for a field triggers a disagreement flag.
TEMP_DISAGREEMENT_C = 2.0
PRECIP_DISAGREEMENT_MM = 3.0
WIND_DISAGREEMENT_KPH = 10.0

# A source's temperature reading more than this many degrees from the
# group mean is treated as noise and dropped before reconciling.
OUTLIER_REJECTION_THRESHOLD_C = 6.0


def score_and_aggregate(city: str, observations: list[WeatherObservation]) -> AggregatedWeather:
    if not observations:
        raise ValueError(f"No weather observations available for {city}; cannot aggregate.")

    valid, rejected = _reject_outliers(observations)
    if not valid:
        # Outlier rejection ate everything — fall back to the raw set rather
        # than failing outright, but confidence will reflect the disagreement.
        valid, rejected = observations, []

    temps = [o.temperature_c for o in valid if o.temperature_c is not None]
    precs = [o.precipitation_mm for o in valid if o.precipitation_mm is not None]
    winds = [o.wind_speed_kph for o in valid if o.wind_speed_kph is not None]

    disagreements = []
    disagreements += _check_disagreement("temperature_c", valid, lambda o: o.temperature_c, TEMP_DISAGREEMENT_C)
    disagreements += _check_disagreement("precipitation_mm", valid, lambda o: o.precipitation_mm, PRECIP_DISAGREEMENT_MM)
    disagreements += _check_disagreement("wind_speed_kph", valid, lambda o: o.wind_speed_kph, WIND_DISAGREEMENT_KPH)

    confidence = _compute_confidence(num_sources=len(valid), num_rejected=len(rejected), disagreements=disagreements)

    return AggregatedWeather(
        city=city,
        forecast_for=valid[0].forecast_for,
        temperature_c=round(mean(temps), 2) if temps else 0.0,
        precipitation_mm=round(mean(precs), 2) if precs else 0.0,
        wind_speed_kph=round(mean(winds), 2) if winds else 0.0,
        confidence_score=confidence,
        sources_used=[o.source for o in valid],
        sources_rejected=[o.source for o in rejected],
        disagreements=disagreements,
    )


def _reject_outliers(
    observations: list[WeatherObservation],
) -> tuple[list[WeatherObservation], list[WeatherObservation]]:
    temps = [o.temperature_c for o in observations if o.temperature_c is not None]
    if len(temps) < 2:
        return observations, []

    group_mean = mean(temps)
    valid, rejected = [], []
    for o in observations:
        if o.temperature_c is not None and abs(o.temperature_c - group_mean) > OUTLIER_REJECTION_THRESHOLD_C:
            rejected.append(o)
        else:
            valid.append(o)
    return valid, rejected


def _check_disagreement(field_name, observations, getter, tolerance) -> list[ForecastDisagreement]:
    values = {o.source: getter(o) for o in observations if getter(o) is not None}
    if len(values) < 2:
        return []
    spread = max(values.values()) - min(values.values())
    return [
        ForecastDisagreement(
            field=field_name,
            values_by_source=values,
            spread=round(spread, 2),
            flagged=spread > tolerance,
        )
    ]


def _compute_confidence(num_sources: int, num_rejected: int, disagreements: list[ForecastDisagreement]) -> float:
    """
    Confidence starts high with multiple agreeing sources and is penalized for:
    - having only one source (no cross-validation possible)
    - rejected outlier sources (signal the data was noisy to begin with)
    - flagged disagreements (sources actively conflict)
    """
    base = 0.95 if num_sources >= 2 else 0.6
    base -= 0.15 * num_rejected
    flagged_count = sum(1 for d in disagreements if d.flagged)
    base -= 0.1 * flagged_count
    return round(max(0.05, min(base, 0.99)), 2)
