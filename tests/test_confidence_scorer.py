from datetime import datetime, timezone

from models.weather import WeatherObservation, WeatherSourceTier
from services.confidence_scorer import score_and_aggregate


def _obs(source: str, temp: float, tier=WeatherSourceTier.GLOBAL_PRIMARY) -> WeatherObservation:
    now = datetime.now(timezone.utc)
    return WeatherObservation(
        city="London",
        source=source,
        tier=tier,
        observed_at=now,
        forecast_for=now,
        temperature_c=temp,
        precipitation_mm=0.0,
        wind_speed_kph=10.0,
        condition="clear",
    )


def test_agreeing_sources_yield_high_confidence():
    obs = [_obs("open-meteo", 18.0), _obs("weatherapi", 18.4)]
    result = score_and_aggregate("London", obs)
    assert result.confidence_score >= 0.9
    assert result.sources_rejected == []
    assert not any(d.flagged for d in result.disagreements)


def test_single_source_lowers_confidence():
    result = score_and_aggregate("London", [_obs("open-meteo", 18.0)])
    assert result.confidence_score <= 0.6


def test_outlier_source_is_rejected_and_flagged():
    obs = [_obs("open-meteo", 18.0), _obs("weatherapi", 18.5), _obs("rogue-source", 32.0)]
    result = score_and_aggregate("London", obs)
    assert "rogue-source" in result.sources_rejected
    assert result.confidence_score < 0.95


def test_disagreeing_sources_flagged_without_outlier_rejection():
    # Spread is large enough to flag but not so large it triggers hard outlier rejection.
    obs = [_obs("open-meteo", 18.0), _obs("weatherapi", 21.5)]
    result = score_and_aggregate("London", obs)
    temp_flags = [d for d in result.disagreements if d.field == "temperature_c"]
    assert temp_flags[0].flagged is True
    assert result.confidence_score < 0.95


def test_no_observations_raises():
    import pytest

    with pytest.raises(ValueError):
        score_and_aggregate("London", [])
