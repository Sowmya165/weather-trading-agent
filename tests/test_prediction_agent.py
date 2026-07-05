import json
from datetime import datetime, timezone

import httpx
import pytest

from agents.prediction_agent import HermesPredictionAgent
from config.settings import Settings
from models.weather import AggregatedWeather


def _weather(confidence=0.9) -> AggregatedWeather:
    return AggregatedWeather(
        city="New York",
        forecast_for=datetime.now(timezone.utc),
        temperature_c=22.5,
        precipitation_mm=0.0,
        wind_speed_kph=12.0,
        confidence_score=confidence,
        sources_used=["open-meteo", "weatherapi"],
        sources_rejected=[],
        disagreements=[],
    )


def _settings() -> Settings:
    return Settings(openrouter_api_key="fake-key-for-tests")


def _client_with_response(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


@pytest.mark.asyncio
async def test_successful_prediction_is_parsed_into_typed_model():
    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "choices": [
                {"message": {"content": json.dumps({
                    "city": "New York",
                    "predicted_probability": 0.73,
                    "reasoning": "Temperature 22.5C with 0mm precipitation and high confidence 0.9 favors clear conditions.",
                })}}
            ]
        }
        return httpx.Response(200, json=body)

    client = _client_with_response(handler)
    agent = HermesPredictionAgent(_settings(), client=client)
    result = await agent.generate_prediction(_weather(), "Will it rain in New York tomorrow?")

    assert result is not None
    assert result.predicted_probability == 0.73
    assert result.probability == 0.73  # backward-compat alias
    assert result.city == "New York"
    assert "0.9" in result.reasoning or "22.5" in result.reasoning
    await agent.aclose()


@pytest.mark.asyncio
async def test_malformed_json_returns_none_not_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"choices": [{"message": {"content": "not valid json at all"}}]}
        return httpx.Response(200, json=body)

    client = _client_with_response(handler)
    agent = HermesPredictionAgent(_settings(), client=client)
    result = await agent.generate_prediction(_weather(), "Will it rain in New York tomorrow?")

    assert result is None
    await agent.aclose()


@pytest.mark.asyncio
async def test_missing_required_field_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        # Missing predicted_probability entirely
        body = {"choices": [{"message": {"content": json.dumps({"city": "New York", "reasoning": "x"})}}]}
        return httpx.Response(200, json=body)

    client = _client_with_response(handler)
    agent = HermesPredictionAgent(_settings(), client=client)
    result = await agent.generate_prediction(_weather(), "Will it rain in New York tomorrow?")

    assert result is None
    await agent.aclose()


@pytest.mark.asyncio
async def test_out_of_range_probability_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"choices": [{"message": {"content": json.dumps({
            "city": "New York", "predicted_probability": 1.5, "reasoning": "x",
        })}}]}
        return httpx.Response(200, json=body)

    client = _client_with_response(handler)
    agent = HermesPredictionAgent(_settings(), client=client)
    result = await agent.generate_prediction(_weather(), "Will it rain in New York tomorrow?")

    assert result is None
    await agent.aclose()


@pytest.mark.asyncio
async def test_rate_limit_429_returns_none_not_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    client = _client_with_response(handler)
    agent = HermesPredictionAgent(_settings(), client=client)
    result = await agent.generate_prediction(_weather(), "Will it rain in New York tomorrow?")

    assert result is None
    await agent.aclose()


@pytest.mark.asyncio
async def test_timeout_returns_none_not_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated timeout", request=request)

    client = _client_with_response(handler)
    agent = HermesPredictionAgent(_settings(), client=client)
    result = await agent.generate_prediction(_weather(), "Will it rain in New York tomorrow?")

    assert result is None
    await agent.aclose()


@pytest.mark.asyncio
async def test_missing_api_key_returns_none_without_network_call():
    settings = Settings(openrouter_api_key="")
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = _client_with_response(handler)
    agent = HermesPredictionAgent(settings, client=client)
    result = await agent.generate_prediction(_weather(), "Will it rain in New York tomorrow?")

    assert result is None
    assert called is False  # should short-circuit before making the request
    await agent.aclose()
