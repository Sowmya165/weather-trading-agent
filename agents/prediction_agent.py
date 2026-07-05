"""
agents/prediction_agent.py

Intelligence layer: turns reconciled weather data into probability
estimates for Polymarket weather markets via OpenRouter.
Uses a single batch API call for all cities to avoid rate limits.
"""
from __future__ import annotations

import json
import asyncio
import logging
from datetime import datetime, timezone

import httpx

from config.settings import Settings
from models.market import Prediction
from models.weather import AggregatedWeather

logger = logging.getLogger("agents.prediction_agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

OPENROUTER_CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"

SYSTEM_PROMPT = """You are a quantitative meteorologist working inside an automated trading system.

Rules:
- Do not invent data that is not in the snapshot.
- Do not discuss trading strategy or money.
- Your reasoning must reference the specific numbers given.
- Return ONLY valid JSON. No prose, no markdown fences.
- Each prediction must have exactly these keys: "city", "market_question", "predicted_probability", "reasoning".
- Example item: {"city": "London", "market_question": "Will it rain?", "predicted_probability": 0.73, "reasoning": "Precip is 0.0mm and confidence is 0.95, indicating low rain probability."}
"""


class OpenRouterError(Exception):
    pass


class HermesPredictionAgent:
    def __init__(
        self,
        settings: Settings,
        model: str = DEFAULT_MODEL,
        request_timeout_secs: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        self._settings = settings
        self._model = model
        self._timeout = request_timeout_secs
        self._client = client or httpx.AsyncClient(timeout=request_timeout_secs)
        self._hermes_agent = self._try_init_hermes_agent()

    def _try_init_hermes_agent(self):
        try:
            from hermes_agent import AIAgent  # type: ignore[import-not-found]
            return AIAgent(
                model=self._model,
                api_base="https://openrouter.ai/api/v1",
                api_key=self._settings.openrouter_api_key,
            )
        except ImportError:
            logger.info("hermes_agent package not installed — using direct OpenRouter client only.")
            return None
        except Exception as e:
            logger.warning("Hermes AIAgent failed to initialize: %s", e)
            return None

    async def generate_predictions_batch(
        self,
        weather_map: dict,
        markets_map: dict,
    ) -> dict:
        """
        One LLM call covering all cities and their top 2 markets.
        Retries up to 5 times with 15-second waits on rate limiting.
        """
        if not self._settings.openrouter_api_key:
            logger.error("OPENROUTER_API_KEY not configured.")
            return {}

        prompt_lines = []
        for city, weather in weather_map.items():
            markets = markets_map.get(city, [])[:2]
            if not markets:
                continue
            for market in markets:
                prompt_lines.append(
                    f"- City: {city} | Market: {market.question} | "
                    f"Temp: {weather.temperature_c}C | Precip: {weather.precipitation_mm}mm | "
                    f"Wind: {weather.wind_speed_kph}kph | Confidence: {weather.confidence_score}"
                )

        if not prompt_lines:
            logger.warning("No markets to predict — prompt_lines is empty.")
            return {}

        user_prompt = (
            "For each market below, estimate the probability it resolves YES.\n"
            'Return a JSON object with key "predictions" containing an array.\n'
            'Each item must have: "city", "market_question", "predicted_probability" (0.0-1.0), "reasoning" (1 sentence citing the numbers).\n\n'
            + "\n".join(prompt_lines)
        )

        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        items = []
        for attempt in range(5):
            try:
                logger.info("OpenRouter call attempt %d/5 ...", attempt + 1)
                response = await self._client.post(
                    OPENROUTER_CHAT_ENDPOINT, headers=headers, json=payload
                )

                if response.status_code == 429:
                    wait_secs = 15 * (attempt + 1)
                    logger.warning(
                        "Rate limited (429) — waiting %ds before retry %d/5 ...",
                        wait_secs, attempt + 1,
                    )
                    await asyncio.sleep(wait_secs)
                    continue

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                # Handle {"predictions": [...]} or bare list or other wrapper keys
                if isinstance(parsed, list):
                    items = parsed
                elif isinstance(parsed, dict):
                    for key in ("predictions", "results", "markets", "data", "items"):
                        if key in parsed and isinstance(parsed[key], list):
                            items = parsed[key]
                            break
                    else:
                        items = next(
                            (v for v in parsed.values() if isinstance(v, list)), []
                        )
                else:
                    items = []

                logger.info("OpenRouter returned %d prediction items.", len(items))
                break  # success — exit retry loop

            except httpx.TimeoutException:
                logger.warning("OpenRouter timed out on attempt %d/5.", attempt + 1)
                await asyncio.sleep(10)
                continue
            except Exception as e:
                if attempt == 4:
                    logger.error("Batch prediction failed after 5 attempts: %s", e)
                    return {}
                logger.warning("Attempt %d failed: %s — retrying in 15s ...", attempt + 1, e)
                await asyncio.sleep(15)
                continue
        else:
            logger.error("All 5 retry attempts exhausted without a successful response.")
            return {}

        results: dict = {}
        now = datetime.now(timezone.utc)
        for item in items:
            city = item.get("city", "")
            if city not in weather_map:
                continue
            try:
                p = Prediction(
                    city=city,
                    predicted_probability=float(item["predicted_probability"]),
                    reasoning=item.get("reasoning", ""),
                    question=item.get("market_question"),
                    confidence=weather_map[city].confidence_score,
                    generated_at=now,
                )
                results.setdefault(city, []).append(p)
            except Exception as e:
                logger.warning("Skipping malformed prediction for %s: %s | raw=%s", city, e, item)

        logger.info(
            "Batch prediction complete — %d predictions across %d cities in 1 API call.",
            sum(len(v) for v in results.values()),
            len(results),
        )
        return results

    async def generate_prediction(
        self, weather: AggregatedWeather, market_question: str
    ) -> Prediction | None:
        """Single-market prediction (kept for backward compatibility)."""
        result = await self.generate_predictions_batch(
            weather_map={weather.city: weather},
            markets_map={},
        )
        preds = result.get(weather.city, [])
        return preds[0] if preds else None

    async def aclose(self) -> None:
        await self._client.aclose()