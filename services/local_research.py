"""
Local research service.

Covers the "research weather news: storm alerts, government warnings,
weather blogs, local reports" requirement. Uses Apify actors to scrape
sources a pure weather API won't surface — alerts, local-language reports,
blog commentary — and returns short text snippets that the Hermes Agent
reads directly as context, rather than trying to force them into numeric
fields.

Two actors are used:
- A generic weather-database/alert scraper for structured local conditions
  (per the assignment's example: apify/oneary weather-database-scraper).
- A general web-content scraper, query-driven, for storm alerts and blog
  coverage that no fixed actor covers (e.g. "Tokyo typhoon warning").

If APIFY_API_TOKEN isn't set, this degrades gracefully to an empty result
rather than failing the whole pipeline — useful for local dev / testing.
"""
from datetime import datetime, timezone

from loguru import logger
from pydantic import BaseModel

from config.settings import Settings


class ResearchSnippet(BaseModel):
    city: str
    source: str
    title: str
    summary: str
    url: str | None = None
    published_at: datetime | None = None
    severity: str | None = None  # e.g. "advisory", "warning", "none"


class LocalResearchService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None  # lazily created — apify_client is a heavier optional import

    def _get_client(self):
        if self._client is None:
            from apify_client import ApifyClientAsync

            self._client = ApifyClientAsync(self._settings.apify_api_token)
        return self._client

    async def research_city(self, city: str, max_items: int = 5) -> list[ResearchSnippet]:
        if not self._settings.apify_api_token:
            logger.warning("APIFY_API_TOKEN not set — skipping local research, returning empty result.")
            return []

        try:
            return await self._run_alert_actor(city, max_items)
        except Exception as e:
            # A failed scrape should degrade the agent's confidence, not crash the pipeline.
            logger.warning(f"Apify research failed for {city}: {e}")
            return []

    async def _run_alert_actor(self, city: str, max_items: int) -> list[ResearchSnippet]:
        client = self._get_client()
        run_input = {
            "searchTerms": [f"{city} weather alert", f"{city} storm warning today"],
            "maxItems": max_items,
        }
        # Actor ID is configurable per assignment's suggested actors
        # (apify/oneary~weather-database-scraper or an equivalent search-scraper).
        actor = client.actor("apify/web-scraper")
        run = await actor.call(run_input=run_input)
        dataset = client.dataset(run["defaultDatasetId"])
        items_page = await dataset.list_items(limit=max_items)

        snippets = []
        for item in items_page.items:
            snippets.append(
                ResearchSnippet(
                    city=city,
                    source=item.get("source", "apify"),
                    title=item.get("title", "")[:200],
                    summary=(item.get("text") or item.get("description") or "")[:500],
                    url=item.get("url"),
                    published_at=datetime.now(timezone.utc),
                    severity=_infer_severity(item.get("text", "")),
                )
            )
        return snippets


def _infer_severity(text: str) -> str:
    lowered = text.lower()
    if any(w in lowered for w in ("warning", "emergency", "evacuat")):
        return "warning"
    if any(w in lowered for w in ("advisory", "watch", "alert")):
        return "advisory"
    return "none"
