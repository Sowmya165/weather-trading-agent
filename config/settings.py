"""
Centralized application configuration.

Loaded once from environment / .env and imported everywhere else as a
typed object, instead of scattering os.getenv() calls through the codebase.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="meta-llama/llama-3.3-70b-instruct:free")

    # Data sources
    apify_api_token: str = Field(default="")
    weatherapi_key: str = Field(default="")

    # Polymarket
    polymarket_host: str = Field(default="https://clob.polymarket.com")
    polymarket_gamma_host: str = Field(default="https://gamma-api.polymarket.com")
    polymarket_paper_trader_db: str = Field(default="database/paper_trades.db")

    # Risk management
    kelly_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    max_position_pct_of_bankroll: float = Field(default=0.05, ge=0.0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    max_portfolio_exposure_pct: float = Field(default=0.40, ge=0.0, le=1.0)
    starting_bankroll_usd: float = Field(default=1000.0, gt=0)

    # App
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite+aiosqlite:///database/app.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Supported cities — name -> (lat, lon, IANA timezone)
    cities: dict[str, tuple[float, float, str]] = {
        "New York": (40.7128, -74.0060, "America/New_York"),
        "London": (51.5074, -0.1278, "Europe/London"),
        "Paris": (48.8566, 2.3522, "Europe/Paris"),
        "Tokyo": (35.6762, 139.6503, "Asia/Tokyo"),
        "Berlin": (52.5200, 13.4050, "Europe/Berlin"),
    }


@lru_cache
def get_settings() -> Settings:
    """Settings are cheap to validate but no reason to re-parse env on every call."""
    return Settings()
