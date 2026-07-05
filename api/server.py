"""
api/server.py

FastAPI server — bridge between the SQLite persistence layer and the
React frontend. All endpoints are read-only GET requests (the trading
pipeline writes; the API only reads), which keeps the API stateless and
trivially cacheable.

CORS: configured to allow the two standard Vite/CRA dev-server origins
(localhost:5173 and localhost:3000) plus any production origin specified
via ALLOWED_ORIGINS env var. In production, tighten this to the exact
deployed frontend URL.

Run standalone (development):
    uvicorn api.server:app --reload --port 8000

The React app should proxy /api/* to http://localhost:8000 in its
vite.config.ts or package.json proxy field.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import get_settings
from database.db import DatabaseManager

logger = logging.getLogger("api.server")

# ── DB singleton ───────────────────────────────────────────────────────────
# Initialised once on startup via lifespan; injected into route handlers
# via module-level reference (no DI framework needed at this scale).
_db: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    if _db is None:
        raise RuntimeError("Database not initialised — did the lifespan handler run?")
    return _db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB on startup, close cleanly on shutdown."""
    global _db
    settings = get_settings()
    _db = DatabaseManager(settings.database_url)
    await _db.init()
    logger.info("API server started — DB ready.")
    yield
    await _db.close()
    logger.info("API server shutdown — DB closed.")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Weather Trading Agent API",
    description="Read-only API exposing predictions, risk decisions, and paper trades to the React dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow both Vite (5173) and CRA (3000) dev servers, plus any
# production origin injected via environment variable.
_extra_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        *_extra_origins,
    ],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],   # read-only API — no POST/PUT/DELETE needed
    allow_headers=["*"],
)


# ── Response models ────────────────────────────────────────────────────────
# Explicit Pydantic response models keep the OpenAPI schema accurate and
# give the React dev typed documentation for free via /docs.

class StatusResponse(BaseModel):
    status: str
    total_predictions: int
    total_trades_placed: int
    filled_trades: int
    total_staked_usd: float
    realized_pnl_usd: float
    last_run_at: str | None


class PredictionResponse(BaseModel):
    id: int
    city: str
    condition_id: str | None
    question: str | None
    predicted_probability: float
    market_implied_probability: float | None
    edge: float | None
    expected_value: float | None
    confidence: float
    reasoning: str
    generated_at: str


class RiskAnalysisResponse(BaseModel):
    id: int
    condition_id: str
    kelly_fraction_applied: float
    raw_kelly_stake_usd: float
    capped_stake_usd: float
    approved: bool
    reasons: list[str]
    hedge_condition_id: str | None
    hedge_stake_usd: float | None
    recorded_at: str


class TradeResponse(BaseModel):
    id: str
    city: str
    condition_id: str
    side: str
    outcome_name: str
    stake_usd: float
    size_shares: float
    entry_price: float
    status: str
    is_hedge: bool
    linked_trade_id: str | None
    placed_at: str
    reasoning_summary: str
    market_question: str
    resolution: str | None
    pnl_usd: float | None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/status", response_model=StatusResponse, tags=["Agent"])
async def get_status():
    """
    Returns the current operational status of the agent.
    The React dashboard uses this to populate the top-level KPI cards
    (total trades, staked amount, realized P&L, last run timestamp).
    """
    try:
        return await get_db().fetch_status()
    except Exception as e:
        logger.error("Failed to fetch status: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch agent status.")


@app.get("/api/predictions", response_model=list[PredictionResponse], tags=["Predictions"])
async def get_predictions(
    limit: int = Query(default=100, ge=1, le=500, description="Max rows to return"),
    city: str | None = Query(default=None, description="Filter by city name"),
):
    """
    Returns the most recent LLM predictions in reverse chronological order.
    Includes the model's full reasoning string so the dashboard can show
    the 'explain every trade' requirement inline.
    """
    try:
        return await get_db().fetch_predictions(limit=limit, city=city)
    except Exception as e:
        logger.error("Failed to fetch predictions: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch predictions.")


@app.get("/api/risk-analysis", response_model=list[RiskAnalysisResponse], tags=["Risk"])
async def get_risk_analysis(
    limit: int = Query(default=100, ge=1, le=500),
):
    """
    Returns the historical Kelly criterion outputs: fraction applied,
    raw vs. capped stake, approval status, and reasons for any caps.
    The React Risk page uses this to render the Kelly allocation chart
    and the rejection-reason breakdown.
    """
    try:
        return await get_db().fetch_risk_decisions(limit=limit)
    except Exception as e:
        logger.error("Failed to fetch risk decisions: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch risk analysis.")


@app.get("/api/trades", response_model=list[TradeResponse], tags=["Trades"])
async def get_trades(
    limit: int = Query(default=100, ge=1, le=500),
    city: str | None = Query(default=None, description="Filter by city"),
    status: str | None = Query(default=None, description="Filter by status: filled | rejected | hedged"),
):
    """
    Returns the full paper-trade ledger in reverse chronological order.
    Hedge legs are included as separate rows with is_hedge=true and
    linked_trade_id pointing to the primary trade, so the frontend can
    group and display them together.
    """
    try:
        return await get_db().fetch_trades(limit=limit, city=city, status=status)
    except Exception as e:
        logger.error("Failed to fetch trades: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch trades.")


@app.get("/api/health", tags=["Internal"])
async def health_check():
    """Lightweight liveness probe — no DB call, just confirms the process is up."""
    return {"ok": True}
