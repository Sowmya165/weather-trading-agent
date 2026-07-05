# 🌩️ Weather Trading Agent

An AI-powered prediction market trading agent that trades weather markets on Polymarket across 5 cities using multi-source weather data, LLM-based probability estimation, and Kelly Criterion risk management.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                            │
│  Open-Meteo (free) + WeatherAPI + Apify (local scraping)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Confidence Scorer                              │
│  Outlier rejection · Disagreement detection · Source merge  │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐     ┌───────────────────────┐
│ Polymarket Data │     │   Hermes Agent         │
│ Gamma API       │     │   OpenRouter LLM       │
│ Market prices   │     │   Batch predictions    │
└────────┬────────┘     └──────────┬────────────┘
         └──────────┬──────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   Risk Manager                              │
│   Kelly Criterion · Exposure limits · Hedging logic         │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Paper Trade Executor                           │
│   Polymarket CLOB payload · Ledger · P&L tracking          │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           FastAPI + SQLite + React Dashboard                │
│   /api/status · /api/predictions · /api/trades · /api/risk  │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

```
1. WeatherCollector     → Fetch Open-Meteo + WeatherAPI concurrently for 5 cities
2. ConfidenceScorer     → Outlier rejection, disagreement detection, confidence score
3. PolymarketData       → Fetch active weather markets via Gamma API
4. HermesPredictionAgent→ Single batch LLM call → probabilities for all cities
5. RiskManager          → Kelly fraction, exposure caps, hedge sizing
6. PaperTrader          → Format CLOB payload, log trade, track P&L
7. DatabaseManager      → Persist full cycle to SQLite
8. FastAPI              → Serve data to React dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | Hermes Agent (NousResearch) |
| LLM Provider | OpenRouter — `meta-llama/llama-3.3-70b-instruct:free` |
| Weather Data | Open-Meteo (free, no key) + WeatherAPI |
| Local Scraping | Apify (`apify/weather-database-scraper`) |
| Market Data | Polymarket Gamma API + CLOB API |
| Risk Engine | Kelly Criterion (fractional, 0.25×) |
| Backend | FastAPI + SQLite (aiosqlite + SQLAlchemy async) |
| Frontend | React + TypeScript + Vite + Tailwind CSS + Recharts |
| Testing | pytest + pytest-asyncio (38 tests) |
| Language | Python 3.12 |

---

## Project Structure

```
weather-trading-agent/
├── agents/
│   ├── prediction_agent.py     # Hermes Agent + OpenRouter LLM integration
│   └── tools.py                # Custom tool registrations for Hermes
├── services/
│   ├── weather_collector.py    # Multi-source async weather fetching
│   ├── confidence_scorer.py    # Outlier rejection + disagreement detection
│   ├── local_research.py       # Apify news/alert scraping
│   ├── polymarket_data.py      # Gamma API market discovery
│   ├── risk_manager.py         # Kelly criterion + hedging logic
│   └── trade_executor.py       # Paper trade execution + CLOB payload
├── models/
│   ├── weather.py              # WeatherObservation, AggregatedWeather
│   └── market.py              # Prediction, RiskDecision, Trade
├── api/
│   └── server.py              # FastAPI endpoints
├── database/
│   └── db.py                  # SQLAlchemy async ORM + DatabaseManager
├── config/
│   └── settings.py            # Typed pydantic-settings configuration
├── dashboard/                  # React + TypeScript frontend
│   └── src/
│       ├── services/api.ts     # Typed API client
│       ├── hooks/useApi.ts     # Polling hook
│       └── components/
│           ├── DashboardLayout.tsx
│           ├── SystemStatus.tsx
│           ├── TradesLedger.tsx
│           ├── RiskChart.tsx
│           └── PredictionsFeed.tsx
├── tests/                      # 38 unit tests (no network calls)
│   ├── test_confidence_scorer.py
│   ├── test_risk_manager.py
│   ├── test_prediction_agent.py
│   ├── test_trade_executor.py
│   └── test_db.py
├── main.py                     # Orchestrator entry point
├── requirements.txt
└── .env.example
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/weather-trading-agent.git
cd weather-trading-agent

# 2. Virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env — add your OPENROUTER_API_KEY and APIFY_API_TOKEN

# 5. Run tests
python -m pytest tests/ -v

# 6. Run the pipeline
python main.py

# 7. Start the API server (new terminal)
uvicorn api.server:app --reload --port 8000

# 8. Start the dashboard (new terminal)
cd dashboard
npm install
npm run dev
```

---

## Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-v1-...         # Required — OpenRouter API key
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
APIFY_API_TOKEN=apify_api_...           # Required — Apify token
WEATHERAPI_KEY=...                      # Optional — adds second weather source
KELLY_FRACTION=0.25                     # Fractional Kelly multiplier
MAX_POSITION_PCT_OF_BANKROLL=0.05       # Max 5% per trade
MAX_DAILY_LOSS_PCT=0.10                 # Stop trading after 10% daily loss
MAX_PORTFOLIO_EXPOSURE_PCT=0.40         # Max 40% total exposure
STARTING_BANKROLL_USD=1000
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/status` | Agent status, total trades, P&L, last run time |
| `GET /api/predictions` | LLM predictions with reasoning, filterable by city |
| `GET /api/trades` | Full paper trade ledger with hedge linking |
| `GET /api/risk-analysis` | Kelly fraction history and stake sizing |
| `GET /docs` | Interactive API documentation (Swagger UI) |

---

## Risk Management Design

**Kelly Criterion (binary market form):**
```
f* = (b×p - q) / b
where:
  p = agent's estimated probability (from LLM)
  q = 1 - p
  b = (1 - market_price) / market_price  (net odds)
  f* = fraction of bankroll to stake
```

**Applied as fractional Kelly (0.25×)** — full Kelly is theoretically optimal but produces large drawdowns on noisy probability estimates. Quarter-Kelly is the standard practical choice.

**Hard limits applied after Kelly sizing:**
- Per-position cap: max 5% of bankroll per trade
- Daily loss limit: no new trades after 10% daily loss
- Portfolio exposure cap: max 40% total open exposure
- Confidence scaling: low-confidence predictions halve the stake

**Hedging:** When two markets share a correlated weather driver (e.g. NYC high temp ranges), the risk engine sizes an offsetting position in the correlated market proportional to confidence uncertainty and edge direction disagreement.

---

## Design Decisions

**Why one batch LLM call instead of one per market?**
OpenRouter's free tier allows ~20 requests/minute. With 79 active markets across 4 cities, per-market calls exhaust the quota immediately. A single prompt covering all cities costs one call and returns all predictions in ~2 seconds.

**Why deterministic confidence scoring instead of LLM judgment?**
Confidence is a pure statistical function of source agreement — keeping it outside the LLM makes it reproducible, auditable, and visible in the dashboard without any risk of the model inventing agreement where none exists.

**Why SQLAlchemy async over raw sqlite3?**
`sqlite3` is synchronous — calling it directly blocks the asyncio event loop on every DB write, stalling the trading pipeline. `aiosqlite` keeps all DB operations non-blocking.

**Why FastAPI read-only endpoints?**
The trading pipeline writes; the API only reads. Restricting CORS to `GET` and `OPTIONS` eliminates an entire class of accidental state mutation from the browser.

---

## Cities Supported

| City | Lat | Lon | Timezone |
|---|---|---|---|
| New York | 40.71 | -74.01 | America/New_York |
| London | 51.51 | -0.13 | Europe/London |
| Paris | 48.86 | 2.35 | Europe/Paris |
| Tokyo | 35.68 | 139.65 | Asia/Tokyo |
| Berlin | 52.52 | 13.41 | Europe/Berlin |

---

## Statistical Results

After running the pipeline:
- Weather data collected from 2 sources per city (Open-Meteo + WeatherAPI)
- Confidence scores: 0.75–0.95 across cities
- Markets discovered: 79 active weather markets (NY: 12, London: 22, Paris: 23, Tokyo: 22)
- Paper trades: logged with full CLOB API payload, Kelly sizing, and hedge legs
- All results queryable via `/api/trades` and `/api/risk-analysis`

---

## Running Modes

```bash
# Single pipeline pass
python main.py

# Continuous loop every 5 minutes
python main.py --loop 300

# API server only (reads from existing DB)
uvicorn api.server:app --reload --port 8000
```

---

## Future Improvements

- Telegram alerts on trade execution
- Backtesting mode with historical Polymarket data
- Auto-generated daily trading journal
- Prometheus + Grafana observability stack
- Multi-model ensemble predictions (average across 2-3 LLMs)
- Resolution tracking — close the P&L loop when markets settle

---

## Submission Notes

- **APIFY Token:** Configured in `.env` — token used for `apify/weather-database-scraper` actor
- **OpenRouter:** `meta-llama/llama-3.3-70b-instruct:free` via direct chat completions API
- **Paper trading only** — no real funds, no wallet signing
- **38 unit tests** covering confidence scoring, Kelly math, prediction validation, trade execution, and database persistence

---

*Built for CrowdWisdomTrading AI Agent Internship Assessment — Sowmya Vadde, IIIT Raichur (CS23B1077)*