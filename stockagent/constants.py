"""Constants shared by the stateful GPT agent."""

DEFAULT_SYMBOLS = [
    "COUR",
    "GOOG",
    "TSLA",
    "NVDA",
    "AAPL",
    "U",
    "ADSK",
    "CRWD",
    "ADBE",
    "NET",
    "COIN",
    "META",
    "AMZN",
    "AMD",
    "INTC",
    "LCID",
    "QUBT",
    "BTCUSD",
    "ETHUSD",
    "UNIUSD",
]

SIMULATION_DAYS = 12
SIMULATION_OPEN_TIME = "09:30"
SIMULATION_CLOSE_TIME = "16:00"

# approx taker fees (per-side) used in simulator
TRADING_FEE = 0.0005  # equities (5 bps per side = ~10 bps round-trip)
CRYPTO_TRADING_FEE = 0.0008  # crypto (8 bps per side = ~16 bps round-trip)

# Shared risk defaults used by prompt guidance and simulator overlays.
DEFAULT_PROBE_TRADE_MULTIPLIER = 0.05
DEFAULT_MIN_PROBE_QUANTITY = 0.01
DEFAULT_MAX_NEW_POSITION_NOTIONAL_FLOOR = 25_000.0
DEFAULT_MAX_NEW_POSITION_EQUITY_FRACTION = 0.05
DEFAULT_RECENT_LOSS_LOOKBACK_DAYS = 2
DEFAULT_RISK_HIGHLIGHT_SYMBOL_LIMIT = 4

# GPT-5 reasoning effort used for plan generation.
DEFAULT_REASONING_EFFORT = "high"
