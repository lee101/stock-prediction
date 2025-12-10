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

# GPT-5 reasoning effort used for plan generation.
DEFAULT_REASONING_EFFORT = "high"
