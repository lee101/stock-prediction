import os
from dataclasses import dataclass, field
from pathlib import Path

CRYPTO_SYMBOLS = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "AAVE", "LTC",
    "XRP", "DOT", "UNI", "NEAR", "APT", "ICP", "BNB",
    "ADA", "FIL", "ARB", "OP", "INJ", "SUI", "TIA", "SEI",
    "ATOM", "ALGO", "BCH", "TRX", "SHIB", "PEPE",
]

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR.parent
DATA_DIR = REPO_DIR / "trainingdata" / "train"
FORECAST_CACHE_DIR = REPO_DIR / "binanceneural" / "forecast_cache" / "h24"
PRED_CACHE_DIR = BASE_DIR / "pred_cache"

OPENPATHS_API_KEY = os.environ.get("OPENPATHS_API_KEY", "")
OPENPATHS_BASE_URL = os.environ.get("OPENPATHS_BASE_URL", "https://openpaths.io/v1")


@dataclass
class GStockConfig:
    symbols: list = field(default_factory=lambda: list(CRYPTO_SYMBOLS))
    leverage: float = 1.0
    model: str = "gemini-flash"  # gemini-flash, gemini-3.1, glm-5, glm-4-plus
    fee_bps: float = 10.0
    margin_annual_rate: float = 0.065
    max_positions: int = 10
    rebalance_hours: int = 24
    initial_capital: float = 10000.0
    data_dir: Path = DATA_DIR
    forecast_cache_dir: Path = FORECAST_CACHE_DIR
    pred_cache_dir: Path = PRED_CACHE_DIR
    lookback_days: int = 7
    temperature: float = 0.2
