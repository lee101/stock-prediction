"""Two-stage portfolio optimization with Chronos2 forecasts.

Stage 1: Portfolio allocation across symbols (which to trade, how much 0-1)
Stage 2: Per-symbol specific buy/sell prices with short/long support

Features:
- Async/parallel API calls with retries
- Crypto: LONG ONLY (no shorting)
- Stocks: Can short OR leverage up to 2x (6.5% annual fee)
- Symbol debiasing (SYMBOL_A instead of BTCUSD)
- Basis points (bps) format for consistency
"""

from .portfolio_allocator import (
    allocate_portfolio,
    async_allocate_portfolio,
    PortfolioAllocation,
    SymbolAllocation,
    is_crypto,
)
from .price_predictor import (
    predict_prices,
    async_predict_prices,
    PricePrediction,
)
from .backtest import run_backtest, run_backtest_async

__all__ = [
    "allocate_portfolio",
    "async_allocate_portfolio",
    "PortfolioAllocation",
    "SymbolAllocation",
    "is_crypto",
    "predict_prices",
    "async_predict_prices",
    "PricePrediction",
    "run_backtest",
    "run_backtest_async",
]
