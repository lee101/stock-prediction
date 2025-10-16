from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest, StockLatestQuoteRequest
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

from src.fixtures import crypto_symbols
from src.stock_utils import remap_symbols

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QuoteResult:
    symbol: str
    bid: Optional[float]
    ask: Optional[float]

    @property
    def spread_ratio(self) -> float:
        if self.bid and self.ask and self.bid > 0.0:
            return self.ask / self.bid
        return 1.0


class SpreadFetcher:
    """Fetch bid/ask spreads for stocks and crypto via Alpaca."""

    def __init__(self) -> None:
        self.stock_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
        self.crypto_client = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

    def fetch(self, symbol: str) -> QuoteResult:
        symbol = symbol.upper()
        if symbol in crypto_symbols or symbol.endswith("USD"):
            return self._fetch_crypto(symbol)
        return self._fetch_stock(symbol)

    def _fetch_stock(self, symbol: str) -> QuoteResult:
        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        response = self.stock_client.get_stock_latest_quote(request)
        if symbol not in response:
            logger.error("Stock symbol %s missing from Alpaca response keys: %s", symbol, list(response.keys()))
            raise KeyError(f"Symbol {symbol} not found in Alpaca response")
        quote = response[symbol]
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)
        return QuoteResult(symbol=symbol, bid=float(bid) if bid else None, ask=float(ask) if ask else None)

    def _fetch_crypto(self, symbol: str) -> QuoteResult:
        remapped = remap_symbols(symbol)
        request = CryptoLatestQuoteRequest(symbol_or_symbols=[remapped])
        response = self.crypto_client.get_crypto_latest_quote(request)
        if remapped not in response:
            logger.error("Crypto symbol %s missing from Alpaca response keys: %s", remapped, list(response.keys()))
            raise KeyError(f"Symbol {remapped} not found in Alpaca response")
        quote = response[remapped]
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)
        return QuoteResult(symbol=symbol, bid=float(bid) if bid else None, ask=float(ask) if ask else None)


__all__ = ["SpreadFetcher", "QuoteResult"]
