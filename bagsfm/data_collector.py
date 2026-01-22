"""Data collection and OHLC aggregation for Solana tokens.

Collects price data at regular intervals and aggregates into OHLC bars
for use with forecasting models.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .bags_api import BagsAPIClient
from .config import DataConfig, TokenConfig, SOL_MINT, BagsConfig

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """A single price observation."""

    timestamp: datetime
    token_mint: str
    token_symbol: str
    price_sol: float  # Price in SOL
    price_usd: Optional[float] = None  # Price in USD (if available)
    quote_amount: int = 0  # Amount used for quote
    out_amount: int = 0  # Output amount from quote


@dataclass
class OHLCBar:
    """OHLC bar for a time interval."""

    timestamp: datetime  # Bar open time
    token_mint: str
    token_symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0  # Sum of trade amounts (if tracked)
    num_ticks: int = 0  # Number of price points in bar

    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range_pct(self) -> float:
        """Bar range as percentage of open."""
        if self.open <= 0:
            return 0.0
        return (self.high - self.low) / self.open * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "token_mint": self.token_mint,
            "token_symbol": self.token_symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "num_ticks": self.num_ticks,
        }


class OHLCAggregator:
    """Aggregates price points into OHLC bars."""

    def __init__(self, interval_minutes: int = 10) -> None:
        self.interval_minutes = interval_minutes
        self._current_bars: Dict[str, OHLCBar] = {}  # token_mint -> current bar
        self._completed_bars: Dict[str, List[OHLCBar]] = {}  # token_mint -> bars

    def _get_bar_start(self, timestamp: datetime) -> datetime:
        """Get the start time of the bar containing timestamp."""
        # Round down to interval boundary
        minutes = timestamp.minute - (timestamp.minute % self.interval_minutes)
        return timestamp.replace(minute=minutes, second=0, microsecond=0)

    def add_price(self, price: PricePoint) -> Optional[OHLCBar]:
        """Add a price point and return completed bar if any.

        Args:
            price: Price point to add

        Returns:
            Completed OHLCBar if a bar was finished, None otherwise
        """
        bar_start = self._get_bar_start(price.timestamp)
        mint = price.token_mint

        # Initialize storage for this token
        if mint not in self._completed_bars:
            self._completed_bars[mint] = []

        current = self._current_bars.get(mint)

        # Check if we need to start a new bar
        if current is None or current.timestamp != bar_start:
            completed = None

            # Save completed bar
            if current is not None and current.num_ticks > 0:
                self._completed_bars[mint].append(current)
                completed = current

            # Start new bar
            self._current_bars[mint] = OHLCBar(
                timestamp=bar_start,
                token_mint=mint,
                token_symbol=price.token_symbol,
                open=price.price_sol,
                high=price.price_sol,
                low=price.price_sol,
                close=price.price_sol,
                num_ticks=1,
            )

            return completed

        # Update current bar
        current.high = max(current.high, price.price_sol)
        current.low = min(current.low, price.price_sol)
        current.close = price.price_sol
        current.num_ticks += 1

        return None

    def get_bars(self, token_mint: str, n: Optional[int] = None) -> List[OHLCBar]:
        """Get completed bars for a token.

        Args:
            token_mint: Token mint address
            n: Number of most recent bars (None for all)

        Returns:
            List of OHLCBars
        """
        bars = self._completed_bars.get(token_mint, [])
        if n is not None:
            return bars[-n:]
        return bars

    def get_all_bars(self) -> Dict[str, List[OHLCBar]]:
        """Get all completed bars for all tokens."""
        return self._completed_bars.copy()

    def get_current_bar(self, token_mint: str) -> Optional[OHLCBar]:
        """Get the current (incomplete) bar for a token."""
        return self._current_bars.get(token_mint)

    def to_dataframe(self, token_mint: str) -> pd.DataFrame:
        """Convert bars to pandas DataFrame.

        Args:
            token_mint: Token mint address

        Returns:
            DataFrame with OHLC data
        """
        bars = self.get_bars(token_mint)
        if not bars:
            return pd.DataFrame(columns=[
                "timestamp", "open", "high", "low", "close", "volume", "num_ticks"
            ])

        data = [b.to_dict() for b in bars]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


class DataCollector:
    """Collects and stores price data for Solana tokens.

    Uses Bags.fm API to get prices via swap quotes, then aggregates
    into OHLC bars for forecasting.
    """

    def __init__(
        self,
        bags_config: BagsConfig,
        data_config: DataConfig,
    ) -> None:
        self.bags_config = bags_config
        self.data_config = data_config
        self.client = BagsAPIClient(bags_config)
        self.aggregator = OHLCAggregator(data_config.ohlc_interval_minutes)

        # Price history (in memory)
        self._price_history: List[PricePoint] = []

        # Ensure data directory exists
        data_config.data_path.mkdir(parents=True, exist_ok=True)

    async def collect_price(self, token: TokenConfig) -> Optional[PricePoint]:
        """Collect current price for a token.

        Gets price by quoting a swap from SOL to the token, then inverting
        to get the token's price in SOL.

        Note: Bags API works better for SOL->token direction than token->SOL.

        Args:
            token: Token configuration

        Returns:
            PricePoint or None if collection failed
        """
        if token.mint == self.data_config.quote_token.mint:
            # Token is the quote token itself (e.g., SOL)
            return PricePoint(
                timestamp=datetime.utcnow(),
                token_mint=token.mint,
                token_symbol=token.symbol,
                price_sol=1.0,
            )

        try:
            # Quote SOL -> token and invert to get price
            # Use 0.01 SOL (10M lamports) as reference amount
            sol_amount = 10_000_000  # 0.01 SOL in lamports

            quote = await self.client.get_quote(
                input_mint=self.data_config.quote_token.mint,  # SOL
                output_mint=token.mint,  # token
                amount=sol_amount,
                slippage_bps=100,
            )

            # We get X tokens for 0.01 SOL
            # So price of 1 token = 0.01 SOL / X tokens
            tokens_received = quote.out_amount / (10 ** token.decimals)
            sol_spent = sol_amount / 1e9  # Convert to SOL

            if tokens_received <= 0:
                logger.warning(f"No tokens received for {token.symbol}")
                return None

            price_sol = sol_spent / tokens_received

            price = PricePoint(
                timestamp=datetime.utcnow(),
                token_mint=token.mint,
                token_symbol=token.symbol,
                price_sol=price_sol,
                quote_amount=sol_amount,
                out_amount=quote.out_amount,
            )

            logger.debug(f"Collected price for {token.symbol}: {price_sol:.10f} SOL")
            return price

        except Exception as e:
            logger.error(f"Failed to collect price for {token.symbol}: {e}")
            return None

    async def collect_all_prices(self) -> List[PricePoint]:
        """Collect prices for all tracked tokens.

        Returns:
            List of collected PricePoints
        """
        prices = []

        for token in self.data_config.tracked_tokens:
            price = await self.collect_price(token)
            if price:
                prices.append(price)

                # Add to history and aggregator
                self._price_history.append(price)
                completed_bar = self.aggregator.add_price(price)

                if completed_bar:
                    logger.info(
                        f"Completed OHLC bar for {token.symbol}: "
                        f"O={completed_bar.open:.8f} H={completed_bar.high:.8f} "
                        f"L={completed_bar.low:.8f} C={completed_bar.close:.8f}"
                    )

        return prices

    async def run_collection_loop(
        self,
        duration_hours: Optional[float] = None,
        callback: Optional[callable] = None,
    ) -> None:
        """Run continuous price collection.

        Args:
            duration_hours: How long to run (None for indefinite)
            callback: Optional callback(prices) after each collection
        """
        interval = self.data_config.collection_interval_minutes * 60
        end_time = None

        if duration_hours:
            end_time = datetime.utcnow() + timedelta(hours=duration_hours)

        logger.info(
            f"Starting price collection every {self.data_config.collection_interval_minutes} minutes"
        )

        while True:
            if end_time and datetime.utcnow() >= end_time:
                logger.info("Collection duration reached, stopping")
                break

            try:
                prices = await self.collect_all_prices()
                logger.info(f"Collected {len(prices)} prices")

                if callback:
                    callback(prices)

                # Save periodically
                self.save_to_disk()

            except Exception as e:
                logger.error(f"Collection error: {e}")

            # Wait for next interval
            await asyncio.sleep(interval)

    def save_to_disk(self) -> None:
        """Save collected data to disk."""
        # Save price history
        if self._price_history:
            price_path = self.data_config.price_history_path
            write_header = not price_path.exists()

            with open(price_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "token_mint", "token_symbol",
                        "price_sol", "price_usd", "quote_amount", "out_amount"
                    ])

                for p in self._price_history:
                    writer.writerow([
                        p.timestamp.isoformat(),
                        p.token_mint,
                        p.token_symbol,
                        p.price_sol,
                        p.price_usd or "",
                        p.quote_amount,
                        p.out_amount,
                    ])

            # Clear in-memory history after saving
            self._price_history.clear()

        # Save OHLC data
        all_bars = self.aggregator.get_all_bars()
        if all_bars:
            ohlc_path = self.data_config.ohlc_path
            write_header = not ohlc_path.exists()

            with open(ohlc_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "token_mint", "token_symbol",
                        "open", "high", "low", "close", "volume", "num_ticks"
                    ])

                for mint, bars in all_bars.items():
                    for bar in bars:
                        writer.writerow([
                            bar.timestamp.isoformat(),
                            bar.token_mint,
                            bar.token_symbol,
                            bar.open,
                            bar.high,
                            bar.low,
                            bar.close,
                            bar.volume,
                            bar.num_ticks,
                        ])

    def load_from_disk(self) -> Tuple[int, int]:
        """Load existing data from disk.

        Returns:
            Tuple of (num_prices_loaded, num_bars_loaded)
        """
        num_prices = 0
        num_bars = 0

        # Load OHLC data
        ohlc_path = self.data_config.ohlc_path
        if ohlc_path.exists():
            try:
                df = pd.read_csv(ohlc_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                for _, row in df.iterrows():
                    bar = OHLCBar(
                        timestamp=row["timestamp"].to_pydatetime(),
                        token_mint=row["token_mint"],
                        token_symbol=row["token_symbol"],
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row.get("volume", 0.0),
                        num_ticks=row.get("num_ticks", 1),
                    )

                    mint = bar.token_mint
                    if mint not in self.aggregator._completed_bars:
                        self.aggregator._completed_bars[mint] = []
                    self.aggregator._completed_bars[mint].append(bar)
                    num_bars += 1

                logger.info(f"Loaded {num_bars} OHLC bars from disk")

            except Exception as e:
                logger.error(f"Failed to load OHLC data: {e}")

        return num_prices, num_bars

    def get_ohlc_dataframe(self, token_mint: str) -> pd.DataFrame:
        """Get OHLC data as DataFrame for a token.

        Args:
            token_mint: Token mint address

        Returns:
            DataFrame with OHLC columns
        """
        return self.aggregator.to_dataframe(token_mint)

    def get_latest_price(self, token_mint: str) -> Optional[float]:
        """Get the latest price for a token.

        Args:
            token_mint: Token mint address

        Returns:
            Latest price in SOL or None
        """
        current_bar = self.aggregator.get_current_bar(token_mint)
        if current_bar:
            return current_bar.close

        bars = self.aggregator.get_bars(token_mint, n=1)
        if bars:
            return bars[-1].close

        return None

    def get_context_bars(
        self,
        token_mint: str,
        n: Optional[int] = None,
    ) -> List[OHLCBar]:
        """Get context bars for forecasting.

        Args:
            token_mint: Token mint address
            n: Number of bars (defaults to config.context_bars)

        Returns:
            List of OHLCBars
        """
        if n is None:
            n = self.data_config.context_bars

        return self.aggregator.get_bars(token_mint, n=n)


async def create_collector(
    bags_config: Optional[BagsConfig] = None,
    data_config: Optional[DataConfig] = None,
) -> DataCollector:
    """Factory function to create a data collector.

    Args:
        bags_config: Bags API configuration (uses defaults if None)
        data_config: Data configuration (uses defaults if None)

    Returns:
        Initialized DataCollector
    """
    if bags_config is None:
        bags_config = BagsConfig()

    if data_config is None:
        data_config = DataConfig()

    collector = DataCollector(bags_config, data_config)
    collector.load_from_disk()

    return collector
