"""Main trading bot for Bags.fm Solana trading.

Combines data collection, forecasting, and execution into
a cohesive trading system with risk management.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional

from .bags_api import BagsAPIClient, SolanaTransactionExecutor, SwapResult, execute_swap
from .config import (
    TradingConfig,
    TokenConfig,
    BagsConfig,
    DataConfig,
    ForecastConfig,
    SOL_MINT,
)
from .data_collector import DataCollector, create_collector
from .forecaster import TokenForecaster, TokenForecast, ForecastBatch, create_forecaster
from .simulator import compute_daily_high_low
from .utils import require_live_trading_enabled

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """Persistent state for the trading bot."""

    # Portfolio
    sol_balance: float = 0.0
    positions: Dict[str, float] = None  # token_mint -> quantity

    # Daily tracking
    day_start_value: float = 0.0
    daily_pnl: float = 0.0
    trades_today: int = 0

    # Historical
    total_trades: int = 0
    total_fees_sol: float = 0.0

    # Last action times
    last_price_check: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    def __post_init__(self):
        if self.positions is None:
            self.positions = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "sol_balance": self.sol_balance,
            "positions": self.positions,
            "day_start_value": self.day_start_value,
            "daily_pnl": self.daily_pnl,
            "trades_today": self.trades_today,
            "total_trades": self.total_trades,
            "total_fees_sol": self.total_fees_sol,
            "last_price_check": (
                self.last_price_check.isoformat() if self.last_price_check else None
            ),
            "last_trade_time": (
                self.last_trade_time.isoformat() if self.last_trade_time else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TradingState":
        """Create from dictionary."""
        state = cls(
            sol_balance=data.get("sol_balance", 0.0),
            positions=data.get("positions", {}),
            day_start_value=data.get("day_start_value", 0.0),
            daily_pnl=data.get("daily_pnl", 0.0),
            trades_today=data.get("trades_today", 0),
            total_trades=data.get("total_trades", 0),
            total_fees_sol=data.get("total_fees_sol", 0.0),
        )

        if data.get("last_price_check"):
            state.last_price_check = datetime.fromisoformat(data["last_price_check"])
        if data.get("last_trade_time"):
            state.last_trade_time = datetime.fromisoformat(data["last_trade_time"])

        return state


@dataclass
class TradeDecision:
    """A decision to execute a trade."""

    token: TokenConfig
    action: str  # "buy" or "sell"
    reason: str
    forecast: Optional[TokenForecast] = None

    # Sizing
    amount_sol: float = 0.0  # For buys: SOL to spend
    amount_tokens: float = 0.0  # For sells: tokens to sell

    # Risk metrics
    predicted_return: float = 0.0
    confidence: float = 0.0


class BagsTrader:
    """Main trading bot for Bags.fm.

    Coordinates:
    - Data collection every 10 minutes
    - OHLC aggregation
    - Chronos2 forecasting
    - Trade execution via Bags.fm
    - Risk management
    """

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self.state = TradingState()

        # Components (initialized lazily)
        self._collector: Optional[DataCollector] = None
        self._forecaster: Optional[TokenForecaster] = None
        self._api_client: Optional[BagsAPIClient] = None
        self._executor: Optional[SolanaTransactionExecutor] = None

        # Token registry
        self._tokens: Dict[str, TokenConfig] = {}
        self._daily_action_day: Optional[date] = None
        self._daily_actions_by_token: Dict[str, int] = {}

        # Load state if exists
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk if available."""
        if self.config.state_file.exists():
            try:
                with open(self.config.state_file) as f:
                    data = json.load(f)
                self.state = TradingState.from_dict(data)
                logger.info(f"Loaded state: {self.state.total_trades} total trades")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            with open(self.config.state_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def initialize(self) -> None:
        """Initialize all components."""
        # Create data collector
        max_rows = max(
            2000,
            self.config.data.context_bars * len(self.config.data.tracked_tokens) * 4,
        )
        self._collector = await create_collector(
            bags_config=self.config.bags,
            data_config=self.config.data,
            max_rows=max_rows,
            required_mints=[token.mint for token in self.config.data.tracked_tokens],
        )

        # Create forecaster
        self._forecaster = create_forecaster(
            data_collector=self._collector,
            config=self.config.forecast,
        )

        # Create API client and executor
        self._api_client = BagsAPIClient(self.config.bags)
        self._executor = SolanaTransactionExecutor(self.config.bags)

        # Register tokens
        for token in self.config.data.tracked_tokens:
            self._tokens[token.mint] = token

        # Update SOL balance
        await self._update_balances()

        logger.info(
            f"Initialized trader: {len(self._tokens)} tokens, "
            f"SOL balance: {self.state.sol_balance:.4f}"
        )

    async def _update_balances(self) -> None:
        """Update wallet balances from chain."""
        # Use public key from config (doesn't require private key)
        pubkey = self.config.bags.public_key
        if not pubkey:
            logger.debug("No public key configured, using cached balances")
            return

        try:
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TokenAccountOpts
            from solders.pubkey import Pubkey

            TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

            async with AsyncClient(self.config.bags.rpc_url) as rpc:
                # Get SOL balance
                response = await rpc.get_balance(Pubkey.from_string(pubkey))
                self.state.sol_balance = response.value / 1e9

                # Get token balances
                opts = TokenAccountOpts(program_id=TOKEN_PROGRAM_ID)
                response = await rpc.get_token_accounts_by_owner_json_parsed(
                    Pubkey.from_string(pubkey),
                    opts,
                )

                # Update positions
                self.state.positions = {}
                if response.value:
                    for account in response.value:
                        data = account.account.data
                        if hasattr(data, "parsed"):
                            info = data.parsed.get("info", {})
                            mint = info.get("mint")
                            amount = info.get("tokenAmount", {}).get("uiAmount", 0)
                            if mint and amount and amount > 0:
                                self.state.positions[mint] = float(amount)

        except ImportError:
            logger.warning("Solana SDK not available, using cached balances")
        except Exception as e:
            logger.error(f"Failed to update balances: {e}")

    async def collect_prices(self) -> None:
        """Collect current prices for all tokens."""
        if self._collector is None:
            return

        prices = await self._collector.collect_all_prices()
        self.state.last_price_check = datetime.utcnow()

        logger.debug(f"Collected {len(prices)} prices")

    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters when the UTC day changes."""
        now = datetime.utcnow()
        if self.state.last_price_check and self.state.last_price_check.date() == now.date():
            return

        self.state.day_start_value = self._get_portfolio_value()
        self.state.trades_today = 0
        self._daily_action_day = now.date()
        self._daily_actions_by_token = {}

    def generate_forecasts(self) -> ForecastBatch:
        """Generate forecasts for all tracked tokens."""
        if self._forecaster is None:
            return ForecastBatch(
                forecast_time=datetime.utcnow(),
                forecasts={},
            )

        return self._forecaster.forecast_all_tokens()

    def _record_daily_action(self, token_mint: str) -> None:
        """Track per-token daily actions for daily-range strategy."""
        now = datetime.utcnow().date()
        if self._daily_action_day != now:
            self._daily_action_day = now
            self._daily_actions_by_token = {}
        self._daily_actions_by_token[token_mint] = self._daily_actions_by_token.get(token_mint, 0) + 1

    def _daily_actions_used(self, token_mint: str) -> int:
        now = datetime.utcnow().date()
        if self._daily_action_day != now:
            return 0
        return self._daily_actions_by_token.get(token_mint, 0)

    def make_daily_range_decisions(self) -> List[TradeDecision]:
        """Generate trading decisions based on prior-day high/low levels."""
        decisions: List[TradeDecision] = []

        if self._collector is None:
            return []

        # Check daily limits
        if self.state.trades_today >= self.config.max_daily_trades:
            logger.info("Daily trade limit reached")
            return []

        # Check daily loss limit
        portfolio_value = self._get_portfolio_value()
        daily_return = (
            (portfolio_value - self.state.day_start_value) / self.state.day_start_value
            if self.state.day_start_value > 0
            else 0
        )
        if daily_return < -self.config.max_daily_loss_pct:
            logger.warning(f"Daily loss limit exceeded: {daily_return*100:.2f}%")
            return []

        # Build daily levels from collected bars
        bars: Dict[str, List] = {}
        for mint, token in self._tokens.items():
            mint_bars = self._collector.aggregator.get_bars(mint)
            if mint_bars:
                bars[mint] = mint_bars

        levels = compute_daily_high_low(bars)

        today = datetime.utcnow().date()
        level_day = today - timedelta(days=1) if self.config.daily_range_use_previous_day else today

        # Enforce total exposure cap
        remaining_capacity = self.config.max_position_sol
        missing_price = False
        for mint, quantity in self.state.positions.items():
            price = self._collector.get_latest_price(mint)
            if price is None:
                missing_price = True
                break
            remaining_capacity -= quantity * price
        if missing_price:
            logger.warning("Skipping new buys: missing latest price for an open position.")
            remaining_capacity = 0.0
        remaining_capacity = max(0.0, remaining_capacity)

        for mint, token in self._tokens.items():
            mint_levels = levels.get(mint)
            if not mint_levels or level_day not in mint_levels:
                continue

            high, low = mint_levels[level_day]
            if low <= 0:
                continue

            range_bps = (high - low) / low * 10000.0
            if range_bps < self.config.daily_range_min_bps:
                continue

            if self._daily_actions_used(mint) >= self.config.daily_range_max_actions_per_day:
                continue

            current_price = self._collector.get_latest_price(mint)
            if current_price is None:
                continue

            has_position = mint in self.state.positions and self.state.positions[mint] > 0

            if not has_position and current_price <= low:
                position_value = portfolio_value * self.config.position_size_pct
                position_value = min(position_value, self.state.sol_balance * 0.9)
                if self.config.max_position_sol is not None:
                    position_value = min(position_value, self.config.max_position_sol)
                position_value = min(position_value, remaining_capacity)

                if position_value >= 0.01:
                    decisions.append(
                        TradeDecision(
                            token=token,
                            action="buy",
                            reason=f"Daily low touch ({level_day})",
                            amount_sol=position_value,
                        )
                    )
                    remaining_capacity -= position_value

            elif has_position and current_price >= high:
                decisions.append(
                    TradeDecision(
                        token=token,
                        action="sell",
                        reason=f"Daily high touch ({level_day})",
                        amount_tokens=self.state.positions[mint],
                    )
                )

        # Sort to keep deterministic order
        decisions.sort(key=lambda d: d.token.symbol)

        max_new_trades = self.config.max_daily_trades - self.state.trades_today
        decisions = decisions[:max_new_trades]
        for decision in decisions:
            self._record_daily_action(decision.token.mint)
        return decisions

    def make_decisions(
        self,
        forecasts: ForecastBatch,
    ) -> List[TradeDecision]:
        """Generate trading decisions based on forecasts.

        Args:
            forecasts: Current forecasts

        Returns:
            List of TradeDecisions
        """
        decisions = []

        # Check daily limits
        if self.state.trades_today >= self.config.max_daily_trades:
            logger.info("Daily trade limit reached")
            return []

        # Check daily loss limit
        portfolio_value = self._get_portfolio_value()
        daily_return = (
            (portfolio_value - self.state.day_start_value) / self.state.day_start_value
            if self.state.day_start_value > 0
            else 0
        )

        if daily_return < -self.config.max_daily_loss_pct:
            logger.warning(f"Daily loss limit exceeded: {daily_return*100:.2f}%")
            return []

        # Enforce total exposure cap
        remaining_capacity = self.config.max_position_sol
        missing_price = False
        if self._collector:
            for mint, quantity in self.state.positions.items():
                price = self._collector.get_latest_price(mint)
                if price is None:
                    missing_price = True
                    break
                remaining_capacity -= quantity * price
        else:
            missing_price = True

        if missing_price:
            logger.warning(
                "Skipping new buys: missing latest price for an open position."
            )
            remaining_capacity = 0.0

        remaining_capacity = max(0.0, remaining_capacity)

        # Evaluate each token
        for mint, forecast in forecasts.forecasts.items():
            if mint not in self._tokens:
                continue

            token = self._tokens[mint]
            has_position = mint in self.state.positions and self.state.positions[mint] > 0

            # BUY signal
            if (
                not has_position
                and forecast.predicted_return >= self.config.min_predicted_return
                and forecast.signal_strength >= self.config.min_confidence
            ):
                # Calculate position size
                position_value = portfolio_value * self.config.position_size_pct
                position_value = min(position_value, self.state.sol_balance * 0.9)
                if self.config.max_position_sol is not None:
                    position_value = min(position_value, self.config.max_position_sol)
                position_value = min(position_value, remaining_capacity)

                if position_value >= 0.01:  # Min 0.01 SOL
                    decisions.append(
                        TradeDecision(
                            token=token,
                            action="buy",
                            reason=f"Predicted return {forecast.predicted_return*100:.2f}%",
                            forecast=forecast,
                            amount_sol=position_value,
                            predicted_return=forecast.predicted_return,
                            confidence=forecast.signal_strength,
                        )
                    )
                    remaining_capacity -= position_value

            # SELL signal
            elif (
                has_position
                and forecast.predicted_return < -self.config.min_predicted_return
            ):
                decisions.append(
                    TradeDecision(
                        token=token,
                        action="sell",
                        reason=f"Predicted return {forecast.predicted_return*100:.2f}%",
                        forecast=forecast,
                        amount_tokens=self.state.positions[mint],
                        predicted_return=forecast.predicted_return,
                        confidence=forecast.signal_strength,
                    )
                )

        # Sort by signal strength
        decisions.sort(key=lambda d: d.confidence, reverse=True)

        # Limit number of trades
        max_new_trades = self.config.max_daily_trades - self.state.trades_today
        return decisions[:max_new_trades]

    async def execute_decision(
        self,
        decision: TradeDecision,
    ) -> Optional[SwapResult]:
        """Execute a trading decision.

        Args:
            decision: TradeDecision to execute

        Returns:
            SwapResult or None if execution failed/skipped
        """
        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] Would {decision.action} {decision.token.symbol}: "
                f"{decision.reason}"
            )
            return None

        try:
            if decision.action == "buy":
                # Convert SOL amount to lamports
                amount_lamports = int(decision.amount_sol * 1e9)

                result = await execute_swap(
                    config=self.config.bags,
                    input_mint=SOL_MINT,
                    output_mint=decision.token.mint,
                    amount=amount_lamports,
                    slippage_bps=self.config.slippage_bps,
                    dry_run=False,
                )

            else:  # sell
                # Convert token amount to smallest units
                amount = int(decision.amount_tokens * (10 ** decision.token.decimals))

                result = await execute_swap(
                    config=self.config.bags,
                    input_mint=decision.token.mint,
                    output_mint=SOL_MINT,
                    amount=amount,
                    slippage_bps=self.config.slippage_bps,
                    dry_run=False,
                )

            if result.success:
                self.state.total_trades += 1
                self.state.trades_today += 1
                self.state.last_trade_time = datetime.utcnow()

                if result.fee_lamports:
                    self.state.total_fees_sol += result.fee_lamports / 1e9

                self._save_state()

                logger.info(
                    f"Executed {decision.action} {decision.token.symbol}: "
                    f"sig={result.signature}"
                )

            else:
                logger.error(
                    f"Failed to execute {decision.action} {decision.token.symbol}: "
                    f"{result.error}"
                )

            return result

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return SwapResult(success=False, error=str(e))

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value in SOL."""
        value = self.state.sol_balance

        for mint, quantity in self.state.positions.items():
            price = self._collector.get_latest_price(mint) if self._collector else None
            if price:
                value += quantity * price

        return value

    async def run_cycle(self) -> None:
        """Run a single trading cycle.

        1. Collect prices
        2. Generate forecasts
        3. Make decisions
        4. Execute trades
        """
        logger.info("Starting trading cycle")
        self._reset_daily_counters_if_needed()

        # Collect prices
        await self.collect_prices()

        # Update balances
        await self._update_balances()

        if self.config.strategy == "daily-range":
            decisions = self.make_daily_range_decisions()
            logger.info(f"Daily-range decisions: {len(decisions)}")
        else:
            # Generate forecasts
            forecasts = self.generate_forecasts()
            logger.info(f"Generated {len(forecasts.forecasts)} forecasts")

            # Make decisions
            decisions = self.make_decisions(forecasts)

        if decisions:
            logger.info(f"Made {len(decisions)} trade decisions")

            for decision in decisions:
                logger.info(
                    f"Decision: {decision.action} {decision.token.symbol} "
                    f"({decision.reason})"
                )
                await self.execute_decision(decision)
        else:
            logger.info("No trade decisions this cycle")

        # Save data
        if self._collector:
            self._collector.save_to_disk()

        self._save_state()

    async def run(
        self,
        duration_hours: Optional[float] = None,
    ) -> None:
        """Run the trading bot continuously.

        Args:
            duration_hours: How long to run (None for indefinite)
        """
        await self.initialize()

        interval = self.config.check_interval_minutes * 60
        end_time = None

        if duration_hours:
            end_time = datetime.utcnow() + timedelta(hours=duration_hours)

        logger.info(
            f"Starting trading bot (check every {self.config.check_interval_minutes} min, "
            f"dry_run={self.config.dry_run})"
        )

        # Reset daily counters at start
        self.state.day_start_value = self._get_portfolio_value()
        self.state.trades_today = 0

        while True:
            if end_time and datetime.utcnow() >= end_time:
                logger.info("Duration reached, stopping")
                break

            try:
                await self.run_cycle()

            except Exception as e:
                logger.error(f"Cycle error: {e}")

            # Wait for next cycle
            await asyncio.sleep(interval)

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._save_state()

        if self._collector:
            self._collector.save_to_disk()

        if self._forecaster:
            self._forecaster.unload()

        logger.info("Trader shutdown complete")


async def run_trader(
    config: Optional[TradingConfig] = None,
    duration_hours: Optional[float] = None,
) -> None:
    """Run the trading bot.

    Args:
        config: Trading configuration (uses defaults if None)
        duration_hours: How long to run (None for indefinite)
    """
    if config is None:
        from .config import load_config_from_env
        config = load_config_from_env()

    trader = BagsTrader(config)

    try:
        await trader.run(duration_hours=duration_hours)
    finally:
        trader.shutdown()


def main():
    """Entry point for running the trader."""
    import argparse

    parser = argparse.ArgumentParser(description="Bags.fm Trading Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (no actual trades)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (execute actual trades)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to run in hours (None for indefinite)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Check interval in minutes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    from .config import is_bagsfm_trading_disabled
    if args.live and is_bagsfm_trading_disabled():
        raise SystemExit(
            "Bags.fm live trading is disabled (BAGSFM_TRADING_DISABLED=1). "
            "Refusing to start in --live mode."
        )

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.live:
        require_live_trading_enabled()

    # Create config
    from .config import load_config_from_env
    config = load_config_from_env()
    config.dry_run = not args.live
    config.check_interval_minutes = args.interval

    # Run
    asyncio.run(run_trader(config, duration_hours=args.duration))


if __name__ == "__main__":
    main()
