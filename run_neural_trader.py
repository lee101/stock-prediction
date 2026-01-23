#!/usr/bin/env python3
"""
Run the Bags.fm Neural Trading Bot for CODEX.

Uses the trained neural model for buy/sell signals instead of Chronos forecaster.

Usage:
    # Dry run mode (default)
    python run_neural_trader.py --dry-run

    # Live trading mode
    python run_neural_trader.py --live

    # Custom position size
    python run_neural_trader.py --live --max-position 0.5

Environment variables:
    BAGS_API_KEY - Bags.fm API key
    SOLANA_PRIVATE_KEY - Base58 encoded private key (required for live trading)
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from bagsfm import BagsAPIClient, BagsConfig, DataCollector, DataConfig, TokenConfig
from bagsfm.config import SOL_MINT
from bagsfm.bags_api import SolanaTransactionExecutor
from bagsneural.dataset import load_ohlc_dataframe, build_window_features, FeatureNormalizer
from bagsneural.model import BagsNeuralModel
from pnl_tracker import PnLTracker

# Data file path
OHLC_DATA_PATH = Path("bagstraining/ohlc_data.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("neural_trader")
logging.getLogger("httpx").setLevel(logging.WARNING)

# CODEX token config
CODEX_TOKEN = TokenConfig(
    symbol="CODEX",
    mint="HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS",
    decimals=9,
    name="CODEX",
    min_trade_amount=1.0,
)

# Default checkpoint path
DEFAULT_CHECKPOINT = Path("bagsneural/checkpoints/bagsneural_HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS_best.pt")


class NeuralTrader:
    """Neural model-based trader for CODEX."""

    def __init__(
        self,
        checkpoint_path: Path,
        bags_config: BagsConfig,
        buy_threshold: float = 0.46,
        sell_threshold: float = 0.42,
        max_position_sol: float = 20.0,  # Total position limit
        max_trade_sol: float = 8.0,  # Per-trade limit (liquidity)
        min_trade_sol: float = 0.01,
        dry_run: bool = True,
        device: str = "cuda",
    ):
        self.bags_config = bags_config
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_position_sol = max_position_sol
        self.max_trade_sol = max_trade_sol
        self.min_trade_sol = min_trade_sol
        self.dry_run = dry_run
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.config = ckpt["config"]
        self.normalizer = FeatureNormalizer.from_dict(ckpt["normalizer"])
        self.context_bars = self.config.get("context", 16)

        input_dim = self.context_bars * 3
        hidden_dims = self.config.get("hidden", [32, 16])
        self.model = BagsNeuralModel(input_dim=input_dim, hidden_dims=hidden_dims)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded: context={self.context_bars}, hidden={hidden_dims}")

        # API clients
        self.api_client = BagsAPIClient(bags_config)
        self.executor = SolanaTransactionExecutor(bags_config) if not dry_run else None

        # State
        self.holding = False
        self.position_tokens = 0.0
        self.entry_price = 0.0
        self.sol_balance = 0.0

        # Price history for features
        self.price_history = []

        # PnL tracking
        self.pnl_tracker = PnLTracker(
            token=CODEX_TOKEN.symbol,
            log_dir="logs/pnl",
            snapshot_interval=60,  # snapshot every minute
        )
        self.last_price = 0.0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5

        # OHLC data settings
        self.ohlc_interval_minutes = 10  # matches collect_bags_data.py
        self.max_gap_minutes = 30  # max allowed gap between bars
        self.data_loaded_from_disk = False

    def load_ohlc_from_disk(self) -> bool:
        """Load OHLC data from disk and check for gaps.

        Returns:
            True if data is good quality, False if too many gaps.
        """
        if not OHLC_DATA_PATH.exists():
            logger.warning(f"OHLC data file not found: {OHLC_DATA_PATH}")
            return False

        try:
            df = pd.read_csv(OHLC_DATA_PATH)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter for CODEX token and deduplicate
            df = df[df["token_symbol"] == CODEX_TOKEN.symbol].copy()
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp")

            if len(df) < self.context_bars:
                logger.warning(f"Not enough bars: {len(df)} < {self.context_bars}")
                return False

            # Get recent bars
            recent = df.tail(self.context_bars + 10)

            # Check for gaps
            gaps = self._check_gaps(recent)
            if gaps > 2:  # Allow up to 2 gaps
                logger.warning(f"Too many gaps in data: {gaps} gaps (max 2)")
                return False

            # Load into price history
            self.price_history = []
            for _, row in recent.iterrows():
                self.price_history.append({
                    "timestamp": row["timestamp"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                })

            logger.info(f"Loaded {len(self.price_history)} bars from disk (gaps: {gaps})")
            self.data_loaded_from_disk = True
            return True

        except Exception as e:
            logger.error(f"Failed to load OHLC from disk: {e}")
            return False

    def _check_gaps(self, df: pd.DataFrame) -> int:
        """Check for gaps in OHLC data.

        Args:
            df: DataFrame with timestamp column

        Returns:
            Number of gaps found
        """
        gaps = 0
        timestamps = df["timestamp"].values
        expected_delta = pd.Timedelta(minutes=self.ohlc_interval_minutes)
        max_delta = pd.Timedelta(minutes=self.max_gap_minutes)

        for i in range(1, len(timestamps)):
            delta = pd.Timestamp(timestamps[i]) - pd.Timestamp(timestamps[i - 1])
            if delta > max_delta:
                gaps += 1
                logger.debug(f"Gap detected: {timestamps[i-1]} -> {timestamps[i]} ({delta})")

        return gaps

    def get_latest_bar_time(self) -> datetime:
        """Get timestamp of latest bar in disk data."""
        if not OHLC_DATA_PATH.exists():
            return None

        try:
            df = pd.read_csv(OHLC_DATA_PATH)
            df = df[df["token_symbol"] == CODEX_TOKEN.symbol]
            if df.empty:
                return None
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df["timestamp"].max()
        except Exception:
            return None

    async def update_balances(self) -> None:
        """Update SOL and token balances."""
        pubkey = self.bags_config.public_key
        if not pubkey:
            logger.warning("No public key configured")
            return

        try:
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TokenAccountOpts
            from solders.pubkey import Pubkey

            TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

            async with AsyncClient(self.bags_config.rpc_url) as rpc:
                # SOL balance
                response = await rpc.get_balance(Pubkey.from_string(pubkey))
                self.sol_balance = response.value / 1e9

                # Token balances
                opts = TokenAccountOpts(program_id=TOKEN_PROGRAM_ID)
                response = await rpc.get_token_accounts_by_owner_json_parsed(
                    Pubkey.from_string(pubkey), opts
                )

                self.position_tokens = 0.0
                if response.value:
                    for account in response.value:
                        data = account.account.data
                        if hasattr(data, "parsed"):
                            info = data.parsed.get("info", {})
                            mint = info.get("mint")
                            if mint == CODEX_TOKEN.mint:
                                amount = info.get("tokenAmount", {}).get("uiAmount", 0)
                                if amount:
                                    self.position_tokens = float(amount)
                                    self.holding = self.position_tokens > 0

            logger.info(f"Balances: {self.sol_balance:.4f} SOL, {self.position_tokens:.2f} CODEX")

        except Exception as e:
            logger.error(f"Failed to update balances: {e}")

    async def get_current_price(self) -> float:
        """Get current CODEX price in SOL."""
        try:
            quote = await self.api_client.get_quote(
                input_mint=SOL_MINT,
                output_mint=CODEX_TOKEN.mint,
                amount=10_000_000,  # 0.01 SOL
                slippage_mode="auto",
            )
            tokens_received = quote.out_amount / (10 ** CODEX_TOKEN.decimals)
            price_sol = 0.01 / tokens_received if tokens_received > 0 else 0
            return price_sol
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return 0.0

    def predict(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> tuple:
        """Get model prediction."""
        features = build_window_features(opens, highs, lows, closes)
        normalized = self.normalizer.transform(features)[None, :]
        x = torch.tensor(normalized, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            signal_logit, size_logit = self.model(x)
            prob = torch.sigmoid(signal_logit).item()
            size = torch.sigmoid(size_logit).item()

        return prob, size

    async def execute_buy(self, sol_amount: float, price: float, signal_strength: float, position_size: float) -> bool:
        """Execute buy order."""
        if self.dry_run:
            logger.info(f"[DRY RUN] BUY {sol_amount:.4f} SOL worth of CODEX at {price:.10f}")
            self.holding = True
            self.entry_price = price
            self.position_tokens = sol_amount / price
            # Log to PnL tracker
            self.pnl_tracker.log_trade(
                action="buy",
                amount=self.position_tokens,
                price=price,
                sol_amount=sol_amount,
                signal_strength=signal_strength,
                position_size=position_size,
            )
            return True

        try:
            # Get quote
            lamports = int(sol_amount * 1e9)
            quote = await self.api_client.get_quote(
                input_mint=SOL_MINT,
                output_mint=CODEX_TOKEN.mint,
                amount=lamports,
                slippage_mode="auto",
            )

            # Build and execute swap
            swap_tx = await self.api_client.build_swap_transaction(
                quote=quote,
                user_public_key=self.bags_config.public_key,
            )

            result = await self.executor.sign_and_send(swap_tx)

            if result.success:
                logger.info(f"BUY executed: {result.signature}")
                self.holding = True
                self.entry_price = price
                tokens_received = quote.out_amount / (10 ** CODEX_TOKEN.decimals)
                # Log to PnL tracker
                self.pnl_tracker.log_trade(
                    action="buy",
                    amount=tokens_received,
                    price=price,
                    sol_amount=sol_amount,
                    signal_strength=signal_strength,
                    position_size=position_size,
                )
                self.consecutive_failures = 0
                return True
            else:
                logger.error(f"BUY failed: {result.error}")
                self.consecutive_failures += 1
                return False

        except Exception as e:
            logger.error(f"BUY execution error: {e}")
            self.consecutive_failures += 1
            return False

    async def execute_sell(self, price: float, signal_strength: float, position_size: float, sell_tokens: float = None) -> bool:
        """Execute sell order. If sell_tokens is None, sells entire position."""
        tokens_to_sell = sell_tokens if sell_tokens is not None else self.position_tokens
        tokens_to_sell = min(tokens_to_sell, self.position_tokens)  # Can't sell more than we have

        if self.dry_run:
            pnl = (price - self.entry_price) / self.entry_price * 100 if self.entry_price > 0 else 0
            sol_received = tokens_to_sell * price
            logger.info(f"[DRY RUN] SELL {tokens_to_sell:.0f} CODEX at {price:.10f} (PnL: {pnl:+.2f}%)")
            # Log to PnL tracker
            self.pnl_tracker.log_trade(
                action="sell",
                amount=tokens_to_sell,
                price=price,
                sol_amount=sol_received,
                signal_strength=signal_strength,
                position_size=position_size,
            )
            self.position_tokens -= tokens_to_sell
            self.holding = self.position_tokens > 0
            if not self.holding:
                self.entry_price = 0.0
            return True

        try:
            # Get token amount
            token_amount = int(tokens_to_sell * (10 ** CODEX_TOKEN.decimals))

            quote = await self.api_client.get_quote(
                input_mint=CODEX_TOKEN.mint,
                output_mint=SOL_MINT,
                amount=token_amount,
                slippage_mode="auto",
            )

            swap_tx = await self.api_client.build_swap_transaction(
                quote=quote,
                user_public_key=self.bags_config.public_key,
            )

            result = await self.executor.sign_and_send(swap_tx)

            if result.success:
                logger.info(f"SELL executed: {result.signature}")
                sol_received = quote.out_amount / 1e9
                # Log to PnL tracker
                self.pnl_tracker.log_trade(
                    action="sell",
                    amount=tokens_to_sell,
                    price=price,
                    sol_amount=sol_received,
                    signal_strength=signal_strength,
                    position_size=position_size,
                )
                self.position_tokens -= tokens_to_sell
                self.holding = self.position_tokens > 0
                if not self.holding:
                    self.entry_price = 0.0
                self.consecutive_failures = 0
                return True
            else:
                logger.error(f"SELL failed: {result.error}")
                self.consecutive_failures += 1
                return False

        except Exception as e:
            logger.error(f"SELL execution error: {e}")
            self.consecutive_failures += 1
            return False

    async def trading_cycle(self) -> None:
        """Run one trading cycle."""
        # Check for too many consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures ({self.consecutive_failures}), pausing trading")
            self.consecutive_failures = 0  # Reset after warning
            return

        # Update balances
        await self.update_balances()

        # Get current price
        current_price = await self.get_current_price()
        if current_price <= 0:
            logger.warning("Could not get price, skipping cycle")
            self.consecutive_failures += 1
            return

        # Edge case: Check for price anomalies (sudden 50%+ moves)
        if self.last_price > 0:
            price_change = abs(current_price - self.last_price) / self.last_price
            if price_change > 0.5:
                logger.warning(f"Price anomaly detected: {price_change*100:.1f}% change, skipping cycle")
                return
        self.last_price = current_price

        # Log PnL snapshot
        self.pnl_tracker.log_snapshot(
            sol_balance=self.sol_balance,
            token_balance=self.position_tokens,
            token_price=current_price,
        )

        # Load OHLC from disk (refreshes each cycle)
        if not self.load_ohlc_from_disk():
            logger.warning("Data quality issue, skipping trading cycle")
            return

        # Check data freshness - last bar should be within 20 minutes
        latest_time = self.get_latest_bar_time()
        if latest_time:
            from datetime import timezone
            now_utc = datetime.now(timezone.utc)
            latest_utc = latest_time.replace(tzinfo=timezone.utc) if latest_time.tzinfo is None else latest_time
            age_minutes = (now_utc - latest_utc).total_seconds() / 60
            if age_minutes > 30:
                logger.warning(f"Data is stale: last bar {age_minutes:.0f} min ago (max 30)")
                return
            logger.info(f"Data freshness: last bar {age_minutes:.1f} min ago")

        # Need enough history
        if len(self.price_history) < self.context_bars:
            logger.info(f"Not enough bars: {len(self.price_history)}/{self.context_bars}")
            return

        # Build arrays for prediction
        recent = self.price_history[-self.context_bars:]
        opens = np.array([p["open"] for p in recent], dtype=np.float32)
        highs = np.array([p["high"] for p in recent], dtype=np.float32)
        lows = np.array([p["low"] for p in recent], dtype=np.float32)
        closes = np.array([p["close"] for p in recent], dtype=np.float32)

        # Get prediction
        prob, size = self.predict(opens, highs, lows, closes)
        logger.info(f"Price: {current_price:.10f} SOL | Signal: {prob:.4f} | Size: {size:.4f} | Holding: {self.holding}")

        # Log current PnL stats periodically
        stats = self.pnl_tracker.get_stats()
        if stats.get("status") != "no data":
            logger.info(f"PnL: {stats['total_return_pct']:+.2f}% | Drawdown: {stats['drawdown_pct']:.2f}% | Trades: {stats['trades']}")

        # Calculate current position value in SOL
        position_value_sol = self.position_tokens * current_price
        room_to_buy = self.max_position_sol - position_value_sol

        # Trading logic - allow gradual position building/exiting
        if prob >= self.buy_threshold and room_to_buy > self.min_trade_sol:
            # Scale trade size by signal strength and available room
            desired_trade = size * self.max_trade_sol
            trade_sol = min(desired_trade, room_to_buy, self.sol_balance * 0.95, self.max_trade_sol)
            if trade_sol >= self.min_trade_sol:
                logger.info(f"BUY signal: prob={prob:.4f} >= {self.buy_threshold} | Position: {position_value_sol:.2f}/{self.max_position_sol:.0f} SOL")
                await self.execute_buy(trade_sol, current_price, prob, size)

        elif prob <= self.sell_threshold and self.position_tokens > 0:
            # Sell portion based on how bearish the signal is
            # Lower signal = sell more (0.42 -> 100%, 0.30 -> 100%, 0.40 -> ~83%)
            sell_fraction = min(1.0, (self.sell_threshold - prob) / self.sell_threshold + 0.5)
            sell_tokens = self.position_tokens * sell_fraction
            sell_value_sol = sell_tokens * current_price
            # Cap at max_trade_sol for liquidity
            if sell_value_sol > self.max_trade_sol:
                sell_tokens = self.max_trade_sol / current_price
                sell_value_sol = self.max_trade_sol
            logger.info(f"SELL signal: prob={prob:.4f} <= {self.sell_threshold} | Selling {sell_fraction*100:.0f}% ({sell_value_sol:.2f} SOL worth)")
            await self.execute_sell(current_price, prob, size, sell_tokens=sell_tokens)

    async def run(self, interval_minutes: int = 10, duration_hours: float = None) -> None:
        """Run trading loop."""
        logger.info(f"Starting neural trader (interval={interval_minutes}min, dry_run={self.dry_run})")
        logger.info(f"Thresholds: buy={self.buy_threshold}, sell={self.sell_threshold}")
        logger.info(f"Max position: {self.max_position_sol} SOL (per-trade: {self.max_trade_sol} SOL)")
        logger.info(f"Data source: {OHLC_DATA_PATH}")

        # Initial data check
        if self.load_ohlc_from_disk():
            latest = self.get_latest_bar_time()
            logger.info(f"Initial data loaded: {len(self.price_history)} bars, latest: {latest}")
        else:
            logger.warning("No valid data at startup - waiting for collector")

        start_time = datetime.utcnow()
        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                logger.info(f"=== Cycle {cycle_count} ===")

                await self.trading_cycle()

                # Check duration
                if duration_hours:
                    elapsed = (datetime.utcnow() - start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        logger.info(f"Duration {duration_hours}h reached, stopping")
                        break

                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Print final PnL stats
            self.pnl_tracker.print_status()
            logger.info(f"PnL log saved to: {self.pnl_tracker.summary_file}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Neural Trading Bot for CODEX")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode (default)")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--interval", type=int, default=10, help="Check interval in minutes")
    parser.add_argument("--duration", type=float, default=None, help="Duration in hours")
    parser.add_argument("--max-position", type=float, default=20.0, help="Max total position in SOL")
    parser.add_argument("--max-trade", type=float, default=8.0, help="Max per-trade size in SOL (liquidity limit)")
    parser.add_argument("--buy-threshold", type=float, default=0.46)
    parser.add_argument("--sell-threshold", type=float, default=0.42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    dry_run = not args.live

    print(f"\n{'='*60}")
    print("Neural Trading Bot - CODEX")
    print(f"{'='*60}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data source: {OHLC_DATA_PATH}")
    print(f"Interval: {args.interval} min")
    print(f"Max position: {args.max_position} SOL (per-trade: {args.max_trade} SOL)")
    print(f"Thresholds: buy={args.buy_threshold}, sell={args.sell_threshold}")
    print(f"{'='*60}")

    # Check if data collector is running
    if not OHLC_DATA_PATH.exists():
        print("\n⚠️  WARNING: OHLC data file not found!")
        print("Make sure collect_bags_data.py is running:")
        print("  python collect_bags_data.py --interval 10")
        print("")

    if not dry_run:
        print("\n⚠️  WARNING: LIVE TRADING MODE")
        print("Real trades will be executed!")

        # Check private key
        from env_real import SOLANA_PRIVATE_KEY
        if not SOLANA_PRIVATE_KEY:
            print("\n❌ ERROR: SOLANA_PRIVATE_KEY not set!")
            print("Export your private key from Bags.fm wallet and set it in env_real.py")
            return

        print("Press Ctrl+C within 10 seconds to cancel...\n")
        try:
            await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    bags_config = BagsConfig()

    trader = NeuralTrader(
        checkpoint_path=args.checkpoint,
        bags_config=bags_config,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        max_position_sol=args.max_position,
        max_trade_sol=args.max_trade,
        dry_run=dry_run,
        device=args.device,
    )

    await trader.run(
        interval_minutes=args.interval,
        duration_hours=args.duration,
    )


if __name__ == "__main__":
    asyncio.run(main())
