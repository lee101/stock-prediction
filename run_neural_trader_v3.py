#!/usr/bin/env python3
"""
Neural Trader V3 - Uses BagsV3LLM transformer model.

Optimized thresholds: buy=0.29, sell=0.23

Usage:
    # Dry run
    python run_neural_trader_v3.py --dry-run

    # Live trading
    python run_neural_trader_v3.py --live
"""

import asyncio
import gc
import io
import os
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bagsfm import BagsAPIClient, BagsConfig, TokenConfig
from bagsfm.config import SOL_MINT
from bagsfm.bags_api import SolanaTransactionExecutor
from bagsfm.utils import require_live_trading_enabled
from bagsv3llm.model import BagsV3Config, BagsV3Transformer
from bagsv3llm.dataset import (
    build_bar_features,
    build_aggregate_features,
    FeatureNormalizerV3,
)
from pnl_tracker import PnLTracker

OHLC_DATA_PATH = Path("bagstraining/ohlc_data.csv")
DEFAULT_CHECKPOINT = Path("bagsv3llm/sweep/ctx96_l4_e64_lr0.001/bagsv3_HAK9cX1j_best.pt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("neural_trader_v3")
logging.getLogger("httpx").setLevel(logging.WARNING)

CODEX_TOKEN = TokenConfig(
    symbol="CODEX",
    mint="HAK9cX1jfYmcNpr6keTkLvxehGPWKELXSu7GH2ofBAGS",
    decimals=9,
    name="CODEX",
    min_trade_amount=1.0,
)


def _read_tail_lines(path: Path, max_lines: int, block_size: int = 1024 * 1024) -> list[str]:
    """Read the last N lines from a file without loading it all into memory."""
    if max_lines <= 0:
        return []

    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end_pos = f.tell()
        buffer = b""
        lines: list[bytes] = []

        while end_pos > 0 and len(lines) <= max_lines:
            read_size = min(block_size, end_pos)
            end_pos -= read_size
            f.seek(end_pos)
            buffer = f.read(read_size) + buffer
            lines = buffer.splitlines()
            if end_pos == 0:
                break

        if len(lines) > max_lines:
            lines = lines[-max_lines:]

    return [line.decode("utf-8", errors="replace") for line in lines]


def load_v3_model(checkpoint_path: Path, device: str = "cuda"):
    """Load V3 model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = checkpoint.get("config", {})
    config = BagsV3Config(
        context_length=config_dict.get("context_length", 96),
        n_layer=config_dict.get("n_layer", 4),
        n_head=config_dict.get("n_head", 8),
        n_embd=config_dict.get("n_embd", 64),
        dropout=config_dict.get("dropout", 0.1),
    )

    model = BagsV3Transformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalizers = {}
    if "normalizers" in checkpoint:
        for key, norm_dict in checkpoint["normalizers"].items():
            normalizers[key] = FeatureNormalizerV3.from_dict(norm_dict)

    return model, config, normalizers


class NeuralTraderV3:
    """V3 Transformer-based trader for CODEX."""

    def __init__(
        self,
        checkpoint_path: Path,
        bags_config: BagsConfig,
        buy_threshold: float = 0.29,
        sell_threshold: float = 0.23,
        max_position_sol: float = 20.0,
        max_trade_sol: float = 8.0,
        min_trade_sol: float = 0.01,
        dry_run: bool = True,
        device: str = "cuda",
        sleep_offload: bool = False,
        sleep_unload: bool = False,
    ):
        self.bags_config = bags_config
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_position_sol = max_position_sol
        self.max_trade_sol = max_trade_sol
        self.min_trade_sol = min_trade_sol
        self.dry_run = dry_run
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.sleep_offload = sleep_offload
        self.sleep_unload = sleep_unload
        self._model_offloaded = False

        # Load V3 model
        logger.info(f"Loading V3 model from {checkpoint_path}")
        self.model, self.config, self.normalizers = load_v3_model(checkpoint_path, device)
        self.model.to(self.device)
        self.context_length = self.config.context_length
        logger.info(f"V3 model loaded: context={self.context_length}, layers={self.config.n_layer}")

        # API clients
        self.api_client = BagsAPIClient(bags_config)
        self.executor = SolanaTransactionExecutor(bags_config) if not dry_run else None

        # State
        self.holding = False
        self.position_tokens = 0.0
        self.entry_price = 0.0
        self.sol_balance = 0.0
        self.last_price = 0.0

        # PnL tracking
        self.pnl_tracker = PnLTracker(
            token=CODEX_TOKEN.symbol,
            log_dir="logs/pnl_v3",
            snapshot_interval=60,
            max_trades_in_memory=2000,
            max_snapshots_in_memory=1440,
        )

        self.consecutive_failures = 0
        self.max_consecutive_failures = 5

    def _model_device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def ensure_model_loaded(self) -> None:
        """Ensure model is loaded on the active device before inference."""
        if self.model is None:
            logger.info("Reloading V3 model for inference")
            self.model, self.config, self.normalizers = load_v3_model(
                self.checkpoint_path, self.device.type
            )
            self.model.to(self.device)
            self.context_length = self.config.context_length
            self._model_offloaded = False
            return

        if self.device.type == "cuda":
            current_device = self._model_device()
            if current_device.type != "cuda":
                self.model.to(self.device)
                self._model_offloaded = False

    def offload_model(self) -> None:
        """Move model to CPU to release GPU memory between cycles."""
        if self.model is None:
            return
        if self.device.type != "cuda":
            return
        if self._model_device().type == "cpu":
            self._model_offloaded = True
            return
        self.model.to("cpu")
        self._model_offloaded = True
        torch.cuda.empty_cache()

    def unload_model(self) -> None:
        """Unload model entirely to release RAM/GPU between cycles."""
        if self.model is None:
            return
        self.model = None
        self._model_offloaded = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def release_model_for_sleep(self) -> None:
        """Optionally offload or unload model during sleep interval."""
        if self.sleep_unload:
            self.unload_model()
        elif self.sleep_offload:
            self.offload_model()

    def load_ohlc_data(self) -> pd.DataFrame:
        """Load and deduplicate OHLC data."""
        if not OHLC_DATA_PATH.exists():
            return None

        max_rows = max(2000, self.context_length * 20)
        max_cap = max_rows * 16
        limit = max_rows

        header_line = None
        with OHLC_DATA_PATH.open("r", encoding="utf-8") as f:
            header_line = f.readline().strip()

        df = None
        while True:
            tail_lines = _read_tail_lines(OHLC_DATA_PATH, limit)
            if tail_lines and tail_lines[0].lower().startswith("timestamp"):
                tail_lines = tail_lines[1:]
            if not tail_lines:
                return None

            if header_line:
                csv_data = "\n".join([header_line] + tail_lines)
            else:
                csv_data = "\n".join(tail_lines)

            df = pd.read_csv(io.StringIO(csv_data))
            if "token_symbol" in df.columns:
                df = df[df["token_symbol"] == CODEX_TOKEN.symbol].copy()

            if len(df) >= self.context_length * 2 or limit >= max_cap:
                break

            limit = min(limit * 2, max_cap)

        if df is None or df.empty:
            return None

        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @torch.no_grad()
    def predict(self, df: pd.DataFrame, idx: int) -> tuple:
        """Get V3 model prediction at given index."""
        if idx < self.context_length:
            return 0.5, 0.5

        start = idx - self.context_length
        opens = df["open"].iloc[start:idx].to_numpy(dtype=np.float32)
        highs = df["high"].iloc[start:idx].to_numpy(dtype=np.float32)
        lows = df["low"].iloc[start:idx].to_numpy(dtype=np.float32)
        closes = df["close"].iloc[start:idx].to_numpy(dtype=np.float32)

        # Build features
        bar_features = build_bar_features(opens, highs, lows, closes)
        agg_features = build_aggregate_features(opens, highs, lows, closes)
        chronos_features = np.zeros((self.context_length, 12), dtype=np.float32)

        # Normalize
        if "bar" in self.normalizers:
            bar_flat = bar_features.reshape(-1, bar_features.shape[-1])
            bar_flat = self.normalizers["bar"].transform(bar_flat)
            bar_features = bar_flat.reshape(bar_features.shape)
        if "agg" in self.normalizers:
            agg_features = self.normalizers["agg"].transform(agg_features)

        # To tensors
        bar_t = torch.tensor(bar_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        chronos_t = torch.tensor(chronos_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        agg_t = torch.tensor(agg_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward
        signal_logit, size_logit = self.model(bar_t, chronos_t, agg_t)
        prob = torch.sigmoid(signal_logit).item()
        size = torch.sigmoid(size_logit).item()

        return prob, size

    async def update_balances(self) -> None:
        """Update SOL and token balances."""
        pubkey = self.bags_config.public_key
        if not pubkey:
            return

        try:
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TokenAccountOpts
            from solders.pubkey import Pubkey

            TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

            async with AsyncClient(self.bags_config.rpc_url) as rpc:
                response = await rpc.get_balance(Pubkey.from_string(pubkey))
                self.sol_balance = response.value / 1e9

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
                amount=10_000_000,
                slippage_mode="auto",
            )
            tokens_received = quote.out_amount / (10 ** CODEX_TOKEN.decimals)
            price_sol = 0.01 / tokens_received if tokens_received > 0 else 0
            return price_sol
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return 0.0

    async def execute_buy(self, sol_amount: float, price: float, prob: float, size: float) -> bool:
        """Execute buy order."""
        if self.dry_run:
            logger.info(f"[DRY RUN] BUY {sol_amount:.4f} SOL worth of CODEX at {price:.10f}")
            self.holding = True
            self.entry_price = price
            self.position_tokens = sol_amount / price
            self.pnl_tracker.log_trade("buy", self.position_tokens, price, sol_amount, prob, size)
            return True

        try:
            lamports = int(sol_amount * 1e9)
            quote = await self.api_client.get_quote(
                input_mint=SOL_MINT,
                output_mint=CODEX_TOKEN.mint,
                amount=lamports,
                slippage_mode="auto",
            )

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
                self.pnl_tracker.log_trade("buy", tokens_received, price, sol_amount, prob, size)
                self.consecutive_failures = 0
                return True
            else:
                logger.warning(f"BUY failed: {result.error}")
                self.consecutive_failures += 1
                return False

        except Exception as e:
            logger.exception(f"BUY exception: {e}")
            self.consecutive_failures += 1
            return False

    async def execute_sell(self, price: float, prob: float, size: float) -> bool:
        """Execute sell order with retry logic for liquidity."""
        # Cap sell at max_trade_sol worth of tokens due to liquidity
        max_tokens = self.max_trade_sol / price if price > 0 else self.position_tokens
        tokens_to_sell = min(self.position_tokens, max_tokens)

        if self.dry_run:
            pnl = (price - self.entry_price) / self.entry_price * 100 if self.entry_price > 0 else 0
            sol_received = tokens_to_sell * price
            logger.info(f"[DRY RUN] SELL {tokens_to_sell:.0f} CODEX at {price:.10f} (PnL: {pnl:+.2f}%)")
            self.pnl_tracker.log_trade("sell", tokens_to_sell, price, sol_received, prob, size)
            self.position_tokens -= tokens_to_sell
            self.holding = self.position_tokens > 0
            if not self.holding:
                self.entry_price = 0.0
            return True

        # Retry with decreasing amounts
        amounts_to_try = [tokens_to_sell]
        for fraction in [0.75, 0.5, 0.25]:
            smaller = tokens_to_sell * fraction
            if smaller * price >= self.min_trade_sol:
                amounts_to_try.append(smaller)

        for attempt, try_tokens in enumerate(amounts_to_try):
            try:
                token_amount = int(try_tokens * (10 ** CODEX_TOKEN.decimals))

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
                    self.pnl_tracker.log_trade("sell", try_tokens, price, sol_received, prob, size)
                    self.position_tokens -= try_tokens
                    self.holding = self.position_tokens > 0
                    if not self.holding:
                        self.entry_price = 0.0
                    self.consecutive_failures = 0
                    return True
                else:
                    logger.warning(f"SELL attempt {attempt+1} failed: {result.error}")
                    if attempt < len(amounts_to_try) - 1:
                        await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"SELL attempt {attempt+1} exception: {e}")
                if attempt < len(amounts_to_try) - 1:
                    await asyncio.sleep(1)

        logger.error(f"SELL failed after {len(amounts_to_try)} attempts")
        self.consecutive_failures += 1
        return False

    async def trading_cycle(self) -> None:
        """Run one trading cycle."""
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures ({self.consecutive_failures}), pausing")
            self.consecutive_failures = 0
            return

        await self.update_balances()

        current_price = await self.get_current_price()
        if current_price <= 0:
            logger.warning("Could not get price")
            self.consecutive_failures += 1
            return

        # Price anomaly check
        if self.last_price > 0:
            price_change = abs(current_price - self.last_price) / self.last_price
            if price_change > 0.5:
                logger.warning(f"Price anomaly: {price_change*100:.1f}% change")
                return
        self.last_price = current_price

        # Log snapshot
        self.pnl_tracker.log_snapshot(self.sol_balance, self.position_tokens, current_price)

        # Load data
        df = self.load_ohlc_data()
        if df is None or len(df) < self.context_length:
            logger.warning(f"Not enough data: {len(df) if df is not None else 0}/{self.context_length}")
            return

        # Check data freshness
        latest_time = df["timestamp"].iloc[-1]
        now_utc = datetime.now(timezone.utc)
        latest_utc = latest_time.replace(tzinfo=timezone.utc) if latest_time.tzinfo is None else latest_time
        age_minutes = (now_utc - latest_utc).total_seconds() / 60
        if age_minutes > 30:
            logger.warning(f"Data stale: {age_minutes:.0f} min old")
            return

        # Ensure model is ready for inference
        self.ensure_model_loaded()

        # Get prediction
        prob, size = self.predict(df, len(df))
        logger.info(f"Price: {current_price:.10f} | Signal: {prob:.4f} | Size: {size:.4f} | Holding: {self.holding}")

        # Log PnL stats
        stats = self.pnl_tracker.get_stats()
        if stats.get("status") != "no data":
            logger.info(f"PnL: {stats['total_return_pct']:+.2f}% | Trades: {stats['trades']}")

        # Trading logic
        position_value_sol = self.position_tokens * current_price
        room_to_buy = self.max_position_sol - position_value_sol

        if prob >= self.buy_threshold and room_to_buy > self.min_trade_sol:
            desired_trade = size * self.max_trade_sol
            trade_sol = min(desired_trade, room_to_buy, self.sol_balance * 0.95, self.max_trade_sol)
            if trade_sol >= self.min_trade_sol:
                logger.info(f"BUY signal: prob={prob:.4f} >= {self.buy_threshold}")
                await self.execute_buy(trade_sol, current_price, prob, size)

        elif prob <= self.sell_threshold and self.position_tokens > 0:
            logger.info(f"SELL signal: prob={prob:.4f} <= {self.sell_threshold}")
            await self.execute_sell(current_price, prob, size)

    async def run(self, interval_minutes: int = 10) -> None:
        """Run trading loop."""
        logger.info(f"Starting V3 trader (interval={interval_minutes}min, dry_run={self.dry_run})")
        logger.info(f"Thresholds: buy={self.buy_threshold}, sell={self.sell_threshold}")
        logger.info(f"Max position: {self.max_position_sol} SOL")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                logger.info(f"=== Cycle {cycle_count} ===")
                await self.trading_cycle()
                self.release_model_for_sleep()
                await asyncio.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.pnl_tracker.print_status()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Neural Trader V3 for CODEX")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--max-position", type=float, default=20.0)
    parser.add_argument("--max-trade", type=float, default=8.0)
    parser.add_argument("--buy-threshold", type=float, default=0.29)
    parser.add_argument("--sell-threshold", type=float, default=0.23)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sleep-offload", action="store_true", help="Move model to CPU between cycles")
    parser.add_argument("--sleep-unload", action="store_true", help="Unload model between cycles")

    args = parser.parse_args()
    dry_run = not args.live

    print(f"\n{'='*60}")
    print("Neural Trader V3 - CODEX")
    print(f"{'='*60}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds: buy={args.buy_threshold}, sell={args.sell_threshold}")
    print(f"Max position: {args.max_position} SOL")
    print(f"{'='*60}\n")

    if not dry_run:
        require_live_trading_enabled()
        print("WARNING: LIVE TRADING MODE - Press Ctrl+C within 10s to cancel")
        try:
            await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("Cancelled.")
            return

    if args.sleep_offload and args.sleep_unload:
        raise SystemExit("Use only one of --sleep-offload or --sleep-unload")

    bags_config = BagsConfig()

    trader = NeuralTraderV3(
        checkpoint_path=args.checkpoint,
        bags_config=bags_config,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        max_position_sol=args.max_position,
        max_trade_sol=args.max_trade,
        dry_run=dry_run,
        device=args.device,
        sleep_offload=args.sleep_offload,
        sleep_unload=args.sleep_unload,
    )

    await trader.run(interval_minutes=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
