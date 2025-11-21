from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

import alpaca_wrapper
from hourlycryptomarketsimulator import HourlyCryptoMarketSimulator, SimulationConfig
from hourlycryptotraining import (
    DailyChronosForecastManager,
    HourlyCryptoDataModule,
    HourlyCryptoTrainer,
    PolicyHeadConfig,
    TrainingConfig,
)
from hourlycryptotraining.data import MultiSymbolDataModule
from hourlycryptotraining.checkpoints import find_best_checkpoint, load_checkpoint
from hourlycryptotraining.model import HourlyCryptoPolicy
from src.process_utils import (
    spawn_close_position_at_maxdiff_takeprofit,
    spawn_open_position_at_maxdiff_takeprofit,
)

logger = logging.getLogger("hourlycrypto")


@dataclass
class TradingPlan:
    timestamp: pd.Timestamp
    buy_price: float
    sell_price: float
    buy_amount: float
    sell_amount: float


@dataclass
class PriceOffsetParams:
    base_pct: float
    span_multiplier: float = 0.0
    max_pct: Optional[float] = None

    def build_tensor(
        self,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
    ) -> torch.Tensor:
        offsets = torch.full_like(reference_close, self.base_pct)
        if self.span_multiplier:
            span = torch.clamp(
                (chronos_high - chronos_low)
                / reference_close.clamp(min=1e-6),
                min=0.0,
            )
            offsets = offsets + self.span_multiplier * span
        if self.max_pct is not None:
            offsets = torch.clamp(offsets, max=self.max_pct)
        return torch.clamp(offsets, min=1e-6)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly Chronos-driven crypto trading loop")
    parser.add_argument("--mode", choices=["train", "simulate", "trade"], default="trade")
    parser.add_argument("--daemon", action="store_true", help="Run continuously, waking every hour aligned to UTC")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate (e.g., 1e-4 for fine-tuning)")
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Skip watcher spawning")
    parser.add_argument("--window-hours", type=int, default=24 * 14, help="Simulation lookback window")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--dry-train-steps", type=int, default=None)
    parser.add_argument("--ema-decay", type=float, default=None, help="EMA decay rate (e.g., 0.999)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (required for long sequences)")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision training")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16", help="AMP dtype (bfloat16 or float16)")
    parser.add_argument("--training-symbols", type=str, nargs="+", default=None, help="Train on multiple symbols (e.g., BTCUSD ETHUSD UNIUSD LINKUSD)")
    parser.add_argument("--preload-checkpoint", type=str, default=None, help="Preload weights from checkpoint for fine-tuning")
    parser.add_argument("--dropout", type=float, default=None, help="Override transformer dropout rate")
    parser.add_argument("--symbol", type=str, default="LINKUSD", help="Primary trading symbol (e.g., UNIUSD)")
    parser.add_argument(
        "--cache-only-forecasts",
        action="store_true",
        help="Use existing Chronos forecast cache without regenerating missing rows (saves GPU; default regenerates)",
    )
    parser.add_argument(
        "--price-offset-pct",
        type=float,
        default=None,
        help="Base limit offset as a decimal fraction (0.0003 = 0.03%%)",
    )
    parser.add_argument(
        "--price-offset-span-multiplier",
        type=float,
        default=0.15,
        help="Linear multiplier on (Chronos_high - Chronos_low)/reference_close to widen offsets",
    )
    parser.add_argument(
        "--price-offset-max-pct",
        type=float,
        default=0.003,
        help="Upper clamp for the effective offset (0.003 = 0.3%%). Use 0 to disable",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _ensure_forecasts(config: TrainingConfig, cache_only: bool = False) -> pd.DataFrame:
    """Load forecast cache, optionally generating missing forecasts.

    Args:
        config: Training configuration
        cache_only: If True, only load from cache without generating missing forecasts (saves GPU memory during training)
    """
    manager = DailyChronosForecastManager(config.forecast_config)
    return manager.ensure_latest(cache_only=cache_only)


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.sequence_length = args.sequence_length
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    cfg.dataset.sequence_length = args.sequence_length
    cfg.dataset.forecast_cache_dir = cfg.forecast_config.cache_dir
    cfg.run_name = cfg.run_name or f"hourlycrypto_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    if args.checkpoint_root:
        cfg.checkpoint_root = Path(args.checkpoint_root)
    if args.checkpoint_path:
        cfg.preload_checkpoint_path = Path(args.checkpoint_path)
    # Use --preload-checkpoint for explicit fine-tuning (takes precedence over checkpoint-path)
    if hasattr(args, 'preload_checkpoint') and args.preload_checkpoint:
        cfg.preload_checkpoint_path = Path(args.preload_checkpoint)
    cfg.force_retrain = bool(args.force_retrain)
    cfg.dry_train_steps = args.dry_train_steps
    if args.ema_decay is not None:
        cfg.ema_decay = args.ema_decay
    if args.no_compile:
        cfg.use_compile = False
    if args.use_amp:
        cfg.use_amp = True
        cfg.amp_dtype = args.amp_dtype
    if args.dropout is not None:
        cfg.transformer_dropout = args.dropout
    if args.price_offset_pct is not None:
        cfg.price_offset_pct = max(1e-6, args.price_offset_pct)
    symbol = (args.symbol or cfg.dataset.symbol).upper()
    cfg.dataset.symbol = symbol
    cfg.forecast_config.symbol = symbol
    return cfg


def _train_policy(config: TrainingConfig, training_symbols: Optional[list[str]] = None) -> tuple[HourlyCryptoDataModule | MultiSymbolDataModule, HourlyCryptoPolicy, Optional[Path]]:
    # Use multi-symbol training if symbols are provided
    if training_symbols and len(training_symbols) > 1:
        logging.info(f"Multi-symbol training with {len(training_symbols)} pairs: {', '.join(training_symbols)}")
        data_module = MultiSymbolDataModule(training_symbols, config.dataset)
    else:
        data_module = HourlyCryptoDataModule(config.dataset)
    trainer = HourlyCryptoTrainer(config, data_module)
    artifacts = trainer.train()
    policy = HourlyCryptoPolicy(
        PolicyHeadConfig(
            input_dim=len(data_module.feature_columns),
            hidden_dim=config.transformer_dim,
            dropout=config.transformer_dropout,
            price_offset_pct=config.price_offset_pct,
            max_trade_qty=config.max_trade_qty,
            min_price_gap_pct=config.min_price_gap_pct,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
        )
    )
    policy.load_state_dict(artifacts.state_dict, strict=False)
    return data_module, policy, artifacts.best_checkpoint


def _find_global_best_checkpoint(checkpoint_root: Path) -> Optional[Path]:
    """Search all checkpoint* directories for the best checkpoint."""
    parent = checkpoint_root.parent
    search_pattern = checkpoint_root.name + "*"
    all_roots = sorted(parent.glob(search_pattern))

    best_path: Optional[Path] = None
    best_loss = float("inf")

    for root in all_roots:
        if not root.is_dir():
            continue
        candidate = find_best_checkpoint(root)
        if candidate is None:
            continue
        # Extract val_loss from checkpoint
        try:
            payload = load_checkpoint(candidate)
            val_loss = payload.get("metrics", {}).get("loss", float("inf"))
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = candidate
        except Exception:
            continue

    if best_path:
        logger.info(f"Global search found best checkpoint: {best_path} (val_loss={best_loss:.6f})")
    return best_path


def _load_pretrained_policy(
    config: TrainingConfig,
    price_offset_override: Optional[float] = None,
) -> Optional[Tuple[HourlyCryptoDataModule, HourlyCryptoPolicy, Path]]:
    if config.preload_checkpoint_path:
        ckpt_path = config.preload_checkpoint_path
    else:
        # Search across all checkpoint* directories for the global best
        ckpt_path = _find_global_best_checkpoint(config.checkpoint_root)

    if ckpt_path is None:
        logger.info("No pretrained checkpoint found under %s*", config.checkpoint_root)
        return None
    payload = load_checkpoint(ckpt_path)
    feature_columns = payload.get("feature_columns") or list(config.dataset.feature_columns or [])
    dataset_cfg = replace(config.dataset, feature_columns=tuple(feature_columns))
    data_module = HourlyCryptoDataModule(dataset_cfg)
    data_module.normalizer = payload["normalizer"]

    payload_cfg = payload.get("config") or {}
    ckpt_offset = payload_cfg.get("price_offset_pct") if isinstance(payload_cfg, dict) else None
    effective_offset = (
        price_offset_override
        if price_offset_override is not None
        else ckpt_offset
        if ckpt_offset is not None
        else config.price_offset_pct
    )
    config.price_offset_pct = max(1e-6, float(effective_offset))

    # Detect max_len from checkpoint's positional encoding for backward compatibility
    state_dict = payload["state_dict"]
    max_len = 2048  # Default for new models
    if "pos_encoding.pe" in state_dict:
        ckpt_max_len = state_dict["pos_encoding.pe"].shape[0]
        if ckpt_max_len != max_len:
            logger.info(f"Detected old checkpoint with max_len={ckpt_max_len} (current default is {max_len})")
            max_len = ckpt_max_len

    policy = HourlyCryptoPolicy(
        PolicyHeadConfig(
            input_dim=len(feature_columns),
            hidden_dim=config.transformer_dim,
            dropout=config.transformer_dropout,
            price_offset_pct=config.price_offset_pct,
            max_trade_qty=config.max_trade_qty,
            min_price_gap_pct=config.min_price_gap_pct,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            max_len=max_len,  # Use detected max_len from checkpoint
        )
    )

    upgraded_state = HourlyCryptoPolicy.upgrade_legacy_state_dict(dict(state_dict))
    missing_keys, unexpected_keys = policy.load_state_dict(upgraded_state, strict=False)

    # Log any compatibility issues
    if missing_keys:
        logger.info("Missing keys in checkpoint (using newly initialized values): %s", missing_keys)
    if unexpected_keys:
        logger.warning("Unexpected keys in checkpoint (ignoring): %s", unexpected_keys)

    logger.info("Loaded pretrained checkpoint %s (val_loss=%.6f)", ckpt_path, payload.get("metrics", {}).get("loss", float("nan")))
    return data_module, policy, ckpt_path


def _infer_actions(
    policy: HourlyCryptoPolicy,
    data_module: HourlyCryptoDataModule,
    config: TrainingConfig,
    offset_params: Optional[PriceOffsetParams] = None,
) -> pd.DataFrame:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = policy.to(device).eval()
    features = data_module.frame[list(data_module.feature_columns)].to_numpy(dtype="float32")
    norm = data_module.normalizer.transform(features)
    closes = data_module.frame["close"].to_numpy(dtype="float32")
    ref_close = data_module.frame["reference_close"].to_numpy(dtype="float32")
    chronos_high = data_module.frame["chronos_high"].to_numpy(dtype="float32")
    chronos_low = data_module.frame["chronos_low"].to_numpy(dtype="float32")
    real_high = data_module.frame["high"].to_numpy(dtype="float32")
    real_low = data_module.frame["low"].to_numpy(dtype="float32")
    timestamps = data_module.frame["timestamp"].to_numpy()
    seq_len = config.sequence_length
    rows = []
    with torch.no_grad():
        for idx in range(seq_len, len(data_module.frame) + 1):
            window = slice(idx - seq_len, idx)
            feat = torch.from_numpy(norm[window]).unsqueeze(0).to(device)
            close_tensor = torch.from_numpy(closes[window]).unsqueeze(0).to(device)
            ref_tensor = torch.from_numpy(ref_close[window]).unsqueeze(0).to(device)
            high_tensor = torch.from_numpy(chronos_high[window]).unsqueeze(0).to(device)
            low_tensor = torch.from_numpy(chronos_low[window]).unsqueeze(0).to(device)
            outputs = policy(feat)
            offset_tensor = (
                offset_params.build_tensor(ref_tensor, high_tensor, low_tensor)
                if offset_params
                else None
            )
            decoded = policy.decode_actions(
                outputs,
                reference_close=ref_tensor,
                chronos_high=high_tensor,
                chronos_low=low_tensor,
                dynamic_offset_pct=offset_tensor,
            )
            ts = pd.Timestamp(timestamps[idx - 1])
            rows.append(
                {
                    "timestamp": ts,
                    "buy_price": float(decoded["buy_price"][0, -1].item()),
                    "sell_price": float(decoded["sell_price"][0, -1].item()),
                    "trade_amount": float(decoded["trade_amount"][0, -1].item()),
                    "buy_amount": float(decoded["buy_amount"][0, -1].item()),
                    "sell_amount": float(decoded["sell_amount"][0, -1].item()),
                    "offset_pct": float(offset_tensor[0, -1].item()) if offset_tensor is not None else float(config.price_offset_pct),
                }
            )
    return pd.DataFrame(rows)


def _simulate(actions: pd.DataFrame, data_module: HourlyCryptoDataModule, window_hours: int) -> None:
    if actions.empty:
        logger.warning("No actions inferred; skipping simulation")
        return
    bars = data_module.frame["timestamp"].isin(actions["timestamp"])
    merged_bars = data_module.frame.loc[bars, ["timestamp", "high", "low", "close"]]
    if window_hours:
        cutoff = merged_bars["timestamp"].max() - pd.Timedelta(hours=window_hours)
        merged_bars = merged_bars[merged_bars["timestamp"] >= cutoff]
        actions = actions[actions["timestamp"] >= cutoff]

    # Run normal simulation
    simulator = HourlyCryptoMarketSimulator(
        SimulationConfig(symbol=data_module.config.symbol.upper())
    )
    result = simulator.run(merged_bars, actions)
    logger.info(
        "Simulation: total_return=%.2f%% sortino=%.2f final_cash=%.2f inventory=%.4f",
        result.metrics.get("total_return", 0.0) * 100,
        result.metrics.get("sortino", 0.0),
        result.final_cash,
        result.final_inventory,
    )

    # Run daily PnL-based probe trading simulation
    try:
        from hourlycryptomarketsimulator import DailyPnlProbeSimulator, DailyPnlProbeConfig
        daily_probe_sim = DailyPnlProbeSimulator(
            SimulationConfig(symbol=data_module.config.symbol.upper()),
            DailyPnlProbeConfig(
                probe_trade_amount=0.01,  # 1% trades in probe mode
                min_daily_pnl_to_exit_probe=0.0,  # Exit probe when daily PnL >= $0
            ),
        )
        daily_probe_result = daily_probe_sim.run(merged_bars, actions)
        logger.info(
            "Daily PnL Probe: total_return=%.2f%% sortino=%.2f final_cash=%.2f inventory=%.4f (%.1f%% time in probe, %d switches)",
            daily_probe_result.metrics.get("total_return", 0.0) * 100,
            daily_probe_result.metrics.get("sortino", 0.0),
            daily_probe_result.final_cash,
            daily_probe_result.final_inventory,
            daily_probe_result.metrics.get("probe_mode_pct", 0.0),
            daily_probe_result.metrics.get("probe_mode_switches", 0),
        )
        daily_improvement = (daily_probe_result.metrics.get("total_return", 0.0) - result.metrics.get("total_return", 0.0)) * 100
        logger.info(
            "Daily PnL Probe Improvement: %+.2f%% return",
            daily_improvement,
        )
    except Exception as e:
        logger.warning("Daily PnL probe trading simulation failed: %s", e)


def _build_trading_plan(actions: pd.DataFrame) -> Optional[TradingPlan]:
    if actions.empty:
        return None
    latest = actions.iloc[-1]
    return TradingPlan(
        timestamp=pd.Timestamp(latest["timestamp"]),
        buy_price=float(latest["buy_price"]),
        sell_price=float(latest["sell_price"]),
        buy_amount=float(latest.get("buy_amount", latest.get("trade_amount", 0.0))),
        sell_amount=float(latest.get("sell_amount", latest.get("trade_amount", 0.0))),
    )


def _current_inventory(symbol: str) -> float:
    try:
        positions = alpaca_wrapper.get_all_positions()
    except Exception:
        return 0.0
    for position in positions:
        if position.symbol.upper() == symbol.upper():
            try:
                return float(position.qty)
            except Exception:
                return 0.0
    return 0.0


def _available_cash() -> float:
    try:
        account = alpaca_wrapper.get_account()
        return float(getattr(account, "cash", 0.0))
    except Exception:
        return 0.0


def _normalize_symbol(symbol: str) -> str:
    """Normalize crypto symbol format (remove slashes)."""
    return symbol.replace("/", "").upper()


def _cancel_existing_orders(symbol: str) -> None:
    """Cancel all open orders for the given symbol."""
    try:
        open_orders = alpaca_wrapper.get_open_orders()
        # Normalize both symbols for comparison (handles BTCUSD vs BTC/USD)
        normalized_symbol = _normalize_symbol(symbol)
        symbol_orders = [o for o in open_orders if _normalize_symbol(o.symbol) == normalized_symbol]
        if not symbol_orders:
            logger.info(f"No existing open orders for {symbol}")
            return
        logger.info(f"Cancelling {len(symbol_orders)} existing orders for {symbol}")
        for order in symbol_orders:
            try:
                alpaca_wrapper.cancel_order(order)
                logger.info(f"Cancelled order {order.id} for {symbol}")
            except Exception as exc:
                logger.warning(f"Failed to cancel order {order.id}: {exc}")
    except Exception as exc:
        logger.error(f"Error fetching/cancelling orders for {symbol}: {exc}")


def _spawn_watchers(plan: TradingPlan, dry_run: bool, symbol: str) -> None:
    symbol = symbol.upper()

    # Cancel existing orders before placing new ones
    if not dry_run:
        _cancel_existing_orders(symbol)

    cash = _available_cash()
    inventory = _current_inventory(symbol)
    buy_amt = max(0.0, min(1.0, plan.buy_amount))
    sell_amt = max(0.0, min(1.0, plan.sell_amount))
    max_buy_qty = cash / plan.buy_price if plan.buy_price > 0 else 0.0
    buy_qty = buy_amt * max(0.0, max_buy_qty)
    sell_qty = sell_amt * max(0.0, inventory)

    # If selling most of the position (>95%), sell ALL to avoid tiny residuals
    # that can block future buy orders
    if inventory > 0 and sell_qty >= inventory * 0.95:
        sell_qty = inventory

    # SAFETY CHECK: Ensure sell_price > buy_price with minimum 3 basis points spread
    # This prevents inverted prices and unprofitable trading
    MIN_SPREAD_PCT = 0.0003  # 3 basis points = 0.03%
    required_min_sell = plan.buy_price * (1.0 + MIN_SPREAD_PCT)
    actual_spread_pct = ((plan.sell_price - plan.buy_price) / plan.buy_price * 100) if plan.buy_price > 0 else 0

    if plan.sell_price <= plan.buy_price or plan.sell_price < required_min_sell:
        logger.error(
            "INVALID PRICE SPREAD for %s: buy=%.2f sell=%.2f (spread=%.4f%%). "
            "Required min spread: %.4f%% (3bp). BLOCKING ALL TRADES!",
            symbol,
            plan.buy_price,
            plan.sell_price,
            actual_spread_pct,
            MIN_SPREAD_PCT * 100,
        )
        return  # Exit immediately - do not place any orders

    logger.info(
        "Trading plan %s amt=%.4f (raw=%.4f) buy=%.4f@%.2f sell=%.4f@%.2f spread=%.4f%% cash=%.2f inv=%.4f",
        plan.timestamp,
        max(buy_amt, sell_amt),
        max(buy_amt, sell_amt),  # Log raw value from model
        buy_qty,
        plan.buy_price,
        sell_qty,
        plan.sell_price,
        actual_spread_pct,
        cash,
        inventory,
    )

    # Warn if model is outputting very low trade intensity
    if max(buy_amt, sell_amt) < 0.01:
        logger.warning(
            "Model output very low trade_amount=%.6f/%.6f → buy_qty=%.6f, sell_qty=%.6f (skipping trades)",
            plan.buy_amount,
            plan.sell_amount,
            buy_qty,
            sell_qty,
        )
        if not dry_run:
            logger.info("No trades will be placed due to low trade intensity")
            return
    if dry_run:
        logger.info("Dry-run: skipping watcher spawn")
        return
    if buy_qty > 0:
        try:
            spawn_open_position_at_maxdiff_takeprofit(
                symbol,
                "buy",
                plan.buy_price,
                buy_qty,
                tolerance_pct=0.0008,
                expiry_minutes=90,
                entry_strategy="hourlycrypto",
                force_immediate=True,  # Place order immediately, don't wait
            )
        except Exception as exc:
            logger.error("Failed to spawn entry watcher: %s", exc)
    if sell_qty > 0:
        try:
            spawn_close_position_at_maxdiff_takeprofit(
                symbol,
                "sell",
                plan.sell_price,
                target_qty=sell_qty,
                entry_strategy="hourlycrypto",
            )
        except Exception as exc:
            logger.error("Failed to spawn exit watcher: %s", exc)


def _seconds_until_next_hour() -> float:
    """Calculate seconds until the next UTC hour boundary, plus a small buffer."""
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    # Add 30 seconds buffer to ensure new data bar is available
    next_hour_with_buffer = next_hour + timedelta(seconds=30)
    delta = (next_hour_with_buffer - now).total_seconds()
    return max(1.0, delta)


def _run_trading_cycle(args: argparse.Namespace, config: TrainingConfig) -> None:
    """Execute one trading cycle: load/train policy, infer, simulate, trade."""
    # Default: regenerate missing Chronos forecasts so train/infer use identical inputs
    cache_only = bool(args.cache_only_forecasts)

    # Ensure fresh forecasts for the latest data
    _ensure_forecasts(config, cache_only=cache_only)

    # Force reload data module to get latest bars
    loaded = None if config.force_retrain else _load_pretrained_policy(config, price_offset_override=args.price_offset_pct)

    # Log the latest timestamp in the data to verify fresh data
    if loaded:
        latest_ts = loaded[0].frame["timestamp"].max()
        logger.info(f"Latest data timestamp: {latest_ts}")
    if loaded is None:
        data_module, policy, checkpoint_path = _train_policy(config, training_symbols=args.training_symbols)
    else:
        data_module, policy, checkpoint_path = loaded
    max_pct = None
    if args.price_offset_max_pct is not None and args.price_offset_max_pct > 0:
        max_pct = args.price_offset_max_pct
    offset_params = PriceOffsetParams(
        base_pct=config.price_offset_pct,
        span_multiplier=max(0.0, args.price_offset_span_multiplier or 0.0),
        max_pct=max_pct,
    )
    actions = _infer_actions(policy, data_module, config, offset_params=offset_params)

    # Log the latest action timestamp to verify we're using fresh data
    if not actions.empty:
        logger.info(f"Generated {len(actions)} actions, latest action timestamp: {actions.iloc[-1]['timestamp']}")
        logger.info(f"Latest action prices: buy={actions.iloc[-1]['buy_price']:.2f}, sell={actions.iloc[-1]['sell_price']:.2f}")

    _simulate(actions, data_module, args.window_hours)
    if args.mode in {"simulate", "train"}:
        return
    plan = _build_trading_plan(actions)
    if not plan:
        logger.warning("No trading plan generated")
        return
    _spawn_watchers(plan, args.dry_run, config.dataset.symbol)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    config = _build_training_config(args)

    if not args.daemon:
        # Single run mode
        _run_trading_cycle(args, config)
        return

    # Daemon mode: run continuously on hourly boundaries
    logger.info("Starting daemon mode - will wake every hour aligned to UTC")
    cycle_count = 0
    while True:
        try:
            now = datetime.now(timezone.utc)
            logger.info(f"Cycle {cycle_count + 1} starting at {now.isoformat()}")

            # Run trading cycle
            _run_trading_cycle(args, config)

            cycle_count += 1

            # Calculate sleep time until next hour
            sleep_seconds = _seconds_until_next_hour()
            next_wake = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
            logger.info(
                f"Cycle {cycle_count} complete. Sleeping {sleep_seconds:.1f}s until {next_wake.isoformat()}"
            )
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
            break
        except Exception as exc:
            logger.error(f"Error in trading cycle {cycle_count + 1}: {exc}", exc_info=True)
            # Sleep before retry to avoid tight error loops
            logger.info("Sleeping 60s before retry...")
            time.sleep(60)


if __name__ == "__main__":
    main()
