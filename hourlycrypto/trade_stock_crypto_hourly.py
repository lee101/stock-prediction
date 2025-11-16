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
    trade_amount: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly Chronos-driven LINKUSD trading loop")
    parser.add_argument("--mode", choices=["train", "simulate", "trade"], default="trade")
    parser.add_argument("--daemon", action="store_true", help="Run continuously, waking every hour aligned to UTC")
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--dropout", type=float, default=None, help="Override transformer dropout rate")
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
    cfg.dataset.sequence_length = args.sequence_length
    cfg.dataset.forecast_cache_dir = cfg.forecast_config.cache_dir
    cfg.run_name = cfg.run_name or f"hourlycrypto_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    if args.checkpoint_root:
        cfg.checkpoint_root = Path(args.checkpoint_root)
    if args.checkpoint_path:
        cfg.preload_checkpoint_path = Path(args.checkpoint_path)
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


def _load_pretrained_policy(config: TrainingConfig) -> Optional[Tuple[HourlyCryptoDataModule, HourlyCryptoPolicy, Path]]:
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

    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)

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
            decoded = policy.decode_actions(
                outputs,
                reference_close=ref_tensor,
                chronos_high=high_tensor,
                chronos_low=low_tensor,
            )
            ts = pd.Timestamp(timestamps[idx - 1])
            rows.append(
                {
                    "timestamp": ts,
                    "buy_price": float(decoded["buy_price"][0, -1].item()),
                    "sell_price": float(decoded["sell_price"][0, -1].item()),
                    "trade_amount": float(decoded["trade_amount"][0, -1].item()),
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
    simulator = HourlyCryptoMarketSimulator(SimulationConfig())
    result = simulator.run(merged_bars, actions)
    logger.info(
        "Simulation: total_return=%.2f%% sortino=%.2f final_cash=%.2f inventory=%.4f",
        result.metrics.get("total_return", 0.0) * 100,
        result.metrics.get("sortino", 0.0),
        result.final_cash,
        result.final_inventory,
    )


def _build_trading_plan(actions: pd.DataFrame) -> Optional[TradingPlan]:
    if actions.empty:
        return None
    latest = actions.iloc[-1]
    return TradingPlan(
        timestamp=pd.Timestamp(latest["timestamp"]),
        buy_price=float(latest["buy_price"]),
        sell_price=float(latest["sell_price"]),
        trade_amount=float(latest.get("trade_amount", 0.0)),
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


def _cancel_existing_orders(symbol: str) -> None:
    """Cancel all open orders for the given symbol."""
    try:
        open_orders = alpaca_wrapper.get_open_orders()
        symbol_orders = [o for o in open_orders if o.symbol == symbol]
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


def _spawn_watchers(plan: TradingPlan, dry_run: bool) -> None:
    symbol = "LINKUSD"

    # Cancel existing orders before placing new ones
    if not dry_run:
        _cancel_existing_orders(symbol)

    cash = _available_cash()
    inventory = _current_inventory(symbol)
    trade_amt = max(0.0, min(1.0, plan.trade_amount))
    max_buy_qty = cash / plan.buy_price if plan.buy_price > 0 else 0.0
    buy_qty = trade_amt * max(0.0, max_buy_qty)
    sell_qty = trade_amt * max(0.0, inventory)

    logger.info(
        "Trading plan %s amt=%.4f (raw=%.4f) buy=%.4f@%.4f sell=%.4f@%.4f cash=%.2f inv=%.4f",
        plan.timestamp,
        trade_amt,
        plan.trade_amount,  # Log raw value from model
        buy_qty,
        plan.buy_price,
        sell_qty,
        plan.sell_price,
        cash,
        inventory,
    )

    # Warn if model is outputting very low trade intensity
    if trade_amt < 0.01:
        logger.warning(
            "Model output very low trade_amount=%.6f → buy_qty=%.6f, sell_qty=%.6f (skipping trades)",
            plan.trade_amount,
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
                "buy",
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
    # Use cache-only mode during training to avoid loading Chronos2 on GPU
    cache_only = args.mode == "train"
    _ensure_forecasts(config, cache_only=cache_only)
    loaded = None if config.force_retrain else _load_pretrained_policy(config)
    if loaded is None:
        data_module, policy, checkpoint_path = _train_policy(config, training_symbols=args.training_symbols)
    else:
        data_module, policy, checkpoint_path = loaded
    actions = _infer_actions(policy, data_module, config)
    _simulate(actions, data_module, args.window_hours)
    if args.mode in {"simulate", "train"}:
        return
    plan = _build_trading_plan(actions)
    if not plan:
        logger.warning("No trading plan generated")
        return
    _spawn_watchers(plan, args.dry_run)


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
