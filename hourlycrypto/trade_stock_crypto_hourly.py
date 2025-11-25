from __future__ import annotations

import argparse
import logging
import sys
import time
from copy import deepcopy
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
from alpaca_data_wrapper import append_recent_crypto_data
from alpaca_wrapper import _get_min_order_notional
from src.process_utils import (
    spawn_close_position_at_maxdiff_takeprofit,
    spawn_open_position_at_maxdiff_takeprofit,
)
from src.price_guard import enforce_gap, record_buy, record_sell

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
    buy_base_pct: float
    sell_base_pct: float
    buy_span_multiplier: float = 0.0
    sell_span_multiplier: float = 0.0
    buy_max_pct: Optional[float] = None
    sell_max_pct: Optional[float] = None

    def build_tensor(
        self,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        span = torch.clamp(
            (chronos_high - chronos_low) / reference_close.clamp(min=1e-6),
            min=0.0,
        )

        buy_offsets = torch.full_like(reference_close, self.buy_base_pct)
        sell_offsets = torch.full_like(reference_close, self.sell_base_pct)

        if self.buy_span_multiplier:
            buy_offsets = buy_offsets + self.buy_span_multiplier * span
        if self.sell_span_multiplier:
            sell_offsets = sell_offsets + self.sell_span_multiplier * span

        if self.buy_max_pct is not None:
            buy_offsets = torch.clamp(buy_offsets, max=self.buy_max_pct)
        if self.sell_max_pct is not None:
            sell_offsets = torch.clamp(sell_offsets, max=self.sell_max_pct)

        buy_offsets = torch.clamp(buy_offsets, min=1e-6)
        sell_offsets = torch.clamp(sell_offsets, min=1e-6)
        return buy_offsets, sell_offsets


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
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (legacy flag; default is already off)")
    parser.add_argument("--enable-compile", action="store_true", help="Enable torch.compile (default: disabled)")
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
        "--buy-price-offset-pct",
        type=float,
        default=None,
        help="Override buy-side base limit offset (defaults to price-offset-pct)",
    )
    parser.add_argument(
        "--sell-price-offset-pct",
        type=float,
        default=None,
        help="Override sell-side base limit offset (defaults to 1.2x price-offset-pct)",
    )
    parser.add_argument(
        "--price-offset-span-multiplier",
        type=float,
        default=0.15,
        help="Linear multiplier on (Chronos_high - Chronos_low)/reference_close to widen offsets",
    )
    parser.add_argument(
        "--buy-price-offset-span-multiplier",
        type=float,
        default=None,
        help="Override buy-side span multiplier (defaults to price-offset-span-multiplier)",
    )
    parser.add_argument(
        "--sell-price-offset-span-multiplier",
        type=float,
        default=None,
        help="Override sell-side span multiplier (defaults to price-offset-span-multiplier)",
    )
    parser.add_argument(
        "--price-offset-max-pct",
        type=float,
        default=0.003,
        help="Upper clamp for the effective offset (0.003 = 0.3%%). Use 0 to disable",
    )
    parser.add_argument(
        "--enable-probe-sim",
        action="store_true",
        help="Run the daily PnL probe variant in simulation (default: off)",
    )
    parser.add_argument(
        "--buy-price-offset-max-pct",
        type=float,
        default=None,
        help="Override buy-side offset cap (defaults to price-offset-max-pct)",
    )
    parser.add_argument(
        "--sell-price-offset-max-pct",
        type=float,
        default=None,
        help="Override sell-side offset cap (defaults to price-offset-max-pct)",
    )
    parser.add_argument(
        "--retrain-to-keep-up-to-date",
        action="store_true",
        help="Refresh latest hourly data, quick full-data retrain, eval vs baseline, promote best, then trade",
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=10,
        help="Days of most recent hourly data to append before retrain (default 10)",
    )
    parser.add_argument(
        "--retrain-epochs",
        type=int,
        default=1,
        help="Epochs for quick catch-up retrain (default 1)",
    )
    parser.add_argument(
        "--retrain-eval-window-hours",
        type=int,
        default=72,
        help="Window (hours) for simulator comparison between old/new checkpoint (default 72)",
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
    # Default: no torch.compile for stability; enable only if requested
    cfg.use_compile = False
    if args.enable_compile:
        cfg.use_compile = True
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


def _eval_checkpoint_total_return(ckpt_path: Path, base_config: TrainingConfig, window_hours: int) -> float:
    """Load checkpoint and compute total_return over given window (using current data)."""
    cfg = replace(base_config)
    cfg.preload_checkpoint_path = ckpt_path
    cfg.force_retrain = False
    data_module = HourlyCryptoDataModule(cfg.dataset)
    loaded = _load_pretrained_policy(cfg, price_offset_override=cfg.price_offset_pct)
    if loaded is None:
        raise RuntimeError(f"Could not load checkpoint {ckpt_path}")
    data_module, policy, _ = loaded
    actions = _infer_actions(policy, data_module, cfg)
    result, _ = _simulate(actions, data_module, window_hours)
    if result is None:
        return float("-inf")
    return float(result.metrics.get("total_return", float("-inf")))


def _quick_retrain_latest(config: TrainingConfig, args: argparse.Namespace, eval_window_hours: int) -> tuple[Optional[Path], float]:
    """Refresh model on full data (no val) for one/few epochs and evaluate."""
    cfg = deepcopy(config)
    cfg.epochs = max(1, int(getattr(args, "retrain_epochs", 1)))
    cfg.dataset.val_fraction = 0.0
    cfg.dataset.validation_days = 0
    cfg.force_retrain = True
    if args.checkpoint_path:
        cfg.preload_checkpoint_path = Path(args.checkpoint_path)
    cfg.run_name = cfg.run_name or f"hourlycrypto_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    cfg.checkpoint_root = Path("hourlycryptotraining") / "checkpoints_256ctx_autorefresh"

    data_module = HourlyCryptoDataModule(cfg.dataset)
    trainer = HourlyCryptoTrainer(cfg, data_module)
    artifacts = trainer.train()
    best_ckpt = artifacts.best_checkpoint or (Path(cfg.checkpoint_root) / cfg.run_name / "epoch0001_valloss-autorefresh.pt")

    # Evaluate
    policy = HourlyCryptoPolicy(
        PolicyHeadConfig(
            input_dim=len(data_module.feature_columns),
            hidden_dim=cfg.transformer_dim,
            dropout=cfg.transformer_dropout,
            price_offset_pct=cfg.price_offset_pct,
            max_trade_qty=cfg.max_trade_qty,
            min_price_gap_pct=cfg.min_price_gap_pct,
            num_heads=cfg.transformer_heads,
            num_layers=cfg.transformer_layers,
        )
    )
    policy.load_state_dict(artifacts.state_dict, strict=False)
    actions = _infer_actions(policy, data_module, cfg)
    result, _ = _simulate(actions, data_module, eval_window_hours)
    score = float(result.metrics.get("total_return", float("-inf"))) if result else float("-inf")
    return best_ckpt, score

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
                offset_params.build_tensor(ref_tensor, high_tensor, low_tensor) if offset_params else None
            )
            decoded = policy.decode_actions(
                outputs,
                reference_close=ref_tensor,
                chronos_high=high_tensor,
                chronos_low=low_tensor,
                dynamic_offset_pct=offset_tensor,
            )
            ts = pd.Timestamp(timestamps[idx - 1])
            if offset_tensor is None:
                buy_off = sell_off = float(config.price_offset_pct)
            else:
                buy_off = float(offset_tensor[0][0, -1].item())
                sell_off = float(offset_tensor[1][0, -1].item())
            rows.append(
                {
                    "timestamp": ts,
                    "buy_price": float(decoded["buy_price"][0, -1].item()),
                    "sell_price": float(decoded["sell_price"][0, -1].item()),
                    "trade_amount": float(decoded["trade_amount"][0, -1].item()),
                    "buy_amount": float(decoded["buy_amount"][0, -1].item()),
                    "sell_amount": float(decoded["sell_amount"][0, -1].item()),
                    "buy_offset_pct": buy_off,
                    "sell_offset_pct": sell_off,
                }
            )
    return pd.DataFrame(rows)


def _simulate(actions: pd.DataFrame, data_module: HourlyCryptoDataModule, window_hours: int, args=None) -> None:
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

    # Optional daily PnL-based probe trading simulation (disabled by default)
    if args is not None and getattr(args, "enable_probe_sim", False):
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


def _current_avg_price(symbol: str) -> float:
    """Return average entry price for current position, or 0 if none/error."""
    try:
        positions = alpaca_wrapper.get_all_positions()
    except Exception:
        return 0.0
    for position in positions:
        if position.symbol.upper() == symbol.upper():
            try:
                return float(getattr(position, "avg_entry_price", 0.0) or 0.0)
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


def _latest_quote(symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[pd.Timestamp]]:
    """Fetch the freshest bid/ask for a symbol and derive midpoint.

    Returns (bid, ask, midpoint, timestamp). Midpoint is only computed when
    both bid and ask are positive to avoid treating single-sided books as
    tradable midpoints.
    """

    try:
        quote = alpaca_wrapper.latest_data(symbol)
    except Exception as exc:  # pragma: no cover - network/API issues
        logger.warning("Failed to fetch latest quote for %s: %s", symbol, exc)
        return None, None, None, None

    bid = getattr(quote, "bid_price", None) or 0.0
    ask = getattr(quote, "ask_price", None) or 0.0
    ts_attr = getattr(quote, "timestamp", None) or getattr(quote, "time", None)
    ts = pd.to_datetime(ts_attr, utc=True, errors="coerce") if ts_attr is not None else None

    bid_val = float(bid) if bid and bid > 0 else None
    ask_val = float(ask) if ask and ask > 0 else None
    midpoint = (bid_val + ask_val) / 2.0 if bid_val is not None and ask_val is not None else None
    return bid_val, ask_val, midpoint, ts


def _adjust_for_maker_liquidity(
    plan: TradingPlan,
    bid: Optional[float],
    ask: Optional[float],
    midpoint: Optional[float],
) -> TradingPlan:
    """Clamp limits to the live top-of-book to avoid taker executions.

    * Buy price -> latest bid (if available)
    * Sell price -> latest ask (if available)

    This keeps orders resting on the book (maker) instead of crossing.
    """

    if bid is None and ask is None and midpoint is None:
        return plan

    buy_price = plan.buy_price
    sell_price = plan.sell_price

    if bid is not None and bid > 0:
        buy_price = min(plan.buy_price, bid)
    elif midpoint is not None and midpoint > 0:
        buy_price = min(plan.buy_price, midpoint)

    if ask is not None and ask > 0:
        sell_price = ask  # hit top-of-book for maker posting
    elif midpoint is not None and midpoint > 0:
        sell_price = min(plan.sell_price, midpoint)

    if buy_price != plan.buy_price or sell_price != plan.sell_price:
        logger.info(
            "Maker guard adjusted plan: buy %.4f->%.4f sell %.4f->%.4f (bid=%s ask=%s mid=%s)",
            plan.buy_price,
            buy_price,
            plan.sell_price,
            sell_price,
            f"{bid:.4f}" if bid is not None else "n/a",
            f"{ask:.4f}" if ask is not None else "n/a",
            f"{midpoint:.4f}" if midpoint is not None else "n/a",
        )

    return replace(plan, buy_price=buy_price, sell_price=sell_price)


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


def _current_avg_price(symbol: str) -> float:
    """Return average entry price for current position, or 0 if none/error."""
    try:
        positions = alpaca_wrapper.get_all_positions()
    except Exception:
        return 0.0
    for position in positions:
        if _normalize_symbol(position.symbol) == _normalize_symbol(symbol):
            try:
                return float(getattr(position, "avg_entry_price", 0.0) or 0.0)
            except Exception:
                return 0.0
    return 0.0


def _current_max_takeprofit(symbol: str) -> float:
    """Max existing TP (sell) limit price for the symbol."""
    try:
        open_orders = alpaca_wrapper.get_open_orders()
    except Exception:
        return 0.0
    norm = _normalize_symbol(symbol)
    tps = []
    for o in open_orders:
        try:
            if _normalize_symbol(o.symbol) != norm:
                continue
            if getattr(o, "side", "").lower() != "sell":
                continue
            lp = getattr(o, "limit_price", None)
            if lp is None:
                continue
            tps.append(float(lp))
        except Exception:
            continue
    return max(tps) if tps else 0.0


def _spawn_watchers(plan: TradingPlan, dry_run: bool, symbol: str) -> None:
    symbol = symbol.upper()

    # Cancel existing orders before placing new ones
    if not dry_run:
        _cancel_existing_orders(symbol)

    cash = _available_cash()
    min_notional = _get_min_order_notional(symbol)
    inventory = _current_inventory(symbol)
    avg_price = _current_avg_price(symbol)
    existing_tp = _current_max_takeprofit(symbol)

    # Pull freshest quote to bias limits away from taker execution
    bid, ask, midpoint, quote_ts = _latest_quote(symbol)
    if bid is not None or ask is not None or midpoint is not None:
        ts_str = quote_ts.isoformat() if quote_ts is not None else "n/a"
        logger.info(
            "Latest live quote %s bid=%s ask=%s mid=%s",
            ts_str,
            f"{bid:.4f}" if bid is not None else "n/a",
            f"{ask:.4f}" if ask is not None else "n/a",
            f"{midpoint:.4f}" if midpoint is not None else "n/a",
        )
    plan = _adjust_for_maker_liquidity(plan, bid, ask, midpoint)

    buy_amt = max(0.0, min(1.0, plan.buy_amount))
    sell_amt = max(0.0, min(1.0, plan.sell_amount))
    max_buy_qty = cash / plan.buy_price if plan.buy_price > 0 else 0.0
    buy_qty = buy_amt * max(0.0, max_buy_qty)
    sell_qty = sell_amt * max(0.0, inventory)

    # Enforce broker minimum notional (crypto ~$10 on Alpaca) to avoid rejected orders
    if buy_qty > 0 and plan.buy_price > 0:
        min_buy_qty = min_notional / plan.buy_price
        if buy_qty < min_buy_qty:
            logger.info(
                "Raising buy_qty to meet $%.2f min notional (%.8f -> %.8f @ %.4f)",
                min_notional,
                buy_qty,
                min_buy_qty,
                plan.buy_price,
            )
            buy_qty = min_buy_qty
    if sell_qty > 0 and plan.sell_price > 0:
        min_sell_qty = min_notional / plan.sell_price
        if sell_qty < min_sell_qty:
            logger.info(
                "Raising sell_qty to meet $%.2f min notional (%.8f -> %.8f @ %.4f)",
                min_notional,
                sell_qty,
                min_sell_qty,
                plan.sell_price,
            )
            sell_qty = min_sell_qty

    # If selling most of the position (>95%), sell ALL to avoid tiny residuals
    # that can block future buy orders
    if inventory > 0 and sell_qty >= inventory * 0.95:
        sell_qty = inventory

    # SAFETY CHECK: Ensure sell_price > buy_price with minimum 3 basis points spread
    # This prevents inverted prices and unprofitable trading
    MIN_SPREAD_PCT = 0.0003  # 3 basis points = 0.03%
    required_min_sell = plan.buy_price * (1.0 + MIN_SPREAD_PCT)
    actual_spread_pct = ((plan.sell_price - plan.buy_price) / plan.buy_price * 100) if plan.buy_price > 0 else 0

    # Extra guards:
    # 1) do not set take-profit below avg entry + small profit
    # 2) do not regress below any existing TP order
    # 3) if buy price is at/above TP, lift TP to maintain a 10bp gap above buy
    PROFIT_FLOOR_PCT = max(MIN_SPREAD_PCT, 0.001)  # at least 0.1% above avg entry
    if inventory > 0 and avg_price > 0:
        min_exit_price = avg_price * (1.0 + PROFIT_FLOOR_PCT)
        min_exit_price = max(min_exit_price, existing_tp)
        if plan.sell_price < min_exit_price:
            logger.warning(
                "Adjusting sell_price up to avoid loss or TP regression: requested=%.4f min_exit=%.4f avg=%.4f inv=%.6f existing_tp=%.4f",
                plan.sell_price,
                min_exit_price,
                avg_price,
                inventory,
                existing_tp,
            )
            plan = replace(plan, sell_price=min_exit_price)
            actual_spread_pct = ((plan.sell_price - plan.buy_price) / plan.buy_price * 100) if plan.buy_price > 0 else 0
            required_min_sell = max(required_min_sell, min_exit_price)

    if existing_tp > 0 and buy_qty > 0 and plan.buy_price >= existing_tp * (1 - 0.0005):
        adjusted_tp = max(plan.sell_price, plan.buy_price * 1.001, existing_tp * 1.001)
        logger.warning(
            "Adjusting TP to maintain 10bp gap above buy: buy=%.4f prev_tp=%.4f -> new_tp=%.4f",
            plan.buy_price,
            existing_tp,
            adjusted_tp,
        )
        plan = replace(plan, sell_price=adjusted_tp)
        actual_spread_pct = ((plan.sell_price - plan.buy_price) / plan.buy_price * 100) if plan.buy_price > 0 else 0
        required_min_sell = max(required_min_sell, adjusted_tp)

    # Guard 4: one-hour anti-inversion window based on recent trades
    adj_buy_price, adj_sell_price = enforce_gap(symbol, plan.buy_price, plan.sell_price, min_gap_pct=0.001)
    if adj_buy_price != plan.buy_price or adj_sell_price != plan.sell_price:
        logger.warning(
            "Price guard adjusted plan for %s: buy %.4f->%.4f sell %.4f->%.4f (1h window)",
            symbol,
            plan.buy_price,
            adj_buy_price,
            plan.sell_price,
            adj_sell_price,
        )
        plan = replace(plan, buy_price=adj_buy_price, sell_price=adj_sell_price)
        actual_spread_pct = ((plan.sell_price - plan.buy_price) / plan.buy_price * 100) if plan.buy_price > 0 else 0
        required_min_sell = max(required_min_sell, plan.buy_price * (1.0 + MIN_SPREAD_PCT), plan.sell_price)

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

    # Enforce broker min notional (Alpaca crypto ~$10)
    if buy_qty * plan.buy_price < min_notional:
        needed_qty = min_notional / plan.buy_price
        logger.info(
            "Raising buy_qty to meet $%.2f min notional (%.8f -> %.8f @ %.4f)",
            min_notional,
            buy_qty,
            needed_qty,
            plan.buy_price,
        )
        buy_qty = needed_qty
    if sell_qty * plan.sell_price < min_notional and sell_qty > 0:
        needed_qty = min_notional / plan.sell_price
        logger.info(
            "Raising sell_qty to meet $%.2f min notional (%.8f -> %.8f @ %.4f)",
            min_notional,
            sell_qty,
            needed_qty,
            plan.sell_price,
        )
        sell_qty = min(sell_qty + (needed_qty - sell_qty), inventory)

    # Warn if model is outputting very low trade intensity; still proceed if we have non‑zero qty after safeguards
    if max(buy_amt, sell_amt) < 0.01:
        logger.warning(
            "Model output very low trade_amount=%.6f/%.6f → buy_qty=%.6f, sell_qty=%.6f",
            plan.buy_amount,
            plan.sell_amount,
            buy_qty,
            sell_qty,
        )
        if not dry_run and max(buy_qty, sell_qty) <= 0:
            logger.info("No trades will be placed because effective qty is zero")
            return

    logger.info(
        "Spawning watchers: inv=%.6f buy_qty=%.6f sell_qty=%.6f buy_price=%.4f sell_price=%.4f",
        inventory,
        buy_qty,
        sell_qty,
        plan.buy_price,
        plan.sell_price,
    )
    if dry_run:
        logger.info("Dry-run: skipping watcher spawn")
        return
    if buy_qty > 0:
        try:
            target_buy_qty = buy_qty + inventory  # aim for current + new so remaining_qty equals desired add-on
            logger.info(
                "Launching entry watcher: target_total=%.6f (add=%.6f, inv=%.6f) @ %.4f",
                target_buy_qty,
                buy_qty,
                inventory,
                plan.buy_price,
            )
            record_buy(symbol, plan.buy_price)
            spawn_open_position_at_maxdiff_takeprofit(
                symbol,
                "buy",
                plan.buy_price,
                target_buy_qty,
                tolerance_pct=0.0008,
                expiry_minutes=90,
                entry_strategy="hourlycrypto",
                force_immediate=False,  # Wait for price to reach limit to stay maker
            )
        except Exception as exc:
            logger.error("Failed to spawn entry watcher: %s", exc)
    if sell_qty > 0:
        try:
            logger.info(
                "Launching exit watcher: target_qty=%.6f @ %.4f (position %.6f)",
                sell_qty,
                plan.sell_price,
                inventory,
            )
            record_sell(symbol, plan.sell_price)
            spawn_close_position_at_maxdiff_takeprofit(
                symbol,
                "buy",  # entry side was buy (long); exit watcher expects entry side
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
    # Build asymmetric offset parameters (enabled by default; sell side slightly wider)
    buy_base = args.buy_price_offset_pct if args.buy_price_offset_pct is not None else config.price_offset_pct
    sell_base_default = config.price_offset_pct * 1.2
    sell_base = args.sell_price_offset_pct if args.sell_price_offset_pct is not None else sell_base_default

    buy_span = (
        args.buy_price_offset_span_multiplier
        if args.buy_price_offset_span_multiplier is not None
        else args.price_offset_span_multiplier or 0.0
    )
    sell_span = (
        args.sell_price_offset_span_multiplier
        if args.sell_price_offset_span_multiplier is not None
        else args.price_offset_span_multiplier or 0.0
    )

    def _cap(val_override, shared):
        if val_override is not None:
            return val_override if val_override > 0 else None
        if shared is not None and shared > 0:
            return shared
        return None

    buy_max = _cap(args.buy_price_offset_max_pct, args.price_offset_max_pct)
    sell_max = _cap(args.sell_price_offset_max_pct, args.price_offset_max_pct)

    offset_params = PriceOffsetParams(
        buy_base_pct=buy_base,
        sell_base_pct=sell_base,
        buy_span_multiplier=max(0.0, buy_span),
        sell_span_multiplier=max(0.0, sell_span),
        buy_max_pct=buy_max,
        sell_max_pct=sell_max,
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

    if args.retrain_to_keep_up_to_date and args.mode == "trade":
        symbols = list({(args.symbol or config.dataset.symbol).upper(), *[s.upper() for s in (args.training_symbols or [])]})
        append_recent_crypto_data(symbols, days=args.refresh_days)
        new_ckpt, new_return = _quick_retrain_latest(config, args, eval_window_hours=args.retrain_eval_window_hours)
        baseline_return = float("-inf")
        if args.checkpoint_path:
            try:
                baseline_return = _eval_checkpoint_total_return(Path(args.checkpoint_path), config, args.retrain_eval_window_hours)
            except Exception as exc:
                logger.warning("Baseline checkpoint evaluation failed: %s", exc)
        if new_ckpt and new_return >= baseline_return:
            logger.info("Promoting refreshed checkpoint %s (return=%.4f) over baseline (%.4f)", new_ckpt, new_return, baseline_return)
            args.checkpoint_path = str(new_ckpt)
            config.preload_checkpoint_path = Path(new_ckpt)
        else:
            logger.info("Keeping baseline checkpoint (%.4f) over refreshed (%.4f); latest refreshed: %s", baseline_return, new_return, new_ckpt)

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
