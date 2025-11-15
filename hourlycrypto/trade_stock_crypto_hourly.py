from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from datetime import timedelta
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly Chronos-driven LINKUSD trading loop")
    parser.add_argument("--mode", choices=["train", "simulate", "trade"], default="trade")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Skip watcher spawning")
    parser.add_argument("--window-hours", type=int, default=24 * 14, help="Simulation lookback window")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--checkpoint-root", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _ensure_forecasts(config: TrainingConfig) -> pd.DataFrame:
    manager = DailyChronosForecastManager(config.forecast_config)
    return manager.ensure_latest()


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
    return cfg


def _train_policy(config: TrainingConfig) -> tuple[HourlyCryptoDataModule, HourlyCryptoPolicy, Optional[Path]]:
    data_module = HourlyCryptoDataModule(config.dataset)
    trainer = HourlyCryptoTrainer(config, data_module)
    artifacts = trainer.train()
    policy = HourlyCryptoPolicy(
        PolicyHeadConfig(
            input_dim=len(data_module.feature_columns),
            price_offset_pct=config.price_offset_pct,
            max_trade_qty=config.max_trade_qty,
        )
    )
    policy.load_state_dict(artifacts.state_dict)
    return data_module, policy, artifacts.best_checkpoint


def _load_pretrained_policy(config: TrainingConfig) -> Optional[Tuple[HourlyCryptoDataModule, HourlyCryptoPolicy, Path]]:
    ckpt_path = config.preload_checkpoint_path or find_best_checkpoint(config.checkpoint_root)
    if ckpt_path is None:
        logger.info("No pretrained checkpoint found under %s", config.checkpoint_root)
        return None
    payload = load_checkpoint(ckpt_path)
    feature_columns = payload.get("feature_columns") or list(config.dataset.feature_columns or [])
    dataset_cfg = replace(config.dataset, feature_columns=tuple(feature_columns))
    data_module = HourlyCryptoDataModule(dataset_cfg)
    data_module.normalizer = payload["normalizer"]
    policy = HourlyCryptoPolicy(
        PolicyHeadConfig(
            input_dim=len(feature_columns),
            price_offset_pct=config.price_offset_pct,
            max_trade_qty=config.max_trade_qty,
        )
    )
    policy.load_state_dict(payload["state_dict"])
    logger.info("Loaded pretrained checkpoint %s (val_loss=%.6f)", ckpt_path, payload.get("metrics", {}).get("loss", float("nan")))
    return data_module, policy, ckpt_path


def _infer_actions(
    policy: HourlyCryptoPolicy,
    data_module: HourlyCryptoDataModule,
    config: TrainingConfig,
) -> pd.DataFrame:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = policy.to(device).eval()
    features = data_module.frame[data_module.feature_columns].to_numpy(dtype="float32")
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
                    "buy_amount": float(decoded["buy_amount"][0, -1].item()),
                    "sell_amount": float(decoded["sell_amount"][0, -1].item()),
                    "close": float(close_tensor[0, -1].item()),
                    "low": float(real_low[idx - 1]),
                    "high": float(real_high[idx - 1]),
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
        buy_amount=float(latest["buy_amount"]),
        sell_amount=float(latest["sell_amount"]),
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


def _spawn_watchers(plan: TradingPlan, dry_run: bool) -> None:
    symbol = "LINKUSD"
    cash = _available_cash()
    inventory = _current_inventory(symbol)
    max_buy_qty = cash / plan.buy_price if plan.buy_price > 0 else 0.0
    buy_qty = min(plan.buy_amount, max(0.0, max_buy_qty))
    sell_qty = min(plan.sell_amount, max(0.0, inventory))
    logger.info(
        "Trading plan %s buy=%.4f@%.4f sell=%.4f@%.4f cash=%.2f inventory=%.4f",
        plan.timestamp,
        buy_qty,
        plan.buy_price,
        sell_qty,
        plan.sell_price,
        cash,
        inventory,
    )
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
            )
        except Exception as exc:
            logger.error("Failed to spawn entry watcher: %s", exc)
    if sell_qty > 0:
        try:
            spawn_close_position_at_maxdiff_takeprofit(
                symbol,
                "buy",
                plan.sell_price,
                entry_strategy="hourlycrypto",
            )
        except Exception as exc:
            logger.error("Failed to spawn exit watcher: %s", exc)


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    config = _build_training_config(args)
    _ensure_forecasts(config)
    loaded = None if config.force_retrain else _load_pretrained_policy(config)
    if loaded is None:
        data_module, policy, checkpoint_path = _train_policy(config)
    else:
        data_module, policy, checkpoint_path = loaded
    actions = _infer_actions(policy, data_module, config)
    _simulate(actions, data_module, args.window_hours)
    if args.mode == "simulate":
        return
    plan = _build_trading_plan(actions)
    if not plan:
        logger.warning("No trading plan generated")
        return
    _spawn_watchers(plan, args.dry_run)


if __name__ == "__main__":
    main()
