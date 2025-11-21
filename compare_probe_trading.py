#!/usr/bin/env python3
"""Compare normal trading vs probe trading strategy for both models."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hourlycryptomarketsimulator import (
    HourlyCryptoMarketSimulator,
    ProbeTradingSimulator,
    ProbeTradeConfig,
    SimulationConfig,
)
from hourlycryptotraining import HourlyCryptoDataModule, PolicyHeadConfig, TrainingConfig
from hourlycryptotraining.checkpoints import load_checkpoint
from hourlycryptotraining.model import HourlyCryptoPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the checkpoint with the lowest (most negative) validation loss from manifest.json."""
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return None

    manifest_path = checkpoint_dir / "manifest.json"

    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            checkpoints = manifest.get("checkpoints", [])
            if checkpoints:
                best = min(checkpoints, key=lambda x: x.get("val_loss", float("inf")))
                best_path = checkpoint_dir / best["path"]
                best_loss = best["val_loss"]

                if best_path.exists():
                    logger.info(
                        f"Best checkpoint: {best_path.name} (val_loss={best_loss:.6f})"
                    )
                    return best_path
        except Exception as e:
            logger.warning(f"Could not read manifest.json: {e}")

    logger.error(f"No valid checkpoints found in {checkpoint_dir}")
    return None


def load_policy_and_data(
    checkpoint_path: Path, symbol: str
) -> Tuple[HourlyCryptoPolicy, HourlyCryptoDataModule, TrainingConfig]:
    """Load policy and data module from checkpoint."""
    payload = load_checkpoint(checkpoint_path)

    # Build config
    config = TrainingConfig()
    config.dataset.symbol = symbol
    config.forecast_config.symbol = symbol

    # Load data module
    feature_columns = payload.get("feature_columns", [])
    data_module = HourlyCryptoDataModule(config.dataset)
    data_module.normalizer = payload["normalizer"]

    # Create and load policy
    payload_cfg = payload.get("config", {})
    price_offset_pct = payload_cfg.get("price_offset_pct", 0.0003)

    policy = HourlyCryptoPolicy(
        PolicyHeadConfig(
            input_dim=len(feature_columns),
            hidden_dim=payload_cfg.get("transformer_dim", 256),
            dropout=payload_cfg.get("transformer_dropout", 0.1),
            price_offset_pct=price_offset_pct,
            max_trade_qty=payload_cfg.get("max_trade_qty", 3.0),
            min_price_gap_pct=payload_cfg.get("min_price_gap_pct", 0.0003),
            num_heads=payload_cfg.get("transformer_heads", 8),
            num_layers=payload_cfg.get("transformer_layers", 4),
        )
    )
    policy.load_state_dict(payload["state_dict"], strict=False)

    logger.info(f"Loaded checkpoint: {checkpoint_path.name}")
    return policy, data_module, config


def generate_actions(
    policy: HourlyCryptoPolicy,
    data_module: HourlyCryptoDataModule,
    config: TrainingConfig,
    sequence_length: int = 256,
) -> pd.DataFrame:
    """Generate trading actions from the policy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device).eval()

    features = data_module.frame[list(data_module.feature_columns)].to_numpy(
        dtype="float32"
    )
    norm = data_module.normalizer.transform(features)
    closes = data_module.frame["close"].to_numpy(dtype="float32")
    ref_close = data_module.frame["reference_close"].to_numpy(dtype="float32")
    chronos_high = data_module.frame["chronos_high"].to_numpy(dtype="float32")
    chronos_low = data_module.frame["chronos_low"].to_numpy(dtype="float32")
    timestamps = data_module.frame["timestamp"].to_numpy()

    rows = []
    with torch.no_grad():
        for idx in range(sequence_length, len(data_module.frame) + 1):
            window = slice(idx - sequence_length, idx)
            feat = torch.from_numpy(norm[window]).unsqueeze(0).to(device)
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


def run_simulations(
    checkpoint_path: Path, symbol: str, window_hours: int
) -> dict:
    """Run both normal and probe trading simulations for a checkpoint."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Simulating: {checkpoint_path.parent.name}/{checkpoint_path.name}")
    logger.info(f"{'='*80}\n")

    # Load policy and generate actions
    policy, data_module, config = load_policy_and_data(checkpoint_path, symbol)
    actions = generate_actions(policy, data_module, config)

    # Prepare bars
    bars = data_module.frame[
        data_module.frame["timestamp"].isin(actions["timestamp"])
    ][["timestamp", "high", "low", "close"]].copy()

    if window_hours:
        cutoff = bars["timestamp"].max() - pd.Timedelta(hours=window_hours)
        bars = bars[bars["timestamp"] >= cutoff]
        actions = actions[actions["timestamp"] >= cutoff]

    # Run normal simulation
    logger.info("Running NORMAL trading simulation...")
    normal_sim = HourlyCryptoMarketSimulator(SimulationConfig(symbol=symbol))
    normal_result = normal_sim.run(bars, actions)

    logger.info(
        f"Normal Trading: return={normal_result.metrics['total_return']*100:.2f}% "
        f"sortino={normal_result.metrics['sortino']:.2f} "
        f"cash=${normal_result.final_cash:.2f} "
        f"inventory={normal_result.final_inventory:.4f}"
    )

    # Run probe trading simulation
    logger.info("\nRunning PROBE trading simulation...")
    probe_sim = ProbeTradingSimulator(
        SimulationConfig(symbol=symbol),
        ProbeTradeConfig(
            probe_trade_amount=0.01,  # 1% probe trades
            lookback_trades=2,
            min_avg_pnl_pct=0.0,
        ),
    )
    probe_result = probe_sim.run(bars, actions)

    logger.info(
        f"Probe Trading: return={probe_result.metrics['total_return']*100:.2f}% "
        f"sortino={probe_result.metrics['sortino']:.2f} "
        f"cash=${probe_result.final_cash:.2f} "
        f"inventory={probe_result.final_inventory:.4f}"
    )
    logger.info(
        f"  Probe trades: {probe_result.metrics.get('total_probe_trades', 0)}, "
        f"Full trades: {probe_result.metrics.get('total_full_trades', 0)}"
    )

    return {
        "checkpoint_dir": checkpoint_path.parent.name,
        "checkpoint_file": checkpoint_path.name,
        "normal_return": normal_result.metrics["total_return"] * 100,
        "normal_sortino": normal_result.metrics["sortino"],
        "normal_cash": normal_result.final_cash,
        "normal_inventory": normal_result.final_inventory,
        "probe_return": probe_result.metrics["total_return"] * 100,
        "probe_sortino": probe_result.metrics["sortino"],
        "probe_cash": probe_result.final_cash,
        "probe_inventory": probe_result.final_inventory,
        "probe_trades": probe_result.metrics.get("total_probe_trades", 0),
        "full_trades": probe_result.metrics.get("total_full_trades", 0),
        "improvement": (
            probe_result.metrics["total_return"] - normal_result.metrics["total_return"]
        )
        * 100,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare normal vs probe trading strategies"
    )
    parser.add_argument(
        "--checkpoint1",
        type=str,
        default="hourlycryptotraining/checkpoints_256ctx_multipair/hourlycrypto_20251117_190541",
        help="First checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint2",
        type=str,
        default="hourlycryptotraining/checkpoints_256ctx_multipair/hourlycrypto_20251116_051625",
        help="Second checkpoint directory",
    )
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Trading symbol")
    parser.add_argument(
        "--window-hours", type=int, default=120, help="Simulation window (hours)"
    )

    args = parser.parse_args()

    # Find best checkpoints
    ckpt1 = find_best_checkpoint(Path(args.checkpoint1))
    ckpt2 = find_best_checkpoint(Path(args.checkpoint2))

    if not ckpt1 or not ckpt2:
        logger.error("Failed to find checkpoints")
        return 1

    # Run simulations
    results1 = run_simulations(ckpt1, args.symbol, args.window_hours)
    results2 = run_simulations(ckpt2, args.symbol, args.window_hours)

    # Display results
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}\n")

    # Create comparison DataFrames
    logger.info("Model 1: " + results1["checkpoint_dir"])
    logger.info(f"  Normal Trading: {results1['normal_return']:.2f}% return")
    logger.info(f"  Probe Trading:  {results1['probe_return']:.2f}% return")
    logger.info(f"  Improvement:    {results1['improvement']:+.2f}%")
    logger.info(
        f"  Trades: {results1['probe_trades']} probe, {results1['full_trades']} full"
    )

    logger.info("\nModel 2: " + results2["checkpoint_dir"])
    logger.info(f"  Normal Trading: {results2['normal_return']:.2f}% return")
    logger.info(f"  Probe Trading:  {results2['probe_return']:.2f}% return")
    logger.info(f"  Improvement:    {results2['improvement']:+.2f}%")
    logger.info(
        f"  Trades: {results2['probe_trades']} probe, {results2['full_trades']} full"
    )

    # Determine winners
    logger.info(f"\n{'='*80}")
    logger.info("WINNERS")
    logger.info(f"{'='*80}\n")

    best_normal = (
        results1 if results1["normal_return"] > results2["normal_return"] else results2
    )
    best_probe = (
        results1 if results1["probe_return"] > results2["probe_return"] else results2
    )

    logger.info(
        f"Best Normal Trading:  {best_normal['checkpoint_dir']} ({best_normal['normal_return']:.2f}%)"
    )
    logger.info(
        f"Best Probe Trading:   {best_probe['checkpoint_dir']} ({best_probe['probe_return']:.2f}%)"
    )

    logger.info(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
