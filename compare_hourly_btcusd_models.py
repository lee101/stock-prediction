#!/usr/bin/env python3
"""Compare two hourly crypto models by running market simulations on BTCUSD."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

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

    # Try reading from manifest.json first (has actual negative losses)
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            checkpoints = manifest.get("checkpoints", [])
            if not checkpoints:
                logger.warning(f"No checkpoints in manifest.json")
            else:
                # min() correctly selects most negative val_loss (best) since loss = -score
                best = min(checkpoints, key=lambda x: x.get("val_loss", float("inf")))
                best_path = checkpoint_dir / best["path"]
                best_loss = best["val_loss"]

                if best_path.exists():
                    logger.info(f"Best checkpoint from manifest: {best_path.name} (val_loss={best_loss:.6f})")
                    return best_path
        except Exception as e:
            logger.warning(f"Could not read manifest.json: {e}, falling back to filename parsing")

    # Fallback: parse from filenames (handles negative losses correctly)
    checkpoints = list(checkpoint_dir.glob("epoch*_valloss*.pt"))
    if not checkpoints:
        logger.error(f"No checkpoints found in {checkpoint_dir}")
        return None

    best_ckpt = None
    best_loss = float("inf")

    for ckpt in checkpoints:
        try:
            # Extract loss from filename like "epoch0001_valloss-5.686103.pt"
            # Split on "valloss" and take everything after it (preserves negative sign)
            loss_str = ckpt.stem.split("valloss")[1]
            loss = float(loss_str)
            # Lower (more negative) losses are better since loss = -score
            if loss < best_loss:
                best_loss = loss
                best_ckpt = ckpt
        except (IndexError, ValueError) as e:
            logger.warning(f"Could not parse loss from {ckpt.name}: {e}")
            continue

    if best_ckpt:
        logger.info(f"Best checkpoint (fallback): {best_ckpt.name} (val_loss={best_loss:.6f})")

    return best_ckpt


def run_simulation(checkpoint_path: Path, symbol: str, window_hours: int = 24) -> dict:
    """Run market simulation for a given checkpoint."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running simulation for checkpoint: {checkpoint_path.parent.name}")
    logger.info(f"Checkpoint file: {checkpoint_path.name}")
    logger.info(f"{'='*80}\n")

    cmd = [
        sys.executable,
        "hourlycrypto/trade_stock_crypto_hourly.py",
        "--mode", "simulate",
        "--checkpoint-path", str(checkpoint_path),
        "--symbol", symbol,
        "--window-hours", str(window_hours),
        "--sequence-length", "256",
        "--log-level", "INFO",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )

        # Parse output for simulation results
        output = result.stdout + result.stderr

        # Extract metrics from log output
        metrics = {
            "checkpoint_dir": checkpoint_path.parent.name,
            "checkpoint_file": checkpoint_path.name,
            "normal_return": None,
            "normal_sortino": None,
            "normal_cash": None,
            "normal_inventory": None,
            "daily_pnl_probe_return": None,
            "daily_pnl_probe_sortino": None,
            "daily_pnl_probe_cash": None,
            "daily_pnl_probe_inventory": None,
            "probe_mode_pct": 0.0,
            "probe_switches": 0,
            "daily_improvement": 0.0,
        }

        import re

        for line in output.split("\n"):
            if "Simulation: total_return=" in line:
                # Parse normal simulation
                parts = line.split("Simulation:")[1].strip()
                for part in parts.split():
                    if part.startswith("total_return="):
                        metrics["normal_return"] = float(part.split("=")[1].replace("%", ""))
                    elif part.startswith("sortino="):
                        metrics["normal_sortino"] = float(part.split("=")[1])
                    elif part.startswith("final_cash="):
                        metrics["normal_cash"] = float(part.split("=")[1])
                    elif part.startswith("inventory="):
                        metrics["normal_inventory"] = float(part.split("=")[1])

            elif "Daily PnL Probe: total_return=" in line:
                # Parse daily PnL probe trading simulation
                # Format: "Daily PnL Probe: total_return=X.XX% sortino=Y.YY final_cash=ZZZZ.ZZ inventory=I.IIII (P.P% time in probe, N switches)"
                parts = line.split("Daily PnL Probe:")[1].strip()

                # Extract probe mode % and switches using regex
                probe_stats = re.search(r'\((\d+\.?\d*)%\s+time in probe,\s+(\d+)\s+switches\)', line)
                if probe_stats:
                    metrics["probe_mode_pct"] = float(probe_stats.group(1))
                    metrics["probe_switches"] = int(probe_stats.group(2))

                for part in parts.split():
                    if part.startswith("total_return="):
                        metrics["daily_pnl_probe_return"] = float(part.split("=")[1].replace("%", ""))
                    elif part.startswith("sortino="):
                        metrics["daily_pnl_probe_sortino"] = float(part.split("=")[1])
                    elif part.startswith("final_cash="):
                        metrics["daily_pnl_probe_cash"] = float(part.split("=")[1])
                    elif part.startswith("inventory="):
                        inv_str = part.split("=")[1]
                        metrics["daily_pnl_probe_inventory"] = float(inv_str.split("(")[0] if "(" in inv_str else inv_str)

            elif "Daily PnL Probe Improvement:" in line:
                parts = line.split("Daily PnL Probe Improvement:")[1].strip()
                improvement_str = parts.split("%")[0].strip()
                metrics["daily_improvement"] = float(improvement_str)

        logger.info(f"\n{checkpoint_path.parent.name} Results:")
        logger.info(f"  Normal Trading:    {metrics['normal_return']:.2f}% return, Sortino: {metrics['normal_sortino']:.2f}")
        logger.info(f"  Daily PnL Probe:   {metrics['daily_pnl_probe_return']:.2f}% return, Sortino: {metrics['daily_pnl_probe_sortino']:.2f}")
        logger.info(f"  Improvement:       {metrics['daily_improvement']:+.2f}%")
        logger.info(f"  Probe mode:        {metrics['probe_mode_pct']:.1f}% of time, {metrics['probe_switches']} switches")

        return metrics

    except subprocess.CalledProcessError as e:
        logger.error(f"Simulation failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return {
            "checkpoint_dir": checkpoint_path.parent.name,
            "checkpoint_file": checkpoint_path.name,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Compare two hourly crypto model checkpoints")
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
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSD",
        help="Trading symbol to simulate",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Simulation window in hours",
    )

    args = parser.parse_args()

    # Find best checkpoints in each directory
    ckpt1_dir = Path(args.checkpoint1)
    ckpt2_dir = Path(args.checkpoint2)

    ckpt1 = find_best_checkpoint(ckpt1_dir)
    ckpt2 = find_best_checkpoint(ckpt2_dir)

    if not ckpt1 or not ckpt2:
        logger.error("Failed to find checkpoints in one or both directories")
        return 1

    # Run simulations
    results1 = run_simulation(ckpt1, args.symbol, args.window_hours)
    results2 = run_simulation(ckpt2, args.symbol, args.window_hours)

    # Compare results
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}\n")

    # Normal Trading Comparison
    logger.info("NORMAL TRADING:")
    logger.info(f"  Model 1 ({results1['checkpoint_dir']}): {results1['normal_return']:.2f}% return, Sortino: {results1['normal_sortino']:.2f}")
    logger.info(f"  Model 2 ({results2['checkpoint_dir']}): {results2['normal_return']:.2f}% return, Sortino: {results2['normal_sortino']:.2f}")

    if results1.get("normal_return") is not None and results2.get("normal_return") is not None:
        if results1["normal_return"] > results2["normal_return"]:
            logger.info(f"  🏆 Winner: {results1['checkpoint_dir']} (+{results1['normal_return'] - results2['normal_return']:.2f}%)")
        elif results2["normal_return"] > results1["normal_return"]:
            logger.info(f"  🏆 Winner: {results2['checkpoint_dir']} (+{results2['normal_return'] - results1['normal_return']:.2f}%)")
        else:
            logger.info(f"  🤝 TIE")

    # Daily PnL Probe Trading Comparison
    logger.info(f"\nDAILY PNL PROBE TRADING:")
    logger.info(f"  Model 1 ({results1['checkpoint_dir']}): {results1['daily_pnl_probe_return']:.2f}% return, Sortino: {results1['daily_pnl_probe_sortino']:.2f}")
    logger.info(f"    Probe mode: {results1['probe_mode_pct']:.1f}% of time, {results1['probe_switches']} switches")
    logger.info(f"    Improvement over normal: {results1['daily_improvement']:+.2f}%")

    logger.info(f"  Model 2 ({results2['checkpoint_dir']}): {results2['daily_pnl_probe_return']:.2f}% return, Sortino: {results2['daily_pnl_probe_sortino']:.2f}")
    logger.info(f"    Probe mode: {results2['probe_mode_pct']:.1f}% of time, {results2['probe_switches']} switches")
    logger.info(f"    Improvement over normal: {results2['daily_improvement']:+.2f}%")

    if results1.get("daily_pnl_probe_return") is not None and results2.get("daily_pnl_probe_return") is not None:
        if results1["daily_pnl_probe_return"] > results2["daily_pnl_probe_return"]:
            logger.info(f"  🏆 Winner: {results1['checkpoint_dir']} (+{results1['daily_pnl_probe_return'] - results2['daily_pnl_probe_return']:.2f}%)")
        elif results2["daily_pnl_probe_return"] > results1["daily_pnl_probe_return"]:
            logger.info(f"  🏆 Winner: {results2['checkpoint_dir']} (+{results2['daily_pnl_probe_return'] - results1['daily_pnl_probe_return']:.2f}%)")
        else:
            logger.info(f"  🤝 TIE")

    # Overall best strategy
    logger.info(f"\nBEST OVERALL STRATEGY:")
    all_strategies = [
        (results1['checkpoint_dir'], "Normal", results1['normal_return']),
        (results1['checkpoint_dir'], "Daily PnL Probe", results1['daily_pnl_probe_return']),
        (results2['checkpoint_dir'], "Normal", results2['normal_return']),
        (results2['checkpoint_dir'], "Daily PnL Probe", results2['daily_pnl_probe_return']),
    ]
    best_strategy = max(all_strategies, key=lambda x: x[2] if x[2] is not None else float('-inf'))
    logger.info(f"  🏆 {best_strategy[0]} with {best_strategy[1]} Trading: {best_strategy[2]:.2f}% return")

    logger.info(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
