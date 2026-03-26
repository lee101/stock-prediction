#!/usr/bin/env python3
"""Evaluate SAP checkpoints on binary-fill marketsim across multiple windows.

Usage:
    python -m sharpnessadjustedproximalpolicy2.evaluate --checkpoint path/to/epoch_005.pt --symbol DOGEUSD
    python -m sharpnessadjustedproximalpolicy2.evaluate --checkpoint-dir path/to/run/ --symbol DOGEUSD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.model import PolicyConfig, build_policy
from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    compute_loss_by_type,
    simulate_hourly_trades_binary,
)
from src.torch_load_utils import torch_load_compat


def evaluate_checkpoint(
    ckpt_path: Path,
    symbol: str,
    windows: list[float] | None = None,
    lag: int = 2,
) -> dict:
    """Evaluate a single checkpoint on binary fills across multiple windows."""
    if windows is None:
        windows = [0.1, 0.25, 0.5, 0.75, 1.0]

    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    feature_cols = ckpt.get("feature_columns", [])

    ds_cfg = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly") / "crypto",
        forecast_cache_root=Path("binanceneural") / "forecast_cache",
        sequence_length=config.get("sequence_length", 72),
        validation_days=70,
        cache_only=True,
    )
    dm = BinanceHourlyDataModule(ds_cfg)

    pc = PolicyConfig(
        input_dim=len(dm.feature_columns),
        hidden_dim=config.get("transformer_dim", 256),
        num_heads=config.get("transformer_heads", 8),
        num_layers=config.get("transformer_layers", 4),
        model_arch=config.get("model_arch", "classic"),
        max_len=max(config.get("sequence_length", 72), 32),
        use_flex_attention=False,
    )
    model = build_policy(pc)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    val_loader = dm.val_dataloader(batch_size=1)
    scale = config.get("trade_amount_scale", 100.0)
    fee = config.get("maker_fee", 0.001)

    results = {"checkpoint": str(ckpt_path), "symbol": symbol, "epoch": ckpt.get("epoch", 0)}
    sharpness_info = ckpt.get("sharpness", {})
    results["sharpness_ema"] = sharpness_info.get("ema", 0)
    results["step_scale"] = sharpness_info.get("step_scale", sharpness_info.get("lr_scale", 1))
    results["lr_scale"] = results["step_scale"]

    all_returns = []
    with torch.inference_mode():
        for batch in val_loader:
            features = batch["features"].to(device)
            highs = batch["high"].to(device)
            lows = batch["low"].to(device)
            closes = batch["close"].to(device)
            opens = batch["open"].to(device) if "open" in batch else None
            ref_close = batch["reference_close"].to(device)
            ch_high = batch["chronos_high"].to(device)
            ch_low = batch["chronos_low"].to(device)

            outputs = model(features)
            actions = model.decode_actions(outputs, reference_close=ref_close, chronos_high=ch_high, chronos_low=ch_low)

            sim = simulate_hourly_trades_binary(
                highs=highs, lows=lows, closes=closes, opens=opens,
                buy_prices=actions["buy_price"], sell_prices=actions["sell_price"],
                trade_intensity=actions["trade_amount"] / scale,
                buy_trade_intensity=actions["buy_amount"] / scale,
                sell_trade_intensity=actions["sell_amount"] / scale,
                maker_fee=fee,
                initial_cash=1.0,
                can_short=False, can_long=True,
                max_leverage=config.get("max_leverage", 1.0),
                fill_buffer_pct=config.get("fill_buffer_pct", 0.0005),
                margin_annual_rate=config.get("margin_annual_rate", 0.0625),
                decision_lag_bars=lag,
            )
            all_returns.append(sim.returns.float().cpu())

    if not all_returns:
        results["error"] = "no batches"
        return results

    returns = torch.cat(all_returns, dim=0)
    T = returns.shape[-1]

    for frac in windows:
        start = int(T * (1 - frac))
        window_ret = returns[..., start:]
        _, score, sortino, annual_ret = compute_loss_by_type(
            window_ret, "sortino", periods_per_year=HOURLY_PERIODS_PER_YEAR,
            return_weight=0.08,
        )
        key = f"w{int(frac*100)}"
        results[f"{key}_sortino"] = sortino.mean().item()
        results[f"{key}_return"] = annual_ret.mean().item()
        results[f"{key}_score"] = score.mean().item()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--symbol", default="DOGEUSD")
    parser.add_argument("--lag", type=int, default=2)
    args = parser.parse_args()

    if args.checkpoint:
        paths = [Path(args.checkpoint)]
    elif args.checkpoint_dir:
        paths = sorted(Path(args.checkpoint_dir).glob("epoch_*.pt"))
    else:
        paths = sorted(Path("sharpnessadjustedproximalpolicy2/checkpoints").rglob("epoch_*.pt"))

    all_results = []
    for p in paths:
        print(f"Evaluating {p}...", flush=True)
        r = evaluate_checkpoint(p, args.symbol, lag=args.lag)
        all_results.append(r)
        w100_sort = r.get("w100_sortino", 0)
        w100_ret = r.get("w100_return", 0)
        print(f"  ep{r.get('epoch', '?')}: Sort={w100_sort:.3f} Ret={w100_ret:.4f} Sharp={r.get('sharpness_ema', 0):.3f}", flush=True)

    out = Path("sharpnessadjustedproximalpolicy2") / "eval_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out}", flush=True)

    # summary
    print(f"\n{'Checkpoint':<60} {'Sort':>8} {'Ret':>10} {'Sharp':>8}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x.get("w100_sortino", -999), reverse=True):
        ckpt = Path(r["checkpoint"]).parent.name + "/" + Path(r["checkpoint"]).name
        print(f"{ckpt:<60} {r.get('w100_sortino', 0):>8.3f} {r.get('w100_return', 0):>10.4f} {r.get('sharpness_ema', 0):>8.3f}")


if __name__ == "__main__":
    main()
