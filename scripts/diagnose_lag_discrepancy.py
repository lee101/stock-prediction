#!/usr/bin/env python3
"""Diagnose discrepancy between training sim and marketsim at lag=2."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from differentiable_loss_utils import simulate_hourly_trades_binary, _sortino_core
from src.torch_load_utils import torch_load_compat


def main():
    symbol = "BTCUSD"
    ckpt_path = "binanceneural/checkpoints/BTCUSD_lag2_s42_20260411_120354/epoch_001.pt"

    dataset_cfg = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=72,
        validation_days=70,
        forecast_horizons=(1, 24),
        cache_only=True,
    )
    data = BinanceHourlyDataModule(dataset_cfg)

    payload = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=len(data.feature_columns))
    cfg = payload.get("config", TrainingConfig())
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=len(data.feature_columns), state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # --- PATH 1: Training validation sim (binary, lag=2) ---
    print("=== PATH 1: Training-style binary sim on val batches ===")
    val_loader = torch.utils.data.DataLoader(
        data.val_dataset, batch_size=16, shuffle=False
    )
    all_returns = []
    with torch.inference_mode():
        for batch in val_loader:
            features = batch["features"]
            highs = batch["high"]
            lows = batch["low"]
            closes = batch["close"]
            reference_close = batch["reference_close"]
            chronos_high = batch["chronos_high"]
            chronos_low = batch["chronos_low"]

            outputs = model(features)
            actions = model.decode_actions(
                outputs,
                reference_close=reference_close,
                chronos_high=chronos_high,
                chronos_low=chronos_low,
            )
            sim = simulate_hourly_trades_binary(
                highs=highs,
                lows=lows,
                closes=closes,
                buy_prices=actions["buy_price"],
                sell_prices=actions["sell_price"],
                trade_intensity=actions["trade_amount"] / 100.0,
                buy_trade_intensity=actions["buy_amount"] / 100.0,
                sell_trade_intensity=actions["sell_amount"] / 100.0,
                maker_fee=0.001,
                initial_cash=1.0,
                max_leverage=1.0,
                decision_lag_bars=2,
                fill_buffer_pct=0.0005,
                margin_annual_rate=0.0625,
            )
            rets = sim.returns.float()
            all_returns.append(rets)

            # Show first batch details
            if len(all_returns) == 1:
                print(f"  Batch shape: {rets.shape}")
                mean_r = rets.mean().item()
                print(f"  Mean hourly return: {mean_r:.6f}")
                print(f"  Annualized: {mean_r * 8760:.2%}")
                vals = sim.portfolio_values.float()
                total_ret = (vals[..., -1] / vals[..., 0] - 1).mean().item()
                print(f"  Mean total return per sequence: {total_ret:.4%}")
                buy_fills = sim.buy_fill_probability.sum(dim=-1).mean().item()
                sell_fills = sim.sell_fill_probability.sum(dim=-1).mean().item()
                print(f"  Avg buy fills/seq: {buy_fills:.1f}, sell fills/seq: {sell_fills:.1f}")
                # Check actions
                print(f"  Buy prices range: [{actions['buy_price'].min():.2f}, {actions['buy_price'].max():.2f}]")
                print(f"  Sell prices range: [{actions['sell_price'].min():.2f}, {actions['sell_price'].max():.2f}]")
                print(f"  Ref close range: [{reference_close.min():.2f}, {reference_close.max():.2f}]")
                spread = (actions['sell_price'] - actions['buy_price']) / reference_close
                print(f"  Spread (sell-buy)/ref: mean={spread.mean():.4%}, min={spread.min():.4%}")

    all_rets = torch.cat(all_returns, dim=0)
    mean_ret = all_rets.mean().item()
    neg_rets = all_rets[all_rets < 0]
    ds = neg_rets.std().item() if len(neg_rets) > 0 else 1e-8
    sortino = mean_ret / ds * np.sqrt(8760)
    annual_ret = mean_ret * 8760
    print(f"\n  Overall: sortino={sortino:.2f}, annual_ret={annual_ret:.2%}")
    print(f"  N batches: {len(all_returns)}, total steps: {all_rets.numel()}")

    # --- PATH 2: Inference + Marketsim ---
    print("\n=== PATH 2: Inference + Marketsim ===")
    val_frame = data.val_dataset.frame.copy()
    print(f"  Val frame: {len(val_frame)} bars")

    actions_df = generate_actions_from_frame(
        model=model,
        frame=val_frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=72,
        horizon=1,
    )
    print(f"  Generated {len(actions_df)} actions")
    print(f"  Buy prices: [{actions_df['buy_price'].min():.2f}, {actions_df['buy_price'].max():.2f}]")
    print(f"  Sell prices: [{actions_df['sell_price'].min():.2f}, {actions_df['sell_price'].max():.2f}]")
    spread = (actions_df['sell_price'] - actions_df['buy_price']) / val_frame['reference_close'].iloc[71:].values[:len(actions_df)]
    print(f"  Spread: mean={spread.mean():.4%}")

    for lag in [0, 1, 2]:
        sim_cfg = SimulationConfig(
            maker_fee=0.001,
            fill_buffer_bps=5.0,
            decision_lag_bars=lag,
            max_hold_hours=24,
            initial_cash=10_000.0,
        )
        sim = BinanceMarketSimulator(sim_cfg)
        result = sim.run(val_frame, actions_df)
        m = result.metrics
        equity = result.combined_equity.values
        peak = np.maximum.accumulate(equity)
        dd = ((equity - peak) / peak).min()
        n_trades = sum(len(sr.trades) for sr in result.per_symbol.values())
        print(f"  lag={lag}: ret={m['total_return']:.2%} sort={m['sortino']:.2f} dd={dd:.2%} trades={n_trades}")

    # --- PATH 3: Run training binary sim on FULL validation (not batched) ---
    print("\n=== PATH 3: Full-sequence training sim (first 200 bars) ===")
    frame = val_frame.iloc[:200].copy()
    features = data.normalizer.transform(
        frame[list(data.feature_columns)].to_numpy(dtype=np.float32)
    )
    features_t = torch.from_numpy(features).unsqueeze(0)
    highs_t = torch.from_numpy(frame["high"].to_numpy(dtype=np.float32)).unsqueeze(0)
    lows_t = torch.from_numpy(frame["low"].to_numpy(dtype=np.float32)).unsqueeze(0)
    closes_t = torch.from_numpy(frame["close"].to_numpy(dtype=np.float32)).unsqueeze(0)
    ref_t = torch.from_numpy(frame["reference_close"].to_numpy(dtype=np.float32)).unsqueeze(0)
    ch_t = torch.from_numpy(frame["predicted_high_p50_h1"].to_numpy(dtype=np.float32)).unsqueeze(0)
    cl_t = torch.from_numpy(frame["predicted_low_p50_h1"].to_numpy(dtype=np.float32)).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(features_t)
        acts = model.decode_actions(outputs, reference_close=ref_t, chronos_high=ch_t, chronos_low=cl_t)

    sim3 = simulate_hourly_trades_binary(
        highs=highs_t, lows=lows_t, closes=closes_t,
        buy_prices=acts["buy_price"],
        sell_prices=acts["sell_price"],
        trade_intensity=acts["trade_amount"] / 100.0,
        buy_trade_intensity=acts["buy_amount"] / 100.0,
        sell_trade_intensity=acts["sell_amount"] / 100.0,
        maker_fee=0.001, initial_cash=1.0, max_leverage=1.0,
        decision_lag_bars=2, fill_buffer_pct=0.0005, margin_annual_rate=0.0625,
    )
    vals = sim3.portfolio_values.float()
    total_ret = (vals[0, -1] / vals[0, 0] - 1).item()
    mean_r = sim3.returns.float().mean().item()
    print(f"  200-bar total return: {total_ret:.4%}")
    print(f"  Mean hourly: {mean_r:.6f}, annualized: {mean_r*8760:.2%}")
    buy_fills = sim3.buy_fill_probability.sum().item()
    sell_fills = sim3.sell_fill_probability.sum().item()
    print(f"  Buy fills: {buy_fills:.0f}, Sell fills: {sell_fills:.0f}")

    # Check first 10 actions in detail
    print("\n  First 10 actions (buy_price, sell_price, ref_close, high, low):")
    for i in range(min(10, acts["buy_price"].shape[-1])):
        bp = acts["buy_price"][0, i].item()
        sp = acts["sell_price"][0, i].item()
        rc = ref_t[0, i].item()
        h = highs_t[0, i+2].item() if i+2 < highs_t.shape[-1] else 0
        l = lows_t[0, i+2].item() if i+2 < lows_t.shape[-1] else 0
        bf = sim3.buy_fill_probability[0, i].item() if i < sim3.buy_fill_probability.shape[-1] else 0
        sf = sim3.sell_fill_probability[0, i].item() if i < sim3.sell_fill_probability.shape[-1] else 0
        print(f"    [{i:3d}] buy={bp:.2f} sell={sp:.2f} ref={rc:.2f} bar+2: h={h:.2f} l={l:.2f} bfill={bf:.0f} sfill={sf:.0f}")


if __name__ == "__main__":
    main()
