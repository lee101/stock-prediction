#!/usr/bin/env python3
"""Train with position-target rebalancing -- no limit orders, no adverse selection.

The model outputs a target allocation (0=cash, 1=fully invested). At each bar
the sim rebalances toward the target at market price (open). Eliminates
the fundamental adverse selection problem that makes limit-order models
catastrophically fail at decision_lag>=2.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import TrainingConfig, DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.trainer import BinanceHourlyTrainer


def train_symbol(symbol: str, epochs: int = 30, seed: int = 42, smooth: float = 50.0,
                  seq_len: int = 168, lr: float = 1e-4, rw: float = 0.5, batch_size: int = 8):
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{symbol}_rebal_sq{seq_len}_sm{smooth}_rw{rw}_s{seed}_{ts}"

    dataset_cfg = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=seq_len,
        validation_days=70,
        forecast_horizons=(1, 24),
        cache_only=True,
    )

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=seq_len,
        learning_rate=lr,
        weight_decay=4e-4,
        transformer_dim=256,
        transformer_heads=8,
        transformer_layers=4,
        model_arch="classic",
        num_outputs=6,
        decision_lag_bars=2,
        decision_lag_range="2",
        validation_lag_aggregation="minimax",
        validation_use_binary_fills=True,
        use_rebalance_sim=True,
        rebalance_entropy_weight=0.01,
        rebalance_smoothness_weight=smooth,
        loss_type="sortino",
        return_weight=rw,
        maker_fee=0.001,
        fill_buffer_pct=0.0,
        fill_temperature=0.01,
        max_leverage=1.0,
        margin_annual_rate=0.0625,
        max_hold_hours=6.0,
        use_causal_attention=True,
        use_qk_norm=True,
        lr_schedule="cosine",
        lr_min_ratio=0.01,
        attention_backend="sdpa",
        use_compile=False,
        use_compiled_sim_loss=False,
        run_name=run_name,
        seed=seed,
    )

    data = BinanceHourlyDataModule(dataset_cfg)
    print(f"\n{symbol}: {len(data.train_dataset)} train, {len(data.val_dataset)} val, {len(data.feature_columns)} features")
    print(f"Rebalance sim: True (position-target, no limit orders)")

    trainer = BinanceHourlyTrainer(config, data)
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    print(f"\n{symbol} done in {elapsed:.0f}s")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth", type=float, default=50.0)
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rw", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train_symbol(args.symbol, args.epochs, args.seed, args.smooth,
                 args.seq_len, args.lr, args.rw, args.batch_size)
