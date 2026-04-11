#!/usr/bin/env python3
"""Train with wide fixed price offsets (not Chronos-bounded) for lag robustness.

Uses use_midpoint_offsets=False with price_offset_pct=0.05 (5%).
This decouples price setting from Chronos predictions, letting the model
learn optimal absolute offset distances that survive 2-hour lag.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import TrainingConfig, DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.trainer import BinanceHourlyTrainer


def train_symbol(symbol: str, epochs: int = 30, seed: int = 42):
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{symbol}_wideoffset_lag2_s{seed}_{ts}"

    dataset_cfg = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=72,
        validation_days=70,
        forecast_horizons=(1, 24),
        cache_only=True,
    )

    config = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        sequence_length=72,
        learning_rate=3e-4,
        weight_decay=4e-4,
        transformer_dim=256,
        transformer_heads=8,
        transformer_layers=4,
        model_arch="classic",
        decision_lag_bars=2,
        decision_lag_range="1,2",
        validation_lag_aggregation="minimax",
        validation_use_binary_fills=True,
        loss_type="sortino",
        return_weight=0.08,
        maker_fee=0.001,
        # Wide offsets: 5% range, 0.3% min gap
        use_midpoint_offsets=False,
        price_offset_pct=0.05,
        min_price_gap_pct=0.003,
        fill_buffer_pct=0.0005,
        fill_temperature=0.1,
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
    print(f"Features: {list(data.feature_columns)}")

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
    args = parser.parse_args()

    train_symbol(args.symbol, args.epochs, args.seed)
