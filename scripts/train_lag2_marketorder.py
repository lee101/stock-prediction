#!/usr/bin/env python3
"""Train with market order entry to eliminate adverse selection at lag>=2.

Root cause: limit order fills at lag>=1 are adversely selected -- they only
fill when price moves against you (downtrend), causing systematic losses.

Fix: use market_order_entry=True so buys execute at the open of bar T+lag.
The model controls WHEN to buy (intensity) and WHERE to sell (sell_price).
This makes the model a trend-predictor rather than a limit-order optimizer.
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
    run_name = f"{symbol}_mktorder_lag2_s{seed}_{ts}"

    dataset_cfg = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=72,
        validation_days=70,
        forecast_horizons=(24, 1),  # h24 primary for sell price anchoring
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
        # Market order entry: buys at open, no limit check
        market_order_entry=True,
        loss_type="sortino",
        return_weight=0.08,
        maker_fee=0.001,
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
    print(f"Market order entry: True (buys at open, sells at limit)")

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
