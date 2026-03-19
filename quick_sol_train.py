#!/usr/bin/env python3
"""Quick SOL training using cached forecasts only."""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from binanceneural.config import TrainingConfig, DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import build_policy, policy_config_from_payload, align_state_dict_input_dim
from src.torch_load_utils import torch_load_compat

SYMBOL = "SOLUSD"
DATA_ROOT = Path("trainingdatahourly/crypto")

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sol_nano_d6_quick_{ts}"

    config = TrainingConfig(
        epochs=40,
        batch_size=16,
        sequence_length=192,
        learning_rate=3e-4,
        weight_decay=1e-4,
        transformer_dim=256,
        transformer_layers=6,
        transformer_heads=8,
        model_arch="nano",
        num_kv_heads=2,
        mlp_ratio=4.0,
        logits_softcap=12.0,
        use_qk_norm=True,
        use_causal_attention=True,
        dry_train_steps=None,
        run_name=run_name,
        use_compile=False,
        use_amp=True,
        use_tf32=True,
        return_weight=0.1,
    )

    dataset_cfg = DatasetConfig(
        symbol=SYMBOL,
        data_root=DATA_ROOT,
        sequence_length=192,
        validation_days=70,
        forecast_horizons=(1, 24),
        cache_only=True,
    )

    print(f"\n=== Training {run_name} ===")
    print(f"Layers: 6, SeqLen: 192, Hidden: 256, Arch: nano")

    data = BinanceHourlyDataModule(dataset_cfg)
    print(f"Features: {len(data.feature_columns)}")
    print(f"Train size: {len(data.train_dataset)}")

    trainer = BinanceHourlyTrainer(config, data)
    artifacts = trainer.train()

    best_ckpt = artifacts.best_checkpoint
    if best_ckpt is None:
        print("No checkpoint saved")
        return

    print(f"\nBest checkpoint: {best_ckpt}")

    # Evaluate
    payload = torch_load_compat(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    input_dim = len(data.feature_columns)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)

    policy_cfg = policy_config_from_payload(
        payload.get("config", {}),
        input_dim=input_dim,
        state_dict=state_dict,
    )
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_frame = data.val_dataset.frame
    actions = generate_actions_from_frame(
        model=model,
        frame=val_frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=192,
        horizon=1,
    )

    sim = BinanceMarketSimulator(SimulationConfig(initial_cash=10000.0))
    result = sim.run(val_frame, actions)
    metrics = result.metrics

    print(f"\n=== Validation Results ===")
    print(f"Return: {metrics['total_return']:.4f}")
    print(f"Sortino: {metrics['sortino']:.4f}")
    print(f"Annualized: {metrics.get('annualized_return', 0):.4f}")


if __name__ == "__main__":
    main()
