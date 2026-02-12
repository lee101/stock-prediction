#!/usr/bin/env python3
"""Train deeper SOL models and deploy best one to supervisor."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
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
import torch
import numpy as np

SYMBOL = "SOLUSD"
DATA_ROOT = Path("trainingdatahourly/crypto")
CHECKPOINT_ROOT = Path("binanceneural/checkpoints")
BEST_RESULTS_PATH = Path("strategy_state/sol_optimization_results.json")

CONFIGS = [
    # Deeper nano models with longer context
    {"name": "sol_nano_d6_s192", "arch": "nano", "layers": 6, "seq_len": 192, "epochs": 60, "hidden": 256, "dry": None},
    {"name": "sol_nano_d8_s256", "arch": "nano", "layers": 8, "seq_len": 256, "epochs": 80, "hidden": 384, "dry": None},
    {"name": "sol_nano_d6_s384", "arch": "nano", "layers": 6, "seq_len": 384, "epochs": 60, "hidden": 256, "dry": None},
    # Even deeper for longer training
    {"name": "sol_nano_d10_s384", "arch": "nano", "layers": 10, "seq_len": 384, "epochs": 100, "hidden": 384, "dry": None},
    {"name": "sol_nano_d12_s512", "arch": "nano", "layers": 12, "seq_len": 512, "epochs": 120, "hidden": 512, "dry": None},
    {"name": "sol_nano_d8_s768", "arch": "nano", "layers": 8, "seq_len": 768, "epochs": 100, "hidden": 384, "dry": None},
    # Higher return weight variants
    {"name": "sol_nano_d8_s384_rw", "arch": "nano", "layers": 8, "seq_len": 384, "epochs": 100, "hidden": 384, "dry": None, "return_weight": 0.2},
    {"name": "sol_nano_d10_s512_rw", "arch": "nano", "layers": 10, "seq_len": 512, "epochs": 120, "hidden": 512, "dry": None, "return_weight": 0.25},
]


def load_best_result():
    if BEST_RESULTS_PATH.exists():
        with open(BEST_RESULTS_PATH) as f:
            return json.load(f)
    return {"best_combined": -999, "best_sortino": -999, "best_sharpe": -999, "best_return": -999, "best_checkpoint": None}


def save_result(result):
    BEST_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)


def compute_sharpe(returns: np.ndarray, periods_per_year: float = 8760) -> float:
    if len(returns) < 2:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret < 1e-10:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(periods_per_year))


def train_config(cfg_dict: dict) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg_dict['name']}_{ts}"
    return_weight = cfg_dict.get("return_weight", 0.1)

    config = TrainingConfig(
        epochs=cfg_dict["epochs"],
        batch_size=16,
        sequence_length=cfg_dict["seq_len"],
        learning_rate=3e-4,
        weight_decay=1e-4,
        transformer_dim=cfg_dict["hidden"],
        transformer_layers=cfg_dict["layers"],
        transformer_heads=8,
        model_arch=cfg_dict["arch"],
        num_kv_heads=2 if cfg_dict["arch"] == "nano" else None,
        mlp_ratio=4.0,
        logits_softcap=12.0,
        use_qk_norm=True,
        use_causal_attention=True,
        dry_train_steps=cfg_dict.get("dry"),
        run_name=run_name,
        use_compile=False,
        use_amp=True,
        use_tf32=True,
        return_weight=return_weight,
    )

    dataset_cfg = DatasetConfig(
        symbol=SYMBOL,
        data_root=DATA_ROOT,
        sequence_length=cfg_dict["seq_len"],
        validation_days=70,
        forecast_horizons=(1, 24),  # use cached only
        cache_only=True,
    )

    print(f"\n=== Training {run_name} ===")
    print(f"Layers: {cfg_dict['layers']}, SeqLen: {cfg_dict['seq_len']}, Hidden: {cfg_dict['hidden']}")

    data = BinanceHourlyDataModule(dataset_cfg)
    trainer = BinanceHourlyTrainer(config, data)
    artifacts = trainer.train()

    # Run simulation on validation data
    best_ckpt = artifacts.best_checkpoint
    if best_ckpt is None:
        print("No checkpoint saved")
        return {"sortino": -999, "return": -999}

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
        sequence_length=cfg_dict["seq_len"],
        horizon=1,
    )

    sim = BinanceMarketSimulator(SimulationConfig(initial_cash=10000.0))
    result = sim.run(val_frame, actions)
    metrics = result.metrics

    equity_curve = result.combined_equity.to_numpy()
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = compute_sharpe(returns)
    else:
        sharpe = 0.0

    sortino = float(metrics.get("sortino", 0.0))
    combined = 0.6 * sortino + 0.4 * sharpe

    print(f"Result: return={metrics['total_return']:.4f}, sortino={sortino:.2f}, sharpe={sharpe:.2f}, combined={combined:.2f}")

    return {
        "sortino": sortino,
        "sharpe": sharpe,
        "combined": combined,
        "return": float(metrics["total_return"]),
        "checkpoint": str(best_ckpt),
        "config": cfg_dict,
    }


def deploy_checkpoint(checkpoint_path: str, sudo_password: str = None):
    """Update supervisor config and restart selector."""
    selector_conf = Path("/etc/supervisor/conf.d/binanceexp1-selector.conf")
    if not selector_conf.exists():
        print("Selector config not found")
        return False

    # Read current config
    content = selector_conf.read_text()

    # Update SOLUSD checkpoint path
    import re
    pattern = r'SOLUSD=[^\s,]+'
    new_ckpt = f'SOLUSD={checkpoint_path}'
    new_content = re.sub(pattern, new_ckpt, content)

    if new_content == content:
        print("Could not update checkpoint in config")
        return False

    # Write new config (needs sudo)
    tmp_conf = Path("/tmp/binanceexp1-selector.conf")
    tmp_conf.write_text(new_content)

    cmd = f"sudo cp {tmp_conf} {selector_conf} && sudo supervisorctl reread && sudo supervisorctl update && sudo supervisorctl restart binanceexp1-selector"
    if sudo_password:
        cmd = f"echo '{sudo_password}' | sudo -S bash -c \"{cmd}\""

    print(f"Deploying {checkpoint_path}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Deploy failed: {result.stderr}")
        return False

    print("Deployed successfully")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="Auto-deploy best model")
    parser.add_argument("--password", help="Sudo password for deploy")
    parser.add_argument("--config-idx", type=int, help="Run specific config by index")
    args = parser.parse_args()

    best = load_best_result()
    print(f"Current best: combined={best.get('best_combined', -999):.2f}, sortino={best.get('best_sortino', -999):.2f}, sharpe={best.get('best_sharpe', -999):.2f}")

    configs_to_run = CONFIGS
    if args.config_idx is not None:
        configs_to_run = [CONFIGS[args.config_idx]]

    for cfg in configs_to_run:
        try:
            result = train_config(cfg)

            is_better = result["combined"] > best.get("best_combined", -999)

            if is_better:
                print(f"\n*** NEW BEST: combined={result['combined']:.2f}, sortino={result['sortino']:.2f}, sharpe={result['sharpe']:.2f}, return={result['return']:.4f} ***")
                best = {
                    "best_combined": result["combined"],
                    "best_sortino": result["sortino"],
                    "best_sharpe": result["sharpe"],
                    "best_return": result["return"],
                    "best_checkpoint": result["checkpoint"],
                    "config": result["config"],
                    "timestamp": datetime.now().isoformat(),
                }
                save_result(best)

                if args.deploy and args.password:
                    deploy_checkpoint(result["checkpoint"], args.password)
        except Exception as e:
            print(f"Error training {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nFinal best: {best}")


if __name__ == "__main__":
    main()
