from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

import torch

import pufferlib.models
import pufferlib.pufferl

from .config import TrainingPlan, load_plan
from .data.loader import load_asset_frames
from .envs.trading_env import make_vecenv, resolve_device
from .logging.logger import CompositeLogger
from .models.policy import TradingPolicy


def _device_string(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return device.type


def build_trainer(plan: TrainingPlan) -> Dict[str, Any]:
    asset_frames = load_asset_frames(plan.data)
    vecenv = make_vecenv(plan, asset_frames)
    device = resolve_device(plan)

    base_cfg = pufferlib.pufferl.load_config("default")
    train_cfg = plan.train.apply_overrides(base_cfg["train"], device=_device_string(device))
    train_cfg["seed"] = plan.vec.seed
    train_cfg["env"] = "stock_trading_env"
    train_cfg["data_dir"] = str(plan.logging.checkpoint_dir)
    train_cfg["tensorboard"] = str(plan.logging.tensorboard_dir)

    logger = CompositeLogger(plan.logging, {"train": train_cfg, "plan": plan.to_dict()})

    policy = TradingPolicy(vecenv.driver_env, plan.model).to(device)
    if plan.model.use_lstm:
        policy = pufferlib.models.LSTMWrapper(
            vecenv.driver_env,
            policy,
            input_size=plan.model.hidden_size,
            hidden_size=plan.model.rnn_hidden_size,
        ).to(device)

    trainer = pufferlib.pufferl.PuffeRL(train_cfg, vecenv, policy, logger)
    return {
        "trainer": trainer,
        "logger": logger,
        "vecenv": vecenv,
        "plan": plan,
    }


def train(plan: TrainingPlan) -> Dict[str, Any]:
    resources = build_trainer(plan)
    trainer: pufferlib.pufferl.PuffeRL = resources["trainer"]
    logger: CompositeLogger = resources["logger"]
    vecenv = resources["vecenv"]
    final_logs: Dict[str, float] = {}

    model_path = ""
    summary: Dict[str, Any] = {}
    try:
        while trainer.global_step < trainer.config["total_timesteps"]:
            trainer.evaluate()
            logs = trainer.train()
            if logs:
                final_logs = {**final_logs, **logs}

        trainer.print_dashboard()
        logs = trainer.mean_and_log()
        if logs:
            final_logs = {**final_logs, **logs}

        model_path = trainer.close()
        summary = {
            "run_id": logger.run_id,
            "model_path": model_path,
            "final_logs": final_logs,
            "plan": resources["plan"].to_dict(),
        }
    except Exception:
        with suppress(Exception):
            trainer.logger.close(model_path)
        raise
    finally:
        with suppress(Exception):
            vecenv.close()
        summary_path = resources["plan"].logging.summary_path
        if not summary:
            summary = {
                "run_id": logger.run_id,
                "model_path": model_path,
                "final_logs": final_logs,
                "plan": resources["plan"].to_dict(),
            }
        summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_with_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    plan = load_plan(path, overrides=overrides)
    return train(plan)


__all__ = ["train", "run_with_config", "build_trainer"]
