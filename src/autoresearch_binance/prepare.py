"""Fixed evaluation harness for Binance PufferLib RL autoresearch.

DO NOT MODIFY THIS FILE during autoresearch runs.
"""
from __future__ import annotations

import logging
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

TIME_BUDGET = int(float(os.getenv("AUTORESEARCH_BINANCE_TIME_BUDGET_SECONDS", "300")))

DATA_PATH = Path("rl-trainingbinance/data/binance6_data.bin")
CHECKPOINT_DIR = Path("rl-trainingbinance/checkpoints")
EVAL_DAYS = (7, 14, 30)
NUM_EVAL_ENVS = 16


@dataclass(frozen=True)
class BinanceTaskConfig:
    data_path: Path = DATA_PATH
    checkpoint_dir: Path = CHECKPOINT_DIR
    eval_days: tuple[int, ...] = EVAL_DAYS
    num_eval_envs: int = NUM_EVAL_ENVS
    max_steps_7d: int = 168
    max_steps_14d: int = 336
    max_steps_30d: int = 720
    fee_rate: float = 0.001
    max_leverage: float = 1.0
    num_symbols: int = 4
    initial_equity: float = 10_000.0


def resolve_task_config(**overrides) -> BinanceTaskConfig:
    fields = {f.name: getattr(BinanceTaskConfig, f.name, None) for f in BinanceTaskConfig.__dataclass_fields__.values()}
    fields.update({k: v for k, v in overrides.items() if k in fields})
    return BinanceTaskConfig(**fields)


def _read_data_header(data_path: Path) -> tuple[int, int]:
    with open(str(data_path.resolve()), "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])
    return num_symbols, num_timesteps


def evaluate_checkpoint(
    checkpoint_path: Path,
    policy_cls,
    config: BinanceTaskConfig,
    *,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    from pufferlib_market import binding

    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Bad checkpoint: {checkpoint_path}")

    num_symbols, _ = _read_data_header(config.data_path)
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols

    first_key = next(k for k in payload["model"] if "encoder" in k or "input_proj" in k)
    obs_size_ckpt = payload["model"][first_key].shape[1] if "weight" in first_key else payload["model"][first_key].shape[0]
    actor_keys = [k for k in payload["model"] if "actor" in k and "weight" in k]
    num_actions_ckpt = payload["model"][actor_keys[-1]].shape[0] if actor_keys else num_actions

    policy = policy_cls(obs_size_ckpt, num_actions_ckpt, hidden=payload.get("hidden_size", 1024))
    policy.load_state_dict(payload["model"])
    policy.to(device).eval()

    data_path_str = str(config.data_path.resolve())
    binding.shared(data_path=data_path_str)

    results = {}
    for days in config.eval_days:
        max_steps = days * 24
        N = config.num_eval_envs

        obs_buf = np.zeros((N, obs_size), dtype=np.float32)
        act_buf = np.zeros((N,), dtype=np.int32)
        rew_buf = np.zeros((N,), dtype=np.float32)
        term_buf = np.zeros((N,), dtype=np.uint8)
        trunc_buf = np.zeros((N,), dtype=np.uint8)

        handle = binding.vec_init(
            obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
            N, 123,
            max_steps=max_steps,
            fee_rate=config.fee_rate,
            max_leverage=config.max_leverage,
            reward_scale=1.0,
            reward_clip=100.0,
            cash_penalty=0.0,
        )
        binding.vec_reset(handle, 123)

        ep_returns = np.zeros(N, dtype=np.float64)
        ep_done = np.zeros(N, dtype=bool)

        for step in range(max_steps):
            obs_t = torch.from_numpy(obs_buf.copy()).to(device)
            with torch.no_grad():
                logits, _ = policy(obs_t)
            actions = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
            act_buf[:] = actions
            binding.vec_step(handle)

            for i in range(N):
                if not ep_done[i]:
                    ep_returns[i] += rew_buf[i]
                if term_buf[i] or trunc_buf[i]:
                    ep_done[i] = True

            if ep_done.all():
                break

        binding.vec_close(handle)

        arr = ep_returns
        mean_ret = float(np.mean(arr))
        downside = arr[arr < 0]
        downside_std = float(np.std(downside)) + 1e-8 if len(downside) > 0 else 1e-8
        sortino = mean_ret / downside_std * np.sqrt(365 / days)
        results[f"return_{days}d"] = round(mean_ret, 4)
        results[f"sortino_{days}d"] = round(sortino, 2)

    r7 = results.get("return_7d", 0)
    s7 = min(results.get("sortino_7d", 0), 100.0)
    r30 = results.get("return_30d", 0)
    s30 = min(results.get("sortino_30d", 0), 100.0)
    results["robust_score"] = round(0.4 * r7 + 0.2 * r30 + 0.3 * max(s7, 0) * 0.01 + 0.1 * max(s30, 0) * 0.01, 6)

    return results


def print_metrics(metrics: dict[str, Any], training_seconds: float, total_seconds: float, peak_vram_mb: float, total_timesteps: int, num_updates: int):
    print("---")
    print(f"robust_score:      {metrics.get('robust_score', 0):.6f}")
    for days in EVAL_DAYS:
        r = metrics.get(f"return_{days}d", 0)
        s = metrics.get(f"sortino_{days}d", 0)
        print(f"val_return_{days}d:     {r:.4f}")
        print(f"val_sortino_{days}d:    {s:.2f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"total_timesteps:   {total_timesteps}")
    print(f"num_updates:       {num_updates}")
