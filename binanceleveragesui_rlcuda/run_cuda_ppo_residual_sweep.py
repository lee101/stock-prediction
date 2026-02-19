#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint

from .residual_env import (
    ResidualLeverageEnv,
    ResidualLeverageEnvConfig,
    evaluate_baseline_episode,
    evaluate_deterministic_episode,
)


@dataclass(frozen=True)
class ResidualSweepConfig:
    symbol: str = "SUIUSDT"
    data_root: str = "trainingdatahourlybinance"
    forecast_cache_root: str = "binancechronossolexperiment/forecast_cache_sui_stable_best"
    horizons: str = "1,6,12"
    context_hours: int = 256
    chronos_batch_size: int = 32
    chronos_model_id: str = "amazon/chronos-t5-small"
    sequence_length: int = 72
    val_days: int = 15
    test_days: int = 7
    baseline_checkpoint: str = "binanceleveragesui/checkpoints/lev5x_rw0.012_s1337/policy_checkpoint.pt"
    window_size: int = 32
    max_leverage: float = 5.0
    maker_fee: float = 0.001
    margin_hourly_rate: float = 0.0000025457
    scale_limit: float = 1.0
    residual_penalty: float = 0.00005
    downside_penalty: float = 0.0
    pnl_smoothness_penalty: float = 0.0
    leverage_cap_smoothness_penalty: float = 0.0
    cap_floor_ratio: float = 0.0
    max_cap_change_per_step: float | None = None
    total_timesteps: int = 120_000
    n_envs: int = 8
    train_episode_hours: int = 24 * 10
    eval_freq: int = 8_000
    learning_rate: float = 2e-4
    n_steps: int = 1024
    batch_size: int = 512
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 256
    device: str = "cuda"
    initial_cash: float = 10_000.0
    drawdown_weight_for_rank: float = 0.35
    sortino_weight_for_rank: float = 0.0
    seeds: str = "1337,2024,7,42,99"
    output: str = "binanceleveragesui_rlcuda/results_cuda_ppo_residual_sweep.json"
    artifacts_root: str = "binanceleveragesui_rlcuda/artifacts_residual"
    baseline_json: str = "binanceleveragesui/results_single_5x_stable_mae.json"


def _parse_horizons(raw: str) -> tuple[int, ...]:
    values = tuple(int(t.strip()) for t in raw.split(",") if t.strip())
    if not values:
        raise ValueError("At least one horizon is required.")
    return values


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(t.strip()) for t in raw.split(",") if t.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token in {"none", "null", ""}:
        return None
    return float(raw)


def _to_index(frame: pd.DataFrame, ts: pd.Timestamp) -> int:
    matches = frame.index[frame["timestamp"] >= ts]
    if len(matches) == 0:
        raise ValueError(f"Timestamp {ts} not found in frame index range.")
    return int(matches[0])


def _standardize(features: np.ndarray, train_end_idx: int) -> np.ndarray:
    train = features[:train_end_idx]
    if len(train) == 0:
        raise ValueError("Training split is empty; cannot standardize features.")
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((features - mean) / std).astype(np.float32)


def _slice_with_context(
    *,
    arrays: dict[str, np.ndarray | list[str]],
    start_idx: int,
    end_idx: int,
    window_size: int,
) -> dict[str, Any]:
    if end_idx <= start_idx:
        raise ValueError(f"Invalid segment bounds start={start_idx} end={end_idx}")
    if end_idx - start_idx < 24:
        raise ValueError(f"Segment too short ({end_idx - start_idx} rows).")

    seg_start = max(0, start_idx - window_size + 1)
    rel_start = start_idx - seg_start
    rel_end = end_idx - seg_start

    out: dict[str, Any] = {
        "start_index": rel_start,
        "end_index": rel_end,
    }
    for key, values in arrays.items():
        out[key] = values[seg_start:end_idx]
    return out


def _build_dataset(cfg: ResidualSweepConfig) -> dict[str, Any]:
    horizons = _parse_horizons(cfg.horizons)

    dm = ChronosSolDataModule(
        symbol=cfg.symbol,
        data_root=Path(cfg.data_root),
        forecast_cache_root=Path(cfg.forecast_cache_root),
        forecast_horizons=horizons,
        context_hours=int(cfg.context_hours),
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=int(cfg.chronos_batch_size),
        model_id=cfg.chronos_model_id,
        sequence_length=int(cfg.sequence_length),
        split_config=SplitConfig(val_days=int(cfg.val_days), test_days=int(cfg.test_days)),
        cache_only=True,
    )

    frame = dm.full_frame.copy().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    model, normalizer, feature_columns, _ = load_policy_checkpoint(cfg.baseline_checkpoint)
    base_actions = generate_actions_from_frame(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=int(cfg.sequence_length),
        horizon=horizons[0],
        require_gpu=(cfg.device == "cuda"),
    )
    base_actions["timestamp"] = pd.to_datetime(base_actions["timestamp"], utc=True)

    merged = frame.merge(base_actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_base"))
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    val_start_idx = _to_index(merged, pd.to_datetime(dm.val_window_start, utc=True))
    test_start_idx = _to_index(merged, pd.to_datetime(dm.test_window_start, utc=True))
    if val_start_idx <= cfg.window_size:
        raise ValueError(
            f"Validation start index {val_start_idx} is too early for window_size={cfg.window_size}."
        )

    feature_cols = list(dm.feature_columns)
    features_raw = merged[feature_cols].to_numpy(dtype=np.float32)
    features = _standardize(features_raw, val_start_idx)

    arrays: dict[str, np.ndarray | list[str]] = {
        "features": features,
        "highs": merged["high"].to_numpy(dtype=np.float32),
        "lows": merged["low"].to_numpy(dtype=np.float32),
        "closes": merged["close"].to_numpy(dtype=np.float32),
        "base_buy_prices": merged["buy_price"].to_numpy(dtype=np.float32),
        "base_sell_prices": merged["sell_price"].to_numpy(dtype=np.float32),
        "base_buy_amounts": merged["buy_amount"].to_numpy(dtype=np.float32),
        "base_sell_amounts": merged["sell_amount"].to_numpy(dtype=np.float32),
        "timestamps": merged["timestamp"].astype(str).tolist(),
    }

    train = _slice_with_context(arrays=arrays, start_idx=cfg.window_size - 1, end_idx=val_start_idx, window_size=cfg.window_size)
    val = _slice_with_context(arrays=arrays, start_idx=val_start_idx, end_idx=test_start_idx, window_size=cfg.window_size)
    test = _slice_with_context(arrays=arrays, start_idx=test_start_idx, end_idx=len(merged), window_size=cfg.window_size)

    return {
        "feature_columns": feature_cols,
        "train": train,
        "val": val,
        "test": test,
        "val_start": str(dm.val_window_start),
        "test_start": str(dm.test_window_start),
    }


def _make_env(data: dict[str, Any], env_cfg: ResidualLeverageEnvConfig) -> ResidualLeverageEnv:
    return ResidualLeverageEnv(
        features=data["features"],
        highs=data["highs"],
        lows=data["lows"],
        closes=data["closes"],
        base_buy_prices=data["base_buy_prices"],
        base_sell_prices=data["base_sell_prices"],
        base_buy_amounts=data["base_buy_amounts"],
        base_sell_amounts=data["base_sell_amounts"],
        timestamps=data["timestamps"],
        config=env_cfg,
        start_index=int(data["start_index"]),
        end_index=int(data["end_index"]),
    )


def _baseline_5x_return(path: str) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return None

    for key in ("lev_5.0x", "lev_5x", "5x"):
        metrics = payload.get(key)
        if isinstance(metrics, dict) and "total_return" in metrics:
            return float(metrics["total_return"])
    return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_single_seed(
    cfg: ResidualSweepConfig,
    data: dict[str, Any],
    seed: int,
    artifacts_root: Path,
) -> dict[str, Any]:
    _set_seed(seed)
    run_dir = artifacts_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env_cfg = ResidualLeverageEnvConfig(
        window_size=cfg.window_size,
        max_leverage=cfg.max_leverage,
        maker_fee=cfg.maker_fee,
        margin_hourly_rate=cfg.margin_hourly_rate,
        initial_cash=cfg.initial_cash,
        scale_limit=cfg.scale_limit,
        residual_penalty=cfg.residual_penalty,
        downside_penalty=cfg.downside_penalty,
        pnl_smoothness_penalty=cfg.pnl_smoothness_penalty,
        leverage_cap_smoothness_penalty=cfg.leverage_cap_smoothness_penalty,
        cap_floor_ratio=cfg.cap_floor_ratio,
        max_cap_change_per_step=cfg.max_cap_change_per_step,
        random_start=True,
        episode_length=cfg.train_episode_hours,
    )

    val_env_cfg = ResidualLeverageEnvConfig(
        window_size=cfg.window_size,
        max_leverage=cfg.max_leverage,
        maker_fee=cfg.maker_fee,
        margin_hourly_rate=cfg.margin_hourly_rate,
        initial_cash=cfg.initial_cash,
        scale_limit=cfg.scale_limit,
        residual_penalty=cfg.residual_penalty,
        downside_penalty=cfg.downside_penalty,
        pnl_smoothness_penalty=cfg.pnl_smoothness_penalty,
        leverage_cap_smoothness_penalty=cfg.leverage_cap_smoothness_penalty,
        cap_floor_ratio=cfg.cap_floor_ratio,
        max_cap_change_per_step=cfg.max_cap_change_per_step,
        random_start=False,
        episode_length=None,
    )

    train_vec_env = DummyVecEnv([
        (lambda train_data=data["train"], ecfg=train_env_cfg: Monitor(_make_env(train_data, ecfg)))
        for _ in range(max(1, int(cfg.n_envs)))
    ])
    val_vec_env = DummyVecEnv([
        (lambda val_data=data["val"], ecfg=val_env_cfg: Monitor(_make_env(val_data, ecfg)))
    ])

    policy_kwargs = {
        "activation_fn": torch.nn.SiLU,
        "net_arch": {
            "pi": [int(cfg.hidden_dim), int(cfg.hidden_dim)],
            "vf": [int(cfg.hidden_dim), int(cfg.hidden_dim)],
        },
    }

    model = PPO(
        "MlpPolicy",
        train_vec_env,
        learning_rate=float(cfg.learning_rate),
        n_steps=int(cfg.n_steps),
        batch_size=int(cfg.batch_size),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_range=float(cfg.clip_range),
        ent_coef=float(cfg.ent_coef),
        vf_coef=float(cfg.vf_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        seed=int(seed),
        device=cfg.device,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    eval_callback = EvalCallback(
        val_vec_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(512, int(cfg.eval_freq)),
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=int(cfg.total_timesteps), callback=eval_callback, progress_bar=False)
    train_seconds = time.time() - t0

    best_model_path = run_dir / "best_model.zip"
    final_model_path = run_dir / "final_model.zip"
    model.save(str(final_model_path))

    model_to_eval = model
    model_path = str(final_model_path)
    if best_model_path.exists():
        model_to_eval = PPO.load(str(best_model_path), device=cfg.device)
        model_path = str(best_model_path)

    rl_metrics: dict[str, dict[str, float]] = {}
    baseline_metrics: dict[str, dict[str, float]] = {}

    for lev in [1.0, 2.0, 3.0, 4.0, 5.0]:
        eval_cfg = ResidualLeverageEnvConfig(
            window_size=cfg.window_size,
            max_leverage=lev,
            maker_fee=cfg.maker_fee,
            margin_hourly_rate=cfg.margin_hourly_rate,
            initial_cash=cfg.initial_cash,
            scale_limit=cfg.scale_limit,
            residual_penalty=cfg.residual_penalty,
            downside_penalty=cfg.downside_penalty,
            pnl_smoothness_penalty=cfg.pnl_smoothness_penalty,
            leverage_cap_smoothness_penalty=cfg.leverage_cap_smoothness_penalty,
            cap_floor_ratio=cfg.cap_floor_ratio,
            max_cap_change_per_step=cfg.max_cap_change_per_step,
            random_start=False,
            episode_length=None,
        )

        test_env_rl = _make_env(data["test"], eval_cfg)
        test_env_base = _make_env(data["test"], eval_cfg)

        rl = evaluate_deterministic_episode(model_to_eval, test_env_rl)
        base = evaluate_baseline_episode(test_env_base)
        rl_metrics[f"lev_{lev:.1f}x"] = rl["metrics"]
        baseline_metrics[f"lev_{lev:.1f}x"] = base["metrics"]

    lev5 = rl_metrics["lev_5.0x"]
    score = float(
        lev5["total_return"]
        + cfg.sortino_weight_for_rank * lev5["sortino"]
        - cfg.drawdown_weight_for_rank * abs(lev5["max_drawdown"])
    )
    return {
        "seed": int(seed),
        "model_path": model_path,
        "train_seconds": float(train_seconds),
        "score": score,
        "rl_metrics": rl_metrics,
        "baseline_metrics": baseline_metrics,
    }


def run_sweep(cfg: ResidualSweepConfig) -> dict[str, Any]:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    dataset = _build_dataset(cfg)
    artifacts_root = Path(cfg.artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(cfg.seeds)
    runs = []
    for seed in seeds:
        runs.append(_train_single_seed(cfg, dataset, seed, artifacts_root))

    runs_sorted = sorted(runs, key=lambda r: r["score"], reverse=True)
    best = runs_sorted[0]
    baseline_return_external = _baseline_5x_return(cfg.baseline_json)

    best_rl_5x = float(best["rl_metrics"]["lev_5.0x"]["total_return"])
    best_internal_baseline_5x = float(best["baseline_metrics"]["lev_5.0x"]["total_return"])

    summary = {
        "config": asdict(cfg),
        "dataset": {
            "feature_columns": dataset["feature_columns"],
            "val_start": dataset["val_start"],
            "test_start": dataset["test_start"],
            "train_rows": len(dataset["train"]["features"]),
            "val_rows": len(dataset["val"]["features"]),
            "test_rows": len(dataset["test"]["features"]),
        },
        "baseline_lev_5x_return_external": baseline_return_external,
        "runs": runs,
        "best": best,
        "best_vs_internal_baseline_5x_delta": best_rl_5x - best_internal_baseline_5x,
    }
    if baseline_return_external is not None:
        summary["best_vs_external_baseline_5x_delta"] = best_rl_5x - baseline_return_external
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA PPO residual sweep on baseline leverage actions.")
    parser.add_argument("--symbol", default="SUIUSDT")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--forecast-cache-root", default="binancechronossolexperiment/forecast_cache_sui_stable_best")
    parser.add_argument("--horizons", default="1,6,12")
    parser.add_argument("--context-hours", type=int, default=256)
    parser.add_argument("--chronos-batch-size", type=int, default=32)
    parser.add_argument("--chronos-model-id", default="amazon/chronos-t5-small")
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--val-days", type=int, default=15)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--baseline-checkpoint", default="binanceleveragesui/checkpoints/lev5x_rw0.012_s1337/policy_checkpoint.pt")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--margin-hourly-rate", type=float, default=0.0000025457)
    parser.add_argument("--scale-limit", type=float, default=1.0)
    parser.add_argument("--residual-penalty", type=float, default=0.00005)
    parser.add_argument("--downside-penalty", type=float, default=0.0)
    parser.add_argument("--pnl-smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--leverage-cap-smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--cap-floor-ratio", type=float, default=0.0)
    parser.add_argument(
        "--max-cap-change-per-step",
        default=None,
        help="Optional cap-ratio movement limit per step, e.g. 0.1 or 'none'.",
    )
    parser.add_argument("--total-timesteps", type=int, default=120_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--train-episode-hours", type=int, default=24 * 10)
    parser.add_argument("--eval-freq", type=int, default=8_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--drawdown-weight-for-rank", type=float, default=0.35)
    parser.add_argument("--sortino-weight-for-rank", type=float, default=0.0)
    parser.add_argument("--seeds", default="1337,2024,7,42,99")
    parser.add_argument("--output", default="binanceleveragesui_rlcuda/results_cuda_ppo_residual_sweep.json")
    parser.add_argument("--artifacts-root", default="binanceleveragesui_rlcuda/artifacts_residual")
    parser.add_argument("--baseline-json", default="binanceleveragesui/results_single_5x_stable_mae.json")
    args = parser.parse_args()

    cfg = ResidualSweepConfig(
        symbol=args.symbol,
        data_root=args.data_root,
        forecast_cache_root=args.forecast_cache_root,
        horizons=args.horizons,
        context_hours=args.context_hours,
        chronos_batch_size=args.chronos_batch_size,
        chronos_model_id=args.chronos_model_id,
        sequence_length=args.sequence_length,
        val_days=args.val_days,
        test_days=args.test_days,
        baseline_checkpoint=args.baseline_checkpoint,
        window_size=args.window_size,
        max_leverage=args.max_leverage,
        maker_fee=args.maker_fee,
        margin_hourly_rate=args.margin_hourly_rate,
        scale_limit=args.scale_limit,
        residual_penalty=args.residual_penalty,
        downside_penalty=args.downside_penalty,
        pnl_smoothness_penalty=args.pnl_smoothness_penalty,
        leverage_cap_smoothness_penalty=args.leverage_cap_smoothness_penalty,
        cap_floor_ratio=args.cap_floor_ratio,
        max_cap_change_per_step=_parse_optional_float(args.max_cap_change_per_step),
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        train_episode_hours=args.train_episode_hours,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        device=args.device,
        initial_cash=args.initial_cash,
        drawdown_weight_for_rank=args.drawdown_weight_for_rank,
        sortino_weight_for_rank=args.sortino_weight_for_rank,
        seeds=args.seeds,
        output=args.output,
        artifacts_root=args.artifacts_root,
        baseline_json=args.baseline_json,
    )

    summary = run_sweep(cfg)
    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    best = summary["best"]
    lev5 = best["rl_metrics"]["lev_5.0x"]
    base5 = best["baseline_metrics"]["lev_5.0x"]
    print("=== CUDA PPO Residual Sweep Complete ===")
    print(f"Best seed: {best['seed']} score={best['score']:.6f}")
    print(
        "Best RL 5x: "
        f"return={lev5['total_return']:.6f} sortino={lev5['sortino']:.3f} "
        f"max_dd={lev5['max_drawdown']:.6f}"
    )
    print(
        "Internal baseline 5x: "
        f"return={base5['total_return']:.6f} sortino={base5['sortino']:.3f} "
        f"max_dd={base5['max_drawdown']:.6f}"
    )
    print(f"Delta vs internal baseline 5x: {summary['best_vs_internal_baseline_5x_delta']:+.6f}")
    if summary.get("best_vs_external_baseline_5x_delta") is not None:
        print(f"Delta vs external baseline 5x: {summary['best_vs_external_baseline_5x_delta']:+.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
