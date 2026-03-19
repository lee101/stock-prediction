#!/usr/bin/env python3
"""Benchmark the fast C++ market env against the Python reference implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


try:  # Optional dependency when only synthetic data is required.
    import pandas as pd
except Exception:  # pragma: no cover - pandas is part of core deps but guarded for safety
    pd = None

from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig

from fastmarketsim import FastMarketEnv


CSV_COLUMNS = [
    "timestamp",
    "symbol",
    "backend",
    "steps",
    "total_time_s",
    "avg_step_ms",
    "reward_sum",
    "equity_final",
    "position_final",
    "reward_mae",
    "reward_max",
    "equity_mae",
    "equity_max",
    "obs_max",
    "speedup",
]


@dataclass(slots=True)
class RunStats:
    backend: str
    symbol: str
    elapsed: float
    step_count: int
    rewards: list[float]
    equities: list[float]
    positions: list[float]
    observations: list[np.ndarray]

    @property
    def avg_step_ms(self) -> float:
        return (self.elapsed / max(self.step_count, 1)) * 1000.0

    @property
    def reward_sum(self) -> float:
        return float(sum(self.rewards))

    @property
    def equity_final(self) -> float:
        return float(self.equities[-1]) if self.equities else 0.0

    @property
    def position_final(self) -> float:
        return float(self.positions[-1]) if self.positions else 0.0


@dataclass(slots=True)
class DeltaStats:
    compared_steps: int
    reward_mae: float
    reward_max: float
    equity_mae: float
    equity_max: float
    obs_max: float
    speedup: float
    python_avg_ms: float
    fast_avg_ms: float


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FastMarketEnv with MarketEnv.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA"],
        help="Symbols to benchmark (default: AAPL MSFT NVDA).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("trainingdata"),
        help="Directory containing CSV price data (default: trainingdata).",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=64,
        help="Context window length passed to both envs (default: 64).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon (default: 1).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=512,
        help="Number of actions to replay (default: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for action generation and synthetic fallbacks (default: 1337).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string for FastMarketEnv (default: cpu).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/bench_fast_vs_python"),
        help="Prefix for CSV/JSON outputs (default: results/bench_fast_vs_python).",
    )
    return parser.parse_args(argv)


def _required_rows(context_len: int, steps: int, horizon: int) -> int:
    return context_len + steps + max(2, horizon) + 8


def _candidate_paths(root: Path, symbol: str) -> list[Path]:
    if not root.exists():
        return []
    upper = symbol.strip().upper()
    patterns = [f"**/{upper}.csv", f"**/{upper}_*.csv"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(root.glob(pattern)))
    if not paths:
        paths = sorted(root.glob("**/*.csv"))
    return paths


def _select_numeric_columns(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:  # type: ignore[name-defined]
    lowered = {col: str(col).lower() for col in frame.columns}
    frame = frame.rename(columns=lowered)
    frame = frame.dropna(axis=0, how="any")
    required = ["open", "high", "low", "close"]
    if any(col not in frame.columns for col in required):
        raise ValueError("missing OHLC columns")
    numeric_cols = list(required)
    for col in frame.columns:
        if col in required:
            continue
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):  # type: ignore[attr-defined]
            numeric_cols.append(col)
    trimmed = frame[numeric_cols].reset_index(drop=True)
    return numeric_cols, trimmed


def _load_prices_from_csv(path: Path, rows_needed: int) -> tuple[torch.Tensor, tuple[str, ...]] | None:
    if pd is None:
        return None
    try:
        frame = pd.read_csv(path, nrows=rows_needed + 2048)
    except Exception:
        frame = pd.read_csv(path)
    try:
        columns, trimmed = _select_numeric_columns(frame)
    except ValueError:
        return None
    if len(trimmed) < rows_needed:
        try:
            frame_full = pd.read_csv(path)
            columns, trimmed = _select_numeric_columns(frame_full)
        except Exception:
            return None
        if len(trimmed) < rows_needed:
            return None
    subset = trimmed.iloc[:rows_needed]
    tensor = torch.from_numpy(subset.to_numpy(dtype=np.float32))
    return tensor, tuple(columns)


def _synth_prices(rows: int, seed: int) -> tuple[torch.Tensor, tuple[str, ...]]:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    steps = torch.randn((rows,), generator=gen) * 0.02
    log_prices = steps.cumsum(dim=0)
    base = torch.exp(log_prices)
    open_ = base
    close = base * torch.exp(torch.randn_like(base, generator=gen) * 0.001)
    high = torch.maximum(open_, close) * (1.0 + torch.rand_like(base, generator=gen) * 0.01)
    low = torch.minimum(open_, close) * (1.0 - torch.rand_like(base, generator=gen) * 0.01)
    tensor = torch.stack([open_, high, low, close], dim=1).to(torch.float32)
    return tensor, ("open", "high", "low", "close")


def load_price_tensor(
    symbol: str,
    data_root: Path,
    rows_needed: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    for path in _candidate_paths(data_root, symbol):
        loaded = _load_prices_from_csv(path, rows_needed)
        if loaded is not None:
            try:
                rel = path.relative_to(data_root)
            except ValueError:
                rel = path
            print(f"[bench] Loaded {rows_needed} rows for {symbol} from {rel}")
            return loaded
    print(f"[bench] Falling back to synthetic prices for {symbol} (no usable CSV found).")
    return _synth_prices(rows_needed, seed)


def _base_env_config(args: argparse.Namespace) -> MarketEnvConfig:
    synth_len = max(_required_rows(args.context_len, args.steps, args.horizon) * 2, 4096)
    return MarketEnvConfig(
        context_len=args.context_len,
        horizon=args.horizon,
        action_space="continuous",
        reward_scale=1.0,
        random_reset=False,
        start_index=args.context_len,
        episode_length=args.steps,
        device="cpu",
        synth_T=synth_len,
    )


def _run_env(name: str, env, actions: Sequence[float], symbol: str) -> RunStats:
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    equities: list[float] = []
    positions: list[float] = []
    start = perf_counter()
    try:
        env.reset()
        for action in actions:
            obs, reward, done, truncated, info = env.step(float(action))
            observations.append(np.array(obs, copy=True))
            rewards.append(float(reward))
            equities.append(float(info.get("equity", getattr(env, "equity", 0.0))))
            positions.append(float(info.get("position", getattr(env, "position", 0.0))))
            if done or truncated:
                break
    finally:
        elapsed = perf_counter() - start
        env.close()
    return RunStats(
        backend=name,
        symbol=symbol,
        elapsed=elapsed,
        step_count=len(rewards),
        rewards=rewards,
        equities=equities,
        positions=positions,
        observations=observations,
    )


def _compare_runs(python_stats: RunStats, fast_stats: RunStats) -> DeltaStats:
    steps = min(python_stats.step_count, fast_stats.step_count)
    if steps == 0:
        return DeltaStats(
            compared_steps=0,
            reward_mae=float("nan"),
            reward_max=float("nan"),
            equity_mae=float("nan"),
            equity_max=float("nan"),
            obs_max=float("nan"),
            speedup=float("nan"),
            python_avg_ms=python_stats.avg_step_ms,
            fast_avg_ms=fast_stats.avg_step_ms,
        )
    py_rewards = np.asarray(python_stats.rewards[:steps], dtype=np.float64)
    fast_rewards = np.asarray(fast_stats.rewards[:steps], dtype=np.float64)
    reward_diff = np.abs(py_rewards - fast_rewards)
    py_equity = np.asarray(python_stats.equities[:steps], dtype=np.float64)
    fast_equity = np.asarray(fast_stats.equities[:steps], dtype=np.float64)
    equity_diff = np.abs(py_equity - fast_equity)
    obs_max = 0.0
    for py_obs, fast_obs in zip(python_stats.observations[:steps], fast_stats.observations[:steps]):
        obs_delta = np.max(np.abs(py_obs - fast_obs))
        if obs_delta > obs_max:
            obs_max = float(obs_delta)
    python_avg_ms = python_stats.avg_step_ms
    fast_avg_ms = fast_stats.avg_step_ms
    speedup = python_avg_ms / fast_avg_ms if fast_avg_ms > 0 and math.isfinite(python_avg_ms) else float("nan")
    return DeltaStats(
        compared_steps=steps,
        reward_mae=float(np.mean(reward_diff)),
        reward_max=float(np.max(reward_diff)),
        equity_mae=float(np.mean(equity_diff)),
        equity_max=float(np.max(equity_diff)),
        obs_max=float(obs_max),
        speedup=float(speedup),
        python_avg_ms=python_avg_ms,
        fast_avg_ms=fast_avg_ms,
    )


def _csv_row(stats: RunStats, timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": stats.symbol,
        "backend": stats.backend,
        "steps": stats.step_count,
        "total_time_s": round(stats.elapsed, 6),
        "avg_step_ms": round(stats.avg_step_ms, 6),
        "reward_sum": round(stats.reward_sum, 6),
        "equity_final": round(stats.equity_final, 6),
        "position_final": round(stats.position_final, 6),
        "reward_mae": "",
        "reward_max": "",
        "equity_mae": "",
        "equity_max": "",
        "obs_max": "",
        "speedup": "",
    }


def _delta_row(symbol: str, delta: DeltaStats, timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "backend": "delta",
        "steps": delta.compared_steps,
        "total_time_s": "",
        "avg_step_ms": "",
        "reward_sum": "",
        "equity_final": "",
        "position_final": "",
        "reward_mae": round(delta.reward_mae, 10),
        "reward_max": round(delta.reward_max, 10),
        "equity_mae": round(delta.equity_mae, 10),
        "equity_max": round(delta.equity_max, 10),
        "obs_max": round(delta.obs_max, 10),
        "speedup": round(delta.speedup, 6),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def benchmark_symbol(
    symbol: str,
    prices: torch.Tensor,
    columns: tuple[str, ...],
    cfg: MarketEnvConfig,
    actions: Sequence[float],
    device: str,
    timestamp: str,
) -> tuple[list[dict], dict]:
    python_env = MarketEnv(prices=prices.clone(), price_columns=columns, cfg=cfg)
    fast_cfg = replace(cfg, device=device)
    fast_env = FastMarketEnv(prices=prices.clone(), price_columns=columns, cfg=fast_cfg, device=device)
    py_stats = _run_env("python", python_env, actions, symbol)
    fast_stats = _run_env("fast", fast_env, actions, symbol)
    delta = _compare_runs(py_stats, fast_stats)
    print(
        f"[bench] {symbol}: python {py_stats.avg_step_ms:.3f} ms | fast {fast_stats.avg_step_ms:.3f} ms | "
        f"speedup {delta.speedup:.2f}x | obs Δ {delta.obs_max:.2e}"
    )
    csv_rows = [_csv_row(py_stats, timestamp), _csv_row(fast_stats, timestamp), _delta_row(symbol, delta, timestamp)]
    summary = {
        "python": {
            "steps": py_stats.step_count,
            "avg_step_ms": py_stats.avg_step_ms,
            "total_time_s": py_stats.elapsed,
            "reward_sum": py_stats.reward_sum,
            "equity_final": py_stats.equity_final,
        },
        "fast": {
            "steps": fast_stats.step_count,
            "avg_step_ms": fast_stats.avg_step_ms,
            "total_time_s": fast_stats.elapsed,
            "reward_sum": fast_stats.reward_sum,
            "equity_final": fast_stats.equity_final,
        },
        "delta": {
            "compared_steps": delta.compared_steps,
            "reward_mae": delta.reward_mae,
            "reward_max": delta.reward_max,
            "equity_mae": delta.equity_mae,
            "equity_max": delta.equity_max,
            "obs_max": delta.obs_max,
            "speedup": delta.speedup,
            "python_avg_ms": delta.python_avg_ms,
            "fast_avg_ms": delta.fast_avg_ms,
        },
    }
    return csv_rows, summary


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now(UTC).isoformat()
    rows_needed = _required_rows(args.context_len, args.steps, args.horizon)
    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(-1.0, 1.0, size=args.steps).astype(np.float32)
    cfg = _base_env_config(args)
    output_csv = args.output_prefix.with_suffix(".csv")
    output_json = args.output_prefix.with_suffix(".json")

    all_rows: list[dict] = []
    summary: dict = {
        "generated_at": timestamp,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "steps": args.steps,
        "symbols": {},
    }

    for symbol in args.symbols:
        prices, columns = load_price_tensor(symbol, args.data_root, rows_needed, seed=args.seed)
        csv_rows, symbol_summary = benchmark_symbol(
            symbol,
            prices,
            columns,
            cfg,
            actions,
            args.device,
            timestamp,
        )
        all_rows.extend(csv_rows)
        summary["symbols"][symbol.upper()] = symbol_summary

    _write_csv(output_csv, all_rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[bench] Wrote benchmark CSV to {output_csv}")
    print(f"[bench] Wrote benchmark JSON to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
