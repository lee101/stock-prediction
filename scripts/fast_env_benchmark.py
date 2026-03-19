from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig  # noqa: E402

from fastmarketsim import FastMarketEnv  # noqa: E402


BENCHMARK_METRICS: tuple[str, ...] = (
    "reward",
    "gross",
    "trading_cost",
    "financing_cost",
    "equity",
)
CSV_FIELDNAMES = [
    "metric",
    "count",
    "python_mean",
    "fast_mean",
    "python_std",
    "fast_std",
    "mean_delta",
    "abs_diff_mean",
    "rel_diff_mean",
    "max_abs_diff",
]


def _build_price_tensor(total_steps: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    steps = torch.randn((total_steps,), generator=generator) * 0.01
    log_prices = steps.cumsum(dim=0)
    base = torch.exp(log_prices)
    open_prices = base
    noise = torch.randn((total_steps,), generator=generator) * 0.001
    close_prices = base * torch.exp(noise)
    swing = torch.rand((total_steps,), generator=generator) * 0.01
    high_prices = torch.maximum(open_prices, close_prices) * (1.0 + swing)
    low_prices = torch.minimum(open_prices, close_prices) * (1.0 - swing)
    stacked = torch.stack([open_prices, high_prices, low_prices, close_prices], dim=1)
    return stacked.to(torch.float32).contiguous()


def _make_actions(num_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=num_steps).astype(np.float32)


@dataclass
class BenchmarkMeta:
    num_steps_requested: int
    num_steps_executed: int
    context_len: int
    horizon: int
    seed: int
    timestamp: str


def _collect_metrics(
    py_env: MarketEnv,
    fast_env: FastMarketEnv,
    actions: Sequence[float],
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[float], float, float, int]:
    py_metrics: dict[str, list[float]] = {key: [] for key in BENCHMARK_METRICS}
    fast_metrics: dict[str, list[float]] = {key: [] for key in BENCHMARK_METRICS}
    observation_diffs: list[float] = []
    py_runtime = 0.0
    fast_runtime = 0.0

    py_obs, _ = py_env.reset()
    fast_obs, _ = fast_env.reset()
    observation_diffs.append(float(np.max(np.abs(py_obs - fast_obs))))

    executed = 0
    for raw_action in actions:
        action_value = float(raw_action)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        py_obs, py_reward, py_done, py_truncated, py_info = py_env.step(action_value)
        py_runtime += time.perf_counter() - start_time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        fast_obs, fast_reward, fast_done, fast_truncated, fast_info = fast_env.step(action_value)
        fast_runtime += time.perf_counter() - start_time

        observation_diffs.append(float(np.max(np.abs(py_obs - fast_obs))))

        py_metrics["reward"].append(py_reward)
        fast_metrics["reward"].append(fast_reward)
        py_metrics["gross"].append(py_info.get("gross_pnl", 0.0))
        fast_metrics["gross"].append(fast_info.get("gross_pnl", 0.0))

        py_metrics["trading_cost"].append(py_info.get("trading_cost", 0.0))
        fast_trade_cost = fast_info.get("trading_cost", 0.0) + fast_info.get("deleverage_cost", 0.0)
        fast_metrics["trading_cost"].append(fast_trade_cost)

        py_metrics["financing_cost"].append(py_info.get("financing_cost", 0.0))
        fast_metrics["financing_cost"].append(fast_info.get("financing_cost", 0.0))

        py_metrics["equity"].append(py_info.get("equity", 0.0))
        fast_metrics["equity"].append(fast_info.get("equity", 0.0))

        executed += 1
        if py_done or py_truncated or fast_done or fast_truncated:
            break

    return py_metrics, fast_metrics, observation_diffs, py_runtime, fast_runtime, executed


def _summarize_metric(name: str, py_values: Iterable[float], fast_values: Iterable[float]) -> dict[str, float]:
    py_arr = np.asarray(list(py_values), dtype=np.float64)
    fast_arr = np.asarray(list(fast_values), dtype=np.float64)
    if py_arr.size == 0 or fast_arr.size == 0:
        return {
            "metric": name,
            "count": 0,
            "python_mean": 0.0,
            "fast_mean": 0.0,
            "python_std": 0.0,
            "fast_std": 0.0,
            "mean_delta": 0.0,
            "abs_diff_mean": 0.0,
            "rel_diff_mean": 0.0,
            "max_abs_diff": 0.0,
        }

    limit = min(py_arr.size, fast_arr.size)
    py_arr = py_arr[:limit]
    fast_arr = fast_arr[:limit]
    diff = fast_arr - py_arr
    abs_diff = np.abs(diff)
    py_mean = float(py_arr.mean())
    fast_mean = float(fast_arr.mean())
    denom = max(abs(py_mean), 1e-9)
    rel_diff = abs(fast_mean - py_mean) / denom

    return {
        "metric": name,
        "count": int(limit),
        "python_mean": py_mean,
        "fast_mean": fast_mean,
        "python_std": float(py_arr.std(ddof=0)),
        "fast_std": float(fast_arr.std(ddof=0)),
        "mean_delta": float(fast_mean - py_mean),
        "abs_diff_mean": float(abs_diff.mean()),
        "rel_diff_mean": float(rel_diff),
        "max_abs_diff": float(abs_diff.max()),
    }


def _summarize_observations(diffs: Sequence[float]) -> dict[str, float]:
    if not diffs:
        max_diff = 0.0
        mean_diff = 0.0
    else:
        arr = np.asarray(diffs, dtype=np.float64)
        max_diff = float(arr.max())
        mean_diff = float(arr.mean())
    return {
        "metric": "observation_max_abs_diff",
        "count": len(diffs),
        "python_mean": 0.0,
        "fast_mean": 0.0,
        "python_std": 0.0,
        "fast_std": 0.0,
        "mean_delta": mean_diff,
        "abs_diff_mean": mean_diff,
        "rel_diff_mean": 0.0,
        "max_abs_diff": max_diff,
    }


def _runtime_row(py_seconds: float, fast_seconds: float, count: int) -> dict[str, float]:
    denom = max(py_seconds, 1e-9)
    rel = (fast_seconds - py_seconds) / denom
    return {
        "metric": "runtime_seconds",
        "count": int(count),
        "python_mean": float(py_seconds),
        "fast_mean": float(fast_seconds),
        "python_std": 0.0,
        "fast_std": 0.0,
        "mean_delta": float(fast_seconds - py_seconds),
        "abs_diff_mean": abs(float(fast_seconds - py_seconds)),
        "rel_diff_mean": float(rel),
        "max_abs_diff": abs(float(fast_seconds - py_seconds)),
    }


def run_benchmark(
    *,
    num_steps: int,
    context_len: int,
    horizon: int,
    seed: int,
) -> tuple[list[dict[str, float]], BenchmarkMeta]:
    total_steps = num_steps + context_len + max(4, horizon + 2)
    prices = _build_price_tensor(total_steps, seed)
    price_columns = ("open", "high", "low", "close")

    cfg = MarketEnvConfig(
        context_len=context_len,
        horizon=horizon,
        device="cpu",
        random_reset=False,
        start_index=context_len,
        episode_length=num_steps + 1,
        synth_T=total_steps,
    )

    py_env = MarketEnv(prices=prices.clone(), price_columns=price_columns, cfg=cfg)
    fast_env = FastMarketEnv(prices=prices.clone(), price_columns=price_columns, cfg=cfg, device="cpu")

    try:
        actions = _make_actions(num_steps, seed)
        (
            py_metrics,
            fast_metrics,
            observation_diffs,
            py_runtime,
            fast_runtime,
            executed,
        ) = _collect_metrics(py_env, fast_env, actions)
    finally:
        py_env.close()
        fast_env.close()

    rows: list[dict[str, float]] = []
    for metric in BENCHMARK_METRICS:
        rows.append(_summarize_metric(metric, py_metrics[metric], fast_metrics[metric]))
    rows.append(_summarize_observations(observation_diffs))
    rows.append(_runtime_row(py_runtime, fast_runtime, executed))

    meta = BenchmarkMeta(
        num_steps_requested=num_steps,
        num_steps_executed=executed,
        context_len=context_len,
        horizon=horizon,
        seed=seed,
        timestamp=datetime.now(UTC).isoformat(),
    )
    return rows, meta


def _write_csv(rows: Sequence[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(meta: BenchmarkMeta, rows: Sequence[dict[str, float]], path: Path) -> None:
    payload = {
        "meta": meta.__dict__,
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fast market env against python reference")
    parser.add_argument("--num-steps", type=int, default=512, help="Number of benchmark steps to execute")
    parser.add_argument("--context-len", type=int, default=64, help="Context window length")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in steps")
    parser.add_argument("--seed", type=int, default=20250318, help="Random seed for reproducibility")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/bench_fast_vs_python.csv"),
        help="Path to write CSV summary",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/bench_fast_vs_python.json"),
        help="Path to write JSON payload",
    )
    args = parser.parse_args()

    rows, meta = run_benchmark(
        num_steps=args.num_steps,
        context_len=args.context_len,
        horizon=args.horizon,
        seed=args.seed,
    )
    _write_csv(rows, args.output_csv)
    _write_json(meta, rows, args.output_json)
    print(f"Wrote benchmark summary to {args.output_csv} and {args.output_json}")


if __name__ == "__main__":
    main()
