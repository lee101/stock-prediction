from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def _load_local_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, CURRENT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load local module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


try:
    from data import FeatureNormalizer, HourlyMarketData, apply_feature_normalizer, load_hourly_market_data
except Exception:
    _data_mod = _load_local_module("rl_trainingbinance_validate_data", "data.py")
    FeatureNormalizer = _data_mod.FeatureNormalizer
    HourlyMarketData = _data_mod.HourlyMarketData
    apply_feature_normalizer = _data_mod.apply_feature_normalizer
    load_hourly_market_data = _data_mod.load_hourly_market_data

try:
    from env import BinanceHourlyEnv, EnvConfig
except Exception:
    _env_mod = _load_local_module("rl_trainingbinance_validate_env", "env.py")
    BinanceHourlyEnv = _env_mod.BinanceHourlyEnv
    EnvConfig = _env_mod.EnvConfig

try:
    from model import PolicyConfig, RiskAwareActorCritic
except Exception:
    _model_mod = _load_local_module("rl_trainingbinance_validate_model", "model.py")
    PolicyConfig = _model_mod.PolicyConfig
    RiskAwareActorCritic = _model_mod.RiskAwareActorCritic

try:
    from presets import parse_symbols
except Exception:
    _presets_mod = _load_local_module("rl_trainingbinance_validate_presets", "presets.py")
    parse_symbols = _presets_mod.parse_symbols

SUMMARY_KEYS = (
    "median_total_return",
    "p10_total_return",
    "median_sortino",
    "p90_max_drawdown",
    "median_volatility",
    "mean_turnover",
)


@dataclass(frozen=True)
class WindowMetric:
    start_index: int
    end_index: int
    total_return: float
    annualized_return: float
    sortino: float
    max_drawdown: float
    volatility: float
    mean_turnover: float


def parse_csv_ints(value: int | str | Iterable[int]) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise ValueError("Expected at least one integer value.")
        return [int(part) for part in parts]
    values = [int(item) for item in value]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_csv_floats(value: float | str | Iterable[float]) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise ValueError("Expected at least one float value.")
        return [float(part) for part in parts]
    values = [float(item) for item in value]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def format_window_label(window_hours: int) -> str:
    hours = int(window_hours)
    if hours % 24 == 0:
        return f"{hours // 24}d"
    return f"{hours}h"


def resolve_window_weights(window_hours: list[int], raw_weights: float | str | Iterable[float] | None) -> list[float]:
    if not window_hours:
        raise ValueError("window_hours must not be empty.")
    if raw_weights is None:
        equal_weight = 1.0 / float(len(window_hours))
        return [equal_weight for _ in window_hours]
    weights = parse_csv_floats(raw_weights)
    if len(weights) != len(window_hours):
        raise ValueError("window_weights must match window_hours length.")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("window_weights must sum to a positive number.")
    return [float(weight / total) for weight in weights]


def resolve_stride_hours(window_hours: list[int], raw_stride: int | str | Iterable[int] | None) -> list[int | None]:
    if raw_stride is None:
        return [None for _ in window_hours]
    if isinstance(raw_stride, (list, tuple)):
        if not raw_stride:
            return [None for _ in window_hours]
        if all(item is None for item in raw_stride):
            return [None for _ in window_hours]
    strides = parse_csv_ints(raw_stride)
    if len(strides) == 1:
        return [int(strides[0]) for _ in window_hours]
    if len(strides) != len(window_hours):
        raise ValueError("stride_hours must be a single value or match window_hours length.")
    return [int(stride) for stride in strides]


def build_walk_forward_windows(
    *,
    num_steps: int,
    lookback: int,
    window_hours: int,
    purge_hours: int,
    stride_hours: int | None = None,
) -> list[tuple[int, int]]:
    if window_hours < 1:
        raise ValueError("window_hours must be >= 1")
    if purge_hours < 0:
        raise ValueError("purge_hours must be >= 0")
    stride = int(stride_hours) if stride_hours is not None else int(window_hours) + int(purge_hours)
    if stride < 1:
        raise ValueError("stride_hours must be >= 1")

    windows: list[tuple[int, int]] = []
    start = int(lookback)
    last_start = int(num_steps) - int(window_hours) - 1
    while start <= last_start:
        windows.append((start, start + int(window_hours)))
        start += stride
    return windows


def _annualized_return(total_return: float, periods: int, *, periods_per_year: float = 365.0 * 24.0) -> float:
    if periods <= 0:
        return 0.0
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    years = float(periods) / float(periods_per_year)
    if years <= 0.0:
        return 0.0
    return float(base ** (1.0 / years) - 1.0)


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if equity_curve.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = 1.0 - equity_curve / np.maximum(peaks, 1e-8)
    return float(np.max(drawdowns))


def _sortino(step_returns: np.ndarray) -> float:
    if step_returns.size == 0:
        return 0.0
    downside = np.minimum(step_returns, 0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 1e-8:
        return float(np.mean(step_returns) * np.sqrt(365.0 * 24.0))
    return float(np.mean(step_returns) / downside_std * np.sqrt(365.0 * 24.0))


def run_window_validation(
    *,
    market: HourlyMarketData,
    model: RiskAwareActorCritic,
    env_config: EnvConfig,
    window: tuple[int, int],
    device: torch.device,
) -> WindowMetric:
    start, end = int(window[0]), int(window[1])
    window_steps = max(1, end - start)
    env = BinanceHourlyEnv(
        market,
        EnvConfig(**{**env_config.to_dict(), "episode_steps": window_steps, "random_reset": False}),
        slice_start=start,
        slice_end=end + 1,
        seed=0,
    )
    obs = env.reset(start_index=start)
    equities = [float(env.equity)]
    step_returns: list[float] = []
    turnovers: list[float] = []
    done = False
    while not done:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            action, _ = model.predict_deterministic(obs_tensor)
        obs, _, done, info = env.step(action.squeeze(0).cpu().numpy())
        equities.append(float(info["equity"]))
        step_returns.append(float(info["step_return"]))
        turnovers.append(float(info["turnover"]))
    equity_curve = np.asarray(equities, dtype=np.float64)
    step_returns_arr = np.asarray(step_returns, dtype=np.float64)
    total_return = float(equity_curve[-1] / max(equity_curve[0], 1e-8) - 1.0)
    return WindowMetric(
        start_index=start,
        end_index=end,
        total_return=total_return,
        annualized_return=_annualized_return(total_return, len(step_returns)),
        sortino=_sortino(step_returns_arr),
        max_drawdown=_max_drawdown(equity_curve),
        volatility=float(np.std(step_returns_arr, ddof=0) * np.sqrt(365.0 * 24.0)) if step_returns_arr.size else 0.0,
        mean_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
    )


def run_window_validation_batch(
    *,
    market: HourlyMarketData,
    model: RiskAwareActorCritic,
    env_config: EnvConfig,
    windows: list[tuple[int, int]],
    device: torch.device,
    batch_size: int | None = None,
) -> list[WindowMetric]:
    if not windows:
        return []
    if batch_size is None:
        batch_size = len(windows)
    chunk_size = max(1, int(batch_size))
    results: list[WindowMetric] = []

    for start_idx in range(0, len(windows), chunk_size):
        chunk = windows[start_idx : start_idx + chunk_size]
        envs: list[BinanceHourlyEnv] = []
        obs_list: list[np.ndarray] = []
        equities: list[list[float]] = []
        step_returns: list[list[float]] = []
        turnovers: list[list[float]] = []
        done_flags: list[bool] = []

        for window in chunk:
            start, end = int(window[0]), int(window[1])
            window_steps = max(1, end - start)
            env = BinanceHourlyEnv(
                market,
                EnvConfig(**{**env_config.to_dict(), "episode_steps": window_steps, "random_reset": False}),
                slice_start=start,
                slice_end=end + 1,
                seed=0,
            )
            obs = env.reset(start_index=start)
            envs.append(env)
            obs_list.append(obs.astype(np.float32, copy=False))
            equities.append([float(env.equity)])
            step_returns.append([])
            turnovers.append([])
            done_flags.append(False)

        while not all(done_flags):
            active_indices = [idx for idx, done in enumerate(done_flags) if not done]
            obs_batch = np.stack([obs_list[idx] for idx in active_indices], axis=0)
            obs_tensor = torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                action_batch, _ = model.predict_deterministic(obs_tensor)
            action_np = action_batch.cpu().numpy()

            for local_idx, env_idx in enumerate(active_indices):
                obs, _, done, info = envs[env_idx].step(action_np[local_idx])
                equities[env_idx].append(float(info["equity"]))
                step_returns[env_idx].append(float(info["step_return"]))
                turnovers[env_idx].append(float(info["turnover"]))
                done_flags[env_idx] = bool(done)
                if not done:
                    obs_list[env_idx] = obs.astype(np.float32, copy=False)

        for env_idx, window in enumerate(chunk):
            start, end = int(window[0]), int(window[1])
            equity_curve = np.asarray(equities[env_idx], dtype=np.float64)
            step_returns_arr = np.asarray(step_returns[env_idx], dtype=np.float64)
            results.append(
                WindowMetric(
                    start_index=start,
                    end_index=end,
                    total_return=float(equity_curve[-1] / max(equity_curve[0], 1e-8) - 1.0),
                    annualized_return=_annualized_return(
                        float(equity_curve[-1] / max(equity_curve[0], 1e-8) - 1.0),
                        len(step_returns_arr),
                    ),
                    sortino=_sortino(step_returns_arr),
                    max_drawdown=_max_drawdown(equity_curve),
                    volatility=float(np.std(step_returns_arr, ddof=0) * np.sqrt(365.0 * 24.0))
                    if step_returns_arr.size
                    else 0.0,
                    mean_turnover=float(np.mean(turnovers[env_idx])) if turnovers[env_idx] else 0.0,
                )
            )
    return results


def summarize_metrics(metrics: list[WindowMetric]) -> dict[str, float]:
    if not metrics:
        summary = {
            "median_total_return": 0.0,
            "p10_total_return": 0.0,
            "median_sortino": 0.0,
            "p90_max_drawdown": 1.0,
            "median_volatility": 0.0,
            "mean_turnover": 0.0,
        }
        summary["score"] = risk_adjusted_score(summary)
        return summary
    total_returns = np.asarray([item.total_return for item in metrics], dtype=np.float64)
    sortinos = np.asarray([item.sortino for item in metrics], dtype=np.float64)
    drawdowns = np.asarray([item.max_drawdown for item in metrics], dtype=np.float64)
    vols = np.asarray([item.volatility for item in metrics], dtype=np.float64)
    turnovers = np.asarray([item.mean_turnover for item in metrics], dtype=np.float64)
    summary = {
        "median_total_return": float(np.median(total_returns)),
        "p10_total_return": float(np.percentile(total_returns, 10)),
        "median_sortino": float(np.median(sortinos)),
        "p90_max_drawdown": float(np.percentile(drawdowns, 90)),
        "median_volatility": float(np.median(vols)),
        "mean_turnover": float(np.mean(turnovers)),
    }
    summary["score"] = risk_adjusted_score(summary)
    return summary


def risk_adjusted_score(summary: dict[str, float]) -> float:
    return float(
        summary.get("p10_total_return", 0.0)
        + 0.15 * max(summary.get("median_sortino", 0.0), 0.0)
        - 1.50 * summary.get("p90_max_drawdown", 0.0)
        - 0.20 * summary.get("median_volatility", 0.0)
        - 0.05 * summary.get("mean_turnover", 0.0)
    )


def aggregate_window_summaries(
    window_summaries: dict[str, dict[str, float]],
    window_weights: dict[str, float],
) -> dict[str, float]:
    if not window_summaries:
        return summarize_metrics([])
    aggregate = {
        key: float(
            sum(
                float(window_weights.get(label, 0.0)) * float(summary.get(key, 0.0))
                for label, summary in window_summaries.items()
            )
        )
        for key in SUMMARY_KEYS
    }
    aggregate["score"] = risk_adjusted_score(aggregate)
    return aggregate


def flatten_window_summaries(window_summaries: dict[str, dict[str, float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for label, summary in window_summaries.items():
        safe_label = str(label).replace("-", "_")
        for key, value in summary.items():
            flat[f"{key}_{safe_label}"] = float(value)
    return flat


def evaluate_validation_plan(
    *,
    market: HourlyMarketData,
    model: RiskAwareActorCritic,
    env_config: EnvConfig,
    window_hours: list[int],
    purge_hours: int,
    stride_hours: list[int | None],
    window_weights: list[float],
    device: torch.device,
    batch_size: int | None = None,
) -> dict[str, Any]:
    if len(window_hours) != len(stride_hours) or len(window_hours) != len(window_weights):
        raise ValueError("window_hours, stride_hours, and window_weights must have matching lengths.")

    labels = [format_window_label(hours) for hours in window_hours]
    if len(set(labels)) != len(labels):
        raise ValueError("window_hours must produce unique labels.")

    window_results: dict[str, dict[str, Any]] = {}
    for label, hours, stride, weight in zip(labels, window_hours, stride_hours, window_weights, strict=True):
        windows = build_walk_forward_windows(
            num_steps=len(market),
            lookback=env_config.lookback,
            window_hours=int(hours),
            purge_hours=int(purge_hours),
            stride_hours=stride,
        )
        metrics = run_window_validation_batch(
            market=market,
            model=model,
            env_config=env_config,
            windows=windows,
            device=device,
            batch_size=batch_size,
        )
        window_results[label] = {
            "window_hours": int(hours),
            "weight": float(weight),
            "num_windows": len(metrics),
            "summary": summarize_metrics(metrics),
            "windows": [asdict(metric) for metric in metrics],
        }

    summaries = {label: dict(result["summary"]) for label, result in window_results.items()}
    weight_map = {label: float(result["weight"]) for label, result in window_results.items()}
    return {
        "aggregate_summary": aggregate_window_summaries(summaries, weight_map),
        "window_summaries": summaries,
        "window_results": window_results,
        "window_weights": weight_map,
    }


def load_checkpoint(path: str | Path, *, device: torch.device) -> tuple[RiskAwareActorCritic, dict[str, Any]]:
    payload = torch.load(str(Path(path)), map_location=device, weights_only=False)
    model_cfg = PolicyConfig(**payload["model_config"])
    model = RiskAwareActorCritic(model_cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation for rl-trainingbinance checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--window-hours", default=None)
    parser.add_argument("--window-weights", default=None)
    parser.add_argument("--purge-hours", type=int, default=24)
    parser.add_argument("--stride-hours", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--ignore-checkpoint-validation-slice",
        action="store_true",
        help="Evaluate on the full aligned market instead of the validation slice stored in the checkpoint.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device=device)
    env_cfg = EnvConfig(**payload["env_config"])
    symbols = parse_symbols(args.symbols, default=payload["symbols"])
    market = load_hourly_market_data(
        data_root=args.data_root,
        symbols=symbols,
        shortable_symbols=payload.get("shortable_symbols"),
    )
    normalizer = FeatureNormalizer.from_dict(payload.get("feature_normalizer"))
    if normalizer is not None:
        market = apply_feature_normalizer(market, normalizer)
    validation_range = payload.get("validation_market_range") if isinstance(payload, dict) else None
    if validation_range and not args.ignore_checkpoint_validation_slice:
        market = market.slice(
            int(validation_range.get("validation_slice_start", 0)),
            int(validation_range.get("validation_slice_end", len(market))),
        )
    validation_window_config = payload.get("validation_window_config") if isinstance(payload, dict) else None
    raw_window_hours = args.window_hours
    if raw_window_hours is None and validation_window_config:
        raw_window_hours = validation_window_config.get("window_hours")
    if raw_window_hours is None:
        raw_window_hours = str(24 * 7)
    raw_window_weights = args.window_weights
    if raw_window_weights is None and validation_window_config:
        raw_window_weights = validation_window_config.get("window_weights")
    raw_stride_hours = args.stride_hours
    if raw_stride_hours is None and validation_window_config:
        raw_stride_hours = validation_window_config.get("stride_hours")

    window_hours = parse_csv_ints(raw_window_hours)
    window_weights = resolve_window_weights(window_hours, raw_window_weights)
    stride_hours = resolve_stride_hours(window_hours, raw_stride_hours)
    evaluation = evaluate_validation_plan(
        market=market,
        model=model,
        env_config=env_cfg,
        window_hours=window_hours,
        purge_hours=int(args.purge_hours),
        stride_hours=stride_hours,
        window_weights=window_weights,
        device=device,
        batch_size=args.batch_size,
    )
    output = {
        "checkpoint": str(Path(args.checkpoint)),
        "symbols": list(market.symbols),
        "summary": evaluation["aggregate_summary"],
        "window_summaries": evaluation["window_summaries"],
        "window_weights": evaluation["window_weights"],
        "window_results": evaluation["window_results"],
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
