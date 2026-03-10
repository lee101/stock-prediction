#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from gymrl import FeatureBuilder, FeatureBuilderConfig
from gymrl.cache_utils import load_feature_cache, save_feature_cache
from gymrl.config import PortfolioEnvConfig
from gymrl.portfolio_env import PortfolioEnv


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str | None, *, base: Path) -> Path | None:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _coerce_symbols(value: Any, fallback_symbol: str) -> list[str]:
    if isinstance(value, str):
        parsed = [token.strip() for token in value.split(",") if token.strip()]
        if parsed:
            return parsed
    if isinstance(value, Iterable):
        parsed = [str(token).strip() for token in value if str(token).strip()]
        if parsed:
            return parsed
    return [fallback_symbol]


def _artifact_root_for_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_dir():
        artifact_root = checkpoint
    else:
        artifact_root = checkpoint.parent

    for candidate in (artifact_root, artifact_root.parent):
        if candidate != candidate.parent and (candidate / "training_metadata.json").exists():
            return candidate

    if artifact_root.name == "topk":
        return artifact_root.parent
    return artifact_root


def _load_training_metadata(checkpoint: Path) -> dict[str, Any]:
    metadata_path = _artifact_root_for_checkpoint(checkpoint) / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing training metadata next to checkpoint: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if not isinstance(metadata, dict):
        raise ValueError(f"Unexpected metadata format in {metadata_path}")
    return metadata


def _resolve_data_dir(raw_data_dir: str, symbols: list[str]) -> Path:
    root = _repo_root()
    candidate = _resolve_path(raw_data_dir, base=root)
    if candidate is None:
        raise ValueError("No data directory supplied in metadata.")
    if candidate.exists():
        symbol_file = candidate / f"{symbols[0]}.csv"
        if not symbol_file.exists():
            crypto_candidate = candidate / "crypto"
            crypto_symbol_file = crypto_candidate / f"{symbols[0]}.csv"
            if crypto_symbol_file.exists():
                return crypto_candidate
        return candidate
    raise FileNotFoundError(f"Training data directory not found: {candidate}")


def _ensure_feature_cache(
    *,
    checkpoint: Path,
    metadata: dict[str, Any],
    output_dir: Path,
    symbol: str,
) -> Path:
    args = metadata.get("args", {})
    if not isinstance(args, dict):
        args = {}

    existing = metadata.get("features_cache_path")
    resolved_existing = _resolve_path(existing, base=_repo_root()) if isinstance(existing, str) and existing else None
    if resolved_existing is not None and resolved_existing.exists():
        return resolved_existing

    default_cache = checkpoint.parent / "features_cache.npz"
    if default_cache.exists():
        return default_cache

    if "data_dir" not in args:
        raise ValueError(
            "Cannot rebuild feature cache: metadata args are missing data_dir. "
            "Re-run training with --cache-features-to or provide metadata with data_dir."
        )

    selected_symbols = _coerce_symbols(args.get("symbols"), symbol)
    data_dir = _resolve_data_dir(str(args["data_dir"]), selected_symbols)

    fill_method = args.get("fill_method", "ffill")
    if isinstance(fill_method, str) and fill_method.lower() == "none":
        fill_method = None

    config = FeatureBuilderConfig(
        forecast_backend=str(args.get("forecast_backend", "bootstrap")),
        num_samples=int(args.get("num_samples", 128)),
        context_window=int(args.get("context_window", 128)),
        prediction_length=int(args.get("prediction_length", 1)),
        realized_horizon=int(args.get("realized_horizon", 1)),
        fill_method=fill_method,
        enforce_common_index=bool(args.get("enforce_common_index", False)),
    )

    backend_kwargs: dict[str, Any] = {}
    for key in (
        "device_map",
        "kronos_device",
        "kronos_temperature",
        "kronos_top_p",
        "kronos_top_k",
        "kronos_sample_count",
        "kronos_max_context",
        "kronos_clip",
        "kronos_oom_retries",
        "kronos_jitter_std",
    ):
        if key in args and args[key] is not None:
            backend_kwargs[key] = args[key]

    builder = FeatureBuilder(config=config, backend_kwargs=backend_kwargs)
    cube = builder.build_from_directory(data_dir, symbols=selected_symbols)

    cache_path = output_dir / "features_cache.npz"
    save_feature_cache(
        cache_path,
        cube,
        extra_metadata={
            "source_checkpoint": str(checkpoint),
            "selected_symbols": selected_symbols,
            "rebuilt_for_evaluation": True,
        },
    )
    return cache_path


def _build_env_config(metadata: dict[str, Any], fill_buffer_bps: float) -> PortfolioEnvConfig:
    raw_cfg = metadata.get("env_config", {})
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    cfg = dict(raw_cfg)
    base_cost = float(cfg.get("costs_bps", 3.0))
    cfg["costs_bps"] = base_cost + float(fill_buffer_bps)

    per_asset = cfg.get("per_asset_costs_bps")
    if isinstance(per_asset, Iterable) and not isinstance(per_asset, (str, bytes)):
        cfg["per_asset_costs_bps"] = [float(v) + float(fill_buffer_bps) for v in per_asset]

    valid_keys = {item.name for item in fields(PortfolioEnvConfig)}
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}
    return PortfolioEnvConfig(**filtered)


def _evaluate_window(
    *,
    model: PPO,
    env: PortfolioEnv,
    periods_per_year: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    obs, _ = env.reset(options={"start_index": env.start_index})

    net_returns: list[float] = []
    turnovers: list[float] = []
    drawdowns: list[float] = []
    trading_costs: list[float] = []
    trajectory_rows: list[dict[str, object]] = []
    trade_steps = 0
    buy_steps = 0
    sell_steps = 0
    prev_crypto_weight = 0.0
    prev_equity = 1.0
    timestamps = getattr(env, "timestamps", None)

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        net = float(info.get("net_return", 0.0))
        turnover = float(info.get("turnover", 0.0))
        drawdown = float(info.get("drawdown", 0.0))
        trading_cost = float(info.get("trading_cost", 0.0))
        crypto_weight = float(info.get("weight_crypto", prev_crypto_weight))

        net_returns.append(net)
        turnovers.append(turnover)
        drawdowns.append(drawdown)
        trading_costs.append(trading_cost)

        if abs(turnover) > 1e-8:
            trade_steps += 1
        delta = crypto_weight - prev_crypto_weight
        if abs(delta) > 1e-8:
            if delta > 0:
                buy_steps += 1
            else:
                sell_steps += 1
        prev_crypto_weight = crypto_weight
        equity = float(info.get("portfolio_value", prev_equity * (1.0 + net)))
        gross_exposure_close = float(info.get("gross_exposure_close", abs(crypto_weight)))
        timestamp_index = int(getattr(env, "start_index", 0)) + len(trajectory_rows)
        timestamp = None
        if timestamps is not None and 0 <= timestamp_index < len(timestamps):
            raw_timestamp = timestamps[timestamp_index]
            timestamp = raw_timestamp.isoformat() if hasattr(raw_timestamp, "isoformat") else str(raw_timestamp)
        trajectory_rows.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "net_return": net,
                "turnover": turnover,
                "drawdown": drawdown,
                "trading_cost": trading_cost,
                "weight_crypto": crypto_weight,
                "gross_exposure_close": gross_exposure_close,
                "in_position": gross_exposure_close > 1e-8,
            }
        )
        prev_equity = equity

        if terminated or truncated:
            break

    returns = np.asarray(net_returns, dtype=np.float64)
    if returns.size:
        total_return = float(np.prod(1.0 + returns) - 1.0)
        mean_return = float(returns.mean())
        downside = np.minimum(returns, 0.0)
        downside_dev = float(np.sqrt(np.mean(np.square(downside))))
        if downside_dev > 1e-12:
            sortino = float((mean_return / downside_dev) * math.sqrt(periods_per_year))
        else:
            sortino = 0.0
    else:
        total_return = 0.0
        mean_return = 0.0
        sortino = 0.0

    max_drawdown = float(np.max(drawdowns)) if drawdowns else 0.0
    mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    mean_trading_cost = float(np.mean(trading_costs)) if trading_costs else 0.0

    metrics = {
        "total_return": total_return,
        "sortino": sortino,
        "mean_hourly_return": mean_return,
        "max_drawdown": max_drawdown,
        "mean_turnover": mean_turnover,
        "mean_trading_cost": mean_trading_cost,
        "fills_total": int(trade_steps),
        "fills_buy": int(buy_steps),
        "fills_sell": int(sell_steps),
    }
    return metrics, pd.DataFrame(trajectory_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one ETH PPO checkpoint across multiple windows and cost buffers.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--symbol", default="ETHUSD")
    parser.add_argument("--windows-hours", default="24,168,720", help="Comma-separated lookback windows in hours.")
    parser.add_argument(
        "--fill-buffers-bps",
        default="0,5,10",
        help="Comma-separated additional cost buffers in basis points (applied to env costs_bps).",
    )
    parser.add_argument("--forecast-cache-root", default="binanceneural/forecast_cache")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-only", action="store_true", help="Kept for CLI compatibility; ignored for GymRL checkpoint evaluation.")
    parser.add_argument("--features-cache", type=Path, default=None, help="Optional feature cache override.")
    parser.add_argument("--periods-per-year", type=float, default=24.0 * 365.0, help="Annualisation factor used for Sortino.")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    windows = _parse_int_list(args.windows_hours)
    buffers = _parse_float_list(args.fill_buffers_bps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_training_metadata(checkpoint)
    features_cache = args.features_cache
    if features_cache is None:
        features_cache = _ensure_feature_cache(checkpoint=checkpoint, metadata=metadata, output_dir=args.output_dir, symbol=args.symbol)
    features_cache = features_cache.expanduser().resolve()

    cube, _ = load_feature_cache(features_cache)
    model = PPO.load(str(checkpoint))
    total_steps = int(cube.features.shape[0])

    rows: list[dict[str, float | int | str]] = []
    for hours in windows:
        episode_steps = max(2, min(int(hours), total_steps - 1))
        start_index = max(0, total_steps - episode_steps - 1)
        for fill_buffer_bps in buffers:
            run_dir = args.output_dir / f"h{hours}_fb{fill_buffer_bps:g}"
            run_dir.mkdir(parents=True, exist_ok=True)

            env_config = _build_env_config(metadata, fill_buffer_bps)
            env = PortfolioEnv(
                cube.features,
                cube.realized_returns,
                config=env_config,
                feature_names=cube.feature_names,
                symbols=cube.symbols,
                timestamps=cube.timestamps,
                forecast_cvar=cube.forecast_cvar,
                forecast_uncertainty=cube.forecast_uncertainty,
                start_index=start_index,
                episode_length=episode_steps,
            )
            metrics, trajectory = _evaluate_window(model=model, env=env, periods_per_year=float(args.periods_per_year))
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            trajectory.to_csv(run_dir / "trajectory.csv", index=False)

            rows.append(
                {
                    "window_hours": int(hours),
                    "eval_steps": int(episode_steps),
                    "fill_buffer_bps": float(fill_buffer_bps),
                    "effective_costs_bps": float(env_config.costs_bps),
                    "total_return": float(metrics["total_return"]),
                    "sortino": float(metrics["sortino"]),
                    "mean_hourly_return": float(metrics["mean_hourly_return"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "mean_turnover": float(metrics["mean_turnover"]),
                    "mean_trading_cost": float(metrics["mean_trading_cost"]),
                    "fills_total": int(metrics["fills_total"]),
                    "fills_buy": int(metrics["fills_buy"]),
                    "fills_sell": int(metrics["fills_sell"]),
                    "features_cache": str(features_cache),
                    "sim_dir": str(run_dir),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["window_hours", "fill_buffer_bps"]).reset_index(drop=True)
    summary_csv = args.output_dir / "summary.csv"
    summary_json = args.output_dir / "summary.json"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
