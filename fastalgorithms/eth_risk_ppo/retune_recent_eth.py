#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastalgorithms.per_stock.meta_selector import (
    MetaSelectorConfig,
    align_equity_curves,
    compute_sortino,
    run_meta_simulation,
)
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "analysis" / "eth_risk_ppo"
DEFAULT_TUNE_SYMBOLS = ("ETHUSD", "BTCUSD", "SOLUSD", "LINKUSD", "UNIUSD")


@dataclass(frozen=True)
class TrainingCandidate:
    name: str
    description: str
    feature_cache_key: str
    env_overrides: dict[str, str]
    family: str
    seed: int


def _parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parse_int_list(raw: str) -> list[int]:
    values = [int(token) for token in _parse_csv_tokens(raw)]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_float_list(raw: str) -> list[float]:
    values = [float(token) for token in _parse_csv_tokens(raw)]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _default_output_dir() -> Path:
    return DEFAULT_ANALYSIS_ROOT / f"recent_retune_{_timestamp_tag()}"


def _candidate(
    name: str,
    description: str,
    feature_cache_key: str,
    *,
    family: str | None = None,
    seed: int = 42,
    **env_overrides: str,
) -> TrainingCandidate:
    return TrainingCandidate(
        name=name,
        description=description,
        feature_cache_key=feature_cache_key,
        env_overrides={str(key): str(value) for key, value in env_overrides.items()},
        family=str(family or name),
        seed=int(seed),
    )


def build_default_candidates(*, seeds: Iterable[int] | None = None) -> list[TrainingCandidate]:
    seed_values = list(seeds or [42])
    base = {
        "LEARNING_RATE": "4e-5",
        "BATCH_SIZE": "256",
        "N_STEPS": "512",
        "N_EPOCHS": "8",
        "GAMMA": "0.996",
        "GAE_LAMBDA": "0.99",
        "TARGET_KL": "0.012",
        "COSTS_BPS": "3.0",
        "TURNOVER_PENALTY": "0.0018",
        "DRAWDOWN_PENALTY": "0.08",
        "CVAR_PENALTY": "0.35",
        "UNCERTAINTY_PENALTY": "0.04",
        "REGIME_DRAWDOWN_THRESHOLD": "0.02",
        "REGIME_LEVERAGE_SCALE": "0.20",
        "LEVERAGE_HEAD": "0",
        "MAX_GROSS_LEVERAGE": "1.0",
        "INTRADAY_LEVERAGE_CAP": "1.15",
        "POLICY_DTYPE": "float32",
        "DEVICE": "cuda",
        "DETERMINISTIC_TRAINING": "1",
        "DISABLE_TF32": "1",
        "TORCH_NUM_THREADS": "1",
    }

    chronos = {
        **base,
        "FORECAST_BACKEND": "chronos2",
        "NUM_SAMPLES": "64",
        "CONTEXT_WINDOW": "192",
        "PREDICTION_LENGTH": "6",
        "REALIZED_HORIZON": "6",
        "CHRONOS2_HORIZONS": "1,6",
        "CHRONOS2_CONTEXT_HOURS": "1024",
        "CHRONOS2_BATCH_SIZE": "16",
    }

    base_candidates = [
        _candidate(
            "bootstrap_h1_core",
            "Bootstrap 1h ETH baseline.",
            "bootstrap_h1",
            family="bootstrap_h1_core",
            **{
                **base,
                "FORECAST_BACKEND": "bootstrap",
                "NUM_SAMPLES": "128",
                "CONTEXT_WINDOW": "128",
                "PREDICTION_LENGTH": "1",
                "REALIZED_HORIZON": "1",
            },
        ),
        _candidate(
            "bootstrap_h1_lowturn",
            "Bootstrap 1h with tighter turnover control.",
            "bootstrap_h1",
            family="bootstrap_h1_lowturn",
            **{
                **base,
                "FORECAST_BACKEND": "bootstrap",
                "NUM_SAMPLES": "128",
                "CONTEXT_WINDOW": "128",
                "PREDICTION_LENGTH": "1",
                "REALIZED_HORIZON": "1",
                "TURNOVER_PENALTY": "0.0040",
                "DRAWDOWN_PENALTY": "0.10",
                "CVAR_PENALTY": "0.45",
                "REGIME_LEVERAGE_SCALE": "0.16",
            },
        ),
        _candidate(
            "chronos2_h6_ctx1024",
            "Chronos2 6h hourly-tuned univariate ETH forecast features with 1024h context.",
            "chronos2_h6_ctx1024",
            family="chronos2_h6_ctx1024",
            **{
                **chronos,
                "TURNOVER_PENALTY": "0.0022",
            },
        ),
        _candidate(
            "chronos2_h6_ctx1024_lowturn",
            "Chronos2 6h univariate ETH features with tighter turnover control.",
            "chronos2_h6_ctx1024",
            family="chronos2_h6_ctx1024_lowturn",
            **{
                **chronos,
                "TURNOVER_PENALTY": "0.0040",
                "DRAWDOWN_PENALTY": "0.10",
                "CVAR_PENALTY": "0.45",
                "REGIME_LEVERAGE_SCALE": "0.16",
            },
        ),
        _candidate(
            "chronos2_h6_joint_ctx1024",
            "Chronos2 6h univariate ETH features with joint-batch Chronos inference enabled.",
            "chronos2_h6_joint_ctx1024",
            family="chronos2_h6_joint_ctx1024",
            **{
                **chronos,
                "CHRONOS2_FORCE_CROSS_LEARNING": "1",
                "TURNOVER_PENALTY": "0.0024",
                "DRAWDOWN_PENALTY": "0.09",
            },
        ),
        _candidate(
            "chronos2_h6_ctx1024_cost8",
            "Chronos2 6h univariate ETH features trained under 8bps costs.",
            "chronos2_h6_ctx1024",
            family="chronos2_h6_ctx1024_cost8",
            **{
                **chronos,
                "COSTS_BPS": "8.0",
                "TURNOVER_PENALTY": "0.0032",
                "DRAWDOWN_PENALTY": "0.10",
                "CVAR_PENALTY": "0.45",
                "REGIME_LEVERAGE_SCALE": "0.16",
            },
        ),
        _candidate(
            "chronos2_h6_regime_tight",
            "Chronos2 6h univariate ETH features with tight risk and cash bias.",
            "chronos2_h6_ctx1024",
            family="chronos2_h6_regime_tight",
            **{
                **chronos,
                "N_EPOCHS": "12",
                "TARGET_KL": "0.008",
                "TURNOVER_PENALTY": "0.0040",
                "DRAWDOWN_PENALTY": "0.12",
                "CVAR_PENALTY": "0.50",
                "REGIME_DRAWDOWN_THRESHOLD": "0.015",
                "REGIME_LEVERAGE_SCALE": "0.15",
                "WEIGHT_CAP": "0.75",
            },
        ),
        _candidate(
            "chronos2_h6_cash_high",
            "Chronos2 6h univariate ETH features with stronger cash preference.",
            "chronos2_h6_ctx1024",
            family="chronos2_h6_cash_high",
            **{
                **chronos,
                "N_EPOCHS": "12",
                "TURNOVER_PENALTY": "0.0035",
                "DRAWDOWN_PENALTY": "0.12",
                "CVAR_PENALTY": "0.45",
                "REGIME_LEVERAGE_SCALE": "0.15",
                "WEIGHT_CAP": "0.55",
            },
        ),
    ]

    candidates: list[TrainingCandidate] = []
    for template in base_candidates:
        for seed in seed_values:
            name = f"{template.name}_s{seed}"
            description = f"{template.description} Seed {seed}."
            candidates.append(
                TrainingCandidate(
                    name=name,
                    description=description,
                    feature_cache_key=template.feature_cache_key,
                    env_overrides={**template.env_overrides, "SEED": str(seed)},
                    family=template.family,
                    seed=int(seed),
                )
            )
    return candidates


def _resolve_symbol_csv(data_root: Path, symbol: str) -> Path:
    candidates = (
        data_root / f"{symbol}.csv",
        data_root / "crypto" / f"{symbol}.csv",
        data_root / "stocks" / f"{symbol}.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing hourly data for {symbol} under {data_root}")


def slice_recent_hourly_data(
    *,
    data_root: Path,
    symbols: Iterable[str],
    recent_hours: int,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}

    for symbol in symbols:
        src_path = _resolve_symbol_csv(data_root, symbol)
        frame = pd.read_csv(src_path)
        if "timestamp" not in frame.columns:
            raise ValueError(f"Missing timestamp column in {src_path}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if recent_hours > 0 and len(frame) > recent_hours:
            frame = frame.iloc[-int(recent_hours):].reset_index(drop=True)
        dest_path = output_dir / f"{symbol}.csv"
        frame.to_csv(dest_path, index=False)
        manifest[symbol] = {
            "source_path": str(src_path),
            "output_path": str(dest_path),
            "rows": int(len(frame)),
            "start": frame["timestamp"].iloc[0].isoformat() if not frame.empty else None,
            "end": frame["timestamp"].iloc[-1].isoformat() if not frame.empty else None,
        }

    return manifest


def build_buy_hold_benchmark(
    *,
    csv_path: Path,
    windows_hours: Iterable[int],
    fill_buffers_bps: Iterable[float],
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    close_col = next((name for name in ("close", "Close", "c", "price", "last") if name in frame.columns), None)
    if close_col is None:
        numeric = [name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])]
        if not numeric:
            raise ValueError(f"No numeric price columns found in {csv_path}")
        close_col = numeric[-1]

    prices = frame[close_col].astype(float).dropna().to_numpy()
    if prices.size < 2:
        raise ValueError(f"Not enough price history in {csv_path}")

    rows: list[dict[str, float | int]] = []
    for window in windows_hours:
        lookback = max(2, min(int(window), int(prices.size)))
        raw_ret = float(prices[-1] / prices[-lookback] - 1.0)
        for fill in fill_buffers_bps:
            round_trip_cost = 2.0 * (float(fill) / 1e4)
            buy_hold_ret = float((1.0 + raw_ret) * (1.0 - round_trip_cost) - 1.0)
            rows.append(
                {
                    "window_hours": int(window),
                    "fill_buffer_bps": float(fill),
                    "buy_hold_return": buy_hold_ret,
                }
            )
    return pd.DataFrame(rows)


def score_summary(summary: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, float | int]:
    subset = summary[summary["fill_buffer_bps"].isin([5.0, 10.0])]
    if subset.empty:
        subset = summary.copy()
    merged = subset.merge(benchmark, on=["window_hours", "fill_buffer_bps"], how="left")
    if "buy_hold_return" not in merged.columns:
        merged["buy_hold_return"] = 0.0
    merged["buy_hold_return"] = merged["buy_hold_return"].fillna(0.0)
    merged["edge_vs_buy_hold"] = merged["total_return"] - merged["buy_hold_return"]

    long_h = merged[merged["window_hours"] >= 168]
    short_h = merged[merged["window_hours"] <= 24]
    if long_h.empty:
        long_h = merged
    if short_h.empty:
        short_h = merged

    long_ret = float(long_h["total_return"].mean()) if not long_h.empty else 0.0
    short_ret = float(short_h["total_return"].mean()) if not short_h.empty else 0.0
    long_edge = float(long_h["edge_vs_buy_hold"].mean()) if not long_h.empty else long_ret
    short_edge = float(short_h["edge_vs_buy_hold"].mean()) if not short_h.empty else short_ret
    long_sort = float(long_h["sortino"].mean()) if not long_h.empty else 0.0
    long_sort_clipped = max(min(long_sort, 5.0), -5.0)
    worst_long = float(long_h["total_return"].min()) if not long_h.empty else 0.0
    mean_drawdown = float(long_h["max_drawdown"].mean()) if ("max_drawdown" in long_h.columns and not long_h.empty) else 0.0
    all_positive = int(bool((merged["total_return"] > 0).all())) if not merged.empty else 0
    mean_fills = float(long_h["fills_total"].mean()) if ("fills_total" in long_h.columns and not long_h.empty) else 0.0
    mean_turnover = float(long_h["mean_turnover"].mean()) if ("mean_turnover" in long_h.columns and not long_h.empty) else 0.0

    robust_score = (
        (long_edge * 180.0)
        + (short_edge * 30.0)
        + (long_ret * 20.0)
        + (0.05 * long_sort_clipped)
        + (worst_long * 80.0)
        - (mean_drawdown * 35.0)
    )
    if all_positive:
        robust_score += 5.0
    if mean_fills < 1.0:
        robust_score -= 100.0
    elif mean_fills < 5.0:
        robust_score -= 40.0
    if mean_turnover < 1e-4:
        robust_score -= 40.0
    elif mean_turnover > 0.45:
        robust_score -= 8.0

    return {
        "robust_score": float(robust_score),
        "long_return": long_ret,
        "short_return": short_ret,
        "long_sortino": long_sort,
        "all_returns_positive": int(all_positive),
        "mean_fills": mean_fills,
        "mean_turnover": mean_turnover,
    }


def _coerce_in_position(frame: pd.DataFrame) -> pd.Series:
    if "in_position" in frame.columns:
        raw = frame["in_position"]
        if raw.dtype == bool:
            return raw
        if pd.api.types.is_numeric_dtype(raw):
            return raw.astype(float).abs() > 1e-8
        return raw.astype(str).str.lower().isin({"1", "true", "yes", "on"})
    if "weight_crypto" in frame.columns:
        return frame["weight_crypto"].astype(float).abs() > 1e-8
    return pd.Series(np.ones(len(frame), dtype=bool), index=frame.index)


def load_strategy_trajectory(path: Path, *, initial_cash: float = 10_000.0) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty trajectory file: {path}")

    if "equity" not in frame.columns:
        if "portfolio_value" in frame.columns:
            frame["equity"] = frame["portfolio_value"]
        else:
            raise ValueError(f"Trajectory missing equity column: {path}")

    if "timestamp" in frame.columns and frame["timestamp"].notna().any():
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    elif "index" in frame.columns:
        timestamps = pd.Series(frame["index"].astype(int).to_numpy())
    else:
        timestamps = pd.Series(np.arange(len(frame), dtype=int))

    equity = frame["equity"].astype(float).to_numpy()
    baseline = float(equity[0]) if abs(float(equity[0])) > 1e-12 else 1.0
    normalized = initial_cash * equity / baseline

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "equity": normalized,
            "in_position": _coerce_in_position(frame).to_numpy(dtype=bool),
        }
    )


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / (peak + 1e-10)
    return float(np.max(drawdown))


def summarize_strategy_frame(name: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    equity = frame["equity"].astype(float).to_numpy()
    total_return = 0.0
    if equity.size >= 2 and abs(float(equity[0])) > 1e-12:
        total_return = float((equity[-1] - equity[0]) / equity[0])
    in_position = frame["in_position"].astype(bool)
    total_bars = int(len(frame))
    bars_in_cash = int((~in_position).sum())
    return {
        "strategy": name,
        "total_return_pct": total_return * 100.0,
        "sortino": compute_sortino(equity),
        "max_drawdown_pct": _max_drawdown(equity) * 100.0,
        "num_switches": 0,
        "bars_in_cash": bars_in_cash,
        "total_bars": total_bars,
        "final_equity": float(equity[-1]),
    }


def _build_equal_weight_frame(strategy_frames: Mapping[str, pd.DataFrame], initial_cash: float) -> pd.DataFrame:
    common_ts, equities, positions = align_equity_curves(dict(strategy_frames))
    num_strategies = max(1, len(equities))
    combined = np.zeros(len(common_ts), dtype=np.float64)
    in_position = np.zeros(len(common_ts), dtype=bool)
    per_alloc = float(initial_cash) / float(num_strategies)

    for name, equity in equities.items():
        baseline = equity[0] if abs(float(equity[0])) > 1e-12 else 1.0
        combined += per_alloc * (equity / baseline)
        in_position |= positions[name]

    return pd.DataFrame({"timestamp": common_ts, "equity": combined, "in_position": in_position})


def run_meta_sweep(
    *,
    strategy_frames: Mapping[str, pd.DataFrame],
    lookbacks: Iterable[int],
    initial_cash: float,
    reeval_every_n_hours: int = 24,
    methods: Iterable[str] = ("winner", "softmax"),
    softmax_temperatures: Iterable[float] = (0.15, 0.35, 0.75),
    softmax_top_ks: Iterable[int] = (2, 3),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [summarize_strategy_frame(name, frame) for name, frame in strategy_frames.items()]
    best_result: dict[str, Any] | None = None

    for lookback in lookbacks:
        for method in methods:
            if method == "winner":
                configs = [
                    (
                        f"meta_winner_lb{int(lookback)}h",
                        MetaSelectorConfig(
                            lookback_hours=int(lookback),
                            sit_out_if_all_negative=True,
                            reeval_every_n_hours=int(reeval_every_n_hours),
                            periodic_reeval_for_active=True,
                            initial_cash=float(initial_cash),
                            selection_method="winner",
                        ),
                    )
                ]
            elif method == "softmax":
                configs = []
                for top_k in softmax_top_ks:
                    for temperature in softmax_temperatures:
                        token = str(temperature).replace(".", "p")
                        configs.append(
                            (
                                f"meta_softmax_lb{int(lookback)}h_t{token}_k{int(top_k)}",
                                MetaSelectorConfig(
                                    lookback_hours=int(lookback),
                                    sit_out_if_all_negative=True,
                                    reeval_every_n_hours=int(reeval_every_n_hours),
                                    periodic_reeval_for_active=True,
                                    initial_cash=float(initial_cash),
                                    selection_method="softmax",
                                    softmax_temperature=float(temperature),
                                    top_k=int(top_k),
                                ),
                            )
                        )
            else:
                raise ValueError(f"Unsupported meta method '{method}'.")

            for strategy_name, config in configs:
                result = run_meta_simulation(dict(strategy_frames), config)
                payload = {
                    "strategy": strategy_name,
                    "total_return_pct": result.total_return * 100.0,
                    "sortino": result.sortino,
                    "max_drawdown_pct": result.max_drawdown * 100.0,
                    "num_switches": result.num_switches,
                    "bars_in_cash": result.bars_in_cash,
                    "total_bars": result.total_bars,
                    "final_equity": float(result.equity_curve[-1]),
                }
                rows.append(payload)
                if best_result is None or (payload["sortino"], payload["total_return_pct"]) > (
                    best_result["sortino"],
                    best_result["total_return_pct"],
                ):
                    best_result = {
                        **payload,
                        "switch_log": result.switch_log,
                    }

    if len(strategy_frames) >= 2:
        equal_weight = _build_equal_weight_frame(strategy_frames, initial_cash=initial_cash)
        rows.append(summarize_strategy_frame("equal_weight", equal_weight))

    summary = pd.DataFrame(rows).sort_values(["sortino", "total_return_pct"], ascending=False).reset_index(drop=True)
    return summary, (best_result or {})


def run_command(cmd: list[str], *, env: Mapping[str, str] | None = None) -> None:
    display = " ".join(str(token) for token in cmd)
    print(f"\n>>> {display}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=dict(env) if env is not None else None, check=True)


def _base_env(*, venv_path: Path, hyperparam_root: Path, preaug_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["VENV_PATH"] = str(venv_path)
    env["HYPERPARAM_ROOT"] = str(hyperparam_root)
    env["CHRONOS2_PREAUG_ROOT"] = str(preaug_root)
    env["CHRONOS2_FREQUENCY"] = "hourly"
    return env


def run_hyperparam_tuning(
    *,
    python_bin: Path,
    env: Mapping[str, str],
    symbols: list[str],
    data_dir: Path,
    output_dir: Path,
    holdout_hours: int,
    prediction_length: int,
    cohort_size: int,
    cohort_min_abs_corr: float,
    device: str,
    quick: bool,
    enable_cross_learning: bool,
    force: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hourly_tuning_results.json"
    if output_path.exists() and not force:
        return output_path

    cmd = [
        str(python_bin),
        "hyperparam_chronos_hourly.py",
        "--symbols",
        *symbols,
        "--holdout-hours",
        str(holdout_hours),
        "--prediction-length",
        str(prediction_length),
        "--cohort-size",
        str(cohort_size),
        "--cohort-min-abs-corr",
        str(cohort_min_abs_corr),
        "--device",
        device,
        "--crypto-data-dir",
        str(data_dir),
        "--stocks-data-dir",
        str(data_dir),
        "--output",
        str(output_path),
        "--save-hyperparams",
    ]
    if quick:
        cmd.append("--quick")
    if enable_cross_learning:
        cmd.append("--enable-cross-learning")
    run_command(cmd, env=env)
    return output_path


def run_preaug_sweep(
    *,
    python_bin: Path,
    env: Mapping[str, str],
    symbols: list[str],
    data_dir: Path,
    hyperparam_dir: Path,
    preaug_root: Path,
    report_dir: Path,
    benchmark_cache_dir: Path,
    strategy_repeats: int,
    device_map: str,
    enable_cross_learning: bool,
    force: bool,
) -> Path:
    output_dir = preaug_root / "chronos2" / "hourly"
    mirror_dir = preaug_root / "best" / "hourly"
    summary_path = output_dir / "ETHUSD.json"
    if summary_path.exists() and not force:
        return summary_path

    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python_bin),
        "preaug_sweeps/evaluate_preaug_chronos.py",
        "--symbols",
        *symbols,
        "--hyperparam-root",
        str(hyperparam_dir),
        "--output-dir",
        str(output_dir),
        "--mirror-best-dir",
        str(mirror_dir),
        "--data-dir",
        str(data_dir),
        "--strategy-repeats",
        str(strategy_repeats),
        "--device-map",
        device_map,
        "--benchmark-cache-dir",
        str(benchmark_cache_dir),
        "--report-dir",
        str(report_dir),
        "--frequency",
        "hourly",
    ]
    if enable_cross_learning:
        cmd.append("--predict-batches-jointly")
    run_command(cmd, env=env)
    return summary_path


def _trajectory_path(eval_dir: Path, *, window_hours: int, fill_buffer_bps: float) -> Path:
    token = format(float(fill_buffer_bps), "g")
    return eval_dir / f"h{int(window_hours)}_fb{token}" / "trajectory.csv"


def candidate_checkpoint_paths(artifact_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in (
        artifact_dir / "best" / "best_model.zip",
        artifact_dir / "ppo_allocator_final.zip",
    ):
        if path.exists():
            candidates.append(path)

    topk_dir = artifact_dir / "topk"
    if topk_dir.exists():
        candidates.extend(sorted(topk_dir.glob("*.zip")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _checkpoint_label(artifact_dir: Path, checkpoint: Path) -> str:
    relative = checkpoint.resolve().relative_to(artifact_dir.resolve())
    return "__".join(relative.with_suffix("").parts)


def train_and_evaluate_candidate(
    *,
    candidate: TrainingCandidate,
    python_bin: Path,
    env: Mapping[str, str],
    venv_path: Path,
    data_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    target_symbol: str,
    num_timesteps: int,
    windows_hours: list[int],
    fill_buffers_bps: list[float],
    meta_window_hours: int,
    meta_fill_buffer_bps: float,
    benchmark: pd.DataFrame,
    force: bool,
) -> dict[str, Any]:
    run_prefix = output_dir.name
    run_name = f"{run_prefix}_{candidate.name}"
    artifact_dir = REPO_ROOT / "fastalgorithms" / "eth_risk_ppo" / "artifacts" / run_name
    final_checkpoint = artifact_dir / "ppo_allocator_final.zip"
    feature_cache_path = feature_cache_dir / f"{candidate.feature_cache_key}.npz"
    eval_dir = output_dir / "candidate_evals" / candidate.name

    run_env = dict(env)
    run_env.update(candidate.env_overrides)
    run_env["VENV_PATH"] = str(venv_path)
    run_env["DATA_DIR"] = str(data_dir)
    run_env["FEATURES_CACHE"] = str(feature_cache_path)

    if force or not final_checkpoint.exists():
        run_command(
            [
                "bash",
                "fastalgorithms/eth_risk_ppo/run_train_local.sh",
                str(int(num_timesteps)),
                run_name,
            ],
            env=run_env,
        )

    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_candidates = candidate_checkpoint_paths(artifact_dir)
    if not checkpoint_candidates:
        raise FileNotFoundError(f"No checkpoints found for candidate {candidate.name} under {artifact_dir}")

    scan_rows: list[dict[str, Any]] = []
    best_choice: dict[str, Any] | None = None
    for checkpoint in checkpoint_candidates:
        checkpoint_label = _checkpoint_label(artifact_dir, checkpoint)
        checkpoint_eval_dir = eval_dir / "checkpoints" / checkpoint_label
        summary_csv = checkpoint_eval_dir / "summary.csv"
        trajectory_csv = _trajectory_path(
            checkpoint_eval_dir,
            window_hours=meta_window_hours,
            fill_buffer_bps=meta_fill_buffer_bps,
        )
        if force or not summary_csv.exists() or not trajectory_csv.exists():
            run_command(
                [
                    str(python_bin),
                    "fastalgorithms/eth_risk_ppo/evaluate_checkpoint_windows.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--symbol",
                    target_symbol,
                    "--windows-hours",
                    ",".join(str(value) for value in windows_hours),
                    "--fill-buffers-bps",
                    ",".join(format(value, "g") for value in fill_buffers_bps),
                    "--features-cache",
                    str(feature_cache_path),
                    "--output-dir",
                    str(checkpoint_eval_dir),
                    "--save-trajectories",
                ],
                env=run_env,
            )

        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing evaluation summary for {candidate.name}: {summary_csv}")
        if not trajectory_csv.exists():
            raise FileNotFoundError(f"Missing meta trajectory for {candidate.name}: {trajectory_csv}")

        summary = pd.read_csv(summary_csv)
        scores = score_summary(summary, benchmark)
        row = {
            "checkpoint_label": checkpoint_label,
            "checkpoint": str(checkpoint),
            "summary_csv": str(summary_csv),
            "trajectory_csv": str(trajectory_csv),
            **scores,
        }
        scan_rows.append(row)
        if best_choice is None or (
            row["robust_score"],
            row["long_return"],
            row["long_sortino"],
        ) > (
            best_choice["robust_score"],
            best_choice["long_return"],
            best_choice["long_sortino"],
        ):
            best_choice = row

    checkpoint_scan = pd.DataFrame(scan_rows).sort_values(
        ["robust_score", "long_return", "long_sortino"],
        ascending=False,
    ).reset_index(drop=True)
    checkpoint_scan_csv = eval_dir / "checkpoint_scan.csv"
    checkpoint_scan_json = eval_dir / "checkpoint_scan.json"
    checkpoint_scan.to_csv(checkpoint_scan_csv, index=False)
    checkpoint_scan_json.write_text(checkpoint_scan.to_json(orient="records", indent=2), encoding="utf-8")
    if best_choice is None:
        raise RuntimeError(f"Failed to score checkpoints for candidate {candidate.name}")

    return {
        "candidate": candidate,
        "run_name": run_name,
        "artifact_dir": artifact_dir,
        "checkpoint": Path(str(best_choice["checkpoint"])),
        "selected_checkpoint_label": str(best_choice["checkpoint_label"]),
        "feature_cache": feature_cache_path,
        "eval_dir": eval_dir,
        "summary_csv": Path(str(best_choice["summary_csv"])),
        "trajectory_csv": Path(str(best_choice["trajectory_csv"])),
        "checkpoint_scan_csv": checkpoint_scan_csv,
        **{key: best_choice[key] for key in ("robust_score", "long_return", "short_return", "long_sortino", "all_returns_positive", "mean_fills", "mean_turnover")},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refit and retune recent ETH strategies, then meta-select over simulator performance.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "trainingdatahourly")
    parser.add_argument("--target-symbol", default="ETHUSD")
    parser.add_argument("--tune-symbols", default="ETHUSD,BTCUSD,SOLUSD,LINKUSD,UNIUSD")
    parser.add_argument("--recent-hours", type=int, default=24 * 180, help="Recent hourly rows to keep for the retune slice.")
    parser.add_argument("--holdout-hours", type=int, default=168)
    parser.add_argument("--prediction-length", type=int, default=6)
    parser.add_argument("--cohort-size", type=int, default=2)
    parser.add_argument("--cohort-min-abs-corr", type=float, default=0.35)
    parser.add_argument("--strategy-repeats", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=12_288)
    parser.add_argument("--seeds", default="42,7,99")
    parser.add_argument("--windows-hours", default="24,168,720")
    parser.add_argument("--fill-buffers-bps", default="0,5,10")
    parser.add_argument("--meta-window-hours", type=int, default=168)
    parser.add_argument("--meta-fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--meta-lookbacks", default="24,48,72,168")
    parser.add_argument("--meta-methods", default="winner,softmax")
    parser.add_argument("--meta-softmax-temperatures", default="0.15,0.35,0.75")
    parser.add_argument("--meta-softmax-top-ks", default="2,3")
    parser.add_argument("--meta-reeval-hours", type=int, default=24)
    parser.add_argument("--meta-max-strategies", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--venv-path", type=Path, default=REPO_ROOT / ".venv313")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-quick-hyperparam", dest="quick_hyperparam", action="store_false")
    parser.add_argument("--disable-cross-learning-tune", dest="enable_cross_learning_tune", action="store_false")
    parser.set_defaults(quick_hyperparam=True, enable_cross_learning_tune=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    python_bin = args.venv_path / "bin" / "python"
    if not python_bin.exists():
        raise FileNotFoundError(f"Missing python executable: {python_bin}")

    tune_symbols = _parse_csv_tokens(args.tune_symbols)
    seeds = _parse_int_list(args.seeds)
    if args.target_symbol not in tune_symbols:
        tune_symbols.insert(0, args.target_symbol)
    windows_hours = _parse_int_list(args.windows_hours)
    fill_buffers_bps = _parse_float_list(args.fill_buffers_bps)
    meta_lookbacks = _parse_int_list(args.meta_lookbacks)
    meta_methods = _parse_csv_tokens(args.meta_methods)
    softmax_temperatures = _parse_float_list(args.meta_softmax_temperatures)
    softmax_top_ks = _parse_int_list(args.meta_softmax_top_ks)
    if int(args.meta_window_hours) not in windows_hours:
        windows_hours = sorted({*windows_hours, int(args.meta_window_hours)})
    if float(args.meta_fill_buffer_bps) not in fill_buffers_bps:
        fill_buffers_bps = sorted({*fill_buffers_bps, float(args.meta_fill_buffer_bps)})

    recent_data_dir = output_dir / "recent_data"
    hyperparam_root = output_dir / "hyperparams"
    hyperparam_dir = hyperparam_root / "chronos2" / "hourly"
    preaug_root = output_dir / "preaugstrategies"
    feature_cache_dir = output_dir / "feature_cache"
    eval_root = output_dir / "candidate_evals"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    env = _base_env(
        venv_path=args.venv_path,
        hyperparam_root=hyperparam_root,
        preaug_root=preaug_root,
    )

    slice_manifest = slice_recent_hourly_data(
        data_root=args.data_root,
        symbols=tune_symbols,
        recent_hours=int(args.recent_hours),
        output_dir=recent_data_dir,
    )
    (output_dir / "recent_data_manifest.json").write_text(json.dumps(slice_manifest, indent=2), encoding="utf-8")

    tuning_results = run_hyperparam_tuning(
        python_bin=python_bin,
        env=env,
        symbols=tune_symbols,
        data_dir=recent_data_dir,
        output_dir=output_dir / "tuning",
        holdout_hours=int(args.holdout_hours),
        prediction_length=int(args.prediction_length),
        cohort_size=int(args.cohort_size),
        cohort_min_abs_corr=float(args.cohort_min_abs_corr),
        device=str(args.device),
        quick=bool(args.quick_hyperparam),
        enable_cross_learning=bool(args.enable_cross_learning_tune),
        force=bool(args.force),
    )

    preaug_result = run_preaug_sweep(
        python_bin=python_bin,
        env=env,
        symbols=[args.target_symbol],
        data_dir=recent_data_dir,
        hyperparam_dir=hyperparam_dir,
        preaug_root=preaug_root,
        report_dir=output_dir / "preaug_reports",
        benchmark_cache_dir=output_dir / "preaug_cache",
        strategy_repeats=int(args.strategy_repeats),
        device_map=str(args.device),
        enable_cross_learning=bool(args.enable_cross_learning_tune),
        force=bool(args.force),
    )

    benchmark = build_buy_hold_benchmark(
        csv_path=recent_data_dir / f"{args.target_symbol}.csv",
        windows_hours=windows_hours,
        fill_buffers_bps=fill_buffers_bps,
    )
    benchmark_path = output_dir / "buy_hold_benchmark.csv"
    benchmark.to_csv(benchmark_path, index=False)

    candidates = build_default_candidates(seeds=seeds)
    candidate_runs: list[dict[str, Any]] = []
    strategy_frames: dict[str, pd.DataFrame] = {}

    for candidate in candidates:
        result = train_and_evaluate_candidate(
            candidate=candidate,
            python_bin=python_bin,
            env=env,
            venv_path=args.venv_path,
            data_dir=recent_data_dir,
            feature_cache_dir=feature_cache_dir,
            output_dir=output_dir,
            target_symbol=args.target_symbol,
            num_timesteps=int(args.num_timesteps),
            windows_hours=windows_hours,
            fill_buffers_bps=fill_buffers_bps,
            meta_window_hours=int(args.meta_window_hours),
            meta_fill_buffer_bps=float(args.meta_fill_buffer_bps),
            benchmark=benchmark,
            force=bool(args.force),
        )

        trajectory = load_strategy_trajectory(Path(result["trajectory_csv"]))
        strategy_frames[candidate.name] = trajectory
        candidate_runs.append(
            {
                "candidate_name": candidate.name,
                "candidate_family": candidate.family,
                "seed": int(candidate.seed),
                "description": candidate.description,
                "run_name": result["run_name"],
                "checkpoint": str(result["checkpoint"]),
                "selected_checkpoint_label": result["selected_checkpoint_label"],
                "feature_cache": str(result["feature_cache"]),
                "summary_csv": str(result["summary_csv"]),
                "trajectory_csv": str(result["trajectory_csv"]),
                "checkpoint_scan_csv": str(result["checkpoint_scan_csv"]),
                **{key: result[key] for key in ("robust_score", "long_return", "short_return", "long_sortino", "all_returns_positive", "mean_fills", "mean_turnover")},
            }
        )

    leaderboard = pd.DataFrame(candidate_runs).sort_values(
        ["robust_score", "long_return", "long_sortino"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard_csv = output_dir / "leaderboard.csv"
    leaderboard_json = output_dir / "leaderboard.json"
    leaderboard.to_csv(leaderboard_csv, index=False)
    leaderboard_json.write_text(leaderboard.to_json(orient="records", indent=2), encoding="utf-8")

    meta_max_strategies = max(1, int(args.meta_max_strategies))
    meta_strategy_names = leaderboard.head(meta_max_strategies)["candidate_name"].tolist()
    meta_frames = {name: strategy_frames[name] for name in meta_strategy_names}
    meta_summary, best_meta = run_meta_sweep(
        strategy_frames=meta_frames,
        lookbacks=meta_lookbacks,
        initial_cash=10_000.0,
        reeval_every_n_hours=int(args.meta_reeval_hours),
        methods=meta_methods,
        softmax_temperatures=softmax_temperatures,
        softmax_top_ks=softmax_top_ks,
    )
    meta_csv = output_dir / "meta_summary.csv"
    meta_json = output_dir / "meta_summary.json"
    meta_summary.to_csv(meta_csv, index=False)
    meta_json.write_text(meta_summary.to_json(orient="records", indent=2), encoding="utf-8")
    best_meta_path = output_dir / "best_meta.json"
    if best_meta:
        best_meta["meta_universe"] = meta_strategy_names
    best_meta_path.write_text(json.dumps(best_meta, indent=2, default=str), encoding="utf-8")

    manifest = {
        "output_dir": str(output_dir),
        "recent_data_dir": str(recent_data_dir),
        "hyperparam_root": str(hyperparam_root),
        "preaug_root": str(preaug_root),
        "tuning_results": str(tuning_results),
        "preaug_result": str(preaug_result),
        "benchmark_csv": str(benchmark_path),
        "leaderboard_csv": str(leaderboard_csv),
        "meta_summary_csv": str(meta_csv),
        "best_meta_json": str(best_meta_path),
        "windows_hours": windows_hours,
        "fill_buffers_bps": fill_buffers_bps,
        "meta_window_hours": int(args.meta_window_hours),
        "meta_fill_buffer_bps": float(args.meta_fill_buffer_bps),
        "meta_lookbacks": meta_lookbacks,
        "meta_methods": meta_methods,
        "meta_softmax_temperatures": softmax_temperatures,
        "meta_softmax_top_ks": softmax_top_ks,
        "meta_reeval_hours": int(args.meta_reeval_hours),
        "meta_max_strategies": meta_max_strategies,
        "meta_strategy_names": meta_strategy_names,
        "num_timesteps": int(args.num_timesteps),
        "seeds": seeds,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Leaderboard ===")
    print(leaderboard.to_string(index=False))
    print("\n=== Meta Summary ===")
    print(meta_summary.to_string(index=False))
    print(f"\nWrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
