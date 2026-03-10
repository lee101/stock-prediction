#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastalgorithms.eth_risk_ppo.retune_recent_eth import (
    DEFAULT_ANALYSIS_ROOT,
    TrainingCandidate,
    _base_env,
    build_buy_hold_benchmark,
    build_default_candidates,
    load_strategy_trajectory,
    summarize_strategy_frame,
    train_and_evaluate_candidate,
)
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity

DEFAULT_FAMILIES = (
    "chronos2_h6_ctx1024",
    "chronos2_h6_ctx1024_lowturn",
    "chronos2_h6_cash_high",
)

MODE_OVERRIDES: dict[str, dict[str, str]] = {
    "fast_cuda": {
        "DEVICE": "cuda",
        "POLICY_DTYPE": "bfloat16",
        "DETERMINISTIC_TRAINING": "0",
        "DISABLE_TF32": "0",
    },
    "deterministic_cuda": {
        "DEVICE": "cuda",
        "POLICY_DTYPE": "float32",
        "DETERMINISTIC_TRAINING": "1",
        "DISABLE_TF32": "1",
    },
    "stable_lowlr_cuda": {
        "DEVICE": "cuda",
        "POLICY_DTYPE": "float32",
        "DETERMINISTIC_TRAINING": "1",
        "DISABLE_TF32": "1",
        "LEARNING_RATE": "2e-5",
        "N_STEPS": "1024",
        "BATCH_SIZE": "256",
        "N_EPOCHS": "6",
        "TARGET_KL": "0.008",
        "MAX_GRAD_NORM": "0.35",
    },
    "deterministic_cpu": {
        "DEVICE": "cpu",
        "POLICY_DTYPE": "float32",
        "DETERMINISTIC_TRAINING": "1",
        "DISABLE_TF32": "1",
    },
    "stable_lowlr_cpu": {
        "DEVICE": "cpu",
        "POLICY_DTYPE": "float32",
        "DETERMINISTIC_TRAINING": "1",
        "DISABLE_TF32": "1",
        "LEARNING_RATE": "2e-5",
        "N_STEPS": "1024",
        "BATCH_SIZE": "256",
        "N_EPOCHS": "6",
        "TARGET_KL": "0.008",
        "MAX_GRAD_NORM": "0.35",
    },
}


@dataclass(frozen=True)
class SweepRunSpec:
    candidate: TrainingCandidate
    family: str
    mode: str
    repeat: int


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


def _latest_reference_output_dir() -> Path:
    candidates = sorted(DEFAULT_ANALYSIS_ROOT.glob("recent_retune_eth_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No recent ETH retune directories found under {DEFAULT_ANALYSIS_ROOT}"
        )
    return candidates[-1]


def _default_output_dir() -> Path:
    return DEFAULT_ANALYSIS_ROOT / f"reproducibility_sweep_eth_{_timestamp_tag()}"


def resolve_mode_overrides(mode: str, *, torch_num_threads: int | None = None) -> dict[str, str]:
    if mode not in MODE_OVERRIDES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(MODE_OVERRIDES)}")
    overrides = dict(MODE_OVERRIDES[mode])
    if torch_num_threads is not None and (
        overrides.get("DETERMINISTIC_TRAINING") == "1" or overrides.get("DEVICE") == "cpu"
    ):
        overrides["TORCH_NUM_THREADS"] = str(max(1, int(torch_num_threads)))
    return overrides


def build_sweep_run_specs(
    *,
    base_candidates: Iterable[TrainingCandidate],
    families: Iterable[str],
    modes: Iterable[str],
    repeats: int,
    torch_num_threads: int | None = None,
) -> list[SweepRunSpec]:
    family_list = [str(family) for family in families]
    family_set = set(family_list)
    base_list = [candidate for candidate in base_candidates if candidate.family in family_set]
    available = {candidate.family for candidate in base_candidates}
    missing = [family for family in family_list if family not in available]
    if missing:
        raise ValueError(f"Unknown candidate families: {missing}. Available families: {sorted(available)}")

    runs: list[SweepRunSpec] = []
    repeat_count = max(1, int(repeats))
    for candidate in base_list:
        for mode in modes:
            mode_overrides = resolve_mode_overrides(mode, torch_num_threads=torch_num_threads)
            for repeat_idx in range(1, repeat_count + 1):
                merged_overrides = dict(candidate.env_overrides)
                merged_overrides.update(mode_overrides)
                run_candidate = TrainingCandidate(
                    name=f"{candidate.name}_{mode}_r{repeat_idx}",
                    description=f"{candidate.description} Mode {mode}. Repeat {repeat_idx}.",
                    feature_cache_key=f"{candidate.feature_cache_key}_{mode}",
                    env_overrides=merged_overrides,
                    family=candidate.family,
                    seed=int(candidate.seed),
                )
                runs.append(
                    SweepRunSpec(
                        candidate=run_candidate,
                        family=candidate.family,
                        mode=str(mode),
                        repeat=int(repeat_idx),
                    )
                )
    return runs


def load_eval_curve_metrics(artifact_dir: Path) -> dict[str, float]:
    eval_path = artifact_dir / "eval" / "evaluations.npz"
    if not eval_path.exists():
        return {
            "eval_count": 0.0,
            "eval_reward_mean": 0.0,
            "eval_reward_std": 0.0,
            "eval_reward_delta_std": 0.0,
            "eval_reward_span": 0.0,
            "eval_reward_up_fraction": 0.0,
        }

    with np.load(eval_path) as data:
        results = np.asarray(data.get("results", []), dtype=np.float64)

    if results.size == 0:
        mean_rewards = np.asarray([], dtype=np.float64)
    elif results.ndim == 1:
        mean_rewards = results.reshape(-1)
    else:
        mean_rewards = results.mean(axis=1)

    deltas = np.diff(mean_rewards) if mean_rewards.size >= 2 else np.asarray([], dtype=np.float64)
    return {
        "eval_count": float(mean_rewards.size),
        "eval_reward_mean": float(mean_rewards.mean()) if mean_rewards.size else 0.0,
        "eval_reward_std": float(mean_rewards.std()) if mean_rewards.size else 0.0,
        "eval_reward_delta_std": float(deltas.std()) if deltas.size else 0.0,
        "eval_reward_span": float(mean_rewards.max() - mean_rewards.min()) if mean_rewards.size else 0.0,
        "eval_reward_up_fraction": float(np.mean(deltas >= 0.0)) if deltas.size else 0.0,
    }


def summarize_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    repeat_rows: list[dict[str, Any]] = []
    for (family, mode, seed), group in results.groupby(["family", "mode", "seed"], dropna=False):
        repeat_rows.append(
            {
                "family": family,
                "mode": mode,
                "seed": int(seed),
                "repeat_count": int(len(group)),
                "robust_score_span": float(group["robust_score"].max() - group["robust_score"].min()),
                "meta_return_span_pct": float(group["meta_return_pct"].max() - group["meta_return_pct"].min()),
                "meta_sortino_span": float(group["meta_sortino"].max() - group["meta_sortino"].min()),
            }
        )
    repeat_summary = pd.DataFrame(repeat_rows)

    rows: list[dict[str, Any]] = []
    for (family, mode), group in results.groupby(["family", "mode"], dropna=False):
        repeat_group = repeat_summary[
            (repeat_summary["family"] == family) & (repeat_summary["mode"] == mode)
        ]
        robust_std = float(group["robust_score"].std(ddof=0))
        smoothness_mean = float(group["pnl_smoothness"].mean())
        repeat_robust_span_mean = float(repeat_group["robust_score_span"].mean()) if not repeat_group.empty else 0.0
        stability_score = (
            float(group["robust_score"].mean())
            - robust_std
            + 0.35 * float(group["robust_score"].min())
            - 1800.0 * smoothness_mean
            - 0.5 * repeat_robust_span_mean
        )
        rows.append(
            {
                "family": family,
                "mode": mode,
                "num_runs": int(len(group)),
                "num_seeds": int(group["seed"].nunique()),
                "repeats_per_seed": int(group.groupby("seed")["repeat"].nunique().max()),
                "robust_score_mean": float(group["robust_score"].mean()),
                "robust_score_std": robust_std,
                "robust_score_min": float(group["robust_score"].min()),
                "long_return_mean": float(group["long_return"].mean()),
                "long_return_std": float(group["long_return"].std(ddof=0)),
                "long_sortino_mean": float(group["long_sortino"].mean()),
                "long_sortino_std": float(group["long_sortino"].std(ddof=0)),
                "meta_return_mean_pct": float(group["meta_return_pct"].mean()),
                "meta_return_std_pct": float(group["meta_return_pct"].std(ddof=0)),
                "meta_sortino_mean": float(group["meta_sortino"].mean()),
                "meta_sortino_std": float(group["meta_sortino"].std(ddof=0)),
                "meta_max_drawdown_mean_pct": float(group["meta_max_drawdown_pct"].mean()),
                "pnl_smoothness_mean": smoothness_mean,
                "pnl_smoothness_std": float(group["pnl_smoothness"].std(ddof=0)),
                "eval_reward_delta_std_mean": float(group["eval_reward_delta_std"].mean()),
                "repeat_robust_span_mean": repeat_robust_span_mean,
                "repeat_return_span_mean_pct": float(repeat_group["meta_return_span_pct"].mean()) if not repeat_group.empty else 0.0,
                "stability_score": float(stability_score),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["stability_score", "robust_score_mean", "robust_score_min"],
        ascending=False,
    ).reset_index(drop=True)
    repeat_summary = repeat_summary.sort_values(
        ["robust_score_span", "meta_return_span_pct"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return summary, repeat_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ETH multi-seed reproducibility and smoothness sweeps on top of recent retune artifacts."
    )
    parser.add_argument("--reference-output-dir", type=Path, default=None, help="Existing recent ETH retune directory to reuse for recent_data/hyperparams/preaug.")
    parser.add_argument("--venv-path", type=Path, default=REPO_ROOT / ".venv313")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-symbol", default="ETHUSD")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--modes", default="fast_cuda,deterministic_cuda,stable_lowlr_cuda")
    parser.add_argument("--seeds", default="42,7,99,123")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=4096)
    parser.add_argument("--windows-hours", default="24,168")
    parser.add_argument("--fill-buffers-bps", default="5,10")
    parser.add_argument("--meta-window-hours", type=int, default=168)
    parser.add_argument("--meta-fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_output_dir = (args.reference_output_dir or _latest_reference_output_dir()).expanduser().resolve()
    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    venv_path = args.venv_path.expanduser()
    python_bin = venv_path / "bin" / "python"
    if not python_bin.exists():
        raise FileNotFoundError(f"Missing python executable: {python_bin}")

    recent_data_dir = reference_output_dir / "recent_data"
    hyperparam_root = reference_output_dir / "hyperparams"
    preaug_root = reference_output_dir / "preaugstrategies"
    if not recent_data_dir.exists():
        raise FileNotFoundError(f"Missing recent_data directory under {reference_output_dir}")
    if not hyperparam_root.exists():
        raise FileNotFoundError(f"Missing hyperparams directory under {reference_output_dir}")
    if not preaug_root.exists():
        raise FileNotFoundError(f"Missing preaugstrategies directory under {reference_output_dir}")

    windows_hours = _parse_int_list(args.windows_hours)
    fill_buffers_bps = _parse_float_list(args.fill_buffers_bps)
    if int(args.meta_window_hours) not in windows_hours:
        windows_hours = sorted({*windows_hours, int(args.meta_window_hours)})
    if float(args.meta_fill_buffer_bps) not in fill_buffers_bps:
        fill_buffers_bps = sorted({*fill_buffers_bps, float(args.meta_fill_buffer_bps)})

    families = _parse_csv_tokens(args.families)
    modes = _parse_csv_tokens(args.modes)
    seeds = _parse_int_list(args.seeds)

    feature_cache_dir = output_dir / "feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    env = _base_env(
        venv_path=venv_path,
        hyperparam_root=hyperparam_root,
        preaug_root=preaug_root,
    )
    benchmark = build_buy_hold_benchmark(
        csv_path=recent_data_dir / f"{args.target_symbol}.csv",
        windows_hours=windows_hours,
        fill_buffers_bps=fill_buffers_bps,
    )
    benchmark.to_csv(output_dir / "buy_hold_benchmark.csv", index=False)

    base_candidates = build_default_candidates(seeds=seeds)
    run_specs = build_sweep_run_specs(
        base_candidates=base_candidates,
        families=families,
        modes=modes,
        repeats=int(args.repeats),
        torch_num_threads=args.torch_num_threads,
    )

    run_manifest = {
        "reference_output_dir": str(reference_output_dir),
        "recent_data_dir": str(recent_data_dir),
        "hyperparam_root": str(hyperparam_root),
        "preaug_root": str(preaug_root),
        "families": families,
        "modes": modes,
        "seeds": seeds,
        "repeats": int(args.repeats),
        "num_timesteps": int(args.num_timesteps),
        "windows_hours": windows_hours,
        "fill_buffers_bps": fill_buffers_bps,
        "meta_window_hours": int(args.meta_window_hours),
        "meta_fill_buffer_bps": float(args.meta_fill_buffer_bps),
        "torch_num_threads": int(args.torch_num_threads) if args.torch_num_threads is not None else None,
        "mode_overrides": {mode: resolve_mode_overrides(mode, torch_num_threads=args.torch_num_threads) for mode in modes},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    for run_spec in run_specs:
        train_result = train_and_evaluate_candidate(
            candidate=run_spec.candidate,
            python_bin=python_bin,
            env=env,
            venv_path=venv_path,
            data_dir=recent_data_dir,
            feature_cache_dir=feature_cache_dir,
            output_dir=output_dir,
            target_symbol=str(args.target_symbol),
            num_timesteps=int(args.num_timesteps),
            windows_hours=windows_hours,
            fill_buffers_bps=fill_buffers_bps,
            meta_window_hours=int(args.meta_window_hours),
            meta_fill_buffer_bps=float(args.meta_fill_buffer_bps),
            benchmark=benchmark,
            force=bool(args.force),
        )
        trajectory = load_strategy_trajectory(Path(train_result["trajectory_csv"]))
        trajectory_summary = summarize_strategy_frame(run_spec.candidate.name, trajectory)
        eval_metrics = load_eval_curve_metrics(Path(train_result["artifact_dir"]))
        result_rows.append(
            {
                "family": run_spec.family,
                "mode": run_spec.mode,
                "seed": int(run_spec.candidate.seed),
                "repeat": int(run_spec.repeat),
                "candidate_name": run_spec.candidate.name,
                "artifact_dir": str(train_result["artifact_dir"]),
                "checkpoint": str(train_result["checkpoint"]),
                "selected_checkpoint_label": str(train_result["selected_checkpoint_label"]),
                "summary_csv": str(train_result["summary_csv"]),
                "trajectory_csv": str(train_result["trajectory_csv"]),
                "robust_score": float(train_result["robust_score"]),
                "long_return": float(train_result["long_return"]),
                "short_return": float(train_result["short_return"]),
                "long_sortino": float(train_result["long_sortino"]),
                "mean_fills": float(train_result["mean_fills"]),
                "mean_turnover": float(train_result["mean_turnover"]),
                "meta_return_pct": float(trajectory_summary["total_return_pct"]),
                "meta_sortino": float(trajectory_summary["sortino"]),
                "meta_max_drawdown_pct": float(trajectory_summary["max_drawdown_pct"]),
                "bars_in_cash": int(trajectory_summary["bars_in_cash"]),
                "total_bars": int(trajectory_summary["total_bars"]),
                "pnl_smoothness": float(
                    compute_pnl_smoothness_from_equity(trajectory["equity"].astype(float).to_numpy())
                ),
                **eval_metrics,
            }
        )
        pd.DataFrame(result_rows).to_csv(output_dir / "candidate_results.csv", index=False)

    results = pd.DataFrame(result_rows).sort_values(
        ["family", "mode", "seed", "repeat"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    summary, repeat_summary = summarize_results(results)

    results.to_csv(output_dir / "candidate_results.csv", index=False)
    summary.to_csv(output_dir / "family_mode_summary.csv", index=False)
    repeat_summary.to_csv(output_dir / "repeat_consistency_summary.csv", index=False)

    best_payload: dict[str, Any] = {}
    if not summary.empty:
        best_payload = summary.iloc[0].to_dict()
    (output_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
