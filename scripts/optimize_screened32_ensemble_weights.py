#!/usr/bin/env python3
"""Optuna search over screened32 ensemble integer weights.

This is a GPU-screening tool, not a production gate. It reuses
scripts/sweep_screened32_gpu_ensemble.py's parity-tested fast evaluator to rank
integer-weighted checkpoint ensembles under binary fills, decision_lag=2,
fill-through buffer, fees, and adverse slippage. The GPU path intentionally
keeps borrow/margin APR at zero for speed; full-cost confirmation must use
scripts/screened32_realism_gate.py before any production consideration.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.evaluate_holdout import load_policy  # noqa: E402
from xgbnew.artifacts import write_json_atomic  # noqa: E402

from scripts.sweep_screened32_gpu_ensemble import (  # noqa: E402
    _as_json,
    _default_baseline_paths,
    _evaluate_candidate_gpu,
    _resolve_repo_path,
    _stage_windows,
    read_mktd,
)


def _candidate_pool(root: Path, include: Sequence[str] | None) -> tuple[Path, ...]:
    all_paths = tuple(sorted(root.glob("*.pt"), key=lambda p: p.stem))
    if include:
        wanted = {name.strip() for name in include if name.strip()}
        selected = tuple(path for path in all_paths if path.stem in wanted or path.name in wanted)
        missing = sorted(wanted - {p.stem for p in selected} - {p.name for p in selected})
        if missing:
            raise ValueError(f"unknown checkpoint(s) in include list: {', '.join(missing)}")
        return selected
    return all_paths


def _counts_from_paths(pool: Sequence[Path], paths: Sequence[Path]) -> dict[str, int]:
    pool_by_stem = {path.stem: path for path in pool}
    counts: Counter[str] = Counter()
    for path in paths:
        stem = Path(path).stem
        if stem in pool_by_stem:
            counts[stem] += 1
    return dict(counts)


def _paths_from_counts(pool: Sequence[Path], counts: dict[str, int]) -> tuple[Path, ...]:
    out: list[Path] = []
    for path in pool:
        out.extend([path] * max(0, int(counts.get(path.stem, 0))))
    return tuple(out)


def _seed_weighted_v8_counts(pool: Sequence[Path]) -> dict[str, int]:
    counts = _counts_from_paths(pool, _default_baseline_paths())
    if "D_s42" in counts:
        counts["D_s42"] = 0
    for stem in ("D_s28", "D_s24", "D_s57", "D_s72"):
        if any(path.stem == stem for path in pool):
            counts[stem] = counts.get(stem, 0) + 1
    return {stem: count for stem, count in counts.items() if count > 0}


def _make_sampler(name: str, seed: int):
    import optuna

    name = str(name).lower()
    if name == "random":
        return optuna.samplers.RandomSampler(seed=int(seed))
    if name == "gps":
        gps = getattr(optuna.samplers, "GPSampler", None)
        if gps is None:
            raise RuntimeError("Optuna GPSampler is unavailable in this environment")
        return gps(seed=int(seed))
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=int(seed))
    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=int(seed),
            multivariate=True,
            group=True,
            n_startup_trials=20,
        )
    raise ValueError(f"unsupported sampler {name!r}")


def _trial_counts(
    trial,
    pool: Sequence[Path],
    *,
    max_weight: int,
    min_members: int,
    max_members: int,
) -> dict[str, int]:
    counts = {
        path.stem: int(trial.suggest_int(path.stem, 0, int(max_weight)))
        for path in pool
    }
    member_count = sum(counts.values())
    # Soft constraints keep TPE informed while making invalid trials very bad.
    if member_count < int(min_members):
        trial.set_user_attr("invalid_reason", f"too_few_members:{member_count}")
    elif member_count > int(max_members):
        trial.set_user_attr("invalid_reason", f"too_many_members:{member_count}")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--checkpoint-root", default="pufferlib_market/prod_ensemble_screened32")
    parser.add_argument("--include-checkpoints", default=None,
                        help="Optional comma-separated checkpoint stems/names to search. Default: all .pt files.")
    parser.add_argument("--window-days", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--ensemble-mode", choices=["softmax_avg", "logit_avg"], default="softmax_avg")
    parser.add_argument("--sampler", choices=["tpe", "random", "gps", "cmaes"], default="tpe")
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--min-members", type=int, default=8)
    parser.add_argument("--max-members", type=int, default=18)
    parser.add_argument("--max-weight", type=int, default=3)
    parser.add_argument("--neg-penalty", type=float, default=0.002)
    parser.add_argument("--dd-penalty", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="analysis/screened32_optuna_weight_search/search.json")
    return parser


def _finite_nonnegative(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        return f"{name} must be finite and non-negative"
    return None


def _finite_positive(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return f"{name} must be finite and positive"
    return None


def validate_args(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if int(args.decision_lag) != 2:
        failures.append("decision_lag must be exactly 2")
    for attr in ("fill_buffer_bps", "slippage_bps", "fee_rate", "neg_penalty", "dd_penalty"):
        failure = _finite_nonnegative(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)
    for attr in ("leverage",):
        failure = _finite_positive(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)
    for attr in ("window_days", "trials", "min_members", "max_members", "max_weight", "top_k"):
        if int(getattr(args, attr)) <= 0:
            failures.append(f"{attr} must be positive")
    if args.max_windows is not None and int(args.max_windows) <= 0:
        failures.append("max_windows must be positive when provided")
    if int(args.max_members) < int(args.min_members):
        failures.append("max_members must be >= min_members")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_failures = validate_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"optimize_screened32_ensemble_weights: {failure}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("optimize_screened32_ensemble_weights: CUDA is unavailable", file=sys.stderr)
        return 2

    import optuna

    val_path = _resolve_repo_path(args.val_data)
    checkpoint_root = _resolve_repo_path(args.checkpoint_root)
    include = None
    if args.include_checkpoints:
        include = [part.strip() for part in str(args.include_checkpoints).split(",") if part.strip()]
    try:
        pool = _candidate_pool(checkpoint_root, include)
    except ValueError as exc:
        print(f"optimize_screened32_ensemble_weights: {exc}", file=sys.stderr)
        return 2
    if not pool:
        print("optimize_screened32_ensemble_weights: empty checkpoint pool", file=sys.stderr)
        return 2
    if int(args.min_members) > int(args.max_weight) * len(pool):
        print(
            "optimize_screened32_ensemble_weights: min_members exceeds maximum possible weighted pool size",
            file=sys.stderr,
        )
        return 2

    data = read_mktd(val_path)
    window_len = int(args.window_days) + 1
    if window_len > int(data.num_timesteps):
        print("optimize_screened32_ensemble_weights: window is longer than val data", file=sys.stderr)
        return 2
    starts = list(range(int(data.num_timesteps) - window_len + 1))
    if args.max_windows is not None:
        starts = starts[: max(1, int(args.max_windows))]

    device = torch.device(str(args.device))
    prices, features, tradable = _stage_windows(data, starts, int(args.window_days), device)
    policies_by_path = {
        path: load_policy(
            path,
            int(data.num_symbols),
            features_per_sym=int(data.features.shape[2]),
            device=device,
        )
        for path in pool
    }

    sampler = _make_sampler(str(args.sampler), int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    baseline_counts = _counts_from_paths(pool, _default_baseline_paths())
    weighted_v8_counts = _seed_weighted_v8_counts(pool)
    for seed_counts in (baseline_counts, weighted_v8_counts):
        if seed_counts:
            params = {path.stem: int(seed_counts.get(path.stem, 0)) for path in pool}
            if int(args.min_members) <= sum(params.values()) <= int(args.max_members):
                study.enqueue_trial(params)

    seen: set[tuple[int, ...]] = set()

    def objective(trial) -> float:
        counts = _trial_counts(
            trial,
            pool,
            max_weight=int(args.max_weight),
            min_members=int(args.min_members),
            max_members=int(args.max_members),
        )
        count_tuple = tuple(int(counts.get(path.stem, 0)) for path in pool)
        member_count = sum(count_tuple)
        trial.set_user_attr("member_count", int(member_count))
        if member_count < int(args.min_members):
            return -10.0 - float(int(args.min_members) - member_count)
        if member_count > int(args.max_members):
            return -10.0 - 0.1 * float(member_count - int(args.max_members))
        if count_tuple in seen:
            trial.set_user_attr("duplicate_counts", True)
            return -9.0
        seen.add(count_tuple)
        paths = _paths_from_counts(pool, counts)
        result = _evaluate_candidate_gpu(
            label=f"trial_{trial.number}",
            paths=paths,
            policies_by_path=policies_by_path,
            prices=prices,
            features=features,
            tradable=tradable,
            num_symbols=int(data.num_symbols),
            features_per_sym=int(data.features.shape[2]),
            window_days=int(args.window_days),
            fill_buffer_bps=float(args.fill_buffer_bps),
            max_leverage=float(args.leverage),
            fee_rate=float(args.fee_rate),
            slippage_bps=float(args.slippage_bps),
            decision_lag=int(args.decision_lag),
            ensemble_mode=str(args.ensemble_mode),
            neg_penalty=float(args.neg_penalty),
            dd_penalty=float(args.dd_penalty),
        )
        trial.set_user_attr("result", _as_json(result))
        trial.set_user_attr("members", [str(path) for path in paths])
        print(
            f"[trial {trial.number:04d}] score={result.score:+.4f} "
            f"med={result.median_monthly_return * 100:+6.2f}% "
            f"p10={result.p10_monthly_return * 100:+6.2f}% "
            f"neg={result.n_neg:3d}/{result.n_windows} "
            f"dd={result.max_drawdown * 100:5.1f}% "
            f"members={member_count}",
            flush=True,
        )
        if not math.isfinite(result.score):
            return -10.0
        return float(result.score)

    study.optimize(objective, n_trials=int(args.trials), gc_after_trial=True)

    completed = [t for t in study.trials if t.value is not None and t.user_attrs.get("result")]
    completed.sort(key=lambda t: float(t.value), reverse=True)
    payload = {
        "val_data": str(val_path),
        "checkpoint_root": str(checkpoint_root),
        "pool": [str(path) for path in pool],
        "window_days": int(args.window_days),
        "n_windows": len(starts),
        "starts": starts,
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "slippage_bps": float(args.slippage_bps),
        "leverage": float(args.leverage),
        "fee_rate": float(args.fee_rate),
        "short_borrow_apr": 0.0,
        "decision_lag": int(args.decision_lag),
        "ensemble_mode": str(args.ensemble_mode),
        "sampler": str(args.sampler),
        "optuna_version": getattr(optuna, "__version__", ""),
        "trials_requested": int(args.trials),
        "trials_completed": len(completed),
        "member_bounds": {
            "min": int(args.min_members),
            "max": int(args.max_members),
            "max_weight": int(args.max_weight),
        },
        "ranking": {
            "score": "median_monthly + 0.5*p10_monthly - neg_penalty*n_neg - dd_penalty*max_drawdown",
            "neg_penalty": float(args.neg_penalty),
            "dd_penalty": float(args.dd_penalty),
        },
        "best": completed[0].user_attrs["result"] if completed else None,
        "results": [
            {
                "trial": int(t.number),
                "value": float(t.value),
                "params": dict(t.params),
                "members": t.user_attrs.get("members", []),
                "result": t.user_attrs["result"],
            }
            for t in completed
        ],
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    write_json_atomic(out_path, payload)

    print(f"\nTop {max(1, int(args.top_k))} Optuna candidates:")
    for item in payload["results"][: max(1, int(args.top_k))]:
        r = item["result"]
        print(
            f"{float(item['value']):+.4f}  {r['median_monthly_return'] * 100:+6.2f}%/mo "
            f"p10={r['p10_monthly_return'] * 100:+6.2f}% "
            f"neg={r['n_neg']:3d}/{r['n_windows']} "
            f"dd={r['max_drawdown'] * 100:5.1f}% "
            f"size={r['ensemble_size']:2d} trial={item['trial']}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
