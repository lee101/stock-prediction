#!/usr/bin/env python3
"""Scaled SAP training: lazy cache, all pairs, RunPod H100 support.

Local:
    python -m sharpnessadjustedproximalpolicy.run_scaled --local --epochs 15
    python -m sharpnessadjustedproximalpolicy.run_scaled --local --symbols DOGEUSD BTCUSD

Remote (RunPod H100):
    python -m sharpnessadjustedproximalpolicy.run_scaled --gpu h100 --epochs 15

All flags:
    --local              Run on local GPU
    --gpu TYPE           RunPod GPU type (h100, 4090, a100, 5090)
    --gpu-fallbacks CSV  Optional fallback GPU aliases/names, or 'none'
    --epochs N           Training epochs per experiment (default 15)
    --configs LIST       Comma-sep config names (default: top performers)
    --symbols LIST       Space-sep symbols (default: all with hourly data)
    --min-rows N         Min data rows to attempt training (default 2000)
    --skip-cache-gen     Don't generate missing caches, skip symbols without cache
    --compile            Enable torch.compile (fast on H100, broken on some setups)
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runpod_remote_utils import (
    render_subprocess_error as _render_subprocess_error,
    run_checked_subprocess as _run_checked_local,
    ssh_cmd as _ssh_cmd,
    ssh_run as _ssh,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE_WORKSPACE = "/workspace/sap"
RUNPOD_SSH_READY_TIMEOUT_SECONDS = 600


@dataclass(frozen=True, slots=True)
class RemoteSweepPlan:
    gpu_preferences: tuple[str, ...]
    config_names: tuple[str, ...]
    symbol_source: str
    symbols: tuple[str, ...]
    remote_dir: str
    wait_timeout_seconds: int
    epochs: int
    compile: bool
    keep_pod: bool
    remote_command_preview: str


def get_all_symbols(min_rows: int = 2000) -> list[str]:
    """All USD crypto symbols with sufficient hourly data."""
    data_root = Path("trainingdatahourly") / "crypto"
    syms = []
    for p in sorted(data_root.glob("*USD.csv")):
        rows = sum(1 for _ in open(p)) - 1
        if rows >= min_rows:
            syms.append(p.stem)
    return syms


def ensure_cache(symbol: str, horizons: tuple[int, ...] = (1, 24)) -> tuple[int, ...]:
    """Generate missing forecast caches lazily. Returns available horizons."""
    cache_root = Path("binanceneural") / "forecast_cache"
    available = []
    missing = []
    for h in horizons:
        if (cache_root / f"h{h}" / f"{symbol}.parquet").exists():
            available.append(h)
        else:
            missing.append(h)

    if not missing:
        return tuple(available)

    # lazy generation
    print(f"  Generating cache for {symbol} horizons {missing}...", flush=True)
    try:
        from binanceneural.forecasts import build_forecast_bundle
        t0 = time.time()
        build_forecast_bundle(
            symbol=symbol,
            data_root=Path("trainingdatahourly") / "crypto",
            cache_root=cache_root,
            horizons=tuple(missing),
            context_hours=24 * 14,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=128,
            cache_only=False,
        )
        print(f"  Cache done in {time.time() - t0:.0f}s", flush=True)
        available.extend(missing)
    except Exception as e:
        print(f"  Cache gen failed: {e}", flush=True)

    return tuple(sorted(available))


TOP_CONFIGS = [
    # Current best baselines
    {"name": "periodic_wd01", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1},
    {"name": "baseline_wd01", "sam_mode": "none", "weight_decay": 0.1},
    # Round 4: bar-shift temporal augmentation
    {"name": "barshift_5_baseline", "sam_mode": "none", "weight_decay": 0.1, "bar_shift_range": 5},
    {"name": "barshift_5_periodic", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1, "bar_shift_range": 5},
    # Round 4b: sparse MoE FFN (same params as dense, learned specialisation)
    {"name": "moe8_baseline", "sam_mode": "none", "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 8},
    {"name": "moe8_barshift5_baseline", "sam_mode": "none", "weight_decay": 0.1,
     "model_arch": "moe", "moe_num_experts": 8, "bar_shift_range": 5},
]


def _available_config_map() -> dict[str, dict]:
    from .config import EXPERIMENTS

    config_map = {str(config["name"]): config for config in TOP_CONFIGS}
    config_map.update({str(config["name"]): config for config in EXPERIMENTS})
    return config_map


def _resolve_requested_configs(configs_arg: str | None) -> list[dict]:
    config_map = _available_config_map()
    if configs_arg is None:
        return list(TOP_CONFIGS)

    requested_names = [name.strip() for name in str(configs_arg).split(",") if name.strip()]
    if not requested_names:
        raise ValueError("--configs must include at least one config name")

    missing = [name for name in requested_names if name not in config_map]
    if missing:
        available = ", ".join(sorted(config_map))
        raise ValueError(
            "Unknown --configs values: "
            f"{', '.join(missing)}. Available configs: {available}"
        )

    return [config_map[name] for name in requested_names]


def _build_remote_sweep_cmd(remote_dir: str, args) -> str:
    quoted_remote_dir = shlex.quote(remote_dir)
    cmd_parts = [
        f"cd {quoted_remote_dir}",
        "source /workspace/venv/bin/activate 2>/dev/null || true",
        (
            "PYTHONUNBUFFERED=1 python -m sharpnessadjustedproximalpolicy.run_scaled"
            f" --local --epochs {shlex.quote(str(args.epochs))}"
        ),
    ]
    if args.compile:
        cmd_parts[-1] += " --compile"
    if args.symbols:
        cmd_parts[-1] += " --symbols " + " ".join(shlex.quote(symbol) for symbol in args.symbols)
    if args.configs:
        cmd_parts[-1] += f" --configs {shlex.quote(args.configs)}"
    return " && ".join(cmd_parts)


def _build_remote_install_cmd(remote_dir: str) -> str:
    return f"cd {shlex.quote(remote_dir)} && pip install -e . 2>&1 | tail -3"


def _build_remote_plan(args, *, resolve_gpu_preferences) -> RemoteSweepPlan:
    configs = _resolve_requested_configs(args.configs)
    symbols = tuple(args.symbols or ())
    return RemoteSweepPlan(
        gpu_preferences=resolve_gpu_preferences(args.gpu, getattr(args, "gpu_fallbacks", None)),
        config_names=tuple(str(config["name"]) for config in configs),
        symbol_source="cli" if symbols else "auto",
        symbols=symbols,
        remote_dir=DEFAULT_REMOTE_WORKSPACE,
        wait_timeout_seconds=RUNPOD_SSH_READY_TIMEOUT_SECONDS,
        epochs=int(args.epochs),
        compile=bool(args.compile),
        keep_pod=bool(args.keep_pod),
        remote_command_preview=_build_remote_sweep_cmd(DEFAULT_REMOTE_WORKSPACE, args),
    )


def _remote_plan_payload(plan: RemoteSweepPlan) -> dict[str, object]:
    return {
        "status": "dry_run",
        "mode": "runpod_remote",
        "gpu_preferences": list(plan.gpu_preferences),
        "config_names": list(plan.config_names),
        "symbol_source": plan.symbol_source,
        "symbols": list(plan.symbols),
        "remote_dir": plan.remote_dir,
        "wait_timeout_seconds": plan.wait_timeout_seconds,
        "epochs": plan.epochs,
        "compile": plan.compile,
        "keep_pod": plan.keep_pod,
        "remote_command_preview": plan.remote_command_preview,
    }


def _format_remote_plan_summary(plan: RemoteSweepPlan) -> str:
    symbols = ", ".join(plan.symbols) if plan.symbols else "auto-detect eligible cached symbols on remote host"
    configs = ", ".join(plan.config_names)
    lines = [
        "SAP RunPod Sweep Plan",
        "Status: dry run",
        f"GPU preferences: {', '.join(plan.gpu_preferences)}",
        f"Configs: {configs}",
        f"Symbols: {symbols}",
        f"Compile: {'enabled' if plan.compile else 'disabled'}",
        f"Pod cleanup: {'keep pod running' if plan.keep_pod else 'terminate pod after sweep'}",
        f"Remote dir: {plan.remote_dir}",
        f"Remote command: {plan.remote_command_preview}",
        "Next step: rerun without --dry-run to provision the pod and execute the sweep.",
    ]
    return "\n".join(lines)


def run_local(args):
    """Run sweep locally on this machine's GPU."""
    from binanceneural.config import DatasetConfig, TrainingConfig
    from binanceneural.data import BinanceHourlyDataModule
    from .config import DEFAULT_TRAINING_OVERRIDES, SAPConfig
    from .trainer import SAPTrainer

    symbols = args.symbols or get_all_symbols(args.min_rows)
    configs = _resolve_requested_configs(args.configs)

    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}", flush=True)
    print(f"Configs ({len(configs)}): {', '.join(c['name'] for c in configs)}", flush=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = REPO / f"sharpnessadjustedproximalpolicy/scaled_results_{timestamp}.json"
    out_csv = REPO / f"sharpnessadjustedproximalpolicy/scaled_leaderboard_{timestamp}.csv"

    overrides = dict(DEFAULT_TRAINING_OVERRIDES)
    overrides["epochs"] = args.epochs
    if args.compile:
        overrides["use_compile"] = True
    else:
        overrides["use_compile"] = False

    all_results = []
    total = len(symbols) * len(configs)
    idx = 0

    for symbol in symbols:
        # lazy cache
        if args.skip_cache_gen:
            cache_root = Path("binanceneural") / "forecast_cache"
            horizons = tuple(h for h in (1, 4, 12, 24) if (cache_root / f"h{h}" / f"{symbol}.parquet").exists())
            if len(horizons) < 2:
                print(f"Skipping {symbol}: insufficient cache ({horizons})", flush=True)
                continue
        else:
            horizons = ensure_cache(symbol, (1, 24))
            if len(horizons) < 2:
                print(f"Skipping {symbol}: cache generation failed", flush=True)
                continue

        for config in configs:
            idx += 1
            name = config["name"]
            print(f"\n[{idx}/{total}] {name} / {symbol}", flush=True)

            tc_kwargs = dict(overrides)
            sap_kwargs = {}
            tc_fields = set(TrainingConfig.__dataclass_fields__.keys())
            sap_fields = set(SAPConfig.__dataclass_fields__.keys())
            for k, v in config.items():
                if k == "name":
                    continue
                if k in tc_fields:
                    tc_kwargs[k] = v
                elif k in sap_fields:
                    sap_kwargs[k] = v

            tc_kwargs["run_name"] = f"sap_{name}_{symbol}_{timestamp}"
            tc_kwargs["checkpoint_root"] = Path("sharpnessadjustedproximalpolicy") / "checkpoints"
            tc_kwargs["dataset"] = DatasetConfig(
                symbol=symbol,
                data_root=Path("trainingdatahourly") / "crypto",
                forecast_cache_root=Path("binanceneural") / "forecast_cache",
                forecast_horizons=horizons,
                sequence_length=tc_kwargs.get("sequence_length", 72),
                validation_days=70,
                cache_only=True,
                bar_shift_range=tc_kwargs.get("bar_shift_range", 0),
            )

            tc = TrainingConfig(**tc_kwargs)
            sc = SAPConfig(**sap_kwargs)

            try:
                dm = BinanceHourlyDataModule(tc.dataset)
            except Exception as e:
                print(f"  Data load failed: {e}", flush=True)
                all_results.append({"name": name, "symbol": symbol, "error": str(e)})
                _save_results(all_results, out_json, out_csv)
                continue

            trainer = SAPTrainer(tc, sc, dm)
            t0 = time.time()
            try:
                artifacts, history = trainer.train()
            except Exception as e:
                print(f"  Training failed: {e}", flush=True)
                traceback.print_exc()
                all_results.append({"name": name, "symbol": symbol, "error": str(e)})
                _save_results(all_results, out_json, out_csv)
                continue

            best = max(history, key=lambda h: h.val_score)
            result = {
                "name": name, "symbol": symbol,
                "best_epoch": best.epoch,
                "best_val_sortino": best.val_sortino,
                "best_val_return": best.val_return,
                "best_val_score": best.val_score,
                "final_sharpness_ema": history[-1].sharpness_ema,
                "final_lr_scale": history[-1].lr_scale,
                "total_epochs": len(history),
                "wall_time_s": time.time() - t0,
                "checkpoint_dir": str(trainer.checkpoint_dir),
                "sam_mode": sc.sam_mode, "rho": sc.rho,
                "weight_decay": tc.weight_decay,
                "horizons": list(horizons),
                "error": None,
            }
            all_results.append(result)
            _save_results(all_results, out_json, out_csv)
            print(f"  Sort={best.val_sortino:.3f} Ret={best.val_return:.4f} @ ep{best.epoch} ({time.time()-t0:.0f}s)", flush=True)

    _print_summary(all_results)
    print(f"\nResults: {out_json}\nLeaderboard: {out_csv}", flush=True)


def run_remote(args):
    """Provision RunPod GPU, sync code+data, run training remotely."""
    from src.runpod_client import RunPodClient, PodConfig, resolve_gpu_preferences

    plan = _build_remote_plan(
        args,
        resolve_gpu_preferences=resolve_gpu_preferences,
    )
    print(f"Provisioning RunPod — preference chain: {plan.gpu_preferences}", flush=True)

    client = RunPodClient()
    config = PodConfig(
        name=f"sap-sweep-{time.strftime('%m%d%H%M')}",
        gpu_type=plan.gpu_preferences[0],
        gpu_count=1,
        image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        volume_size=50,
    )
    print(f"Provisioning pod and waiting for SSH...", flush=True)
    pod = client.create_ready_pod_with_fallback(
        config,
        plan.gpu_preferences,
        timeout=plan.wait_timeout_seconds,
    )
    host, port = pod.ssh_host, pod.ssh_port
    print(f"SSH ready: {host}:{port}", flush=True)

    remote_dir = plan.remote_dir
    stage_error: BaseException | None = None
    try:
        try:
            _bootstrap_remote(host, port, remote_dir)
        except Exception as exc:
            raise _render_remote_stage_error(
                stage="bootstrap/install",
                pod=pod,
                remote_dir=remote_dir,
                detail=str(exc),
            ) from exc
        try:
            _run_remote_sweep(host, port, remote_dir, args)
        except Exception as exc:
            raise _render_remote_stage_error(
                stage="remote sweep",
                pod=pod,
                remote_dir=remote_dir,
                detail=str(exc),
            ) from exc
        try:
            _download_results(host, port, remote_dir)
        except Exception as exc:
            raise _render_remote_stage_error(
                stage="result download",
                pod=pod,
                remote_dir=remote_dir,
                detail=str(exc),
            ) from exc
    except Exception as exc:
        stage_error = exc
        raise
    finally:
        if not args.keep_pod:
            print("Terminating pod...", flush=True)
            if stage_error is None:
                client.terminate_pod(pod.id)
            else:
                try:
                    client.terminate_pod(pod.id)
                except Exception as cleanup_exc:
                    stage_error.add_note(
                        f"failed to terminate pod {pod.id} after remote failure: {cleanup_exc}"
                    )


def _bootstrap_remote(host: str, port: int, remote_dir: str):
    """Sync code and install deps on remote pod."""
    print("Syncing code...", flush=True)
    exclude_patterns = [
        "__pycache__/",
        ".git/",
        ".venv*/",
        "*.pyc",
        "pufferlib_market/data/",
        "pufferlib_market/checkpoints/",
        "sharpnessadjustedproximalpolicy/checkpoints/",
    ]
    _run_checked_local(
        [
            "rsync",
            "-az",
            "--delete",
            *[value for pattern in exclude_patterns for value in ("--exclude", pattern)],
            "-e",
            f"ssh -o StrictHostKeyChecking=no -p {port}",
            ".",
            f"root@{host}:{remote_dir}/",
        ],
        description="Failed to sync repository to remote pod",
    )

    # sync forecast caches
    print("Syncing forecast caches...", flush=True)
    _run_checked_local(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -p {port}",
            "binanceneural/forecast_cache/",
            f"root@{host}:{remote_dir}/binanceneural/forecast_cache/",
        ],
        description="Failed to sync forecast caches to remote pod",
    )

    # sync training data
    print("Syncing training data...", flush=True)
    _run_checked_local(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -p {port}",
            "trainingdatahourly/crypto/",
            f"root@{host}:{remote_dir}/trainingdatahourly/crypto/",
        ],
        description="Failed to sync training data to remote pod",
    )

    print("Installing deps...", flush=True)
    install_cmd = _build_remote_install_cmd(remote_dir)
    result = _ssh(host, port, install_cmd)
    if result.returncode != 0:
        raise _render_bootstrap_install_error(
            host=host,
            port=port,
            remote_dir=remote_dir,
            result=result,
            install_cmd=install_cmd,
        )


def _render_bootstrap_install_error(
    *,
    host: str,
    port: int,
    remote_dir: str,
    result,
    install_cmd: str,
) -> RuntimeError:
    base_error = _render_subprocess_error(
        description="Remote dependency installation failed",
        cmd=_ssh_cmd(
            ssh_host=host,
            ssh_port=port,
            remote_cmd=install_cmd,
        ),
        result=result,
    )
    return RuntimeError(
        "\n".join(
            [
                "RunPod SAP remote bootstrap/install failed",
                f"ssh_host: {host}",
                f"ssh_port: {port}",
                f"remote_dir: {remote_dir}",
                str(base_error),
            ]
        )
    )


def _run_remote_sweep(host: str, port: int, remote_dir: str, args):
    """Execute sweep on remote pod."""
    cmd = _build_remote_sweep_cmd(remote_dir, args)
    print(f"Running: {cmd}", flush=True)
    result = _ssh(host, port, cmd)
    if result.returncode != 0:
        raise _render_subprocess_error(
            description="Remote training sweep failed",
            cmd=_ssh_cmd(
                ssh_host=host,
                ssh_port=port,
                remote_cmd=cmd,
            ),
            result=result,
        )


def _download_results(host: str, port: int, remote_dir: str):
    """Download results from remote pod."""
    print("Downloading results...", flush=True)
    _run_checked_local(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -p {port}",
            f"root@{host}:{remote_dir}/sharpnessadjustedproximalpolicy/scaled_*.json",
            f"root@{host}:{remote_dir}/sharpnessadjustedproximalpolicy/scaled_*.csv",
            "sharpnessadjustedproximalpolicy/",
        ],
        description="Failed to download remote result summaries",
    )
    _run_checked_local(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -p {port}",
            f"root@{host}:{remote_dir}/sharpnessadjustedproximalpolicy/checkpoints/",
            "sharpnessadjustedproximalpolicy/checkpoints/",
        ],
        description="Failed to download remote checkpoints",
    )


def _render_remote_stage_error(*, stage: str, pod, remote_dir: str, detail: str) -> RuntimeError:
    message = [
        f"RunPod SAP remote {stage} failed",
        f"pod_id: {pod.id}",
        f"gpu_type: {pod.gpu_type}",
        f"ssh_host: {pod.ssh_host}",
        f"ssh_port: {pod.ssh_port}",
        f"remote_dir: {remote_dir}",
        detail,
    ]
    return RuntimeError("\n".join(message))

def _save_results(results: list[dict], json_path: Path, csv_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "config", "best_epoch", "val_sortino", "val_return", "sharpness", "wd_scale", "wall_s", "error"])
        for r in sorted(results, key=lambda x: x.get("best_val_sortino", -999), reverse=True):
            if r.get("error"):
                w.writerow([r.get("symbol"), r.get("name"), "", "", "", "", "", "", r["error"][:60]])
            else:
                w.writerow([
                    r["symbol"], r["name"], r["best_epoch"],
                    f"{r['best_val_sortino']:.3f}", f"{r['best_val_return']:.4f}",
                    f"{r.get('final_sharpness_ema', 0):.1f}",
                    f"{r.get('final_lr_scale', 1):.2f}",
                    f"{r.get('wall_time_s', 0):.0f}", "",
                ])


def _print_summary(results: list[dict]):
    print(f"\n{'='*90}", flush=True)
    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]
    print(f"Completed: {len(ok)} OK, {len(err)} errors", flush=True)

    # per-symbol best
    by_sym: dict[str, dict] = {}
    for r in ok:
        s = r["symbol"]
        if s not in by_sym or r["best_val_sortino"] > by_sym[s]["best_val_sortino"]:
            by_sym[s] = r

    print(f"\n{'Symbol':<12} {'Config':<20} {'Sort':>8} {'Ret':>8} {'Ep':>3} {'WD':>5}", flush=True)
    print("-" * 60, flush=True)
    for sym in sorted(by_sym):
        r = by_sym[sym]
        print(f"{sym:<12} {r['name']:<20} {r['best_val_sortino']:>8.1f} {r['best_val_return']:>8.3f} {r['best_epoch']:>3} {r.get('weight_decay', 0.04):>5.3f}", flush=True)

    # SAM win rate
    sam_wins = baseline_wins = ties = 0
    for sym in set(r["symbol"] for r in ok):
        sym_r = [r for r in ok if r["symbol"] == sym]
        base = [r for r in sym_r if r.get("sam_mode") == "none"]
        sam = [r for r in sym_r if r.get("sam_mode") != "none"]
        if base and sam:
            best_b = max(r["best_val_sortino"] for r in base)
            best_s = max(r["best_val_sortino"] for r in sam)
            if best_s > best_b * 1.01:
                sam_wins += 1
            elif best_b > best_s * 1.01:
                baseline_wins += 1
            else:
                ties += 1

    total = sam_wins + baseline_wins + ties
    if total:
        print(f"\nSAM wins: {sam_wins}/{total} | Baseline wins: {baseline_wins}/{total} | Ties: {ties}/{total}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local or RunPod-backed scaled SAP training sweeps.")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--gpu", type=str, default=None, help="RunPod GPU type")
    parser.add_argument(
        "--gpu-fallbacks",
        type=str,
        default=None,
        help="Comma-separated fallback GPU aliases/names, or 'none' to disable fallbacks",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--min-rows", type=int, default=2000)
    parser.add_argument("--skip-cache-gen", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved remote RunPod execution plan and exit without provisioning a pod.",
    )
    parser.add_argument(
        "--dry-run-text",
        action="store_true",
        help="When used with --dry-run, also print a human-readable summary to stderr.",
    )
    args = parser.parse_args(argv)
    if args.dry_run and not args.gpu:
        parser.error("--dry-run requires --gpu TYPE")
    if args.dry_run_text and not args.dry_run:
        parser.error("--dry-run-text requires --dry-run")
    if args.gpu_fallbacks and not args.gpu:
        parser.error("--gpu-fallbacks requires --gpu TYPE")
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.gpu:
        if args.dry_run:
            from src.runpod_client import resolve_gpu_preferences

            plan = _build_remote_plan(
                args,
                resolve_gpu_preferences=resolve_gpu_preferences,
            )
            print(json.dumps(_remote_plan_payload(plan), indent=2, sort_keys=True))
            if args.dry_run_text:
                print(_format_remote_plan_summary(plan), file=sys.stderr)
            return
        run_remote(args)
    elif args.local:
        run_local(args)
    else:
        print("Specify --local or --gpu TYPE")
        sys.exit(1)


if __name__ == "__main__":
    main()
