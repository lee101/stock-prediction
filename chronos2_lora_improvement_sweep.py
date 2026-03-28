#!/usr/bin/env python3
"""Chronos2 LoRA improvement sweep targeting high-MAE symbols.

Sweeps lora_r, lr, preaug, context_length, and wider LoRA targets
(gate_proj, up_proj, down_proj inspired by Qwen finetuning recipe).
Reports val MAE% and promotes configs that beat baseline by >5%.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.remote_training_pipeline import normalize_symbols, parse_csv_tokens

DASHBOARD_CSV = REPO / "chronos2_mae_dashboard.csv"
DEFAULT_MAE_THRESHOLD = 2.0
DEFAULT_IMPROVEMENT_THRESHOLD = 5.0

NARROW_LORA_TARGETS = ("q", "k", "v", "o")
WIDE_LORA_TARGETS = ("q", "k", "v", "o", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class ImprovementSweepConfig:
    symbol: str
    preaug: str
    context_length: int
    batch_size: int
    learning_rate: float
    num_steps: int
    prediction_length: int
    lora_r: int
    lora_alpha: int
    lora_targets: tuple[str, ...]
    lr_scheduler: str  # "cosine" or "linear"
    warmup_ratio: float


@dataclass(frozen=True)
class ImprovementResult:
    config: ImprovementSweepConfig
    status: str
    run_name: str | None = None
    result_path: str | None = None
    output_dir: str | None = None
    val_mae_percent: float | None = None
    test_mae_percent: float | None = None
    baseline_mae_percent: float | None = None
    improvement_pct: float | None = None
    promoted: bool = False
    elapsed_s: float | None = None
    error: str | None = None


def load_dashboard_baselines(csv_path: Path) -> dict[str, float]:
    """Load symbol -> mae_pct from the dashboard CSV."""
    baselines: dict[str, float] = {}
    if not csv_path.exists():
        return baselines
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get("symbol", "").strip().upper()
            mae_str = row.get("mae_pct", "")
            if not sym or not mae_str:
                continue
            try:
                baselines[sym] = float(mae_str)
            except (ValueError, TypeError):
                pass
    return baselines


def find_high_mae_symbols(
    baselines: dict[str, float],
    threshold: float = DEFAULT_MAE_THRESHOLD,
    extra_symbols: Sequence[str] = (),
) -> list[str]:
    """Return symbols with MAE > threshold, plus any explicitly requested."""
    high = {sym for sym, mae in baselines.items() if mae > threshold}
    for sym in normalize_symbols(extra_symbols):
        high.add(sym)
    return sorted(high)


def generate_sweep_configs(
    symbols: Sequence[str],
    lora_rs: Sequence[int] = (8, 16),
    learning_rates: Sequence[float] = (1e-5, 5e-5, 1e-4),
    preaugs: Sequence[str] = ("baseline", "percent_change", "differencing"),
    context_lengths: Sequence[int] = (128, 256, 512),
    lora_target_sets: Sequence[tuple[str, ...]] = (NARROW_LORA_TARGETS, WIDE_LORA_TARGETS),
    batch_size: int = 32,
    num_steps: int = 1000,
    prediction_length: int = 24,
    warmup_ratio: float = 0.1,
) -> list[ImprovementSweepConfig]:
    configs: list[ImprovementSweepConfig] = []
    for symbol in normalize_symbols(symbols):
        for preaug in preaugs:
            for ctx in context_lengths:
                for lr in learning_rates:
                    for r in lora_rs:
                        for targets in lora_target_sets:
                            configs.append(
                                ImprovementSweepConfig(
                                    symbol=symbol,
                                    preaug=preaug,
                                    context_length=ctx,
                                    batch_size=batch_size,
                                    learning_rate=lr,
                                    num_steps=num_steps,
                                    prediction_length=prediction_length,
                                    lora_r=r,
                                    lora_alpha=r * 2,
                                    lora_targets=targets,
                                    lr_scheduler="cosine",
                                    warmup_ratio=warmup_ratio,
                                )
                            )
    return configs


def build_train_cmd(
    *,
    run_id: str,
    cfg: ImprovementSweepConfig,
    data_root: Path,
    output_root: Path,
    results_dir: Path,
) -> list[str]:
    run_prefix = _config_run_prefix(run_id, cfg)
    cmd = [
        sys.executable,
        "scripts/train_crypto_lora_sweep.py",
        "--symbol", cfg.symbol,
        "--data-root", str(data_root),
        "--output-root", str(output_root),
        "--results-dir", str(results_dir),
        "--context-length", str(cfg.context_length),
        "--prediction-length", str(cfg.prediction_length),
        "--batch-size", str(cfg.batch_size),
        "--learning-rate", str(cfg.learning_rate),
        "--num-steps", str(cfg.num_steps),
        "--lora-r", str(cfg.lora_r),
        "--lora-alpha", str(cfg.lora_alpha),
        "--lora-targets", ",".join(cfg.lora_targets),
        "--lr-scheduler-type", cfg.lr_scheduler,
        "--warmup-ratio", str(cfg.warmup_ratio),
        "--preaug", cfg.preaug,
        "--run-prefix", run_prefix,
    ]
    return cmd


def _config_run_prefix(run_id: str, cfg: ImprovementSweepConfig) -> str:
    targets_label = "wide" if len(cfg.lora_targets) > 4 else "narrow"
    return f"{run_id}_{targets_label}"


def _newest_matching_result(results_dir: Path, run_id: str, cfg: ImprovementSweepConfig) -> Path | None:
    run_prefix = _config_run_prefix(run_id, cfg)
    pattern = (
        f"{run_prefix}_{cfg.symbol}_lora_{cfg.preaug}_ctx{cfg.context_length}_"
        f"lr*_r{cfg.lora_r}_*.json"
    )
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_improvement(baseline: float | None, val_mae: float | None) -> float | None:
    if baseline is None or val_mae is None or baseline <= 0:
        return None
    return (baseline - val_mae) / baseline * 100.0


def run_sweep(
    *,
    run_id: str,
    configs: Sequence[ImprovementSweepConfig],
    baselines: dict[str, float],
    data_root: Path,
    output_root: Path,
    results_dir: Path,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
    stop_on_error: bool = False,
) -> list[ImprovementResult]:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[ImprovementResult] = []
    for idx, cfg in enumerate(configs, 1):
        targets_label = "wide" if len(cfg.lora_targets) > 4 else "narrow"
        print(
            f"[{idx}/{len(configs)}] {cfg.symbol} preaug={cfg.preaug} "
            f"ctx={cfg.context_length} lr={cfg.learning_rate:.0e} r={cfg.lora_r} "
            f"targets={targets_label}",
            flush=True,
        )
        cmd = build_train_cmd(
            run_id=run_id, cfg=cfg, data_root=data_root,
            output_root=output_root, results_dir=results_dir,
        )
        started_at = time.time()
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        elapsed_s = time.time() - started_at
        result_path = _newest_matching_result(results_dir, run_id, cfg)

        baseline_mae = baselines.get(cfg.symbol)

        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[-1000:]
            results.append(ImprovementResult(
                config=cfg, status="error", baseline_mae_percent=baseline_mae,
                elapsed_s=elapsed_s, error=error,
            ))
            if stop_on_error:
                break
            continue

        if result_path is None or not result_path.exists():
            results.append(ImprovementResult(
                config=cfg, status="missing_report", baseline_mae_percent=baseline_mae,
                elapsed_s=elapsed_s, error="no result JSON produced",
            ))
            if stop_on_error:
                break
            continue

        payload = json.loads(result_path.read_text())
        val_raw = payload.get("val")
        val = val_raw if isinstance(val_raw, dict) else {}
        test_raw = payload.get("test")
        test = test_raw if isinstance(test_raw, dict) else {}
        val_mae = _to_float(val.get("mae_percent_mean"))
        test_mae = _to_float(test.get("mae_percent_mean"))
        improvement = compute_improvement(baseline_mae, val_mae)
        promoted = improvement is not None and improvement > improvement_threshold

        results.append(ImprovementResult(
            config=cfg, status="ok",
            run_name=str(payload.get("run_name", "")),
            result_path=str(result_path),
            output_dir=str(payload.get("output_dir", "")),
            val_mae_percent=val_mae,
            test_mae_percent=test_mae,
            baseline_mae_percent=baseline_mae,
            improvement_pct=improvement,
            promoted=promoted,
            elapsed_s=elapsed_s,
        ))
        print(
            f"  -> val_mae={val_mae or float('nan'):.4f}% "
            f"baseline={baseline_mae or float('nan'):.4f}% "
            f"improvement={improvement or float('nan'):.1f}% "
            f"{'PROMOTED' if promoted else ''}",
            flush=True,
        )

    return results


def write_summary(path: Path, *, run_id: str, results: Sequence[ImprovementResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "promoted": sum(1 for r in results if r.promoted),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_summary_csv(path: Path, results: Sequence[ImprovementResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol", "preaug", "context_length", "lora_r", "learning_rate",
        "lora_targets", "status", "val_mae_percent", "test_mae_percent",
        "baseline_mae_percent", "improvement_pct", "promoted", "elapsed_s", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "symbol": r.config.symbol,
                "preaug": r.config.preaug,
                "context_length": r.config.context_length,
                "lora_r": r.config.lora_r,
                "learning_rate": r.config.learning_rate,
                "lora_targets": ",".join(r.config.lora_targets),
                "status": r.status,
                "val_mae_percent": r.val_mae_percent,
                "test_mae_percent": r.test_mae_percent,
                "baseline_mae_percent": r.baseline_mae_percent,
                "improvement_pct": r.improvement_pct,
                "promoted": r.promoted,
                "elapsed_s": r.elapsed_s,
                "error": r.error or "",
            })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos2 LoRA improvement sweep for high-MAE symbols")
    p.add_argument("--run-id", required=True)
    p.add_argument("--symbols", default=None, help="Comma-sep symbols. Defaults to auto-detected high-MAE.")
    p.add_argument("--mae-threshold", type=float, default=DEFAULT_MAE_THRESHOLD)
    p.add_argument("--improvement-threshold", type=float, default=DEFAULT_IMPROVEMENT_THRESHOLD)
    p.add_argument("--dashboard-csv", type=Path, default=DASHBOARD_CSV)
    p.add_argument("--data-root", type=Path, default=Path("trainingdatahourly"))
    p.add_argument("--output-root", type=Path, default=Path("chronos2_finetuned"))
    p.add_argument("--results-dir", type=Path, default=Path("hyperparams/lora_improvement_sweep"))
    p.add_argument("--lora-rs", default="8,16")
    p.add_argument("--learning-rates", default="1e-5,5e-5,1e-4")
    p.add_argument("--preaugs", default="baseline,percent_change,differencing")
    p.add_argument("--context-lengths", default="128,256,512")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--prediction-length", type=int, default=24)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--wide-targets", action="store_true", default=True,
                    help="Include wide LoRA targets (gate_proj, up_proj, down_proj)")
    p.add_argument("--no-wide-targets", action="store_false", dest="wide_targets")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    baselines = load_dashboard_baselines(args.dashboard_csv)

    if args.symbols:
        symbols = normalize_symbols(parse_csv_tokens(args.symbols))
    else:
        symbols = find_high_mae_symbols(
            baselines, args.mae_threshold,
            extra_symbols=["AAVEUSD", "AVAXUSD", "GOOG"],
        )

    if not symbols:
        print("No symbols to sweep.", flush=True)
        return 1

    print(f"Sweeping {len(symbols)} symbols: {', '.join(symbols)}", flush=True)

    lora_rs = [int(x) for x in parse_csv_tokens(args.lora_rs, cast=int)]
    learning_rates = [float(x) for x in parse_csv_tokens(args.learning_rates, cast=float)]
    preaugs = [s.strip() for s in parse_csv_tokens(args.preaugs)]
    context_lengths = [int(x) for x in parse_csv_tokens(args.context_lengths, cast=int)]

    target_sets: list[tuple[str, ...]] = [NARROW_LORA_TARGETS]
    if args.wide_targets:
        target_sets.append(WIDE_LORA_TARGETS)

    configs = generate_sweep_configs(
        symbols=symbols,
        lora_rs=lora_rs,
        learning_rates=learning_rates,
        preaugs=preaugs,
        context_lengths=context_lengths,
        lora_target_sets=target_sets,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        prediction_length=args.prediction_length,
        warmup_ratio=args.warmup_ratio,
    )
    print(f"Generated {len(configs)} configs", flush=True)

    results = run_sweep(
        run_id=args.run_id,
        configs=configs,
        baselines=baselines,
        data_root=args.data_root,
        output_root=args.output_root,
        results_dir=args.results_dir,
        improvement_threshold=args.improvement_threshold,
        stop_on_error=args.stop_on_error,
    )

    prefix = args.results_dir / f"{args.run_id}_improvement_summary"
    write_summary(prefix.with_suffix(".json"), run_id=args.run_id, results=results)
    write_summary_csv(prefix.with_suffix(".csv"), results)

    ok = [r for r in results if r.status == "ok"]
    promoted = [r for r in results if r.promoted]
    print(f"Runs: {len(ok)}/{len(results)} OK, {len(promoted)} promoted", flush=True)

    if promoted:
        print("Promoted configs (beat baseline by >5%):", flush=True)
        for r in sorted(promoted, key=lambda x: x.improvement_pct or 0, reverse=True):
            print(
                f"  {r.config.symbol} preaug={r.config.preaug} ctx={r.config.context_length} "
                f"lr={r.config.learning_rate:.0e} r={r.config.lora_r} "
                f"val_mae={r.val_mae_percent:.4f}% improvement={r.improvement_pct:.1f}%",
                flush=True,
            )

    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
