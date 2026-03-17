#!/usr/bin/env python3
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


@dataclass(frozen=True)
class SweepConfig:
    symbol: str
    preaug: str
    context_length: int
    learning_rate: float
    num_steps: int
    prediction_length: int
    lora_r: int


@dataclass(frozen=True)
class SweepResult:
    config: SweepConfig
    status: str
    run_name: str | None = None
    result_path: str | None = None
    output_dir: str | None = None
    val_mae_percent: float | None = None
    test_mae_percent: float | None = None
    val_consistency_score: float | None = None
    test_consistency_score: float | None = None
    elapsed_s: float | None = None
    error: str | None = None


def build_train_cmd(
    *,
    run_id: str,
    cfg: SweepConfig,
    data_root: Path,
    output_root: Path,
    results_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/train_crypto_lora_sweep.py",
        "--symbol",
        cfg.symbol,
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--results-dir",
        str(results_dir),
        "--context-length",
        str(int(cfg.context_length)),
        "--prediction-length",
        str(int(cfg.prediction_length)),
        "--learning-rate",
        str(float(cfg.learning_rate)),
        "--num-steps",
        str(int(cfg.num_steps)),
        "--lora-r",
        str(int(cfg.lora_r)),
        "--preaug",
        cfg.preaug,
        "--run-prefix",
        run_id,
    ]


def iter_sweep_configs(
    *,
    symbols: Sequence[str],
    preaugs: Sequence[str],
    context_lengths: Sequence[int],
    learning_rates: Sequence[float],
    num_steps: int,
    prediction_length: int,
    lora_r: int,
) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for symbol in normalize_symbols(symbols):
        for preaug in [str(item).strip() for item in preaugs if str(item).strip()]:
            for context_length in context_lengths:
                for learning_rate in learning_rates:
                    configs.append(
                        SweepConfig(
                            symbol=symbol,
                            preaug=preaug,
                            context_length=int(context_length),
                            learning_rate=float(learning_rate),
                            num_steps=int(num_steps),
                            prediction_length=int(prediction_length),
                            lora_r=int(lora_r),
                        )
                    )
    return configs


def _newest_matching_result(results_dir: Path, run_id: str, cfg: SweepConfig) -> Path | None:
    pattern = (
        f"{run_id}_{cfg.symbol}_lora_{cfg.preaug}_ctx{int(cfg.context_length)}_"
        f"lr*_r{int(cfg.lora_r)}_*.json"
    )
    matches = sorted(results_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _result_from_payload(cfg: SweepConfig, payload: dict[str, Any], *, elapsed_s: float) -> SweepResult:
    val = payload.get("val") if isinstance(payload.get("val"), dict) else {}
    test = payload.get("test") if isinstance(payload.get("test"), dict) else {}
    return SweepResult(
        config=cfg,
        status="ok",
        run_name=str(payload.get("run_name") or "") or None,
        result_path=str(payload.get("result_path") or "") or None,
        output_dir=str(payload.get("output_dir") or "") or None,
        val_mae_percent=_to_float(val.get("mae_percent_mean")),
        test_mae_percent=_to_float(test.get("mae_percent_mean")),
        val_consistency_score=_to_float(payload.get("val_consistency_score")),
        test_consistency_score=_to_float(payload.get("test_consistency_score")),
        elapsed_s=float(elapsed_s),
    )


def _write_summary_json(path: Path, *, run_id: str, results: Sequence[SweepResult]) -> None:
    payload = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_summary_csv(path: Path, results: Sequence[SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "preaug",
        "context_length",
        "learning_rate",
        "num_steps",
        "prediction_length",
        "lora_r",
        "status",
        "run_name",
        "result_path",
        "output_dir",
        "val_mae_percent",
        "test_mae_percent",
        "val_consistency_score",
        "test_consistency_score",
        "elapsed_s",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "symbol": result.config.symbol,
                    "preaug": result.config.preaug,
                    "context_length": result.config.context_length,
                    "learning_rate": result.config.learning_rate,
                    "num_steps": result.config.num_steps,
                    "prediction_length": result.config.prediction_length,
                    "lora_r": result.config.lora_r,
                    "status": result.status,
                    "run_name": result.run_name or "",
                    "result_path": result.result_path or "",
                    "output_dir": result.output_dir or "",
                    "val_mae_percent": result.val_mae_percent,
                    "test_mae_percent": result.test_mae_percent,
                    "val_consistency_score": result.val_consistency_score,
                    "test_consistency_score": result.test_consistency_score,
                    "elapsed_s": result.elapsed_s,
                    "error": result.error or "",
                }
            )


def _write_summary_md(path: Path, *, run_id: str, results: Sequence[SweepResult]) -> None:
    lines = [f"# Crypto LoRA Batch ({run_id})", ""]
    ok = [result for result in results if result.status == "ok"]
    lines.append(f"- Total runs: {len(results)}")
    lines.append(f"- Successful: {len(ok)}")
    lines.append(f"- Errors: {len(results) - len(ok)}")
    lines.append("")
    lines.append("| Symbol | Preaug | Ctx | LR | Val MAE% | Test MAE% | Val Consistency | Output Dir |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for result in sorted(ok, key=lambda item: (item.val_mae_percent is None, item.val_mae_percent or float("inf"))):
        lines.append(
            "| "
            + " | ".join(
                [
                    result.config.symbol,
                    result.config.preaug,
                    str(result.config.context_length),
                    f"{result.config.learning_rate:.0e}",
                    f"{result.val_mae_percent:.4f}" if result.val_mae_percent is not None else "n/a",
                    f"{result.test_mae_percent:.4f}" if result.test_mae_percent is not None else "n/a",
                    f"{result.val_consistency_score:.4f}" if result.val_consistency_score is not None else "n/a",
                    result.output_dir or "",
                ]
            )
            + " |"
        )
    failures = [result for result in results if result.status != "ok"]
    if failures:
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        for result in failures:
            lines.append(
                f"- {result.config.symbol} {result.config.preaug} ctx={result.config.context_length} "
                f"lr={result.config.learning_rate:.0e}: {result.error or result.status}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_batch(
    *,
    run_id: str,
    configs: Sequence[SweepConfig],
    data_root: Path,
    output_root: Path,
    results_dir: Path,
    stop_on_error: bool = False,
) -> list[SweepResult]:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[SweepResult] = []
    for idx, cfg in enumerate(configs, 1):
        print(
            f"[{idx}/{len(configs)}] {cfg.symbol} preaug={cfg.preaug} "
            f"ctx={cfg.context_length} lr={cfg.learning_rate:.0e}",
            flush=True,
        )
        cmd = build_train_cmd(
            run_id=run_id,
            cfg=cfg,
            data_root=data_root,
            output_root=output_root,
            results_dir=results_dir,
        )
        started_at = time.time()
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        elapsed_s = time.time() - started_at
        result_path = _newest_matching_result(results_dir, run_id, cfg)

        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[-1000:]
            results.append(
                SweepResult(
                    config=cfg,
                    status="error",
                    result_path=str(result_path) if result_path else None,
                    elapsed_s=float(elapsed_s),
                    error=error,
                )
            )
            if stop_on_error:
                break
            continue

        if result_path is None or not result_path.exists():
            results.append(
                SweepResult(
                    config=cfg,
                    status="missing_report",
                    elapsed_s=float(elapsed_s),
                    error="train_crypto_lora_sweep completed without a result JSON",
                )
            )
            if stop_on_error:
                break
            continue

        payload = json.loads(result_path.read_text())
        result = _result_from_payload(cfg, payload, elapsed_s=elapsed_s)
        results.append(result)
        print(
            "  -> "
            f"val_mae={result.val_mae_percent if result.val_mae_percent is not None else float('nan'):.4f}% "
            f"test_mae={result.test_mae_percent if result.test_mae_percent is not None else float('nan'):.4f}%",
            flush=True,
        )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tagged batch of Chronos2 crypto LoRA sweeps.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--output-root", type=Path, default=Path("chronos2_finetuned"))
    parser.add_argument("--results-dir", type=Path, default=Path("hyperparams/crypto_lora_sweep"))
    parser.add_argument("--preaugs", default="baseline,percent_change,log_returns")
    parser.add_argument("--context-lengths", default="128,256")
    parser.add_argument("--learning-rates", default="5e-5,1e-4")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = normalize_symbols(parse_csv_tokens(args.symbols))
    preaugs = [str(item).strip() for item in parse_csv_tokens(args.preaugs)]
    context_lengths = [int(item) for item in parse_csv_tokens(args.context_lengths, cast=int)]
    learning_rates = [float(item) for item in parse_csv_tokens(args.learning_rates, cast=float)]
    configs = iter_sweep_configs(
        symbols=symbols,
        preaugs=preaugs,
        context_lengths=context_lengths,
        learning_rates=learning_rates,
        num_steps=int(args.num_steps),
        prediction_length=int(args.prediction_length),
        lora_r=int(args.lora_r),
    )
    if not configs:
        raise SystemExit("No sweep configurations generated.")

    results = run_batch(
        run_id=str(args.run_id),
        configs=configs,
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        results_dir=Path(args.results_dir),
        stop_on_error=bool(args.stop_on_error),
    )

    prefix = Path(args.results_dir) / f"{args.run_id}_batch_summary"
    _write_summary_json(prefix.with_suffix(".json"), run_id=str(args.run_id), results=results)
    _write_summary_csv(prefix.with_suffix(".csv"), results)
    _write_summary_md(prefix.with_suffix(".md"), run_id=str(args.run_id), results=results)

    ok = [result for result in results if result.status == "ok"]
    print(f"Summary written: {prefix.with_suffix('.json')}", flush=True)
    print(f"Successful runs: {len(ok)}/{len(results)}", flush=True)
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
