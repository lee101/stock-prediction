#!/usr/bin/env python3
"""Run Chronos2 LoRA sweeps with non-regression promotion gating.

For each symbol:
1) Evaluate the currently-promoted model from rebuild_all_caches.BEST_MODELS.
2) Train a LoRA grid from the base model.
3) Promote only if the best candidate beats the current model by configured
   absolute/relative MAE% thresholds.
4) Rebuild forecast cache for the selected model (promoted or retained).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_hourly_experiment.rebuild_all_caches import BEST_MODELS


DEFAULT_DATA_ROOT = Path("trainingdatahourly/stocks")
DEFAULT_OUTPUT_ROOT = Path("chronos2_finetuned")
DEFAULT_HYPERPARAM_LORA_DIR = Path("hyperparams/chronos2/hourly_lora")
DEFAULT_HYPERPARAM_BASE_DIR = Path("hyperparams/chronos2/hourly_finetune")
DEFAULT_EXPERIMENT_DIR = Path("experiments")


@dataclass
class CandidateConfig:
    context_length: int
    learning_rate: float
    num_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    save_name: str


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in parse_csv_list(value)]


def float_token(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".").replace(".", "p")


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _report_path(symbol: str, finetune_mode: str, save_name: str) -> Path:
    root = DEFAULT_HYPERPARAM_LORA_DIR if finetune_mode == "lora" else DEFAULT_HYPERPARAM_BASE_DIR
    return root / f"{symbol}_{finetune_mode}_{save_name}.json"


def should_promote(
    *,
    current_test_mae_percent: float,
    candidate_test_mae_percent: float,
    min_improvement_abs: float,
    min_improvement_rel: float,
) -> bool:
    if candidate_test_mae_percent >= current_test_mae_percent:
        return False
    improvement_abs = current_test_mae_percent - candidate_test_mae_percent
    if improvement_abs < min_improvement_abs:
        return False
    if current_test_mae_percent <= 0:
        return True
    improvement_rel = improvement_abs / current_test_mae_percent
    return improvement_rel >= min_improvement_rel


def evaluate_current_model(
    *,
    symbol: str,
    run_id: str,
    data_root: Path,
    val_hours: int,
    test_hours: int,
    batch_size: int,
    torch_dtype: str,
) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    current_model = BEST_MODELS.get(symbol)
    if not current_model:
        errors.append(f"missing_best_model_mapping({symbol})")
        return None, errors

    model_id = DEFAULT_OUTPUT_ROOT / current_model / "finetuned-ckpt"
    if not model_id.exists():
        errors.append(f"missing_current_model_ckpt({model_id})")
        return None, errors

    save_name = f"{symbol}_current_eval_{run_id}"
    cmd = [
        sys.executable,
        "chronos2_trainer.py",
        "--symbol",
        symbol,
        "--data-root",
        str(data_root),
        "--output-root",
        str(DEFAULT_OUTPUT_ROOT),
        "--save-name",
        save_name,
        "--finetune-mode",
        "none",
        "--model-id",
        str(model_id),
        "--context-length",
        "512",
        "--batch-size",
        str(batch_size),
        "--val-hours",
        str(val_hours),
        "--test-hours",
        str(test_hours),
        "--torch-dtype",
        torch_dtype,
    ]
    proc = run_cmd(cmd)
    report_path = _report_path(symbol, "none", save_name)
    if proc.returncode != 0 or not report_path.exists():
        errors.append(
            f"current_eval_failed(returncode={proc.returncode}, report_exists={report_path.exists()}, model={current_model})"
        )
        return None, errors

    payload = load_json(report_path)
    return {
        "model_name": current_model,
        "model_id": str(model_id),
        "save_name": save_name,
        "report_path": str(report_path),
        "val_mae_percent": float(payload["val_metrics"]["mae_percent"]),
        "test_mae_percent": float(payload["test_metrics"]["mae_percent"]),
    }, errors


def train_candidate(
    *,
    symbol: str,
    cfg: CandidateConfig,
    data_root: Path,
    val_hours: int,
    test_hours: int,
    batch_size: int,
    torch_dtype: str,
) -> tuple[dict | None, str | None]:
    cmd = [
        sys.executable,
        "chronos2_trainer.py",
        "--symbol",
        symbol,
        "--data-root",
        str(data_root),
        "--output-root",
        str(DEFAULT_OUTPUT_ROOT),
        "--save-name",
        cfg.save_name,
        "--finetune-mode",
        "lora",
        "--context-length",
        str(cfg.context_length),
        "--learning-rate",
        str(cfg.learning_rate),
        "--num-steps",
        str(cfg.num_steps),
        "--lora-r",
        str(cfg.lora_r),
        "--lora-alpha",
        str(cfg.lora_alpha),
        "--lora-dropout",
        str(cfg.lora_dropout),
        "--batch-size",
        str(batch_size),
        "--val-hours",
        str(val_hours),
        "--test-hours",
        str(test_hours),
        "--torch-dtype",
        torch_dtype,
    ]
    proc = run_cmd(cmd)
    report_path = _report_path(symbol, "lora", cfg.save_name)
    if proc.returncode != 0 or not report_path.exists():
        return None, f"candidate_failed(save={cfg.save_name}, returncode={proc.returncode}, report_exists={report_path.exists()})"

    payload = load_json(report_path)
    return {
        "save_name": cfg.save_name,
        "context_length": cfg.context_length,
        "learning_rate": cfg.learning_rate,
        "num_steps": cfg.num_steps,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "report_path": str(report_path),
        "output_dir": str(payload.get("output_dir", "")),
        "val_mae_percent": float(payload["val_metrics"]["mae_percent"]),
        "test_mae_percent": float(payload["test_metrics"]["mae_percent"]),
    }, None


def rebuild_cache_for_model(
    *,
    symbol: str,
    model_name: str,
    data_root: Path,
) -> tuple[bool, str | None]:
    ckpt = DEFAULT_OUTPUT_ROOT / model_name / "finetuned-ckpt"
    if not ckpt.exists():
        return False, f"selected_model_ckpt_missing({ckpt})"

    cmd = [
        sys.executable,
        "-m",
        "alpacanewccrosslearning.build_forecasts",
        "--symbols",
        symbol,
        "--finetuned-model",
        str(ckpt),
        "--forecast-cache-root",
        "unified_hourly_experiment/forecast_cache",
        "--stock-data-root",
        str(data_root),
        "--horizons",
        "1,24",
        "--lookback-hours",
        "8000",
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        return False, f"cache_rebuild_failed(symbol={symbol}, model={model_name}, returncode={proc.returncode})"
    return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos non-regression LoRA sweep.")
    parser.add_argument("--symbols", default="NVDA,PLTR,GOOG,DBX,TRIP,MTCH")
    parser.add_argument("--contexts", default="512")
    parser.add_argument("--learning-rates", default="5e-5,1e-4")
    parser.add_argument("--steps", default="200,400")
    parser.add_argument("--lora-ranks", default="16,32")
    parser.add_argument("--lora-alphas", default="32")
    parser.add_argument("--lora-dropouts", default="0.05")
    parser.add_argument("--val-hours", type=int, default=336)
    parser.add_argument("--test-hours", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--min-improvement-abs", type=float, default=0.0)
    parser.add_argument("--min-improvement-rel", type=float, default=0.0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    symbols = [s.upper() for s in parse_csv_list(args.symbols)]
    contexts = parse_int_list(args.contexts)
    learning_rates = parse_float_list(args.learning_rates)
    steps = parse_int_list(args.steps)
    lora_ranks = parse_int_list(args.lora_ranks)
    lora_alphas = parse_int_list(args.lora_alphas)
    lora_dropouts = parse_float_list(args.lora_dropouts)

    summary: dict = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "symbols": symbols,
        "grid": {
            "contexts": contexts,
            "learning_rates": learning_rates,
            "steps": steps,
            "lora_ranks": lora_ranks,
            "lora_alphas": lora_alphas,
            "lora_dropouts": lora_dropouts,
            "val_hours": int(args.val_hours),
            "test_hours": int(args.test_hours),
            "batch_size": int(args.batch_size),
            "torch_dtype": str(args.torch_dtype),
            "min_improvement_abs": float(args.min_improvement_abs),
            "min_improvement_rel": float(args.min_improvement_rel),
        },
        "results": {},
    }

    for symbol in symbols:
        logger.info("=== {}: evaluating current promoted model ===", symbol)
        current_eval, errors = evaluate_current_model(
            symbol=symbol,
            run_id=run_id,
            data_root=args.data_root,
            val_hours=args.val_hours,
            test_hours=args.test_hours,
            batch_size=args.batch_size,
            torch_dtype=args.torch_dtype,
        )

        symbol_result: dict = {
            "current_eval": current_eval,
            "candidates": [],
            "best_candidate": None,
            "promoted": False,
            "selected_model": (current_eval["model_name"] if current_eval else None),
            "cache_rebuilt": False,
            "errors": list(errors),
        }

        if current_eval is None:
            summary["results"][symbol] = symbol_result
            continue

        for ctx in contexts:
            for lr in learning_rates:
                for st in steps:
                    for rank in lora_ranks:
                        for alpha in lora_alphas:
                            for dropout in lora_dropouts:
                                save_name = (
                                    f"{symbol}_lora_nonreg_{run_id}_ctx{ctx}"
                                    f"_lr{float_token(lr)}_st{st}_r{rank}"
                                    f"_a{alpha}_d{float_token(dropout)}"
                                )
                                cfg = CandidateConfig(
                                    context_length=int(ctx),
                                    learning_rate=float(lr),
                                    num_steps=int(st),
                                    lora_r=int(rank),
                                    lora_alpha=int(alpha),
                                    lora_dropout=float(dropout),
                                    save_name=save_name,
                                )
                                logger.info(
                                    "[{}] ctx={} lr={} steps={} r={} alpha={} dropout={}",
                                    symbol,
                                    cfg.context_length,
                                    cfg.learning_rate,
                                    cfg.num_steps,
                                    cfg.lora_r,
                                    cfg.lora_alpha,
                                    cfg.lora_dropout,
                                )
                                result, err = train_candidate(
                                    symbol=symbol,
                                    cfg=cfg,
                                    data_root=args.data_root,
                                    val_hours=args.val_hours,
                                    test_hours=args.test_hours,
                                    batch_size=args.batch_size,
                                    torch_dtype=args.torch_dtype,
                                )
                                if err is not None:
                                    symbol_result["errors"].append(err)
                                    continue
                                symbol_result["candidates"].append(result)

        if symbol_result["candidates"]:
            best = min(symbol_result["candidates"], key=lambda row: float(row["test_mae_percent"]))
            symbol_result["best_candidate"] = best

            promote = should_promote(
                current_test_mae_percent=float(current_eval["test_mae_percent"]),
                candidate_test_mae_percent=float(best["test_mae_percent"]),
                min_improvement_abs=float(args.min_improvement_abs),
                min_improvement_rel=float(args.min_improvement_rel),
            )
            symbol_result["promoted"] = bool(promote)
            symbol_result["selected_model"] = best["save_name"] if promote else current_eval["model_name"]

        if args.rebuild_cache and symbol_result["selected_model"]:
            ok, err = rebuild_cache_for_model(
                symbol=symbol,
                model_name=str(symbol_result["selected_model"]),
                data_root=args.data_root,
            )
            symbol_result["cache_rebuilt"] = bool(ok)
            if err is not None:
                symbol_result["errors"].append(err)

        summary["results"][symbol] = symbol_result

    output_path = args.output
    if output_path is None:
        DEFAULT_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_EXPERIMENT_DIR / f"chronos_nonreg_sweep_{run_id}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved non-regression sweep summary -> {}", output_path)


if __name__ == "__main__":
    main()
