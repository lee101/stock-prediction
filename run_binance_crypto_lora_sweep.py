#!/usr/bin/env python3
"""Run sequential Binance crypto LoRA sweeps and summarize the winners."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATA_ROOT = Path("trainingdatahourlybinance")
DEFAULT_OUTPUT_ROOT = Path("chronos2_finetuned")
DEFAULT_RESULTS_ROOT = Path("experiments/binance_crypto_lora_sweep")


def _load_train_helpers():
    from scripts.train_crypto_lora_sweep import TrainConfig, train_and_evaluate

    return TrainConfig, train_and_evaluate


def _parse_multi_value(raw_values: Optional[Sequence[str]], *, normalizer) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in raw_values or ():
        for part in str(raw).split(","):
            normalized = normalizer(part)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
    return values


def parse_symbols(raw_values: Optional[Sequence[str]]) -> list[str]:
    return _parse_multi_value(raw_values, normalizer=lambda value: str(value).strip().upper())


def parse_preaugs(raw_values: Optional[Sequence[str]]) -> list[str]:
    return _parse_multi_value(raw_values, normalizer=lambda value: str(value).strip().lower())


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def summarize_result(result: dict[str, Any], *, result_path: Path) -> dict[str, Any]:
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    val = result.get("val") if isinstance(result.get("val"), dict) else {}
    test = result.get("test") if isinstance(result.get("test"), dict) else {}
    return {
        "symbol": str(config.get("symbol") or "").upper(),
        "preaug": str(config.get("preaug") or ""),
        "run_name": str(result.get("run_name") or ""),
        "output_dir": str(result.get("output_dir") or ""),
        "result_path": str(result_path),
        "val_consistency_score": _coerce_float(result.get("val_consistency_score")),
        "test_consistency_score": _coerce_float(result.get("test_consistency_score")),
        "val_mae_percent_mean": _coerce_float(val.get("mae_percent_mean")),
        "test_mae_percent_mean": _coerce_float(test.get("mae_percent_mean")),
    }


def rank_results(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(record: dict[str, Any]) -> tuple[float, float, str, str]:
        score = _coerce_float(record.get("val_consistency_score"))
        val_mae = _coerce_float(record.get("val_mae_percent_mean"))
        return (
            score if score is not None else float("inf"),
            val_mae if val_mae is not None else float("inf"),
            str(record.get("symbol") or ""),
            str(record.get("preaug") or ""),
        )

    return sorted((dict(record) for record in records), key=_sort_key)


def best_by_symbol(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ranked = rank_results(records)
    best: dict[str, dict[str, Any]] = {}
    for record in ranked:
        symbol = str(record.get("symbol") or "")
        if symbol and symbol not in best:
            best[symbol] = record
    return best


def render_summary_md(records: Iterable[dict[str, Any]]) -> str:
    ranked = rank_results(records)
    best = best_by_symbol(ranked)
    lines = [
        "# Binance Crypto LoRA Sweep",
        "",
        f"- Total runs: {len(ranked)}",
        "",
        "## Best by Symbol",
        "",
    ]
    for symbol, record in sorted(best.items()):
        score = record.get("val_consistency_score")
        val_mae = record.get("val_mae_percent_mean")
        test_mae = record.get("test_mae_percent_mean")
        lines.append(
            f"- `{symbol}`: `{record.get('preaug')}` "
            f"(val_consistency={score:.4f}, val_mae%={val_mae:.4f}, test_mae%={test_mae:.4f})"
        )
    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| Rank | Symbol | Preaug | Val Consistency | Val MAE% | Test MAE% | Run Name |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for idx, record in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(record.get("symbol") or ""),
                    str(record.get("preaug") or ""),
                    f"{float(record.get('val_consistency_score')):.4f}",
                    f"{float(record.get('val_mae_percent_mean')):.4f}",
                    f"{float(record.get('test_mae_percent_mean')):.4f}",
                    str(record.get("run_name") or ""),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_sweep(
    *,
    symbols: Sequence[str],
    preaugs: Sequence[str],
    data_root: Path,
    output_root: Path,
    results_dir: Path,
    context_length: int,
    prediction_length: int,
    learning_rate: float,
    num_steps: int,
    lora_r: int,
) -> dict[str, Any]:
    TrainConfig, train_and_evaluate = _load_train_helpers()
    results_dir.mkdir(parents=True, exist_ok=True)

    summarized_runs: list[dict[str, Any]] = []
    for symbol in symbols:
        data_path = Path(data_root) / f"{symbol}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found for {symbol}: {data_path}")
        for preaug in preaugs:
            cfg = TrainConfig(
                symbol=symbol,
                context_length=int(context_length),
                prediction_length=int(prediction_length),
                learning_rate=float(learning_rate),
                num_steps=int(num_steps),
                lora_r=int(lora_r),
                preaug=preaug,
            )
            result = train_and_evaluate(cfg, data_path, Path(output_root))
            result_path = Path(results_dir) / f"{result['run_name']}.json"
            _write_json(result_path, result)
            summarized_runs.append(summarize_result(result, result_path=result_path))

    ranked = rank_results(summarized_runs)
    best = best_by_symbol(ranked)
    summary = {
        "symbols": list(symbols),
        "preaugs": list(preaugs),
        "config": {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "results_dir": str(results_dir),
            "context_length": int(context_length),
            "prediction_length": int(prediction_length),
            "learning_rate": float(learning_rate),
            "num_steps": int(num_steps),
            "lora_r": int(lora_r),
        },
        "results": ranked,
        "best_by_symbol": best,
    }
    _write_json(Path(results_dir) / "summary.json", summary)
    (Path(results_dir) / "summary.md").write_text(render_summary_md(ranked))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", dest="symbols", action="append", required=True, help="Repeatable or comma-separated Binance symbols.")
    parser.add_argument("--preaug", dest="preaugs", action="append", required=True, help="Repeatable or comma-separated preaugmentation names.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lora-r", type=int, default=16)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = parse_symbols(args.symbols)
    preaugs = parse_preaugs(args.preaugs)
    if not symbols:
        raise SystemExit("At least one symbol is required.")
    if not preaugs:
        raise SystemExit("At least one preaug is required.")

    run_sweep(
        symbols=symbols,
        preaugs=preaugs,
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        results_dir=Path(args.results_dir),
        context_length=int(args.context_length),
        prediction_length=int(args.prediction_length),
        learning_rate=float(args.learning_rate),
        num_steps=int(args.num_steps),
        lora_r=int(args.lora_r),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
