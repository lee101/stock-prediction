#!/usr/bin/env python3
"""Compare Chronos2 precision / compile modes on the repo benchmark harness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModeSpec:
    name: str
    torch_compile: bool
    torch_dtype: str
    pipeline_backend: str = "chronos"


MODE_REGISTRY: dict[str, ModeSpec] = {
    "eager_fp32": ModeSpec(name="eager_fp32", torch_compile=False, torch_dtype="float32", pipeline_backend="chronos"),
    "compiled_fp32": ModeSpec(name="compiled_fp32", torch_compile=True, torch_dtype="float32", pipeline_backend="chronos"),
    "eager_bf16": ModeSpec(name="eager_bf16", torch_compile=False, torch_dtype="bfloat16", pipeline_backend="chronos"),
    "compiled_bf16": ModeSpec(name="compiled_bf16", torch_compile=True, torch_dtype="bfloat16", pipeline_backend="chronos"),
    "cute_fp32": ModeSpec(name="cute_fp32", torch_compile=False, torch_dtype="float32", pipeline_backend="cutechronos"),
    "cute_compiled_fp32": ModeSpec(name="cute_compiled_fp32", torch_compile=True, torch_dtype="float32", pipeline_backend="cutechronos"),
}


def resolve_modes(mode_names: Iterable[str]) -> list[ModeSpec]:
    resolved: list[ModeSpec] = []
    for name in mode_names:
        key = str(name).strip().lower()
        if key not in MODE_REGISTRY:
            raise KeyError(f"Unknown Chronos2 benchmark mode: {name}")
        resolved.append(MODE_REGISTRY[key])
    if not resolved:
        raise ValueError("At least one Chronos2 mode is required.")
    return resolved


def build_benchmark_command(args: argparse.Namespace, *, symbol: str, mode: ModeSpec, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "benchmark_chronos2.py",
        "--symbols",
        symbol,
        "--context-lengths",
        *(str(v) for v in args.context_lengths),
        "--batch-sizes",
        *(str(v) for v in args.batch_sizes),
        "--aggregations",
        *(str(v) for v in args.aggregations),
        "--sample-counts",
        *(str(v) for v in args.sample_counts),
        "--scalers",
        *(str(v) for v in args.scalers),
        "--device-map",
        args.device_map,
        "--torch-dtype",
        mode.torch_dtype,
        "--pipeline-backend",
        mode.pipeline_backend,
        "--output-dir",
        str(output_dir),
    ]
    if args.verbose:
        cmd.append("--verbose")
    if mode.torch_compile:
        cmd.append("--torch-compile")
    return cmd


def load_latest_benchmark(output_dir: Path, *, symbol: str) -> dict:
    bench_dir = output_dir / symbol
    if not bench_dir.exists():
        raise FileNotFoundError(f"No benchmark output found for {symbol} under {output_dir}")
    files = sorted(bench_dir.glob(f"{symbol}_chronos2_bench_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark JSON files found for {symbol} under {bench_dir}")
    payload = json.loads(files[-1].read_text())
    if not payload:
        raise ValueError(f"Benchmark payload for {symbol} is empty: {files[-1]}")
    return payload[0]


def compare_reports(baseline: dict, candidate: dict) -> dict[str, float]:
    base_test = baseline["test"]
    cand_test = candidate["test"]
    base_val = baseline["validation"]
    cand_val = candidate["validation"]

    def _delta_pct(cand: float, base: float) -> float:
        if float(base) == 0.0:
            return 0.0
        return ((float(cand) - float(base)) / float(base)) * 100.0

    return {
        "test_price_mae_delta_pct": _delta_pct(cand_test["price_mae"], base_test["price_mae"]),
        "test_pct_return_mae_delta_pct": _delta_pct(cand_test["pct_return_mae"], base_test["pct_return_mae"]),
        "validation_price_mae_delta_pct": _delta_pct(cand_val["price_mae"], base_val["price_mae"]),
        "validation_pct_return_mae_delta_pct": _delta_pct(cand_val["pct_return_mae"], base_val["pct_return_mae"]),
        "test_latency_speedup": (
            float(base_test["latency_s"]) / float(cand_test["latency_s"])
            if float(cand_test["latency_s"]) > 0.0
            else 0.0
        ),
    }


def report_passes_gate(
    deltas: dict[str, float],
    *,
    max_price_mae_regression_pct: float,
    max_return_mae_regression_pct: float,
) -> bool:
    return (
        deltas["test_price_mae_delta_pct"] <= max_price_mae_regression_pct
        and deltas["validation_price_mae_delta_pct"] <= max_price_mae_regression_pct
        and deltas["test_pct_return_mae_delta_pct"] <= max_return_mae_regression_pct
        and deltas["validation_pct_return_mae_delta_pct"] <= max_return_mae_regression_pct
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD"])
    parser.add_argument("--modes", nargs="+", default=["eager_fp32", "compiled_fp32"])
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--aggregations", nargs="+", default=["median"])
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[0])
    parser.add_argument("--scalers", nargs="+", default=["none"])
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--output-dir", default="chronos2_compile_benchmarks")
    parser.add_argument("--max-price-mae-regression-pct", type=float, default=0.0)
    parser.add_argument("--max-return-mae-regression-pct", type=float, default=0.0)
    parser.add_argument("--allow-mode-failure", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = resolve_modes(args.modes)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "symbols": list(args.symbols),
        "modes": [asdict(mode) for mode in modes],
        "max_price_mae_regression_pct": args.max_price_mae_regression_pct,
        "max_return_mae_regression_pct": args.max_return_mae_regression_pct,
        "results": {},
    }
    all_ok = True

    for symbol in args.symbols:
        symbol_summary: dict[str, object] = {}
        baseline_payload: dict | None = None
        baseline_mode: str | None = None
        for mode in modes:
            mode_output_dir = output_root / mode.name
            mode_output_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_benchmark_command(args, symbol=symbol, mode=mode, output_dir=mode_output_dir)
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                symbol_summary[mode.name] = {"status": "failed", "returncode": int(result.returncode)}
                all_ok = False
                if not args.allow_mode_failure:
                    summary["results"][symbol] = symbol_summary
                    summary_path = output_root / "summary.json"
                    summary_path.write_text(json.dumps(summary, indent=2))
                    return result.returncode
                continue

            payload = load_latest_benchmark(mode_output_dir, symbol=symbol)
            mode_result: dict[str, object] = {"status": "ok", "report": payload}
            if baseline_payload is None:
                baseline_payload = payload
                baseline_mode = mode.name
                mode_result["baseline"] = True
            else:
                deltas = compare_reports(baseline_payload, payload)
                passes = report_passes_gate(
                    deltas,
                    max_price_mae_regression_pct=args.max_price_mae_regression_pct,
                    max_return_mae_regression_pct=args.max_return_mae_regression_pct,
                )
                mode_result["baseline"] = False
                mode_result["compared_to"] = baseline_mode
                mode_result["deltas"] = deltas
                mode_result["passes_gate"] = passes
                if not passes:
                    all_ok = False
            symbol_summary[mode.name] = mode_result
        summary["results"][symbol] = symbol_summary

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print("Chronos2 Precision / Compile Audit")
    print("=" * 72)
    print(f"Summary JSON: {summary_path}")
    for symbol, symbol_summary in (summary["results"] or {}).items():
        print(f"\n{symbol}")
        for mode in modes:
            entry = symbol_summary.get(mode.name, {})
            status = entry.get("status", "missing")
            if status != "ok":
                print(f"  {mode.name}: {status}")
                continue
            report = entry["report"]
            test = report["test"]
            line = (
                f"  {mode.name}: test_price_mae={float(test['price_mae']):.6f} "
                f"test_pct_return_mae={float(test['pct_return_mae']):.6f} "
                f"latency_s={float(test['latency_s']):.4f}"
            )
            if entry.get("baseline"):
                print(line + " baseline")
            else:
                deltas = entry.get("deltas", {})
                print(
                    line
                    + (
                        f" Δprice={float(deltas.get('test_price_mae_delta_pct', 0.0)):+.4f}% "
                        f"Δreturn={float(deltas.get('test_pct_return_mae_delta_pct', 0.0)):+.4f}% "
                        f"speedup={float(deltas.get('test_latency_speedup', 0.0)):.2f}x "
                        f"gate={'pass' if entry.get('passes_gate') else 'FAIL'}"
                    )
                )

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
