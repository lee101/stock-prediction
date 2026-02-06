#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class RunResult:
    symbol: str
    status: str  # ok|error|missing_report
    report_path: Optional[str] = None
    output_dir: Optional[str] = None
    preaug_strategy: Optional[str] = None
    val_mae_percent: Optional[float] = None
    test_mae_percent: Optional[float] = None
    val_pct_return_mae: Optional[float] = None
    test_pct_return_mae: Optional[float] = None
    error: Optional[str] = None


def _discover_symbols(data_root: Path) -> List[str]:
    symbols: List[str] = []
    for path in sorted(Path(data_root).glob("*.csv")):
        if path.name == "download_summary.csv":
            continue
        symbols.append(path.stem.upper())
    return symbols


def _load_report(path: Path) -> Dict[str, object]:
    with path.open() as fp:
        return json.load(fp)


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if numeric != numeric:  # NaN
        return "nan"
    return f"{numeric:.4f}"


def _write_summary_md(path: Path, *, run_id: str, results: Sequence[RunResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    err = [r for r in results if r.status != "ok"]
    lines: List[str] = []
    lines.append(f"# Chronos2 LoRA Binance Batch ({run_id})")
    lines.append("")
    lines.append(f"- Total symbols: {len(results)}")
    lines.append(f"- OK: {len(ok)}")
    lines.append(f"- Errors: {len(err)}")
    lines.append("")
    lines.append("| Symbol | Preaug | Val MAE% | Test MAE% | Val ret MAE | Test ret MAE | Output Dir |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for r in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.symbol,
                    r.preaug_strategy or "baseline",
                    _format_float(r.val_mae_percent),
                    _format_float(r.test_mae_percent),
                    _format_float(r.val_pct_return_mae),
                    _format_float(r.test_pct_return_mae),
                    r.output_dir or "",
                ]
            )
            + " |"
        )
    if err:
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        for r in err:
            lines.append(f"- {r.symbol}: {r.status} {r.error or ''}".rstrip())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain Chronos2 LoRA models for all Binance hourly pairs.")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourlybinance"))
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to retrain (default: discover from CSVs).")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--run-id", default=None, help="Run identifier (default: timestamp).")
    parser.add_argument("--output-root", type=Path, default=Path("chronos2_finetuned"))
    parser.add_argument("--report-dir", type=Path, default=Path("hyperparams/chronos2/hourly_lora"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/chronos2_lora_binance"))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--preaug-eval", action="store_true")
    parser.add_argument("--preaug-strategy", default=None)
    args = parser.parse_args(argv)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    symbols = [s.upper() for s in (args.symbols or _discover_symbols(args.data_root))]
    if args.max_symbols is not None:
        symbols = symbols[: max(0, int(args.max_symbols))]
    if not symbols:
        print("No symbols to retrain.")
        return 2

    args.logs_dir.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    for idx, symbol in enumerate(symbols, 1):
        save_name = f"binance_lora_{run_id}_{symbol}"
        log_path = args.logs_dir / f"{save_name}.log"
        report_path = args.report_dir / f"{symbol}_lora_{save_name}.json"

        cmd = [
            sys.executable,
            "chronos2_trainer.py",
            "--symbol",
            symbol,
            "--data-root",
            str(args.data_root),
            "--output-root",
            str(args.output_root),
            "--finetune-mode",
            "lora",
            "--learning-rate",
            str(args.learning_rate),
            "--num-steps",
            str(args.num_steps),
            "--context-length",
            str(args.context_length),
            "--batch-size",
            str(args.batch_size),
            "--val-hours",
            str(args.val_hours),
            "--test-hours",
            str(args.test_hours),
            "--torch-dtype",
            str(args.torch_dtype),
            "--save-name",
            save_name,
        ]
        if args.preaug_eval:
            cmd.append("--preaug-eval")
        if args.preaug_strategy:
            cmd.extend(["--preaug-strategy", str(args.preaug_strategy)])

        print(f"[{idx}/{len(symbols)}] {symbol} -> {save_name}")
        with log_path.open("w") as fp:
            proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, text=True)

        if proc.returncode != 0:
            results.append(
                RunResult(
                    symbol=symbol,
                    status="error",
                    report_path=str(report_path) if report_path.exists() else None,
                    error=f"chronos2_trainer exited {proc.returncode} (log: {log_path})",
                )
            )
            continue

        if not report_path.exists():
            results.append(
                RunResult(
                    symbol=symbol,
                    status="missing_report",
                    report_path=str(report_path),
                    error=f"Missing report JSON after successful run (log: {log_path})",
                )
            )
            continue

        payload = _load_report(report_path)
        val = payload.get("val_metrics", {}) if isinstance(payload, dict) else {}
        test = payload.get("test_metrics", {}) if isinstance(payload, dict) else {}
        results.append(
            RunResult(
                symbol=symbol,
                status="ok",
                report_path=str(report_path),
                output_dir=str(payload.get("output_dir") or ""),
                preaug_strategy=str(payload.get("preaug_strategy") or "") or None,
                val_mae_percent=float(val.get("mae_percent")) if isinstance(val, dict) and val.get("mae_percent") is not None else None,
                test_mae_percent=float(test.get("mae_percent")) if isinstance(test, dict) and test.get("mae_percent") is not None else None,
                val_pct_return_mae=float(val.get("pct_return_mae")) if isinstance(val, dict) and val.get("pct_return_mae") is not None else None,
                test_pct_return_mae=float(test.get("pct_return_mae")) if isinstance(test, dict) and test.get("pct_return_mae") is not None else None,
            )
        )

    summary_path = Path("reports/chronos2_lora_binance") / f"binance_lora_{run_id}_summary.md"
    _write_summary_md(summary_path, run_id=run_id, results=results)
    print(f"Summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

