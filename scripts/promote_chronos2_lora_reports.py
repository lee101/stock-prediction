#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LoRAReport:
    symbol: str
    report_path: Path
    output_dir: Path
    finetuned_ckpt: Path
    val_mae: Optional[float]
    val_mae_percent: Optional[float]
    val_pct_return_mae: Optional[float]
    test_mae: Optional[float]
    test_mae_percent: Optional[float]
    test_pct_return_mae: Optional[float]
    trainer_config: Dict[str, object]
    preaug_strategy: Optional[str]
    preaug_source: Optional[str]


def _is_finite(value: Optional[float]) -> bool:
    if value is None:
        return False
    return math.isfinite(float(value))


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _load_report(path: Path) -> Optional[LoRAReport]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    output_dir_raw = payload.get("output_dir")
    if not isinstance(output_dir_raw, str) or not output_dir_raw.strip():
        return None
    output_dir = Path(output_dir_raw)
    finetuned_ckpt = output_dir / "finetuned-ckpt"
    val = payload.get("val_metrics") if isinstance(payload.get("val_metrics"), dict) else {}
    test = payload.get("test_metrics") if isinstance(payload.get("test_metrics"), dict) else {}
    trainer_cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    return LoRAReport(
        symbol=symbol.strip().upper(),
        report_path=path,
        output_dir=output_dir,
        finetuned_ckpt=finetuned_ckpt,
        val_mae=_parse_float(val.get("mae") if isinstance(val, dict) else None),
        val_mae_percent=_parse_float(val.get("mae_percent") if isinstance(val, dict) else None),
        val_pct_return_mae=_parse_float(val.get("pct_return_mae") if isinstance(val, dict) else None),
        test_mae=_parse_float(test.get("mae") if isinstance(test, dict) else None),
        test_mae_percent=_parse_float(test.get("mae_percent") if isinstance(test, dict) else None),
        test_pct_return_mae=_parse_float(test.get("pct_return_mae") if isinstance(test, dict) else None),
        trainer_config={str(k): v for k, v in trainer_cfg.items()} if isinstance(trainer_cfg, dict) else {},
        preaug_strategy=str(payload.get("preaug_strategy")) if payload.get("preaug_strategy") is not None else None,
        preaug_source=str(payload.get("preaug_source")) if payload.get("preaug_source") is not None else None,
    )


def _discover_reports(report_dir: Path) -> List[LoRAReport]:
    reports: List[LoRAReport] = []
    for path in sorted(Path(report_dir).glob("*_lora_*.json")):
        report = _load_report(path)
        if report is None:
            continue
        reports.append(report)
    return reports


def _score_report(report: LoRAReport, metric: str) -> Optional[float]:
    if metric == "val_mae_percent":
        return report.val_mae_percent
    if metric == "val_pct_return_mae":
        return report.val_pct_return_mae
    if metric == "val_mae":
        return report.val_mae
    raise ValueError(f"Unknown metric '{metric}'")


def _select_best_reports(
    reports: Sequence[LoRAReport],
    *,
    symbols: Optional[Iterable[str]] = None,
    run_id: Optional[str] = None,
    metric: str = "val_mae_percent",
) -> Dict[str, LoRAReport]:
    wanted: Optional[set[str]] = None
    if symbols is not None:
        wanted = {str(sym).strip().upper() for sym in symbols if str(sym).strip()}
        if not wanted:
            wanted = None

    best: Dict[str, Tuple[float, LoRAReport]] = {}
    for report in reports:
        if wanted is not None and report.symbol not in wanted:
            continue
        if run_id and run_id not in report.output_dir.name and run_id not in report.report_path.name:
            continue
        score = _score_report(report, metric)
        if not _is_finite(score):
            continue
        current = best.get(report.symbol)
        if current is None or float(score) < current[0]:
            best[report.symbol] = (float(score), report)
    return {symbol: entry[1] for symbol, entry in best.items()}


def _load_template(template_path: Optional[Path]) -> Dict[str, object]:
    if template_path is None:
        return {}
    payload = json.loads(template_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Template must be a JSON object")
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Template must contain a 'config' object")
    return {str(k): v for k, v in cfg.items()}


def _write_hourly_config(
    *,
    symbol: str,
    report: LoRAReport,
    output_path: Path,
    template_config: Dict[str, object],
) -> None:
    cfg = dict(template_config)
    cfg.setdefault("name", "hourly_ctx512_skip1_single")
    cfg["model_id"] = str(report.finetuned_ckpt)
    cfg.setdefault("device_map", report.trainer_config.get("device_map", "cuda"))
    cfg.setdefault("context_length", int(report.trainer_config.get("context_length", 512) or 512))
    cfg.setdefault("batch_size", int(report.trainer_config.get("batch_size", 32) or 32))
    cfg.setdefault("quantile_levels", [0.1, 0.5, 0.9])
    cfg.setdefault("aggregation", "median")
    cfg.setdefault("sample_count", 0)
    cfg.setdefault("scaler", "none")
    cfg.setdefault("predict_kwargs", {})
    cfg.setdefault("skip_rates", [1])
    cfg.setdefault("aggregation_method", "single")
    cfg.setdefault("use_multivariate", False)

    payload: Dict[str, object] = {
        "symbol": symbol,
        "model": "chronos2",
        "config": cfg,
        "validation": {
            "price_mae": report.val_mae,
            "pct_return_mae": report.val_pct_return_mae,
            "mae_percent": report.val_mae_percent,
        },
        "test": {
            "price_mae": report.test_mae,
            "pct_return_mae": report.test_pct_return_mae,
            "mae_percent": report.test_mae_percent,
        },
        "windows": {
            "val_window": int(report.trainer_config.get("val_hours", 168) or 168),
            "test_window": int(report.trainer_config.get("test_hours", 168) or 168),
            "forecast_horizon": int(report.trainer_config.get("prediction_length", 1) or 1),
        },
        "metadata": {
            "source": "promote_chronos2_lora_reports",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "report_path": str(report.report_path),
            "output_dir": str(report.output_dir),
            "preaug_strategy": report.preaug_strategy,
            "preaug_source": report.preaug_source,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote Chronos2 LoRA fine-tune reports into hourly hyperparam configs.",
    )
    parser.add_argument("--report-dir", type=Path, default=Path("hyperparams/chronos2/hourly_lora"))
    parser.add_argument("--output-dir", type=Path, default=Path("hyperparams/chronos2/hourly"))
    parser.add_argument("--symbols", nargs="*", default=None, help="Symbols to promote (default: promote all found).")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Only consider reports whose output_dir/report filename contains this substring (recommended).",
    )
    parser.add_argument(
        "--metric",
        default="val_mae_percent",
        choices=("val_mae_percent", "val_pct_return_mae", "val_mae"),
        help="Metric used to select the best report per symbol.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("hyperparams/chronos2/hourly/BTCUSD.json"),
        help="Template config file; its 'config' keys are used as defaults.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    reports = _discover_reports(args.report_dir)
    if not reports:
        print(f"No LoRA reports found under {args.report_dir}")
        return 2

    template_cfg: Dict[str, object] = {}
    if args.template and args.template.exists():
        template_cfg = _load_template(args.template)

    selected = _select_best_reports(
        reports,
        symbols=args.symbols,
        run_id=args.run_id,
        metric=args.metric,
    )
    if not selected:
        print("No promotable reports found after filtering.")
        return 3

    for symbol, report in sorted(selected.items()):
        if not report.finetuned_ckpt.exists():
            print(f"Skipping {symbol}: missing {report.finetuned_ckpt}")
            continue
        output_path = args.output_dir / f"{symbol}.json"
        _write_hourly_config(symbol=symbol, report=report, output_path=output_path, template_config=template_cfg)
        print(f"Wrote {output_path} (model_id={report.finetuned_ckpt})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

