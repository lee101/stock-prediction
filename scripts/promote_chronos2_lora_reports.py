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


@dataclass(frozen=True)
class PromotionChoice:
    report: LoRAReport
    selection_strategy: str
    selection_score: float
    family_size: int
    family_score_mean: Optional[float] = None
    family_score_std: Optional[float] = None
    family_key: Optional[str] = None


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
    trainer_cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        nested_symbol = trainer_cfg.get("symbol") if isinstance(trainer_cfg, dict) else None
        symbol = nested_symbol if isinstance(nested_symbol, str) else None
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    output_dir_raw = payload.get("output_dir")
    if not isinstance(output_dir_raw, str) or not output_dir_raw.strip():
        return None
    output_dir = Path(output_dir_raw)
    finetuned_ckpt = output_dir / "finetuned-ckpt"
    val = payload.get("val_metrics") if isinstance(payload.get("val_metrics"), dict) else None
    test = payload.get("test_metrics") if isinstance(payload.get("test_metrics"), dict) else None
    if val is None:
        val = payload.get("val") if isinstance(payload.get("val"), dict) else {}
    if test is None:
        test = payload.get("test") if isinstance(payload.get("test"), dict) else {}
    preaug_strategy = payload.get("preaug_strategy")
    if preaug_strategy is None and isinstance(trainer_cfg, dict):
        preaug_strategy = trainer_cfg.get("preaug")
    val_mae_percent = _parse_float(val.get("mae_percent")) if isinstance(val, dict) else None
    if val_mae_percent is None and isinstance(val, dict):
        val_mae_percent = _parse_float(val.get("mae_percent_mean"))
    test_mae_percent = _parse_float(test.get("mae_percent")) if isinstance(test, dict) else None
    if test_mae_percent is None and isinstance(test, dict):
        test_mae_percent = _parse_float(test.get("mae_percent_mean"))
    return LoRAReport(
        symbol=symbol.strip().upper(),
        report_path=path,
        output_dir=output_dir,
        finetuned_ckpt=finetuned_ckpt,
        val_mae=_parse_float(val.get("mae") if isinstance(val, dict) else None),
        val_mae_percent=val_mae_percent,
        val_pct_return_mae=_parse_float(val.get("pct_return_mae") if isinstance(val, dict) else None),
        test_mae=_parse_float(test.get("mae") if isinstance(test, dict) else None),
        test_mae_percent=test_mae_percent,
        test_pct_return_mae=_parse_float(test.get("pct_return_mae") if isinstance(test, dict) else None),
        trainer_config={str(k): v for k, v in trainer_cfg.items()} if isinstance(trainer_cfg, dict) else {},
        preaug_strategy=str(preaug_strategy) if preaug_strategy is not None else None,
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


def _family_parts(report: LoRAReport) -> Tuple[Tuple[str, str], ...]:
    trainer_cfg = report.trainer_config
    parts: list[tuple[str, str]] = [
        ("symbol", report.symbol),
        ("preaug", str(report.preaug_strategy or trainer_cfg.get("preaug") or "")),
    ]
    family_keys = (
        "context_length",
        "batch_size",
        "learning_rate",
        "num_steps",
        "prediction_length",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
    )
    for key in family_keys:
        value = trainer_cfg.get(key)
        if value is None:
            continue
        parts.append((key, str(value)))
    return tuple(parts)


def _format_family_key(parts: Sequence[Tuple[str, str]]) -> str:
    return "|".join(f"{key}={value}" for key, value in parts)


def _select_best_reports(
    reports: Sequence[LoRAReport],
    *,
    symbols: Optional[Iterable[str]] = None,
    run_id: Optional[str] = None,
    metric: str = "val_mae_percent",
    selection_strategy: str = "best_single",
    stability_penalty: float = 0.25,
    min_family_size: int = 2,
) -> Dict[str, PromotionChoice]:
    wanted: Optional[set[str]] = None
    if symbols is not None:
        wanted = {str(sym).strip().upper() for sym in symbols if str(sym).strip()}
        if not wanted:
            wanted = None

    candidates: Dict[str, List[Tuple[float, LoRAReport]]] = {}
    for report in reports:
        if wanted is not None and report.symbol not in wanted:
            continue
        if run_id and run_id not in report.output_dir.name and run_id not in report.report_path.name:
            continue
        score = _score_report(report, metric)
        if not _is_finite(score):
            continue
        candidates.setdefault(report.symbol, []).append((float(score), report))

    selected: Dict[str, PromotionChoice] = {}
    for symbol, entries in candidates.items():
        best_score, best_report = min(entries, key=lambda item: item[0])
        fallback = PromotionChoice(
            report=best_report,
            selection_strategy="best_single",
            selection_score=float(best_score),
            family_size=1,
        )
        if selection_strategy != "stable_family":
            selected[symbol] = fallback
            continue

        families: Dict[Tuple[Tuple[str, str], ...], List[Tuple[float, LoRAReport]]] = {}
        for score, report in entries:
            families.setdefault(_family_parts(report), []).append((score, report))

        stable_choice: PromotionChoice | None = None
        for family_parts, family_entries in families.items():
            if len(family_entries) < int(min_family_size):
                continue
            family_scores = [float(score) for score, _ in family_entries]
            mean_score = sum(family_scores) / len(family_scores)
            variance = sum((score - mean_score) ** 2 for score in family_scores) / len(family_scores)
            std_score = math.sqrt(max(variance, 0.0))
            penalized_score = mean_score + float(stability_penalty) * std_score
            best_family_score, best_family_report = min(family_entries, key=lambda item: item[0])
            candidate = PromotionChoice(
                report=best_family_report,
                selection_strategy="stable_family",
                selection_score=float(penalized_score),
                family_size=len(family_entries),
                family_score_mean=float(mean_score),
                family_score_std=float(std_score),
                family_key=_format_family_key(family_parts),
            )
            if stable_choice is None:
                stable_choice = candidate
                continue
            if candidate.selection_score < stable_choice.selection_score:
                stable_choice = candidate
                continue
            if candidate.selection_score == stable_choice.selection_score and best_family_score < _score_report(stable_choice.report, metric):
                stable_choice = candidate

        selected[symbol] = stable_choice or fallback
    return selected


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
    choice: PromotionChoice,
    output_path: Path,
    template_config: Dict[str, object],
) -> None:
    report = choice.report
    cfg = dict(template_config)
    cfg.setdefault("name", "hourly_ctx512_skip1_single")
    cfg["model_id"] = str(report.finetuned_ckpt)
    cfg["device_map"] = str(report.trainer_config.get("device_map", cfg.get("device_map", "cuda")) or "cuda")
    cfg["context_length"] = int(report.trainer_config.get("context_length", cfg.get("context_length", 512)) or 512)
    cfg["batch_size"] = int(report.trainer_config.get("batch_size", cfg.get("batch_size", 32)) or 32)
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
            "selection_strategy": choice.selection_strategy,
            "selection_score": choice.selection_score,
            "selection_family_size": choice.family_size,
            "selection_family_score_mean": choice.family_score_mean,
            "selection_family_score_std": choice.family_score_std,
            "selection_family_key": choice.family_key,
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
        "--selection-strategy",
        default="best_single",
        choices=("best_single", "stable_family"),
        help="How to choose the promoted checkpoint for each symbol.",
    )
    parser.add_argument(
        "--stability-penalty",
        type=float,
        default=0.25,
        help="Penalty multiplier applied to score stddev when using stable_family selection.",
    )
    parser.add_argument(
        "--min-family-size",
        type=int,
        default=2,
        help="Minimum reports required for a stable_family group before it is considered.",
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
        selection_strategy=str(args.selection_strategy),
        stability_penalty=float(args.stability_penalty),
        min_family_size=int(args.min_family_size),
    )
    if not selected:
        print("No promotable reports found after filtering.")
        return 3

    for symbol, choice in sorted(selected.items()):
        report = choice.report
        if not report.finetuned_ckpt.exists():
            print(f"Skipping {symbol}: missing {report.finetuned_ckpt}")
            continue
        output_path = args.output_dir / f"{symbol}.json"
        _write_hourly_config(symbol=symbol, choice=choice, output_path=output_path, template_config=template_cfg)
        print(
            f"Wrote {output_path} (model_id={report.finetuned_ckpt}, "
            f"selection_strategy={choice.selection_strategy}, selection_score={choice.selection_score:.6f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
