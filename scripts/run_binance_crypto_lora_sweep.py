#!/usr/bin/env python3
"""Run sequential Binance crypto LoRA sweeps and summarize the winners."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATA_ROOT = Path("trainingdatahourlybinance")
DEFAULT_OUTPUT_ROOT = Path("chronos2_finetuned")
DEFAULT_RESULTS_ROOT = Path("experiments/binance_crypto_lora_sweep")
DEFAULT_EVAL_BASELINE_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD")


def _load_train_helpers():
    from scripts.train_crypto_lora_sweep import TrainConfig, train_and_evaluate

    return TrainConfig, train_and_evaluate


def _load_eval_helpers():
    from scripts.evaluate_binance_lora_candidate import main as evaluate_candidate_main

    return evaluate_candidate_main


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


def _pnl_gate_passed(record: dict[str, Any]) -> bool:
    if not record.get("pnl_eval_present"):
        return True
    if bool(record.get("pnl_all_windows_accept")):
        return True
    accepted = int(record.get("pnl_accepted_window_count") or 0)
    min_pnl = _coerce_float(record.get("pnl_min_new_symbol_pnl"))
    min_sortino = _coerce_float(record.get("pnl_min_sortino_delta"))
    return accepted > 0 and (min_pnl is None or min_pnl > 0.0) and (min_sortino is None or min_sortino >= -0.5)


def rank_results(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def _sort_key(record: dict[str, Any]) -> tuple[int, float, float, float, float, float, str, str]:
        score = _coerce_float(record.get("val_consistency_score"))
        val_mae = _coerce_float(record.get("val_mae_percent_mean"))
        mean_sortino = _coerce_float(record.get("pnl_mean_sortino_delta"))
        mean_return = _coerce_float(record.get("pnl_mean_return_delta"))
        mean_symbol_pnl = _coerce_float(record.get("pnl_mean_new_symbol_pnl"))
        return (
            0 if _pnl_gate_passed(record) else 1,
            score if score is not None else float("inf"),
            val_mae if val_mae is not None else float("inf"),
            -(mean_sortino if mean_sortino is not None else float("-inf")),
            -(mean_return if mean_return is not None else float("-inf")),
            -(mean_symbol_pnl if mean_symbol_pnl is not None else float("-inf")),
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


def _format_metric(value: Any, *, signed: bool = False, digits: int = 4, default: str = "n/a") -> str:
    numeric = _coerce_float(value)
    if numeric is None:
        return default
    return f"{numeric:+.{digits}f}" if signed else f"{numeric:.{digits}f}"


def render_summary_md(records: Iterable[dict[str, Any]]) -> str:
    ranked = rank_results(records)
    best = best_by_symbol(ranked)
    include_pnl = any(record.get("pnl_eval_present") for record in ranked)
    lines = [
        "# Binance Crypto LoRA Sweep",
        "",
        f"- Total runs: {len(ranked)}",
        "",
        "## Best by Symbol",
        "",
    ]
    for symbol, record in sorted(best.items()):
        base = (
            f"- `{symbol}`: `{record.get('preaug')}` "
            f"(val_consistency={_format_metric(record.get('val_consistency_score'))}, "
            f"val_mae%={_format_metric(record.get('val_mae_percent_mean'))}, "
            f"test_mae%={_format_metric(record.get('test_mae_percent_mean'))})"
        )
        if include_pnl and record.get("pnl_eval_present"):
            base += (
                " "
                f"(pnl_gate={'pass' if _pnl_gate_passed(record) else 'fail'}, "
                f"accept={int(record.get('pnl_accepted_window_count') or 0)}/{int(record.get('pnl_window_count') or 0)}, "
                f"mean_sortino_delta={_format_metric(record.get('pnl_mean_sortino_delta'), signed=True)}, "
                f"mean_return_delta={_format_metric(record.get('pnl_mean_return_delta'), signed=True)}%, "
                f"mean_symbol_pnl={_format_metric(record.get('pnl_mean_new_symbol_pnl'), signed=True, digits=2)})"
            )
        lines.append(base)
    lines.extend(["", "## All Runs", ""])
    if include_pnl:
        lines.extend(
            [
                "| Rank | Symbol | Preaug | PnL Gate | PnL Accept | Mean Sortino Δ | Mean Return Δ% | Mean Symbol PnL | Val Consistency | Val MAE% | Test MAE% | Run Name |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
    else:
        lines.extend(
            [
                "| Rank | Symbol | Preaug | Val Consistency | Val MAE% | Test MAE% | Run Name |",
                "|---:|---|---|---:|---:|---:|---|",
            ]
        )
    for idx, record in enumerate(ranked, start=1):
        if include_pnl:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(idx),
                        str(record.get("symbol") or ""),
                        str(record.get("preaug") or ""),
                        "pass" if _pnl_gate_passed(record) else "fail",
                        f"{int(record.get('pnl_accepted_window_count') or 0)}/{int(record.get('pnl_window_count') or 0)}",
                        _format_metric(record.get("pnl_mean_sortino_delta"), signed=True),
                        _format_metric(record.get("pnl_mean_return_delta"), signed=True),
                        _format_metric(record.get("pnl_mean_new_symbol_pnl"), signed=True, digits=2),
                        _format_metric(record.get("val_consistency_score")),
                        _format_metric(record.get("val_mae_percent_mean")),
                        _format_metric(record.get("test_mae_percent_mean")),
                        str(record.get("run_name") or ""),
                    ]
                )
                + " |"
            )
        else:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(idx),
                        str(record.get("symbol") or ""),
                        str(record.get("preaug") or ""),
                        _format_metric(record.get("val_consistency_score")),
                        _format_metric(record.get("val_mae_percent_mean")),
                        _format_metric(record.get("test_mae_percent_mean")),
                        str(record.get("run_name") or ""),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _parse_symbol_list(raw: Optional[str], *, default: Sequence[str]) -> list[str]:
    if raw is None:
        return [str(symbol).strip().upper() for symbol in default if str(symbol).strip()]
    values = parse_symbols([raw])
    return values or [str(symbol).strip().upper() for symbol in default if str(symbol).strip()]


def _flatten_pnl_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "pnl_eval_present": True,
        "pnl_window_count": int(summary.get("window_count") or 0),
        "pnl_accepted_window_count": int(summary.get("accepted_window_count") or 0),
        "pnl_rejected_window_count": int(summary.get("rejected_window_count") or 0),
        "pnl_all_windows_accept": bool(summary.get("all_windows_accept")),
        "pnl_mean_return_delta": _coerce_float(summary.get("mean_return_delta")),
        "pnl_min_return_delta": _coerce_float(summary.get("min_return_delta")),
        "pnl_mean_sortino_delta": _coerce_float(summary.get("mean_sortino_delta")),
        "pnl_min_sortino_delta": _coerce_float(summary.get("min_sortino_delta")),
        "pnl_mean_max_dd_delta": _coerce_float(summary.get("mean_max_dd_delta")),
        "pnl_max_max_dd_delta": _coerce_float(summary.get("max_max_dd_delta")),
        "pnl_mean_new_symbol_pnl": _coerce_float(summary.get("mean_new_symbol_pnl")),
        "pnl_min_new_symbol_pnl": _coerce_float(summary.get("min_new_symbol_pnl")),
    }


def evaluate_pnl_for_result(
    *,
    result_path: Path,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    evaluate_candidate_main = _load_eval_helpers()
    local_output_root = Path(evaluation_config["local_output_root"])
    report_name = Path(result_path).stem
    argv = [
        "--report-path",
        str(result_path),
        "--remote-host",
        str(evaluation_config["remote_host"]),
        "--remote-root",
        str(evaluation_config["remote_root"]),
        "--remote-venv",
        str(evaluation_config["remote_venv"]),
        "--remote-data-root",
        str(evaluation_config["remote_data_root"]),
        "--remote-output-root",
        str(evaluation_config["remote_output_root"]),
        "--local-output-root",
        str(local_output_root),
        "--baseline-symbols",
        *[str(symbol).strip().upper() for symbol in evaluation_config["baseline_symbols"]],
        "--windows",
        str(evaluation_config["windows"]),
        "--end",
        str(evaluation_config["end"]),
        "--signal-mode",
        str(evaluation_config["signal_mode"]),
        "--data-root",
        str(evaluation_config["data_root"]),
        "--horizons",
        str(evaluation_config["horizons"]),
        "--lookback-hours",
        str(float(evaluation_config["lookback_hours"])),
        "--forecast-rule-total-cost-bps",
        str(float(evaluation_config["forecast_rule_total_cost_bps"])),
        "--forecast-rule-min-reward-risk",
        str(float(evaluation_config["forecast_rule_min_reward_risk"])),
        "--model",
        str(evaluation_config["model"]),
        "--thinking",
        str(evaluation_config["thinking"]),
        "--rate-limit",
        str(float(evaluation_config["rate_limit"])),
    ]
    if evaluation_config.get("add_symbol_forecast_rule_total_cost_bps") is not None:
        argv.extend(
            [
                "--add-symbol-forecast-rule-total-cost-bps",
                str(float(evaluation_config["add_symbol_forecast_rule_total_cost_bps"])),
            ]
        )
    if evaluation_config.get("add_symbol_forecast_rule_min_reward_risk") is not None:
        argv.extend(
            [
                "--add-symbol-forecast-rule-min-reward-risk",
                str(float(evaluation_config["add_symbol_forecast_rule_min_reward_risk"])),
            ]
        )
    if evaluation_config.get("add_symbol_max_pos") is not None:
        argv.extend(["--add-symbol-max-pos", str(float(evaluation_config["add_symbol_max_pos"]))])

    exit_code = evaluate_candidate_main(argv)
    if exit_code not in (0, None):
        raise RuntimeError(f"Candidate PnL evaluation failed for {result_path} with exit code {exit_code}")

    summary_path = local_output_root / report_name / "candidate_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected candidate summary at {summary_path}")
    payload = json.loads(summary_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {summary_path}")
    return payload


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
    evaluation_config: Optional[dict[str, Any]] = None,
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
            summary_record = summarize_result(result, result_path=result_path)
            if evaluation_config is not None:
                pnl_payload = evaluate_pnl_for_result(result_path=result_path, evaluation_config=evaluation_config)
                pnl_summary = pnl_payload.get("eval_summary") if isinstance(pnl_payload.get("eval_summary"), dict) else {}
                summary_record["pnl_eval_path"] = str(pnl_payload.get("eval_result_path") or "")
                summary_record["pnl_eval_summary_path"] = str(
                    Path(evaluation_config["local_output_root"]) / result_path.stem / "candidate_summary.json"
                )
                summary_record.update(_flatten_pnl_summary(pnl_summary))
            summarized_runs.append(summary_record)

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
            "evaluation_config": evaluation_config,
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
    parser.add_argument("--evaluate-pnl", action="store_true", help="Run post-train candidate evaluation and include PnL gates in ranking.")
    parser.add_argument("--eval-baseline-symbols", default="BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument("--eval-windows", default="120,60,30,7,1")
    parser.add_argument("--eval-end", default="2026-03-18")
    parser.add_argument("--eval-signal-mode", choices=("forecast_rule", "gemini"), default="forecast_rule")
    parser.add_argument("--eval-remote-host", default="administrator@93.127.141.100")
    parser.add_argument("--eval-remote-root", type=Path, default=Path("/nvme0n1-disk/code/stock-prediction"))
    parser.add_argument("--eval-remote-venv", default=".venv313")
    parser.add_argument("--eval-remote-data-root", default="trainingdatahourlybinance")
    parser.add_argument("--eval-remote-output-root", default="analysis/lora_candidate_eval")
    parser.add_argument("--eval-local-output-root", type=Path, default=Path("analysis/lora_candidate_eval"))
    parser.add_argument("--eval-data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--eval-horizons", default="1,24")
    parser.add_argument("--eval-lookback-hours", type=float, default=5000.0)
    parser.add_argument("--eval-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--eval-thinking", default="HIGH")
    parser.add_argument("--eval-rate-limit", type=float, default=0.2)
    parser.add_argument("--eval-forecast-rule-total-cost-bps", type=float, default=20.0)
    parser.add_argument("--eval-forecast-rule-min-reward-risk", type=float, default=1.10)
    parser.add_argument("--eval-add-symbol-forecast-rule-total-cost-bps", type=float, default=None)
    parser.add_argument("--eval-add-symbol-forecast-rule-min-reward-risk", type=float, default=None)
    parser.add_argument("--eval-add-symbol-max-pos", type=float, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = parse_symbols(args.symbols)
    preaugs = parse_preaugs(args.preaugs)
    if not symbols:
        raise SystemExit("At least one symbol is required.")
    if not preaugs:
        raise SystemExit("At least one preaug is required.")

    evaluation_config: Optional[dict[str, Any]] = None
    if args.evaluate_pnl:
        evaluation_config = {
            "baseline_symbols": _parse_symbol_list(args.eval_baseline_symbols, default=DEFAULT_EVAL_BASELINE_SYMBOLS),
            "windows": str(args.eval_windows),
            "end": str(args.eval_end),
            "signal_mode": str(args.eval_signal_mode),
            "remote_host": str(args.eval_remote_host),
            "remote_root": Path(args.eval_remote_root),
            "remote_venv": str(args.eval_remote_venv),
            "remote_data_root": str(args.eval_remote_data_root),
            "remote_output_root": str(args.eval_remote_output_root),
            "local_output_root": Path(args.eval_local_output_root),
            "data_root": str(args.eval_data_root),
            "horizons": str(args.eval_horizons),
            "lookback_hours": float(args.eval_lookback_hours),
            "model": str(args.eval_model),
            "thinking": str(args.eval_thinking),
            "rate_limit": float(args.eval_rate_limit),
            "forecast_rule_total_cost_bps": float(args.eval_forecast_rule_total_cost_bps),
            "forecast_rule_min_reward_risk": float(args.eval_forecast_rule_min_reward_risk),
            "add_symbol_forecast_rule_total_cost_bps": args.eval_add_symbol_forecast_rule_total_cost_bps,
            "add_symbol_forecast_rule_min_reward_risk": args.eval_add_symbol_forecast_rule_min_reward_risk,
            "add_symbol_max_pos": args.eval_add_symbol_max_pos,
        }

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
        evaluation_config=evaluation_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
