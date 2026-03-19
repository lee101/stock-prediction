from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from loguru import logger

# Allow running as `python scripts/...` without needing PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import binance_data_wrapper
from src.binan import binance_wrapper
from src.binance_symbol_utils import split_stable_quote_symbol
from src.forecast_cache_metrics import compute_forecast_cache_mae_for_paths


@dataclass(frozen=True)
class SelectorSweepResult:
    intensity_scale: float
    price_offset_pct: float
    min_edge: float
    risk_weight: float
    total_return: float
    sortino: float
    final_cash: float


def _pairs_to_symbols(pairs: Sequence[str]) -> List[str]:
    symbols: List[str] = []
    for pair in pairs:
        raw = str(pair).strip().upper()
        if not raw:
            continue
        raw = raw.replace("-", "/").replace("_", "/")
        if "/" in raw:
            base, quote = raw.split("/", 1)
            symbols.append(f"{base}{quote}")
        else:
            symbols.append(raw)
    return symbols


def _filter_trading_symbols(symbols: Sequence[str]) -> List[str]:
    client = binance_wrapper.get_client()
    kept: List[str] = []
    dropped: List[Tuple[str, object]] = []
    for sym in symbols:
        sym = str(sym).strip().upper()
        if not sym:
            continue
        info = client.get_symbol_info(sym)
        status = info.get("status") if isinstance(info, Mapping) else None
        if status == "TRADING":
            kept.append(sym)
        else:
            dropped.append((sym, status))
    if dropped:
        summary = ", ".join(f"{sym}({status})" for sym, status in dropped)
        logger.warning("Skipping non-trading symbols: {}", summary)
    return kept


def _run_cmd(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> str:
    """Run a command, streaming stdout/stderr, and return captured text."""
    cmd_list = [str(c) for c in cmd]
    logger.info("Running: {}", " ".join(cmd_list))
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        lines.append(line)
        # Preserve real-time logs for long-running training.
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    out = "".join(lines)
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd_list)}")
    return out


def _parse_key_value(output: str, key: str) -> Optional[str]:
    needle = f"{key}:"
    for line in output.splitlines():
        if line.startswith(needle):
            return line[len(needle) :].strip()
    return None


def _parse_float(output: str, key: str) -> Optional[float]:
    value = _parse_key_value(output, key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _iter_sweep(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _compute_mae_summary(
    *,
    symbols: Sequence[str],
    cache_root: Path,
    data_root: Path,
    horizons: Sequence[int],
) -> dict:
    summary: dict = {"cache_root": str(cache_root), "symbols": list(symbols), "horizons": list(horizons), "mae": {}}
    for sym in symbols:
        sym_key = str(sym).upper()
        per_symbol: dict = {}
        for h in horizons:
            parquet_path = cache_root / f"h{int(h)}" / f"{sym_key}.parquet"
            if not parquet_path.exists():
                continue
            history_csv = data_root / f"{sym_key}.csv"
            metric = compute_forecast_cache_mae_for_paths(
                symbol=sym_key,
                horizon_hours=int(h),
                history_csv=history_csv,
                forecast_parquet=parquet_path,
            )
            per_symbol[str(int(h))] = {
                "count": metric.count,
                "mae": metric.mae,
                "mae_percent": metric.mae_percent,
                "start_timestamp": metric.start_timestamp.isoformat(),
                "end_timestamp": metric.end_timestamp.isoformat(),
            }
        summary["mae"][sym_key] = per_symbol
    return summary


def _append_progress(progress_path: Path, snippet: str) -> None:
    path = Path(progress_path)
    existing = path.read_text() if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + snippet + ("\n" if not snippet.endswith("\n") else ""))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="End-to-end Binance Chronos2->policy->selector experiment runner.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (e.g. BTCFDUSD,ETHFDUSD,...)")
    parser.add_argument(
        "--pair-list",
        choices=("auto", "fdusd", "u"),
        default="auto",
        help=(
            "Default Binance symbol list to use when --symbols is omitted. "
            "`auto` respects BINANCE_DEFAULT_QUOTE (U -> `u`, otherwise `fdusd`)."
        ),
    )
    parser.add_argument("--data-root", default="trainingdatahourlybinance")

    parser.add_argument(
        "--update-data",
        action="store_true",
        help="Refresh Binance hourly CSVs for the selected stable-quote symbols (FDUSD/U/etc).",
    )
    parser.add_argument("--update-data-years", type=int, default=10)

    parser.add_argument("--finetune", action="store_true", help="Run multi-symbol Chronos2 fine-tune.")
    parser.add_argument("--finetuned-model", default=None, help="Use an existing finetuned model (skip --finetune).")
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--finetune-batch-size", type=int, default=64)
    parser.add_argument("--finetune-learning-rate", type=float, default=5e-5)
    parser.add_argument("--finetune-steps", type=int, default=600)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--preaug-strategy", default=None)
    parser.add_argument("--preaug-params", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="binancecrosslearning/chronos_finetuned")

    parser.add_argument("--build-forecasts", action="store_true", help="Build forecast caches with finetuned Chronos2.")
    parser.add_argument("--forecast-cache-root", default=None)
    parser.add_argument("--forecast-lookback-hours", type=float, default=5000)
    parser.add_argument("--forecast-context-hours", type=int, default=1024)
    parser.add_argument("--forecast-batch-size", type=int, default=64)
    parser.add_argument("--forecast-horizons", default="1,4,24")

    parser.add_argument("--train-policy", action="store_true", help="Train global policy using the forecast cache.")
    parser.add_argument("--policy-run-name", default=None)
    parser.add_argument("--policy-epochs", type=int, default=8)
    parser.add_argument("--policy-sequence-length", type=int, default=96)
    parser.add_argument("--policy-batch-size", type=int, default=128)
    parser.add_argument("--policy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-optimizer", default="muon_mix")
    parser.add_argument("--policy-model-arch", default="nano")
    parser.add_argument("--policy-attention-window", type=int, default=64)
    parser.add_argument("--policy-skip-scale-init", type=float, default=0.1)
    parser.add_argument("--policy-no-compile", action="store_true")
    parser.add_argument("--policy-eval-days", type=float, default=30)
    parser.add_argument("--policy-horizon", type=int, default=1)

    parser.add_argument("--selector-sweep", action="store_true", help="Sweep selector overrides on a trained checkpoint.")
    parser.add_argument("--selector-intensity", default="1.0,1.2,1.5,1.8")
    parser.add_argument("--selector-offset", default="0.0,0.0001,0.0002")
    parser.add_argument("--selector-min-edge", default="0.0")
    parser.add_argument("--selector-risk-weight", default="0.5")
    parser.add_argument("--selector-eval-days", type=float, default=30)

    parser.add_argument("--progress-file", default=None, help="Append a markdown snippet to this progress file.")
    parser.add_argument("--output-dir", default=None, help="Write run artifacts (JSON/MD) here.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.symbols:
        symbols = [tok.strip().upper() for tok in str(args.symbols).split(",") if tok.strip()]
    else:
        pair_list = str(args.pair_list or "auto").strip().lower()
        if pair_list == "auto":
            preferred_quote = os.getenv("BINANCE_DEFAULT_QUOTE", "FDUSD").strip().upper() or "FDUSD"
            pair_list = "u" if preferred_quote == "U" else "fdusd"
        if pair_list == "u":
            symbols = _pairs_to_symbols(binance_data_wrapper.DEFAULT_BINANCE_U_PAIRS)
        else:
            symbols = _pairs_to_symbols(binance_data_wrapper.DEFAULT_BINANCE_FDUSD_PAIRS)

    symbols = _filter_trading_symbols(symbols)
    if not symbols:
        raise RuntimeError("No TRADING symbols selected.")

    horizons = [int(x) for x in str(args.forecast_horizons).split(",") if x.strip()]
    if not horizons:
        raise ValueError("At least one horizon is required.")

    run_name = args.run_name or time.strftime("binance_auto_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.update_data:
        pairs: List[str] = []
        for sym in symbols:
            base, quote = split_stable_quote_symbol(sym)
            if not quote:
                continue
            pairs.append(f"{base}/{quote}")
        if not pairs:
            logger.warning("No stable-quote symbols selected; skipping Binance data refresh.")
        else:
            logger.info("Updating Binance hourly data for {} pair(s).", len(pairs))
            binance_data_wrapper.download_all_pairs(
                pairs=pairs,
                output_dir=Path(args.data_root),
                history_years=int(args.update_data_years),
                skip_if_exists=True,
                fallback_quotes=[],
            )

    needs_finetuned_model = bool(args.finetune or args.build_forecasts or args.train_policy or args.selector_sweep)
    if not needs_finetuned_model:
        # Allow running this script as a data-refresh utility without requiring a model.
        return

    finetuned_model = args.finetuned_model
    if args.finetune:
        finetune_out = Path(args.output_root) / run_name
        finetune_cmd = [
            sys.executable,
            "-m",
            "binancecrosslearning.chronos_finetune_multi",
            "--symbols",
            ",".join(symbols),
            "--data-root",
            str(args.data_root),
            "--prediction-length",
            str(int(args.prediction_length)),
            "--context-length",
            str(int(args.context_length)),
            "--batch-size",
            str(int(args.finetune_batch_size)),
            "--learning-rate",
            str(float(args.finetune_learning_rate)),
            "--num-steps",
            str(int(args.finetune_steps)),
            "--val-hours",
            str(int(args.val_hours)),
            "--finetune-mode",
            "lora",
            "--torch-dtype",
            "bfloat16",
            "--run-name",
            run_name,
            "--output-root",
            str(Path(args.output_root)),
        ]
        if args.preaug_strategy:
            finetune_cmd += ["--preaug-strategy", str(args.preaug_strategy)]
        if args.preaug_params:
            finetune_cmd += ["--preaug-params", str(args.preaug_params)]
        _run_cmd(finetune_cmd, cwd=_REPO_ROOT)
        finetuned_model = str(finetune_out / "finetuned")

    if not finetuned_model:
        raise RuntimeError("No finetuned model available. Provide --finetuned-model or pass --finetune.")

    forecast_cache_root: Optional[Path] = Path(args.forecast_cache_root) if args.forecast_cache_root else None
    if forecast_cache_root is None:
        forecast_cache_root = Path("binancecrosslearning") / "forecast_cache_auto" / run_name

    mae_summary: Optional[dict] = None
    if args.build_forecasts:
        build_cmd = [
            sys.executable,
            "-m",
            "binancecrosslearning.build_forecasts",
            "--symbols",
            ",".join(symbols),
            "--finetuned-model",
            str(finetuned_model),
            "--forecast-cache-root",
            str(forecast_cache_root),
            "--data-root",
            str(args.data_root),
            "--horizons",
            ",".join(str(int(h)) for h in horizons),
            "--context-hours",
            str(int(args.forecast_context_hours)),
            "--batch-size",
            str(int(args.forecast_batch_size)),
            "--quantiles",
            "0.1,0.5,0.9",
            "--lookback-hours",
            str(float(args.forecast_lookback_hours)),
        ]
        _run_cmd(build_cmd, cwd=_REPO_ROOT)

        mae_summary = _compute_mae_summary(
            symbols=symbols,
            cache_root=forecast_cache_root,
            data_root=Path(args.data_root),
            horizons=horizons,
        )
        if output_dir is not None:
            (output_dir / "mae_summary.json").write_text(json.dumps(mae_summary, indent=2))

    checkpoint_path: Optional[str] = None
    policy_metrics: Optional[dict] = None
    if args.train_policy:
        policy_run_name = args.policy_run_name or f"{run_name}_policy"
        train_cmd = [
            sys.executable,
            "-m",
            "binancecrosslearning.train_global_policy",
            "--symbols",
            ",".join(symbols),
            "--epochs",
            str(int(args.policy_epochs)),
            "--sequence-length",
            str(int(args.policy_sequence_length)),
            "--batch-size",
            str(int(args.policy_batch_size)),
            "--learning-rate",
            str(float(args.policy_learning_rate)),
            "--weight-decay",
            str(float(args.policy_weight_decay)),
            "--optimizer",
            str(args.policy_optimizer),
            "--model-arch",
            str(args.policy_model_arch),
            "--attention-window",
            str(int(args.policy_attention_window)),
            "--skip-scale-init",
            str(float(args.policy_skip_scale_init)),
            "--forecast-horizons",
            ",".join(str(int(h)) for h in horizons),
            "--forecast-cache-root",
            str(forecast_cache_root),
            "--data-root",
            str(args.data_root),
            "--cache-only",
            "--run-name",
            str(policy_run_name),
            "--horizon",
            str(int(args.policy_horizon)),
            "--eval-days",
            str(float(args.policy_eval_days)),
        ]
        if args.policy_no_compile:
            train_cmd.append("--no-compile")
        train_out = _run_cmd(train_cmd, cwd=_REPO_ROOT)

        checkpoint_path = _parse_key_value(train_out, "Checkpoint")
        total_return = _parse_float(train_out, "total_return")
        sortino = _parse_float(train_out, "sortino")
        policy_metrics = {
            "run_name": policy_run_name,
            "checkpoint": checkpoint_path,
            "total_return": total_return,
            "sortino": sortino,
        }
        if output_dir is not None:
            (output_dir / "policy_metrics.json").write_text(json.dumps(policy_metrics, indent=2))

    sweep_results: List[SelectorSweepResult] = []
    best_sweep: Optional[SelectorSweepResult] = None
    if args.selector_sweep:
        if checkpoint_path is None:
            raise RuntimeError("--selector-sweep requires --train-policy (to produce a checkpoint).")

        intensity_scales = _iter_sweep([x for x in str(args.selector_intensity).split(",") if x.strip()])
        offsets = _iter_sweep([x for x in str(args.selector_offset).split(",") if x.strip()])
        min_edges = _iter_sweep([x for x in str(args.selector_min_edge).split(",") if x.strip()])
        risk_weights = _iter_sweep([x for x in str(args.selector_risk_weight).split(",") if x.strip()])
        if not intensity_scales:
            intensity_scales = [1.0]
        if not offsets:
            offsets = [0.0]
        if not min_edges:
            min_edges = [0.0]
        if not risk_weights:
            risk_weights = [0.5]

        for intensity in intensity_scales:
            for offset in offsets:
                for min_edge in min_edges:
                    for risk_weight in risk_weights:
                        out = _run_cmd(
                            [
                                sys.executable,
                                "-m",
                                "binancecrosslearning.run_global_selector",
                                "--symbols",
                                ",".join(symbols),
                                "--checkpoint",
                                str(checkpoint_path),
                                "--sequence-length",
                                str(int(args.policy_sequence_length)),
                                "--horizon",
                                str(int(args.policy_horizon)),
                                "--forecast-horizons",
                                ",".join(str(int(h)) for h in horizons),
                                "--forecast-cache-root",
                                str(forecast_cache_root),
                                "--data-root",
                                str(args.data_root),
                                "--cache-only",
                                "--intensity-scale",
                                str(float(intensity)),
                                "--price-offset-pct",
                                str(float(offset)),
                                "--min-edge",
                                str(float(min_edge)),
                                "--risk-weight",
                                str(float(risk_weight)),
                                "--eval-days",
                                str(float(args.selector_eval_days)),
                            ],
                            cwd=_REPO_ROOT,
                        )

                        total_return = _parse_float(out, "total_return") or 0.0
                        sortino = _parse_float(out, "sortino") or 0.0
                        final_cash = _parse_float(out, "final_cash") or 0.0
                        result = SelectorSweepResult(
                            intensity_scale=float(intensity),
                            price_offset_pct=float(offset),
                            min_edge=float(min_edge),
                            risk_weight=float(risk_weight),
                            total_return=float(total_return),
                            sortino=float(sortino),
                            final_cash=float(final_cash),
                        )
                        sweep_results.append(result)
                        if best_sweep is None or result.total_return > best_sweep.total_return:
                            best_sweep = result
                            logger.info(
                                "New best sweep: return={:.4f} sortino={:.2f} intensity={} offset={} min_edge={} risk_weight={}",
                                best_sweep.total_return,
                                best_sweep.sortino,
                                best_sweep.intensity_scale,
                                best_sweep.price_offset_pct,
                                best_sweep.min_edge,
                                best_sweep.risk_weight,
                            )

        if output_dir is not None:
            payload = [r.__dict__ for r in sweep_results]
            (output_dir / "selector_sweep.json").write_text(json.dumps(payload, indent=2))

    if args.progress_file:
        snippet_lines = [
            "",
            f"### Auto pipeline: {run_name}",
            f"- Symbols: `{','.join(symbols)}`",
            f"- Finetuned model: `{finetuned_model}`",
            f"- Forecast cache root: `{forecast_cache_root}`",
        ]
        if mae_summary:
            snippet_lines.append("- Forecast MAE%:")
            for sym in symbols:
                per = (mae_summary.get("mae") or {}).get(sym, {})
                parts = []
                for h in horizons:
                    entry = per.get(str(int(h)))
                    if not entry:
                        continue
                    parts.append(f"h{int(h)} {float(entry['mae_percent']):.4f}")
                if parts:
                    snippet_lines.append(f"  - {sym}: " + ", ".join(parts))
        if policy_metrics:
            snippet_lines.append(f"- Policy checkpoint: `{policy_metrics.get('checkpoint')}`")
            snippet_lines.append(
                f"- Policy eval ({args.policy_eval_days}d): total_return={policy_metrics.get('total_return')}, sortino={policy_metrics.get('sortino')}"
            )
        if best_sweep:
            snippet_lines.append(
                f"- Best selector ({args.selector_eval_days}d): total_return={best_sweep.total_return:.4f}, sortino={best_sweep.sortino:.4f}, "
                f"intensity={best_sweep.intensity_scale}, offset={best_sweep.price_offset_pct}, min_edge={best_sweep.min_edge}, risk_weight={best_sweep.risk_weight}"
            )
        snippet_lines.append("")
        _append_progress(Path(args.progress_file), "\n".join(snippet_lines))

    if output_dir is not None:
        bundle = {
            "run_name": run_name,
            "symbols": symbols,
            "finetuned_model": finetuned_model,
            "forecast_cache_root": str(forecast_cache_root),
            "mae_summary": mae_summary,
            "policy": policy_metrics,
            "best_selector": best_sweep.__dict__ if best_sweep else None,
        }
        (output_dir / "run_bundle.json").write_text(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    main()
