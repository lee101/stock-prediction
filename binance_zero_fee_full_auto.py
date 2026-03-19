#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from loguru import logger

# Allow running as `python scripts/...` without needing PYTHONPATH tweaks.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import binance_data_wrapper
from src.binan import binance_wrapper
from src.binance_symbol_utils import split_stable_quote_symbol


@dataclass(frozen=True)
class SweepBest:
    total_return: float
    sortino: float
    intensity: float
    offset: float
    min_edge: float
    risk_weight: float
    max_volume_fraction: Optional[float]
    final_cash: float
    open_symbol: Optional[str]


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


def _append_progress(progress_path: Path, snippet: str) -> None:
    path = Path(progress_path)
    existing = path.read_text() if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + snippet + ("\n" if not snippet.endswith("\n") else ""))


def _load_sweep_best(output_dir: Path) -> Optional[SweepBest]:
    best_path = Path(output_dir) / "sweep_best.json"
    if not best_path.exists():
        return None
    payload = json.loads(best_path.read_text())
    if not isinstance(payload, dict):
        return None
    return SweepBest(
        total_return=float(payload.get("total_return", 0.0) or 0.0),
        sortino=float(payload.get("sortino", 0.0) or 0.0),
        intensity=float(payload.get("intensity_scale", payload.get("intensity", 1.0)) or 1.0),
        offset=float(payload.get("price_offset_pct", payload.get("offset", 0.0)) or 0.0),
        min_edge=float(payload.get("min_edge", 0.0) or 0.0),
        risk_weight=float(payload.get("risk_weight", 0.5) or 0.5),
        max_volume_fraction=payload.get("max_volume_fraction", None),
        final_cash=float(payload.get("final_cash", 0.0) or 0.0),
        open_symbol=str(payload.get("open_symbol") or "") or None,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Full auto loop for Binance zero-fee stable-quote pairs: refresh data -> LoRA -> cache -> policy -> selector sweep.",
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (e.g. BTCU,ETHU,...)")
    parser.add_argument(
        "--pair-list",
        choices=("auto", "fdusd", "u", "both"),
        default="auto",
        help=(
            "Default Binance symbol list when --symbols is omitted. "
            "`auto` respects BINANCE_DEFAULT_QUOTE (U -> `u`, otherwise `fdusd`)."
        ),
    )
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--update-data-years", type=int, default=10)

    parser.add_argument("--run-id", default=None, help="Run identifier (default: UTC timestamp).")

    # Steps: default to all if none are explicitly selected.
    parser.add_argument("--update-data", action="store_true")
    parser.add_argument("--retrain-lora", action="store_true")
    parser.add_argument("--promote-lora", action="store_true")
    parser.add_argument("--build-forecast-cache", action="store_true")
    parser.add_argument("--train-policy", action="store_true")
    parser.add_argument("--sweep-selector", action="store_true")

    # LoRA params
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-num-steps", type=int, default=600)
    parser.add_argument("--lora-context-length", type=int, default=None)
    parser.add_argument("--lora-batch-size", type=int, default=64)
    parser.add_argument("--lora-val-hours", type=int, default=None)
    parser.add_argument("--lora-test-hours", type=int, default=None)

    # Forecast cache params
    parser.add_argument("--forecast-cache-root", default=None)
    parser.add_argument("--forecast-horizons", default=None)
    parser.add_argument("--forecast-lookback-hours", type=float, default=None)
    parser.add_argument("--forecast-context-hours", type=int, default=None)
    parser.add_argument("--forecast-batch-size", type=int, default=64)
    parser.add_argument("--force-rebuild-forecasts", action="store_true")

    # Policy params
    parser.add_argument("--policy-run-name", default=None)
    parser.add_argument("--policy-epochs", type=int, default=None)
    parser.add_argument("--policy-sequence-length", type=int, default=None)
    parser.add_argument("--policy-batch-size", type=int, default=64)
    parser.add_argument("--policy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-optimizer", default="muon_mix")
    parser.add_argument("--policy-model-arch", default="nano")
    parser.add_argument("--policy-no-compile", action="store_true")
    parser.add_argument("--policy-horizon", type=int, default=1)
    parser.add_argument("--policy-maker-fee", type=float, default=0.0)
    parser.add_argument("--policy-moving-average-windows", default=None)
    parser.add_argument("--policy-min-history-hours", type=int, default=None)
    parser.add_argument("--policy-val-fraction", type=float, default=None)
    parser.add_argument("--policy-validation-days", type=int, default=None)
    parser.add_argument("--policy-feature-max-window-hours", type=int, default=None)

    # Selector sweep params (runs the fast sweep helper).
    parser.add_argument("--selector-edge-mode", default="close", choices=("high_low", "high", "close"))
    parser.add_argument("--selector-eval-days", type=float, default=7.0)
    parser.add_argument("--selector-frame-split", default="full", choices=("val", "full"))
    parser.add_argument(
        "--selector-initial-cash",
        type=float,
        default=10_000.0,
        help="Initial cash used for selector simulations (affects volume-cap binding/capacity).",
    )
    parser.add_argument(
        "--selector-max-hold-hours",
        type=int,
        default=None,
        help="Max hold in hours for selector simulations (None disables the max-hold forced close).",
    )
    parser.add_argument("--selector-intensity-scales", default="1,2,5,10,20")
    parser.add_argument("--selector-price-offset-pcts", default="0,0.0001,0.00025,0.0005")
    parser.add_argument("--selector-min-edges", default="0,0.001,0.002,0.004,0.006")
    parser.add_argument("--selector-risk-weights", default="0.5")
    parser.add_argument("--selector-max-volume-fractions", default="none,0.1")

    parser.add_argument("--checkpoint", default=None, help="Use an existing policy checkpoint (skip --train-policy).")
    parser.add_argument("--progress-file", default=None, help="Append a markdown snippet to this file.")
    parser.add_argument("--output-dir", default=None, help="Write run artifacts here (JSON/MD).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    step_flags = [
        args.update_data,
        args.retrain_lora,
        args.promote_lora,
        args.build_forecast_cache,
        args.train_policy,
        args.sweep_selector,
    ]
    if not any(step_flags):
        args.update_data = True
        args.retrain_lora = True
        args.promote_lora = True
        args.build_forecast_cache = True
        args.train_policy = True
        args.sweep_selector = True

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = [tok.strip().upper() for tok in str(args.symbols).split(",") if tok.strip()]
    else:
        pair_list = str(args.pair_list or "auto").strip().lower()
        if pair_list == "auto":
            preferred_quote = os.getenv("BINANCE_DEFAULT_QUOTE", "FDUSD").strip().upper() or "FDUSD"
            pair_list = "u" if preferred_quote == "U" else "fdusd"

        if pair_list == "both":
            pairs = list(binance_data_wrapper.DEFAULT_BINANCE_FDUSD_PAIRS) + list(binance_data_wrapper.DEFAULT_BINANCE_U_PAIRS)
        elif pair_list == "u":
            pairs = list(binance_data_wrapper.DEFAULT_BINANCE_U_PAIRS)
        else:
            pairs = list(binance_data_wrapper.DEFAULT_BINANCE_FDUSD_PAIRS)
        symbols = _pairs_to_symbols(pairs)

    symbols = _filter_trading_symbols(symbols)
    if not symbols:
        raise RuntimeError("No TRADING symbols selected.")

    # Heuristic defaults based on whether we're primarily U-quoted.
    quotes = {split_stable_quote_symbol(sym)[1] for sym in symbols}
    is_u_run = (quotes == {"U"}) or ("U" in quotes and "FDUSD" not in quotes)

    default_horizons = "1,4" if is_u_run else "1,4,24"
    horizons = str(args.forecast_horizons or default_horizons)

    lora_ctx = int(args.lora_context_length) if args.lora_context_length is not None else (192 if is_u_run else 1024)
    lora_val_hours = int(args.lora_val_hours) if args.lora_val_hours is not None else (48 if is_u_run else 168)
    lora_test_hours = int(args.lora_test_hours) if args.lora_test_hours is not None else (48 if is_u_run else 168)

    forecast_lookback = float(args.forecast_lookback_hours) if args.forecast_lookback_hours is not None else (800.0 if is_u_run else 5000.0)
    forecast_ctx = int(args.forecast_context_hours) if args.forecast_context_hours is not None else lora_ctx

    forecast_cache_root = Path(args.forecast_cache_root) if args.forecast_cache_root else None
    if forecast_cache_root is None:
        tag = "u" if is_u_run else "fdusd"
        htag = "h" + horizons.replace(",", "")
        forecast_cache_root = Path("binancecrosslearning") / f"forecast_cache_{tag}_lora_{run_id}_{htag}"

    checkpoint_path = args.checkpoint
    policy_run_name = args.policy_run_name
    if policy_run_name is None:
        policy_run_name = f"binance_cross_global_{'u' if is_u_run else 'fdusd'}_{run_id}"

    policy_seq_len = int(args.policy_sequence_length) if args.policy_sequence_length is not None else (48 if is_u_run else 96)
    policy_epochs = int(args.policy_epochs) if args.policy_epochs is not None else (80 if is_u_run else 8)
    policy_min_history_hours = (
        int(args.policy_min_history_hours)
        if args.policy_min_history_hours is not None
        else (48 if is_u_run else 24 * 30)
    )
    policy_validation_days = (
        int(args.policy_validation_days)
        if args.policy_validation_days is not None
        else (3 if is_u_run else None)
    )
    policy_feature_max_window = (
        int(args.policy_feature_max_window_hours)
        if args.policy_feature_max_window_hours is not None
        else (72 if is_u_run else None)
    )
    policy_ma_windows = args.policy_moving_average_windows or ("24,72" if is_u_run else None)
    selector_max_hold_hours = (
        int(args.selector_max_hold_hours)
        if args.selector_max_hold_hours is not None
        else (6 if is_u_run else None)
    )
    selector_initial_cash = float(args.selector_initial_cash)

    if args.update_data:
        pairs: List[str] = []
        for sym in symbols:
            base, quote = split_stable_quote_symbol(sym)
            if not quote:
                continue
            pairs.append(f"{base}/{quote}")
        if pairs:
            logger.info("Updating Binance hourly data for {} pair(s).", len(pairs))
            binance_data_wrapper.download_all_pairs(
                pairs=pairs,
                output_dir=Path(args.data_root),
                history_years=int(args.update_data_years),
                skip_if_exists=True,
                fallback_quotes=[],
            )
        else:
            logger.warning("No stable-quote symbols selected; skipping Binance data refresh.")

    if args.retrain_lora:
        cmd = [
            sys.executable,
            "scripts/retrain_chronos2_lora_binance_pairs.py",
            "--data-root",
            str(args.data_root),
            "--symbols",
            *symbols,
            "--run-id",
            run_id,
            "--learning-rate",
            str(float(args.lora_learning_rate)),
            "--num-steps",
            str(int(args.lora_num_steps)),
            "--context-length",
            str(int(lora_ctx)),
            "--batch-size",
            str(int(args.lora_batch_size)),
            "--val-hours",
            str(int(lora_val_hours)),
            "--test-hours",
            str(int(lora_test_hours)),
            "--torch-dtype",
            "bfloat16",
            "--preaug-eval",
        ]
        _run_cmd(cmd, cwd=_REPO_ROOT)

    if args.promote_lora:
        cmd = [
            sys.executable,
            "scripts/promote_chronos2_lora_reports.py",
            "--run-id",
            run_id,
            "--symbols",
            *symbols,
        ]
        _run_cmd(cmd, cwd=_REPO_ROOT)

    mae_summary_path = None
    if args.build_forecast_cache:
        mae_summary_path = Path(output_dir) / "mae_summary.json" if output_dir is not None else None
        cmd = [
            sys.executable,
            "scripts/build_hourly_forecast_caches.py",
            "--symbols",
            ",".join(symbols),
            "--data-root",
            str(args.data_root),
            "--forecast-cache-root",
            str(forecast_cache_root),
            "--horizons",
            horizons,
            "--lookback-hours",
            str(float(forecast_lookback)),
            "--context-hours",
            str(int(forecast_ctx)),
            "--batch-size",
            str(int(args.forecast_batch_size)),
        ]
        if args.force_rebuild_forecasts:
            cmd.append("--force-rebuild")
        if mae_summary_path is not None:
            cmd += ["--output-json", str(mae_summary_path)]
        _run_cmd(cmd, cwd=_REPO_ROOT)

    policy_metrics: Optional[dict] = None
    if args.train_policy:
        cmd = [
            sys.executable,
            "-m",
            "binancecrosslearning.train_global_policy",
            "--symbols",
            ",".join(symbols),
            "--target-symbol",
            symbols[0],
            "--epochs",
            str(int(policy_epochs)),
            "--batch-size",
            str(int(args.policy_batch_size)),
            "--sequence-length",
            str(int(policy_seq_len)),
            "--learning-rate",
            str(float(args.policy_learning_rate)),
            "--weight-decay",
            str(float(args.policy_weight_decay)),
            "--optimizer",
            str(args.policy_optimizer),
            "--model-arch",
            str(args.policy_model_arch),
            "--forecast-horizons",
            horizons,
            "--forecast-cache-root",
            str(forecast_cache_root),
            "--data-root",
            str(args.data_root),
            "--cache-only",
            "--run-name",
            str(policy_run_name),
            "--horizon",
            str(int(args.policy_horizon)),
            "--maker-fee",
            str(float(args.policy_maker_fee)),
            "--min-history-hours",
            str(int(policy_min_history_hours)),
        ]
        if policy_ma_windows:
            cmd += ["--moving-average-windows", str(policy_ma_windows)]
        if args.policy_val_fraction is not None:
            cmd += ["--val-fraction", str(float(args.policy_val_fraction))]
        if policy_validation_days is not None:
            cmd += ["--validation-days", str(int(policy_validation_days))]
        if policy_feature_max_window is not None:
            cmd += ["--feature-max-window-hours", str(int(policy_feature_max_window))]
        if args.policy_no_compile:
            cmd.append("--no-compile")
        train_out = _run_cmd(cmd, cwd=_REPO_ROOT)
        checkpoint_path = _parse_key_value(train_out, "Checkpoint")
        policy_metrics = {
            "checkpoint": checkpoint_path,
            "total_return": _parse_float(train_out, "total_return"),
            "sortino": _parse_float(train_out, "sortino"),
        }
        if output_dir is not None:
            (output_dir / "policy_metrics.json").write_text(json.dumps(policy_metrics, indent=2) + "\n")

    sweep_best: Optional[SweepBest] = None
    sweep_output_dir = None
    if args.sweep_selector:
        if not checkpoint_path:
            raise RuntimeError("--sweep-selector requires --checkpoint or --train-policy.")
        sweep_output_dir = Path(output_dir) / "selector_sweep" if output_dir is not None else None
        cmd = [
            sys.executable,
            "-m",
            "binancecrosslearning.sweep_global_selector",
            "--symbols",
            ",".join(symbols),
            "--checkpoint",
            str(checkpoint_path),
            "--sequence-length",
            str(int(policy_seq_len)),
            "--horizon",
            str(int(args.policy_horizon)),
            "--frame-split",
            str(args.selector_frame_split),
            "--forecast-horizons",
            horizons,
            "--forecast-cache-root",
            str(forecast_cache_root),
            "--data-root",
            str(args.data_root),
            "--cache-only",
            "--min-history-hours",
            str(int(policy_min_history_hours)),
            "--validation-days",
            str(int(policy_validation_days or 0)),
            "--edge-mode",
            str(args.selector_edge_mode),
            "--initial-cash",
            str(selector_initial_cash),
            "--maker-fee",
            str(float(args.policy_maker_fee)),
            "--eval-days",
            str(float(args.selector_eval_days)),
            "--intensity-scales",
            str(args.selector_intensity_scales),
            "--price-offset-pcts",
            str(args.selector_price_offset_pcts),
            "--min-edges",
            str(args.selector_min_edges),
            "--risk-weights",
            str(args.selector_risk_weights),
            "--max-volume-fractions",
            str(args.selector_max_volume_fractions),
        ]
        if selector_max_hold_hours is not None:
            cmd += ["--max-hold-hours", str(int(selector_max_hold_hours))]
        if policy_ma_windows:
            cmd += ["--moving-average-windows", str(policy_ma_windows)]
        if policy_feature_max_window is not None:
            cmd += ["--feature-max-window-hours", str(int(policy_feature_max_window))]
        if sweep_output_dir is not None:
            cmd += ["--output-dir", str(sweep_output_dir)]
        _run_cmd(cmd, cwd=_REPO_ROOT)
        if sweep_output_dir is not None:
            sweep_best = _load_sweep_best(sweep_output_dir)

    if args.progress_file:
        snippet_lines = [
            "",
            f"### Binance zero-fee auto run: {run_id}",
            f"- Symbols: `{','.join(symbols)}`",
            f"- Data root: `{args.data_root}`",
            f"- Forecast cache root: `{forecast_cache_root}`",
        ]
        if mae_summary_path and Path(mae_summary_path).exists():
            snippet_lines.append(f"- Forecast MAE summary: `{mae_summary_path}`")
        if policy_metrics and policy_metrics.get("checkpoint"):
            snippet_lines.append(f"- Policy checkpoint: `{policy_metrics.get('checkpoint')}`")
            snippet_lines.append(
                f"- Policy eval: total_return={policy_metrics.get('total_return')}, sortino={policy_metrics.get('sortino')}"
            )
        if sweep_best is not None:
            snippet_lines.append(
                f"- Best selector: total_return={sweep_best.total_return:.4f}, sortino={sweep_best.sortino:.4f}, "
                f"intensity={sweep_best.intensity}, offset={sweep_best.offset}, min_edge={sweep_best.min_edge}, "
                f"risk_weight={sweep_best.risk_weight}, max_volume_fraction={sweep_best.max_volume_fraction}"
            )
            if sweep_output_dir is not None:
                snippet_lines.append(f"- Sweep artifacts: `{sweep_output_dir}`")
        snippet_lines.append("")
        _append_progress(Path(args.progress_file), "\n".join(snippet_lines))

    if output_dir is not None:
        bundle = {
            "run_id": run_id,
            "symbols": symbols,
            "data_root": str(args.data_root),
            "forecast_cache_root": str(forecast_cache_root),
            "policy": policy_metrics,
            "sweep_best": asdict(sweep_best) if sweep_best is not None else None,
        }
        (output_dir / "run_bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")


if __name__ == "__main__":
    main()
