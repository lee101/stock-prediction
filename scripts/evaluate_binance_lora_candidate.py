#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import json
import math
import shlex
import subprocess
import sys
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.binance_symbol_utils import proxy_symbol_to_usd


DEFAULT_REMOTE_HOST = "administrator@93.127.141.100"
DEFAULT_REMOTE_ROOT = Path("/nvme0n1-disk/code/stock-prediction")
SSH_OPTIONS = ("-o", "StrictHostKeyChecking=no", "-o", "ControlMaster=no", "-o", "ControlPath=none")


@dataclass(frozen=True)
class CandidateConfig:
    symbol: str
    model_id: str
    context_length: int
    batch_size: int
    report_path: Path
    report_name: str


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _parse_window_days(window: dict[str, Any]) -> float | None:
    label = str(window.get("window") or "").strip().lower()
    if label.endswith("d"):
        try:
            days = float(label[:-1])
        except ValueError:
            days = None
        else:
            if days > 0:
                return days

    start_raw = str(window.get("start") or "").strip()
    end_raw = str(window.get("end") or "").strip()
    if start_raw and end_raw:
        try:
            start_dt = date.fromisoformat(start_raw)
            end_dt = date.fromisoformat(end_raw)
        except ValueError:
            return None
        delta_days = (end_dt - start_dt).days
        if delta_days > 0:
            return float(delta_days)
    return None


def _annualize_return_pct(return_pct: Any, *, window_days: float | None) -> float | None:
    numeric = _coerce_float(return_pct)
    if numeric is None or window_days is None or window_days <= 0:
        return None
    gross_return = 1.0 + numeric / 100.0
    if gross_return <= 0.0:
        return -100.0
    annualized = math.pow(gross_return, 365.0 / window_days) - 1.0
    if not math.isfinite(annualized):
        return None
    return annualized * 100.0


def _annualize_linear_metric(value: Any, *, window_days: float | None) -> float | None:
    numeric = _coerce_float(value)
    if numeric is None or window_days is None or window_days <= 0:
        return None
    annualized = numeric * (365.0 / window_days)
    return annualized if math.isfinite(annualized) else None


def build_ssh_command(*parts: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, *parts]


def build_rsync_remote_shell() -> str:
    return " ".join(["ssh", *SSH_OPTIONS])



def summarize_eval_windows(payload: dict[str, Any]) -> dict[str, Any]:
    windows = payload.get("windows")
    if not isinstance(windows, list):
        return {
            "window_count": 0,
            "accepted_window_count": 0,
            "rejected_window_count": 0,
            "all_windows_accept": False,
            "mean_return_delta": None,
            "min_return_delta": None,
            "mean_sortino_delta": None,
            "min_sortino_delta": None,
            "mean_max_dd_delta": None,
            "max_max_dd_delta": None,
            "mean_new_symbol_pnl": None,
            "min_new_symbol_pnl": None,
        }

    return_deltas: list[float] = []
    sortino_deltas: list[float] = []
    max_dd_deltas: list[float] = []
    new_symbol_pnls: list[float] = []
    annualized_return_deltas: list[float] = []
    annualized_new_symbol_pnls: list[float] = []
    annualized_weights: list[float] = []
    accepted = 0
    rejected = 0

    for window in windows:
        if not isinstance(window, dict):
            continue
        comparison = window.get("comparison")
        if not isinstance(comparison, dict):
            continue
        verdict = str(comparison.get("verdict") or "").upper()
        if verdict == "ACCEPT":
            accepted += 1
        elif verdict == "REJECT":
            rejected += 1
        return_delta = _coerce_float(comparison.get("return_delta"))
        sortino_delta = _coerce_float(comparison.get("sortino_delta"))
        max_dd_delta = _coerce_float(comparison.get("max_dd_delta"))
        new_symbol_pnl = _coerce_float(comparison.get("new_symbol_pnl"))
        if return_delta is not None:
            return_deltas.append(return_delta)
        if sortino_delta is not None:
            sortino_deltas.append(sortino_delta)
        if max_dd_delta is not None:
            max_dd_deltas.append(max_dd_delta)
        if new_symbol_pnl is not None:
            new_symbol_pnls.append(new_symbol_pnl)
        window_days = _parse_window_days(window)
        annualized_return_delta = _annualize_return_pct(return_delta, window_days=window_days)
        annualized_new_symbol_pnl = _annualize_linear_metric(new_symbol_pnl, window_days=window_days)
        if annualized_return_delta is not None and annualized_new_symbol_pnl is not None and window_days is not None:
            annualized_return_deltas.append(annualized_return_delta)
            annualized_new_symbol_pnls.append(annualized_new_symbol_pnl)
            annualized_weights.append(window_days)

    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
        if not values or not weights or len(values) != len(weights):
            return None
        total_weight = sum(weights)
        if total_weight <= 0:
            return None
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight

    return {
        "window_count": len(windows),
        "accepted_window_count": accepted,
        "rejected_window_count": rejected,
        "all_windows_accept": bool(windows) and accepted == len(windows),
        "mean_return_delta": _mean(return_deltas),
        "min_return_delta": min(return_deltas) if return_deltas else None,
        "mean_sortino_delta": _mean(sortino_deltas),
        "min_sortino_delta": min(sortino_deltas) if sortino_deltas else None,
        "mean_max_dd_delta": _mean(max_dd_deltas),
        "max_max_dd_delta": max(max_dd_deltas) if max_dd_deltas else None,
        "mean_new_symbol_pnl": _mean(new_symbol_pnls),
        "min_new_symbol_pnl": min(new_symbol_pnls) if new_symbol_pnls else None,
        "mean_annualized_return_delta": _mean(annualized_return_deltas),
        "weighted_annualized_return_delta": _weighted_mean(annualized_return_deltas, annualized_weights),
        "mean_annualized_new_symbol_pnl": _mean(annualized_new_symbol_pnls),
        "weighted_annualized_new_symbol_pnl": _weighted_mean(annualized_new_symbol_pnls, annualized_weights),
    }


def load_candidate_config(report_path: Path) -> CandidateConfig:
    payload = _load_json(Path(report_path))
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Missing config object in {report_path}")
    symbol = payload.get("symbol") or config.get("symbol")
    output_dir = payload.get("output_dir")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"Missing symbol in {report_path}")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(f"Missing output_dir in {report_path}")
    return CandidateConfig(
        symbol=str(symbol).strip().upper(),
        model_id=f"{output_dir.rstrip('/')}/finetuned-ckpt",
        context_length=int(config.get("context_length") or 512),
        batch_size=int(config.get("batch_size") or 32),
        report_path=Path(report_path),
        report_name=Path(report_path).stem,
    )


def build_remote_cache_command(
    *,
    remote_root: Path,
    remote_venv: str,
    candidate: CandidateConfig,
    remote_cache_root: str,
    remote_data_root: str,
    horizons: str,
    lookback_hours: float,
) -> str:
    parts = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(remote_root))}",
        f"source {shlex.quote(remote_venv)}/bin/activate",
        'export PYTHONPATH="$PWD:$PWD/PufferLib:${PYTHONPATH:-}"',
        "python -u scripts/build_hourly_forecast_caches.py "
        f"--symbols {shlex.quote(candidate.symbol)} "
        f"--data-root {shlex.quote(remote_data_root)} "
        f"--forecast-cache-root {shlex.quote(remote_cache_root + '/forecast_cache')} "
        f"--horizons {shlex.quote(horizons)} "
        f"--lookback-hours {lookback_hours} "
        f"--output-json {shlex.quote(remote_cache_root + '/forecast_mae.json')} "
        "--force-rebuild "
        f"--model-id {shlex.quote(candidate.model_id)} "
        f"--context-hours {candidate.context_length} "
        f"--batch-size {candidate.batch_size}",
    ]
    return "; ".join(parts)


def build_eval_command(
    *,
    python_executable: str,
    baseline_symbols: Sequence[str],
    add_symbol: str,
    end: str,
    windows: str,
    signal_mode: str,
    forecast_cache_root: Path,
    data_root: str,
    model: str,
    thinking: str,
    rate_limit: float,
    forecast_rule_total_cost_bps: float,
    forecast_rule_min_reward_risk: float,
    add_symbol_forecast_rule_total_cost_bps: float | None,
    add_symbol_forecast_rule_min_reward_risk: float | None,
    add_symbol_max_pos: float | None,
) -> list[str]:
    cmd = [
        python_executable,
        "rl-trading-agent-binance/eval_new_symbol.py",
        "--symbols",
        *[str(symbol).strip().upper() for symbol in baseline_symbols],
        "--add-symbol",
        str(add_symbol).strip().upper(),
        "--end",
        str(end),
        "--windows",
        str(windows),
        "--signal-mode",
        str(signal_mode),
        "--forecast-cache-root",
        str(forecast_cache_root),
        "--data-root",
        str(data_root),
        "--forecast-rule-total-cost-bps",
        str(forecast_rule_total_cost_bps),
        "--forecast-rule-min-reward-risk",
        str(forecast_rule_min_reward_risk),
    ]
    if add_symbol_forecast_rule_total_cost_bps is not None:
        cmd.extend(
            [
                "--add-symbol-forecast-rule-total-cost-bps",
                str(add_symbol_forecast_rule_total_cost_bps),
            ]
        )
    if add_symbol_forecast_rule_min_reward_risk is not None:
        cmd.extend(
            [
                "--add-symbol-forecast-rule-min-reward-risk",
                str(add_symbol_forecast_rule_min_reward_risk),
            ]
        )
    if add_symbol_max_pos is not None:
        cmd.extend(["--add-symbol-max-pos", str(add_symbol_max_pos)])
    if str(signal_mode) == "gemini":
        cmd.extend(
            [
                "--model",
                str(model),
                "--thinking",
                str(thinking),
                "--rate-limit",
                str(rate_limit),
            ]
        )
    return cmd


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(list(cmd), cwd=cwd, check=True)


def _find_latest_eval_result(*, add_symbol: str, signal_mode: str) -> Path:
    candidates = sorted(
        (REPO / "analysis").glob(f"hybrid_symbol_eval_*/eval_{add_symbol}_{signal_mode}.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No eval result found for {add_symbol} {signal_mode}")
    return candidates[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Binance Chronos2 LoRA candidate using real cache MAE and hybrid portfolio windows.")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--remote-venv", default=".venv313")
    parser.add_argument("--remote-data-root", default="trainingdatahourlybinance")
    parser.add_argument("--remote-output-root", default="analysis/lora_candidate_eval")
    parser.add_argument("--local-output-root", type=Path, default=Path("analysis/lora_candidate_eval"))
    parser.add_argument("--baseline-symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--add-symbol", default=None)
    parser.add_argument("--windows", default="120,60,30,7,1")
    parser.add_argument("--end", default="2026-03-18")
    parser.add_argument("--signal-mode", choices=("forecast_rule", "gemini"), default="forecast_rule")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--rate-limit", type=float, default=0.2)
    parser.add_argument("--thinking", default="HIGH")
    parser.add_argument("--data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--horizons", default="1,24")
    parser.add_argument("--lookback-hours", type=float, default=5000.0)
    parser.add_argument("--forecast-rule-total-cost-bps", type=float, default=20.0)
    parser.add_argument("--forecast-rule-min-reward-risk", type=float, default=1.10)
    parser.add_argument("--add-symbol-forecast-rule-total-cost-bps", type=float, default=None)
    parser.add_argument("--add-symbol-forecast-rule-min-reward-risk", type=float, default=None)
    parser.add_argument("--add-symbol-max-pos", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate = load_candidate_config(args.report_path)
    add_symbol = str(args.add_symbol or proxy_symbol_to_usd(candidate.symbol)).strip().upper()

    remote_eval_root = f"{args.remote_output_root.rstrip('/')}/{candidate.report_name}"
    local_eval_root = Path(args.local_output_root) / candidate.report_name
    local_remote_root = local_eval_root / "remote_run"
    local_remote_root.mkdir(parents=True, exist_ok=True)

    ssh_cmd = build_remote_cache_command(
        remote_root=Path(args.remote_root),
        remote_venv=str(args.remote_venv),
        candidate=candidate,
        remote_cache_root=remote_eval_root,
        remote_data_root=str(args.remote_data_root),
        horizons=str(args.horizons),
        lookback_hours=float(args.lookback_hours),
    )
    _run(
        build_ssh_command(str(args.remote_host), ssh_cmd)
    )

    _run(
        [
            "rsync",
            "-az",
            "-e",
            build_rsync_remote_shell(),
            f"{args.remote_host}:{args.remote_root}/{remote_eval_root}/",
            str(local_remote_root) + "/",
        ]
    )

    eval_cmd = build_eval_command(
        python_executable=sys.executable,
        baseline_symbols=args.baseline_symbols,
        add_symbol=add_symbol,
        end=str(args.end),
        windows=str(args.windows),
        signal_mode=str(args.signal_mode),
        forecast_cache_root=local_remote_root / "forecast_cache",
        data_root=str(args.data_root),
        model=str(args.model),
        thinking=str(args.thinking),
        rate_limit=float(args.rate_limit),
        forecast_rule_total_cost_bps=float(args.forecast_rule_total_cost_bps),
        forecast_rule_min_reward_risk=float(args.forecast_rule_min_reward_risk),
        add_symbol_forecast_rule_total_cost_bps=args.add_symbol_forecast_rule_total_cost_bps,
        add_symbol_forecast_rule_min_reward_risk=args.add_symbol_forecast_rule_min_reward_risk,
        add_symbol_max_pos=args.add_symbol_max_pos,
    )
    _run(eval_cmd, cwd=REPO)
    eval_result_path = _find_latest_eval_result(add_symbol=add_symbol, signal_mode=str(args.signal_mode))
    local_eval_result = local_eval_root / eval_result_path.name
    local_eval_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(eval_result_path, local_eval_result)
    eval_payload = _load_json(local_eval_result)
    eval_summary = summarize_eval_windows(eval_payload)

    summary = {
        "candidate_report": str(candidate.report_path),
        "symbol": candidate.symbol,
        "add_symbol": add_symbol,
        "model_id": candidate.model_id,
        "context_length": candidate.context_length,
        "batch_size": candidate.batch_size,
        "remote_eval_root": remote_eval_root,
        "local_remote_root": str(local_remote_root),
        "forecast_mae_path": str(local_remote_root / "forecast_mae.json"),
        "eval_result_path": str(local_eval_result),
        "eval_summary": eval_summary,
    }
    (local_eval_root / "candidate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
