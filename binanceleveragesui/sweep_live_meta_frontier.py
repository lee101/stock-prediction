#!/usr/bin/env python3
"""Sweep selector/risk configs around a fixed two-model meta pair and validate top challengers."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from binanceleveragesui import trade_margin_meta as live_meta


DEFAULT_BUNDLE = (
    REPO / "deployments/binance-meta-margin/20260308_h1h6_e3_calmar_s016_gate24_full_v2"
)
DEFAULT_MODEL_A_CHECKPOINT = DEFAULT_BUNDLE / "checkpoints/doge_epoch_003.pt"
DEFAULT_MODEL_B_CHECKPOINT = DEFAULT_BUNDLE / "checkpoints/aave_epoch_003.pt"
DEFAULT_FORECAST_CACHE = DEFAULT_BUNDLE / "forecast_cache"
DEFAULT_DATA_ROOT = REPO / "trainingdatahourlybinance"
DEFAULT_EXPERIMENT_ROOT = (
    REPO / "experiments" / f"meta_live_frontier_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
)
DEFAULT_SEEDED_INVENTORIES = {"doge": 53.835, "aave": 0.043992}

SUPPORTED_SELECTION_METRICS = (
    "return",
    "sortino",
    "sharpe",
    "calmar",
    "omega",
    "gain_pain",
    "p10",
    "median",
)
SUPPORTED_SELECTION_MODES = ("winner_cash", "winner")
SUPPORTED_RANK_OBJECTIVES = ("return", "sortino")


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _parse_str_list(raw: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for token in str(raw).split(","):
        value = token.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _build_seeded_scenarios(
    model_specs: list[dict],
    seeded_inventories: list[float],
) -> tuple[tuple[str, dict], ...]:
    scenarios: list[tuple[str, dict]] = [
        ("flat_1d", {"days": 1, "initial_model": "", "initial_inv": 0.0}),
        ("flat_7d", {"days": 7, "initial_model": "", "initial_inv": 0.0}),
    ]
    for spec, raw_qty in zip(model_specs, seeded_inventories):
        qty = abs(float(raw_qty))
        if qty <= 0.0:
            continue
        name = str(spec["name"])
        scenarios.append((f"{name}_long_7d", {"days": 7, "initial_model": name, "initial_inv": qty}))
        scenarios.append((f"{name}_short_7d", {"days": 7, "initial_model": name, "initial_inv": -qty}))
    return tuple(scenarios)


def _build_backtest_cmd(
    *,
    model_specs: list[dict],
    output_path: Path,
    data_root: Path,
    forecast_cache: Path,
    selection_mode: str,
    selection_metric: str,
    lookback: int,
    short_max_leverage: float,
    switch_margin: float,
    min_score_gap: float,
    profit_gate_mode: str,
    days: int,
    initial_model: str = "",
    initial_inv: float = 0.0,
    initial_entry_ts: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "binanceleveragesui.backtest_trade_margin_meta",
        "--days",
        str(int(days)),
    ]
    for index, spec in enumerate(model_specs):
        label = "a" if index == 0 else "b"
        cmd.extend([
            f"--model-{label}-name",
            str(spec["name"]),
            f"--model-{label}-symbol",
            str(spec["symbol"]),
            f"--model-{label}-data-symbol",
            str(spec["data_symbol"]),
            f"--model-{label}-base-asset",
            str(spec["base_asset"]),
            f"--model-{label}-maker-fee",
            f"{float(spec['maker_fee']):.6f}",
        ])
        checkpoint = spec.get("checkpoint")
        if checkpoint is not None:
            cmd.extend([f"--model-{label}-checkpoint", str(checkpoint)])
    cmd.extend([
        "--selection-mode",
        str(selection_mode),
        "--selection-metric",
        str(selection_metric),
        "--lookback",
        str(int(lookback)),
        "--allow-short",
        "--max-leverage",
        "2.3",
        "--long-max-leverage",
        "2.3",
        "--short-max-leverage",
        f"{float(short_max_leverage):.2f}",
        "--cash-threshold",
        "0.0",
        "--switch-margin",
        f"{float(switch_margin):.3f}",
        "--min-score-gap",
        f"{float(min_score_gap):.3f}",
        "--profit-gate-lookback-hours",
        "24",
        "--profit-gate-min-return",
        "0.0",
        "--profit-gate-mode",
        str(profit_gate_mode),
        "--sequence-length",
        "72",
        "--horizon",
        "1",
        "--intensity-scale",
        "5.0",
        "--max-hold-hours",
        "6",
        "--data-root",
        str(data_root),
        "--forecast-cache",
        str(forecast_cache),
        "--output-json",
        str(output_path),
    ])
    if initial_model:
        cmd.extend(["--initial-model", str(initial_model)])
    if abs(float(initial_inv)) > 0.0:
        cmd.extend(["--initial-inv", f"{float(initial_inv):.12f}"])
    if initial_entry_ts:
        cmd.extend(["--initial-entry-ts", str(initial_entry_ts)])
    return cmd


def _config_name(config: dict) -> str:
    return (
        f"{config['selection_mode']}_{config['selection_metric']}"
        f"_lb{int(config['lookback'])}"
        f"_s{int(round(float(config['short_max_leverage']) * 1000)):03d}"
        f"_sm{int(round(float(config['switch_margin']) * 1000)):03d}"
        f"_gap{int(round(float(config['min_score_gap']) * 1000)):03d}"
    )


def _metric_summary(report: dict) -> dict:
    meta = report["meta"]
    summary = {
        "return_pct": float(meta["return_pct"]),
        "max_drawdown_pct": float(meta["max_drawdown_pct"]),
        "trade_count": int(meta["trade_count"]),
        "switch_count": int(meta["switch_count"]),
        "final_equity": float(meta["final_equity"]),
    }
    for key in ("sortino_ratio", "sharpe_ratio"):
        value = meta.get(key)
        if value is not None:
            summary[key] = float(value)
    return summary


def _coerce_metric_summary(report_or_summary: dict) -> dict:
    if "meta" in report_or_summary:
        return _metric_summary(report_or_summary)
    summary = {
        "return_pct": float(report_or_summary["return_pct"]),
        "max_drawdown_pct": float(report_or_summary["max_drawdown_pct"]),
        "trade_count": int(report_or_summary["trade_count"]),
        "switch_count": int(report_or_summary["switch_count"]),
        "final_equity": float(report_or_summary["final_equity"]),
    }
    for key in ("sortino_ratio", "sharpe_ratio"):
        value = report_or_summary.get(key)
        if value is not None:
            summary[key] = float(value)
    return summary


def _window_signature(summary: dict) -> tuple:
    return (
        round(float(summary["return_pct"]), 6),
        round(float(summary["max_drawdown_pct"]), 6),
        int(summary["trade_count"]),
        int(summary["switch_count"]),
    )


def _rank_ratio(summary: dict, key: str) -> float:
    value = summary.get(key)
    if value is None:
        return float("-inf")
    return float(value)


def _candidate_rank(row: dict, objective: str = "return") -> tuple:
    windows = row["windows"]
    ret_1d = float(windows.get("1d", {}).get("return_pct", -999.0))
    ret_7d = float(windows.get("7d", {}).get("return_pct", -999.0))
    ret_30d = float(windows.get("30d", {}).get("return_pct", -999.0))
    dd_30d = float(windows.get("30d", {}).get("max_drawdown_pct", -999.0))
    sortino_1d = _rank_ratio(windows.get("1d", {}), "sortino_ratio")
    sortino_7d = _rank_ratio(windows.get("7d", {}), "sortino_ratio")
    sortino_30d = _rank_ratio(windows.get("30d", {}), "sortino_ratio")
    gate_1d = 1 if ret_1d > 0.0 else 0
    gate_7d = 1 if ret_7d > 0.0 else 0
    gate_dd = 1 if dd_30d > -20.0 else 0
    if str(objective) == "sortino":
        return (
            gate_1d,
            gate_7d,
            gate_dd,
            sortino_7d,
            ret_7d,
            sortino_30d,
            ret_30d,
            dd_30d,
            sortino_1d,
            ret_1d,
        )
    return (
        gate_1d,
        gate_7d,
        gate_dd,
        ret_7d,
        ret_1d,
        ret_30d,
        dd_30d,
        sortino_7d,
        sortino_30d,
    )


def _long_window_rank(row: dict, objective: str = "return") -> tuple:
    windows = row["windows"]
    ret_1d = float(windows.get("1d", {}).get("return_pct", -999.0))
    ret_7d = float(windows.get("7d", {}).get("return_pct", -999.0))
    ret_30d = float(windows.get("30d", {}).get("return_pct", -999.0))
    dd_7d = float(windows.get("7d", {}).get("max_drawdown_pct", 0.0))
    dd_30d = float(windows.get("30d", {}).get("max_drawdown_pct", -999.0))
    sortino_7d = _rank_ratio(windows.get("7d", {}), "sortino_ratio")
    sortino_30d = _rank_ratio(windows.get("30d", {}), "sortino_ratio")
    gate_1d = 1 if ret_1d > 0.0 else 0
    gate_7d = 1 if ret_7d > 0.0 else 0
    gate_7d_dd = 1 if dd_7d > -20.0 else 0
    gate_30d_dd = 1 if dd_30d > -20.0 else 0
    if str(objective) == "sortino":
        return (
            gate_1d,
            gate_7d,
            gate_7d_dd,
            gate_30d_dd,
            sortino_30d,
            ret_30d,
            sortino_7d,
            ret_7d,
            dd_30d,
            dd_7d,
        )
    return (
        gate_1d,
        gate_7d,
        gate_7d_dd,
        gate_30d_dd,
        ret_30d,
        ret_7d,
        ret_1d,
        dd_30d,
        dd_7d,
    )


def _seeded_abs_return(row: dict) -> float:
    scenarios = row.get("seeded", {})
    if not scenarios:
        return float("inf")
    return max(abs(float(result["return_pct"])) for result in scenarios.values())


def _validated_rank(row: dict, objective: str = "return") -> tuple:
    windows = row["windows"]
    ret_1d = float(windows.get("1d", {}).get("return_pct", -999.0))
    ret_7d = float(windows.get("7d", {}).get("return_pct", -999.0))
    ret_30d = float(windows.get("30d", {}).get("return_pct", -999.0))
    dd_7d = float(windows.get("7d", {}).get("max_drawdown_pct", 0.0))
    dd_30d = float(windows.get("30d", {}).get("max_drawdown_pct", -999.0))
    sortino_7d = _rank_ratio(windows.get("7d", {}), "sortino_ratio")
    sortino_30d = _rank_ratio(windows.get("30d", {}), "sortino_ratio")
    seeded_abs = _seeded_abs_return(row)
    if str(objective) == "sortino":
        return (
            1 if ret_1d > 0.0 else 0,
            1 if ret_7d > 0.0 else 0,
            1 if dd_7d > -20.0 else 0,
            1 if dd_30d > -20.0 else 0,
            1 if seeded_abs <= 0.05 else 0,
            sortino_7d,
            sortino_30d,
            ret_30d,
            ret_7d,
            ret_1d,
            -seeded_abs,
            dd_30d,
            dd_7d,
        )
    return (
        1 if ret_1d > 0.0 else 0,
        1 if ret_7d > 0.0 else 0,
        1 if dd_7d > -20.0 else 0,
        1 if dd_30d > -20.0 else 0,
        1 if seeded_abs <= 0.05 else 0,
        ret_30d,
        ret_7d,
        ret_1d,
        -seeded_abs,
        dd_30d,
        dd_7d,
    )


def _run_backtest(
    *,
    experiment_root: Path,
    scenario_name: str,
    config_name: str,
    model_specs: list[dict],
    forecast_cache: Path,
    data_root: Path,
    selection_mode: str,
    selection_metric: str,
    lookback: int,
    short_max_leverage: float,
    switch_margin: float,
    min_score_gap: float,
    profit_gate_mode: str,
    days: int,
    initial_model: str = "",
    initial_inv: float = 0.0,
    initial_entry_ts: str | None = None,
    force: bool = False,
) -> dict:
    output_dir = experiment_root / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config_name}.json"
    if output_path.exists() and not force:
        return json.loads(output_path.read_text())
    log_path = output_dir / f"{config_name}.log"

    cmd = _build_backtest_cmd(
        model_specs=model_specs,
        output_path=output_path,
        data_root=data_root,
        forecast_cache=forecast_cache,
        selection_mode=selection_mode,
        selection_metric=selection_metric,
        lookback=lookback,
        short_max_leverage=short_max_leverage,
        switch_margin=switch_margin,
        min_score_gap=min_score_gap,
        profit_gate_mode=profit_gate_mode,
        days=days,
        initial_model=initial_model,
        initial_inv=initial_inv,
        initial_entry_ts=initial_entry_ts,
    )

    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with log_path.open("w") as log_file:
        subprocess.run(
            cmd,
            check=True,
            cwd=REPO,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return json.loads(output_path.read_text())


def _summarize_windows(reports: dict[str, dict]) -> dict[str, dict]:
    return {name: _coerce_metric_summary(report) for name, report in reports.items()}


def _build_coarse_grid(args) -> list[dict]:
    configs = []
    for selection_mode, selection_metric, lookback in itertools.product(
        _parse_str_list(args.coarse_modes),
        _parse_str_list(args.coarse_metrics),
        _parse_int_list(args.coarse_lookbacks),
    ):
        configs.append(
            {
                "selection_mode": selection_mode,
                "selection_metric": selection_metric,
                "lookback": int(lookback),
                "short_max_leverage": float(args.base_short_max_leverage),
                "switch_margin": float(args.base_switch_margin),
                "min_score_gap": float(args.base_min_score_gap),
            }
        )
    return configs


def _build_refine_grid(seed_configs: list[dict], args) -> list[dict]:
    configs: list[dict] = []
    seen: set[tuple] = set()
    for seed in seed_configs:
        for short_cap, switch_margin, min_gap in itertools.product(
            _parse_float_list(args.refine_short_caps),
            _parse_float_list(args.refine_switch_margins),
            _parse_float_list(args.refine_min_score_gaps),
        ):
            candidate = {
                "selection_mode": str(seed["selection_mode"]),
                "selection_metric": str(seed["selection_metric"]),
                "lookback": int(seed["lookback"]),
                "short_max_leverage": float(short_cap),
                "switch_margin": float(switch_margin),
                "min_score_gap": float(min_gap),
            }
            key = (
                candidate["selection_mode"],
                candidate["selection_metric"],
                candidate["lookback"],
                round(candidate["short_max_leverage"], 6),
                round(candidate["switch_margin"], 6),
                round(candidate["min_score_gap"], 6),
            )
            if key in seen:
                continue
            seen.add(key)
            configs.append(candidate)
    return configs


def _dedupe_by_signature(rows: list[dict], window_name: str, rank_fn=_candidate_rank) -> list[dict]:
    kept: list[dict] = []
    seen: set[tuple] = set()
    for row in sorted(rows, key=rank_fn, reverse=True):
        signature = _window_signature(row["windows"][window_name])
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(row)
    return kept


def _group_by_signature(rows: list[dict], window_name: str, rank_fn=_candidate_rank) -> list[tuple[tuple, list[dict]]]:
    grouped: dict[tuple, list[dict]] = {}
    for row in sorted(rows, key=rank_fn, reverse=True):
        signature = _window_signature(row["windows"][window_name])
        grouped.setdefault(signature, []).append(row)
    ordered = sorted(grouped.items(), key=lambda item: rank_fn(item[1][0]), reverse=True)
    return ordered


def _family_key(config_or_row: dict) -> tuple[str, str, int]:
    config = config_or_row.get("config", config_or_row)
    return (
        str(config["selection_mode"]),
        str(config["selection_metric"]),
        int(config["lookback"]),
    )


def _best_by_family(rows: list[dict], rank_fn) -> list[dict]:
    kept: list[dict] = []
    seen: set[tuple[str, str, int]] = set()
    for row in sorted(rows, key=rank_fn, reverse=True):
        key = _family_key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _merge_unique_rows(*groups: list[dict], limit: int | None = None) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for row in group:
            name = str(row["name"])
            if name in seen:
                continue
            seen.add(name)
            merged.append(row)
            if limit is not None and len(merged) >= int(limit):
                return merged
    return merged


def _passes_primary_gate(row: dict) -> bool:
    windows = row["windows"]
    ret_1d = float(windows.get("1d", {}).get("return_pct", -999.0))
    ret_7d = float(windows.get("7d", {}).get("return_pct", -999.0))
    dd_30d = float(windows.get("30d", {}).get("max_drawdown_pct", -999.0))
    return ret_1d > 0.0 and ret_7d > 0.0 and dd_30d > -20.0


def _select_refine_seed_rows(
    rows: list[dict],
    top_unique: int,
    *,
    rank_fn=_candidate_rank,
    long_rank_fn=_long_window_rank,
) -> list[dict]:
    if not rows:
        return []
    pool = [row for row in rows if _passes_primary_gate(row)] or list(rows)
    top_n = max(1, int(top_unique))
    limit = min(len(pool), top_n * 2)
    return _merge_unique_rows(
        _dedupe_by_signature(_best_by_family(pool, rank_fn), "7d", rank_fn=rank_fn)[:top_n],
        _dedupe_by_signature(_best_by_family(pool, long_rank_fn), "30d", rank_fn=long_rank_fn)[:top_n],
        limit=limit,
    )


def _select_finalist_rows(
    rows: list[dict],
    top_finalists: int,
    *,
    rank_fn=_candidate_rank,
    long_rank_fn=_long_window_rank,
) -> list[dict]:
    if not rows:
        return []
    pool = [row for row in rows if _passes_primary_gate(row)] or list(rows)
    top_n = max(1, int(top_finalists))
    limit = min(len(pool), top_n * 2)
    return _merge_unique_rows(
        _dedupe_by_signature(_best_by_family(pool, rank_fn), "7d", rank_fn=rank_fn)[:top_n],
        _dedupe_by_signature(_best_by_family(pool, long_rank_fn), "30d", rank_fn=long_rank_fn)[:top_n],
        limit=limit,
    )


def _select_phase2c_rows(rows: list[dict], top_finalists: int, *, rank_fn=_candidate_rank) -> list[dict]:
    if not rows:
        return []
    pool = [
        row
        for row in rows
        if float(row["windows"].get("7d", {}).get("return_pct", -999.0)) > 0.0
    ] or list(rows)
    top_n = max(1, int(top_finalists))
    return _dedupe_by_signature(
        _best_by_family(pool, rank_fn),
        "7d",
        rank_fn=rank_fn,
    )[:top_n]


def _validate_candidates(
    *,
    experiment_root: Path,
    candidate_rows: list[dict],
    model_specs: list[dict],
    seeded_scenarios: tuple[tuple[str, dict], ...],
    forecast_cache: Path,
    data_root: Path,
    profit_gate_mode: str,
    force: bool,
    validated_rank_fn=_validated_rank,
) -> list[dict]:
    validated: list[dict] = []
    for row in candidate_rows:
        config = dict(row["config"])
        config_name = _config_name(config)
        seeded_reports = {}
        for scenario_name, scenario in seeded_scenarios:
            seeded_reports[scenario_name] = _run_backtest(
                experiment_root=experiment_root,
                scenario_name="phase3_seeded_validation",
                config_name=f"{config_name}__{scenario_name}",
                model_specs=model_specs,
                forecast_cache=forecast_cache,
                data_root=data_root,
                selection_mode=str(config["selection_mode"]),
                selection_metric=str(config["selection_metric"]),
                lookback=int(config["lookback"]),
                short_max_leverage=float(config["short_max_leverage"]),
                switch_margin=float(config["switch_margin"]),
                min_score_gap=float(config["min_score_gap"]),
                profit_gate_mode=str(profit_gate_mode),
                days=int(scenario["days"]),
                initial_model=str(scenario["initial_model"]),
                initial_inv=float(scenario["initial_inv"]),
                force=force,
            )
        validated_row = dict(row)
        validated_row["seeded"] = _summarize_windows(seeded_reports)
        validated.append(validated_row)
    return sorted(validated, key=validated_rank_fn, reverse=True)


def _run_job_batch(jobs: list[dict], max_workers: int) -> dict[str, dict]:
    if not jobs:
        return {}
    results: dict[str, dict] = {}
    worker_count = max(1, int(max_workers))
    if worker_count == 1:
        for job in jobs:
            results[str(job["job_id"])] = _run_backtest(**job["kwargs"])
        return results
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_run_backtest, **job["kwargs"]): str(job["job_id"])
            for job in jobs
        }
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep a live two-model meta frontier around a fixed checkpoint pair.")
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--model-a-name", default=live_meta.MODEL_SLOT_DEFAULTS[0][1])
    parser.add_argument("--model-a-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["symbol"])
    parser.add_argument("--model-a-data-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["data_symbol"])
    parser.add_argument("--model-a-base-asset", default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["base_asset"])
    parser.add_argument("--model-a-maker-fee", type=float, default=live_meta.MODEL_SLOT_DEFAULTS[0][2]["maker_fee"])
    parser.add_argument(
        "--model-a-checkpoint",
        "--doge-checkpoint",
        dest="model_a_checkpoint",
        default=str(DEFAULT_MODEL_A_CHECKPOINT),
    )
    parser.add_argument(
        "--model-a-seeded-inv",
        type=float,
        default=float(DEFAULT_SEEDED_INVENTORIES.get(live_meta.MODEL_SLOT_DEFAULTS[0][1], 0.0)),
    )
    parser.add_argument("--model-b-name", default=live_meta.MODEL_SLOT_DEFAULTS[1][1])
    parser.add_argument("--model-b-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["symbol"])
    parser.add_argument("--model-b-data-symbol", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["data_symbol"])
    parser.add_argument("--model-b-base-asset", default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["base_asset"])
    parser.add_argument("--model-b-maker-fee", type=float, default=live_meta.MODEL_SLOT_DEFAULTS[1][2]["maker_fee"])
    parser.add_argument(
        "--model-b-checkpoint",
        "--aave-checkpoint",
        dest="model_b_checkpoint",
        default=str(DEFAULT_MODEL_B_CHECKPOINT),
    )
    parser.add_argument(
        "--model-b-seeded-inv",
        type=float,
        default=float(DEFAULT_SEEDED_INVENTORIES.get(live_meta.MODEL_SLOT_DEFAULTS[1][1], 0.0)),
    )
    parser.add_argument("--forecast-cache", default=str(DEFAULT_FORECAST_CACHE))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--coarse-modes", default="winner_cash,winner")
    parser.add_argument("--coarse-metrics", default="return,sortino,sharpe,calmar,omega,gain_pain,p10,median")
    parser.add_argument("--coarse-lookbacks", default="1,2,3,5,6,12,24")
    parser.add_argument("--base-short-max-leverage", type=float, default=0.16)
    parser.add_argument("--base-switch-margin", type=float, default=0.0)
    parser.add_argument("--base-min-score-gap", type=float, default=0.0)
    parser.add_argument("--profit-gate-mode", default="live_like", choices=("hypothetical", "live_like"))
    parser.add_argument("--rank-objective", default="return", choices=SUPPORTED_RANK_OBJECTIVES)
    parser.add_argument("--top-unique", type=int, default=3)
    parser.add_argument("--top-finalists", type=int, default=3)
    parser.add_argument("--refine-short-caps", default="0.08,0.10,0.12,0.16,0.20,0.25")
    parser.add_argument("--refine-switch-margins", default="0.000,0.002,0.005")
    parser.add_argument("--refine-min-score-gaps", default="0.000,0.002,0.005")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    model_specs = live_meta.build_model_specs_from_args(args)
    seeded_scenarios = _build_seeded_scenarios(
        model_specs,
        [float(args.model_a_seeded_inv), float(args.model_b_seeded_inv)],
    )
    forecast_cache = Path(args.forecast_cache)
    data_root = Path(args.data_root)
    rank_fn = lambda row: _candidate_rank(row, objective=str(args.rank_objective))
    long_rank_fn = lambda row: _long_window_rank(row, objective=str(args.rank_objective))
    validated_rank_fn = lambda row: _validated_rank(row, objective=str(args.rank_objective))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": [
            {
                **{k: v for k, v in spec.items() if k != "checkpoint"},
                "checkpoint": "" if spec.get("checkpoint") is None else str(Path(spec["checkpoint"]).resolve()),
            }
            for spec in model_specs
        ],
        "seeded_scenarios": [
            {"name": name, **scenario}
            for name, scenario in seeded_scenarios
        ],
        "forecast_cache": str(forecast_cache.resolve()),
        "data_root": str(data_root.resolve()),
        "coarse_modes": _parse_str_list(args.coarse_modes),
        "coarse_metrics": _parse_str_list(args.coarse_metrics),
        "coarse_lookbacks": _parse_int_list(args.coarse_lookbacks),
        "base_short_max_leverage": float(args.base_short_max_leverage),
        "base_switch_margin": float(args.base_switch_margin),
        "base_min_score_gap": float(args.base_min_score_gap),
        "profit_gate_mode": str(args.profit_gate_mode),
        "rank_objective": str(args.rank_objective),
        "top_unique": int(args.top_unique),
        "top_finalists": int(args.top_finalists),
        "refine_short_caps": _parse_float_list(args.refine_short_caps),
        "refine_switch_margins": _parse_float_list(args.refine_switch_margins),
        "refine_min_score_gaps": _parse_float_list(args.refine_min_score_gaps),
    }
    (experiment_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    coarse_configs = _build_coarse_grid(args)
    coarse_jobs = []
    for index, config in enumerate(coarse_configs):
        config_name = _config_name(config)
        coarse_jobs.append(
            {
                "job_id": str(index),
                "config": config,
                "name": config_name,
                "kwargs": {
                    "experiment_root": experiment_root,
                    "scenario_name": "phase1_coarse_7d",
                    "config_name": config_name,
                    "model_specs": model_specs,
                    "forecast_cache": forecast_cache,
                    "data_root": data_root,
                    "selection_mode": str(config["selection_mode"]),
                    "selection_metric": str(config["selection_metric"]),
                    "lookback": int(config["lookback"]),
                    "short_max_leverage": float(config["short_max_leverage"]),
                    "switch_margin": float(config["switch_margin"]),
                    "min_score_gap": float(config["min_score_gap"]),
                    "profit_gate_mode": str(args.profit_gate_mode),
                    "days": 7,
                    "force": bool(args.force),
                },
            }
        )
    coarse_results = _run_job_batch(
        [{"job_id": job["job_id"], "kwargs": job["kwargs"]} for job in coarse_jobs],
        max_workers=int(args.max_workers),
    )
    coarse_rows: list[dict] = []
    for job in coarse_jobs:
        report_7d = coarse_results[str(job["job_id"])]
        coarse_rows.append(
            {
                "name": job["name"],
                "config": job["config"],
                "windows": {"7d": _metric_summary(report_7d)},
            }
        )

    coarse_unique = _dedupe_by_signature(coarse_rows, "7d", rank_fn=rank_fn)
    coarse_groups = _group_by_signature(coarse_rows, "7d", rank_fn=rank_fn)
    shortlisted_signatures = coarse_groups[: max(1, int(args.top_unique))]
    shortlisted = [row for _signature, group in shortlisted_signatures for row in group]

    phase2_jobs = []
    for row in shortlisted:
        config = dict(row["config"])
        config_name = _config_name(config)
        for window_name, days in (("1d", 1), ("7d", 7), ("30d", 30)):
            if window_name == "7d":
                continue
            phase2_jobs.append(
                {
                    "job_id": f"{config_name}__{window_name}",
                    "config_name": config_name,
                    "window_name": window_name,
                    "config": config,
                    "kwargs": {
                        "experiment_root": experiment_root,
                        "scenario_name": "phase2_flat_windows",
                        "config_name": f"{config_name}__{window_name}",
                        "model_specs": model_specs,
                        "forecast_cache": forecast_cache,
                        "data_root": data_root,
                        "selection_mode": str(config["selection_mode"]),
                        "selection_metric": str(config["selection_metric"]),
                        "lookback": int(config["lookback"]),
                        "short_max_leverage": float(config["short_max_leverage"]),
                        "switch_margin": float(config["switch_margin"]),
                        "min_score_gap": float(config["min_score_gap"]),
                        "profit_gate_mode": str(args.profit_gate_mode),
                        "days": int(days),
                        "force": bool(args.force),
                    },
                }
            )
    phase2_results = _run_job_batch(
        [{"job_id": job["job_id"], "kwargs": job["kwargs"]} for job in phase2_jobs],
        max_workers=int(args.max_workers),
    )
    phase2_rows: list[dict] = []
    for row in shortlisted:
        config = dict(row["config"])
        config_name = _config_name(config)
        reports = {"7d": row["windows"]["7d"]}
        for job in phase2_jobs:
            if job["config_name"] != config_name:
                continue
            reports[job["window_name"]] = phase2_results[str(job["job_id"])]
        phase2_rows.append(
            {
                "name": config_name,
                "config": config,
                "windows": _summarize_windows(reports),
            }
        )

    phase2_ranked = sorted(phase2_rows, key=rank_fn, reverse=True)
    refine_seeds = _select_refine_seed_rows(
        phase2_rows,
        int(args.top_unique),
        rank_fn=rank_fn,
        long_rank_fn=long_rank_fn,
    )
    refine_configs = _build_refine_grid([row["config"] for row in refine_seeds], args)

    refine_jobs = []
    for index, config in enumerate(refine_configs):
        config_name = _config_name(config)
        refine_jobs.append(
            {
                "job_id": str(index),
                "config": config,
                "name": config_name,
                "kwargs": {
                    "experiment_root": experiment_root,
                    "scenario_name": "phase2b_refine_7d",
                    "config_name": config_name,
                    "model_specs": model_specs,
                    "forecast_cache": forecast_cache,
                    "data_root": data_root,
                    "selection_mode": str(config["selection_mode"]),
                    "selection_metric": str(config["selection_metric"]),
                    "lookback": int(config["lookback"]),
                    "short_max_leverage": float(config["short_max_leverage"]),
                    "switch_margin": float(config["switch_margin"]),
                    "min_score_gap": float(config["min_score_gap"]),
                    "profit_gate_mode": str(args.profit_gate_mode),
                    "days": 7,
                    "force": bool(args.force),
                },
            }
        )
    refine_results = _run_job_batch(
        [{"job_id": job["job_id"], "kwargs": job["kwargs"]} for job in refine_jobs],
        max_workers=int(args.max_workers),
    )
    refine_rows: list[dict] = []
    for job in refine_jobs:
        report_7d = refine_results[str(job["job_id"])]
        refine_rows.append(
            {
                "name": job["name"],
                "config": job["config"],
                "windows": {"7d": _metric_summary(report_7d)},
            }
        )

    refine_unique = _dedupe_by_signature(refine_rows, "7d", rank_fn=rank_fn)
    finalists = _select_phase2c_rows(refine_unique, int(args.top_finalists), rank_fn=rank_fn)

    final_jobs = []
    for row in finalists:
        config = dict(row["config"])
        config_name = _config_name(config)
        for window_name, days in (("1d", 1), ("7d", 7), ("30d", 30)):
            final_jobs.append(
                {
                    "job_id": f"{config_name}__{window_name}",
                    "config_name": config_name,
                    "window_name": window_name,
                    "config": config,
                    "kwargs": {
                        "experiment_root": experiment_root,
                        "scenario_name": "phase2c_final_flat_windows",
                        "config_name": f"{config_name}__{window_name}",
                        "model_specs": model_specs,
                        "forecast_cache": forecast_cache,
                        "data_root": data_root,
                        "selection_mode": str(config["selection_mode"]),
                        "selection_metric": str(config["selection_metric"]),
                        "lookback": int(config["lookback"]),
                        "short_max_leverage": float(config["short_max_leverage"]),
                        "switch_margin": float(config["switch_margin"]),
                        "min_score_gap": float(config["min_score_gap"]),
                        "profit_gate_mode": str(args.profit_gate_mode),
                        "days": int(days),
                        "force": bool(args.force),
                    },
                }
            )
    final_results = _run_job_batch(
        [{"job_id": job["job_id"], "kwargs": job["kwargs"]} for job in final_jobs],
        max_workers=int(args.max_workers),
    )
    final_phase_rows: list[dict] = []
    for row in finalists:
        config = dict(row["config"])
        config_name = _config_name(config)
        reports = {
            job["window_name"]: final_results[str(job["job_id"])]
            for job in final_jobs
            if job["config_name"] == config_name
        }
        final_phase_rows.append(
            {
                "name": config_name,
                "config": config,
                "windows": _summarize_windows(reports),
            }
        )

    final_phase_ranked = sorted(final_phase_rows, key=rank_fn, reverse=True)
    validation_pool = _select_finalist_rows(
        final_phase_rows,
        int(args.top_finalists),
        rank_fn=rank_fn,
        long_rank_fn=long_rank_fn,
    )
    validated_rows = _validate_candidates(
        experiment_root=experiment_root,
        candidate_rows=validation_pool,
        model_specs=model_specs,
        seeded_scenarios=seeded_scenarios,
        forecast_cache=forecast_cache,
        data_root=data_root,
        profit_gate_mode=str(args.profit_gate_mode),
        force=bool(args.force),
        validated_rank_fn=validated_rank_fn,
    )

    summary = {
        "manifest": manifest,
        "coarse_count": len(coarse_rows),
        "coarse_unique": coarse_unique,
        "coarse_groups": [
            {
                "signature": list(signature),
                "members": [member["name"] for member in members],
            }
            for signature, members in coarse_groups
        ],
        "phase2_candidate_pool_count": len(shortlisted),
        "phase2_ranked": phase2_ranked,
        "refine_count": len(refine_rows),
        "refine_unique": refine_unique,
        "final_phase_ranked": final_phase_ranked,
        "validated_ranked": validated_rows,
        "selected": validated_rows[0] if validated_rows else None,
    }
    (experiment_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
