from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

from binanceneural.inference import generate_actions_from_frame
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation_merged
from src.fees import get_fee_for_symbol
from src.robust_trading_metrics import summarize_scenario_results

from .config import DatasetConfig
from .data import BinanceExp1DataModule
from .sweep_multiasset_selector_robust import (
    build_start_state_kwargs,
    compute_selection_score,
    format_window_label,
    load_model,
    parse_float_map,
    parse_float_list,
    parse_symbols,
    resolve_device,
    slice_eval_window,
)


@dataclass(frozen=True)
class CheckpointCandidate:
    symbol: str
    checkpoint: str
    label: str


def _parse_candidate_spec(raw: str, symbols: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    specs = [token.strip() for token in raw.split(";") if token.strip()]
    if not specs:
        raise ValueError("At least one checkpoint candidate mapping is required.")
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected SYMBOL=PATH|PATH mapping, got {spec!r}.")
        symbol_raw, values_raw = spec.split("=", 1)
        symbol = symbol_raw.strip().upper()
        if symbol not in result:
            raise ValueError(f"Unexpected symbol {symbol!r} in candidate mapping.")
        values = [token.strip() for token in values_raw.split("|") if token.strip()]
        if not values:
            raise ValueError(f"No paths supplied for symbol {symbol!r}.")
        result[symbol].extend(values)
    for symbol, values in result.items():
        if not values:
            raise ValueError(f"Missing checkpoint candidates for symbol {symbol}.")
    return result


def _resolve_checkpoint_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Checkpoint file must be a .pt file, got {path}.")
        return path.resolve()
    if path.is_dir():
        checkpoints = sorted(path.glob("epoch_*.pt"), key=lambda item: int(item.stem.split("_")[1]))
        if not checkpoints:
            raise ValueError(f"No epoch checkpoints found in {path}.")
        return checkpoints[-1].resolve()
    raise FileNotFoundError(f"Checkpoint candidate does not exist: {raw}")


def _candidate_label(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _load_candidates(raw: str, symbols: Sequence[str]) -> dict[str, list[CheckpointCandidate]]:
    spec = _parse_candidate_spec(raw, symbols)
    resolved: dict[str, list[CheckpointCandidate]] = {}
    for symbol in symbols:
        seen: set[Path] = set()
        items: list[CheckpointCandidate] = []
        for token in spec[symbol]:
            checkpoint = _resolve_checkpoint_path(token)
            if checkpoint in seen:
                continue
            seen.add(checkpoint)
            items.append(
                CheckpointCandidate(
                    symbol=symbol,
                    checkpoint=str(checkpoint),
                    label=_candidate_label(checkpoint),
                )
            )
        if not items:
            raise ValueError(f"No usable checkpoint candidates resolved for {symbol}.")
        resolved[symbol] = items
    return resolved


def _combo_name(candidates: Sequence[CheckpointCandidate]) -> str:
    return " | ".join(f"{candidate.symbol}:{candidate.label}" for candidate in candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search BTC/ETH/SOL checkpoint combinations under robust selector scoring.")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument(
        "--candidate-checkpoints",
        required=True,
        help="Semicolon-separated SYMBOL=PATH|PATH mappings. Directories resolve to their latest epoch.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--data-root", default=str(DatasetConfig().data_root))
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--validation-days", type=float, default=float(DatasetConfig().validation_days))
    parser.add_argument("--max-history-hours", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--window-hours", default="336")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--seed-position-fraction", type=float, default=1.0)
    parser.add_argument("--default-intensity", type=float, default=6.0)
    parser.add_argument("--default-offset", type=float, default=0.0)
    parser.add_argument("--intensity-map", default=None)
    parser.add_argument("--offset-map", default=None)
    parser.add_argument("--min-edge", type=float, default=0.0015)
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--decision-lag-bars", type=int, default=2)
    parser.add_argument("--fill-buffer-bps", type=float, default=20.0)
    parser.add_argument("--max-volume-fraction", type=float, default=0.1)
    parser.add_argument("--max-concurrent-positions", type=int, default=1)
    parser.add_argument("--sortino-clip", type=float, default=10.0)
    parser.add_argument("--min-trade-count-mean", type=float, default=6.0)
    parser.add_argument("--require-all-positive", action="store_true")
    parser.add_argument("--realistic-selection", action="store_true")
    parser.add_argument("--work-steal", action="store_true")
    parser.add_argument("--work-steal-min-profit-pct", type=float, default=0.001)
    parser.add_argument("--work-steal-min-edge", type=float, default=0.005)
    parser.add_argument("--work-steal-edge-margin", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    symbols = parse_symbols(args.symbols)
    candidates_by_symbol = _load_candidates(args.candidate_checkpoints, symbols)
    intensity_map = parse_float_map(args.intensity_map)
    offset_map = parse_float_map(args.offset_map)
    windows = parse_float_list(args.window_hours)
    forecast_horizons = tuple(int(token) for token in str(args.forecast_horizons).split(",") if token.strip())
    if not forecast_horizons:
        raise ValueError("At least one forecast horizon is required.")
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / datetime.now(UTC).strftime(
        "binance_checkpoint_search_%Y%m%d_%H%M%S"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    fee_by_symbol: dict[str, float] = {}
    periods_by_symbol: dict[str, float] = {}
    merged_by_candidate: dict[str, pd.DataFrame] = {}
    manifest: dict[str, Any] = {"symbols": symbols, "candidates": {}}

    logger.info("Loading data and actions for {} symbols on {}", len(symbols), device)
    for symbol in symbols:
        data = BinanceExp1DataModule(
            DatasetConfig(
                symbol=symbol,
                data_root=Path(args.data_root),
                forecast_cache_root=Path(args.forecast_cache_root),
                sequence_length=int(args.sequence_length),
                forecast_horizons=forecast_horizons,
                cache_only=bool(args.cache_only),
                validation_days=float(args.validation_days),
                max_history_hours=int(args.max_history_hours) if args.max_history_hours is not None else None,
            )
        )
        frame = data.val_dataset.frame.copy()
        bars_by_symbol[symbol] = frame
        fee_by_symbol[symbol] = float(get_fee_for_symbol(symbol))
        periods_by_symbol[symbol] = 24.0 * 365.0
        manifest["candidates"][symbol] = []
        for candidate in candidates_by_symbol[symbol]:
            model = load_model(Path(candidate.checkpoint), len(data.feature_columns), int(args.sequence_length))
            actions = generate_actions_from_frame(
                model=model,
                frame=frame,
                feature_columns=data.feature_columns,
                normalizer=data.normalizer,
                sequence_length=int(args.sequence_length),
                horizon=int(args.horizon),
                device=device,
                require_gpu=device.type == "cuda",
            )
            intensity = float(intensity_map.get(symbol, args.default_intensity))
            offset = float(offset_map.get(symbol, args.default_offset))
            if intensity != 1.0 or offset != 0.0:
                from .sweep import apply_action_overrides

                actions = apply_action_overrides(actions, intensity_scale=float(intensity), price_offset_pct=float(offset))
            merged = frame.merge(actions, on=["timestamp", "symbol"], how="inner")
            merged_by_candidate[candidate.checkpoint] = merged.reset_index(drop=True)
            manifest["candidates"][symbol].append({"label": candidate.label, "checkpoint": candidate.checkpoint})

    summary_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    combos = list(itertools.product(*(candidates_by_symbol[symbol] for symbol in symbols)))
    logger.info("Evaluating {} checkpoint combos", len(combos))

    for index, combo in enumerate(combos, start=1):
        combo_name = _combo_name(combo)
        merged_combo = pd.concat([merged_by_candidate[item.checkpoint] for item in combo], ignore_index=True)
        window_frames = {format_window_label(hours): slice_eval_window(merged_combo, hours) for hours in windows}
        combo_scenarios: list[dict[str, Any]] = []
        for window_label, window_df in window_frames.items():
            if window_df.empty:
                raise ValueError(f"Evaluation window {window_label} is empty for combo {combo_name}.")
            for start_symbol in [None, *symbols]:
                start_state = "flat" if start_symbol is None else start_symbol
                initial_kwargs = build_start_state_kwargs(
                    window_df,
                    initial_cash=float(args.initial_cash),
                    start_symbol=start_symbol,
                    position_fraction=float(args.seed_position_fraction),
                )
                sim_cfg = SelectionConfig(
                    initial_cash=float(initial_kwargs["initial_cash"]),
                    initial_inventory=float(initial_kwargs["initial_inventory"]),
                    initial_symbol=initial_kwargs["initial_symbol"],
                    initial_open_price=initial_kwargs["initial_open_price"],
                    initial_open_ts=initial_kwargs["initial_open_ts"],
                    min_edge=float(args.min_edge),
                    risk_weight=float(args.risk_weight),
                    edge_mode=str(args.edge_mode),
                    max_hold_hours=int(args.max_hold_hours) if args.max_hold_hours is not None else None,
                    symbols=symbols,
                    allow_reentry_same_bar=False,
                    max_volume_fraction=float(args.max_volume_fraction) if args.max_volume_fraction is not None else None,
                    select_fillable_only=not bool(args.realistic_selection),
                    fee_by_symbol=fee_by_symbol,
                    periods_per_year_by_symbol=periods_by_symbol,
                    max_concurrent_positions=int(args.max_concurrent_positions),
                    work_steal_enabled=bool(args.work_steal),
                    work_steal_min_profit_pct=float(args.work_steal_min_profit_pct),
                    work_steal_min_edge=float(args.work_steal_min_edge),
                    work_steal_edge_margin=float(args.work_steal_edge_margin),
                    decision_lag_bars=int(args.decision_lag_bars),
                    bar_margin=float(args.fill_buffer_bps) / 10_000.0,
                )
                result = run_best_trade_simulation_merged(window_df, sim_cfg, horizon=int(args.horizon))
                metrics = result.metrics
                scenario_row = {
                    "combo_name": combo_name,
                    "period": window_label,
                    "start_state": start_state,
                    "return_pct": float(metrics.get("total_return", 0.0) * 100.0),
                    "sortino": float(metrics.get("sortino", 0.0)),
                    "calmar": float(metrics.get("calmar", 0.0)),
                    "max_drawdown_pct": float(abs(metrics.get("max_drawdown", 0.0)) * 100.0),
                    "pnl_smoothness": float(metrics.get("pnl_smoothness", 0.0)),
                    "trade_count": int(round(float(metrics.get("trade_count", len(result.trades))))),
                    "open_symbol": result.open_symbol or "",
                }
                combo_scenarios.append(scenario_row)
                scenario_rows.append(
                    {
                        **{f"{item.symbol.lower()}_checkpoint": item.label for item in combo},
                        **scenario_row,
                    }
                )

        summary = summarize_scenario_results(combo_scenarios, sortino_clip=float(args.sortino_clip))
        selection_score = compute_selection_score(
            summary,
            min_trade_count_mean=float(args.min_trade_count_mean),
            require_all_positive=bool(args.require_all_positive),
        )
        row = {
            "combo_name": combo_name,
            "selection_score": float(selection_score),
            "all_profitable": bool(float(summary.get("return_worst_pct", 0.0)) > 0.0),
            **{f"{item.symbol.lower()}_checkpoint": item.label for item in combo},
            **summary,
        }
        summary_rows.append(row)
        logger.info(
            "[{}/{}] {} score={:.3f} worst_ret={:+.2f}% mean_ret={:+.2f}% worst_dd={:.2f}%",
            index,
            len(combos),
            combo_name,
            row["selection_score"],
            row["return_worst_pct"],
            row["return_mean_pct"],
            row["max_drawdown_worst_pct"],
        )

    ranking_df = pd.DataFrame(summary_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    scenarios_df = pd.DataFrame(scenario_rows)
    ranking_df.to_csv(output_dir / "ranking.csv", index=False)
    scenarios_df.to_csv(output_dir / "scenarios.csv", index=False)
    best = ranking_df.iloc[0].to_dict() if not ranking_df.empty else {}
    manifest["best"] = best
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.success(
        "Best combo {} | score={:.3f} | worst_ret={:+.2f}% | mean_ret={:+.2f}% | worst_dd={:.2f}%",
        best.get("combo_name", ""),
        float(best.get("selection_score", 0.0)),
        float(best.get("return_worst_pct", 0.0)),
        float(best.get("return_mean_pct", 0.0)),
        float(best.get("max_drawdown_worst_pct", 0.0)),
    )
    logger.info("Saved ranking to {}", output_dir / "ranking.csv")
    logger.info("Saved scenarios to {}", output_dir / "scenarios.csv")


if __name__ == "__main__":
    main()
