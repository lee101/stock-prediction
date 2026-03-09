from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import torch
from loguru import logger

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation_merged
from src.action_frame_cache import load_or_generate_action_frame
from src.fees import get_fee_for_symbol
from src.robust_trading_metrics import summarize_scenario_results
from src.torch_load_utils import torch_load_compat

from .config import DatasetConfig
from .data import BinanceExp1DataModule
from .sweep import apply_action_overrides


@dataclass(frozen=True)
class SweepConfig:
    default_intensity: float
    default_offset: float
    min_edge: float
    risk_weight: float
    edge_mode: str
    max_hold_hours: Optional[int]
    decision_lag_bars: int
    fill_buffer_bps: float
    max_volume_fraction: Optional[float]
    limit_fill_model: str
    touch_fill_fraction: float
    max_concurrent_positions: int


def parse_csv_list(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def parse_symbols(value: str) -> list[str]:
    symbols = [token.upper() for token in parse_csv_list(value)]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def parse_float_list(value: str) -> list[float]:
    values = [float(token) for token in parse_csv_list(value)]
    if not values:
        raise ValueError(f"Expected at least one float, got {value!r}")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(token) for token in parse_csv_list(value)]
    if not values:
        raise ValueError(f"Expected at least one integer, got {value!r}")
    return values


def parse_optional_float_list(value: str) -> list[float | None]:
    values: list[float | None] = []
    for token in parse_csv_list(value):
        if token.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(token))
    if not values:
        raise ValueError(f"Expected at least one float/None token, got {value!r}")
    return values


def parse_optional_int_list(value: str) -> list[int | None]:
    values: list[int | None] = []
    for token in parse_csv_list(value):
        if token.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(int(token))
    if not values:
        raise ValueError(f"Expected at least one integer/None token, got {value!r}")
    return values


def parse_fill_model_list(value: str) -> list[str]:
    values = [token.strip().lower() for token in parse_csv_list(value)]
    if not values:
        raise ValueError(f"Expected at least one fill model, got {value!r}")
    invalid = sorted({token for token in values if token not in {"binary", "penetration"}})
    if invalid:
        raise ValueError(f"Unsupported fill model(s): {invalid}. Expected 'binary' or 'penetration'.")
    return values


def parse_kv_pairs(raw: Optional[str]) -> dict[str, str]:
    if not raw:
        return {}
    pairs: dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected KEY=VALUE pair, got {token!r}.")
        key, value = token.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid KEY=VALUE pair {token!r}.")
        pairs[key] = value
    return pairs


def parse_float_map(raw: Optional[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, value in parse_kv_pairs(raw).items():
        parsed[key] = float(value)
    return parsed


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, input_dim: int, sequence_length: int) -> torch.nn.Module:
    payload = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", {})
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    policy_cfg.max_len = max(policy_cfg.max_len, sequence_length)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def format_window_label(eval_hours: float) -> str:
    hours = float(eval_hours)
    if hours > 0 and abs(hours / 24.0 - round(hours / 24.0)) < 1e-9:
        return f"{int(round(hours / 24.0))}d"
    return f"{hours:g}h"


def slice_eval_window(merged: pd.DataFrame, eval_hours: float) -> pd.DataFrame:
    if eval_hours <= 0 or merged.empty:
        return merged.reset_index(drop=True)
    ts = pd.to_datetime(merged["timestamp"], utc=True)
    ts_end = ts.max()
    if pd.isna(ts_end):
        return merged.reset_index(drop=True)
    ts_start = ts_end - pd.Timedelta(hours=float(eval_hours))
    window = merged.loc[ts >= ts_start].copy()
    return window.reset_index(drop=True)


def build_start_state_kwargs(
    merged: pd.DataFrame,
    *,
    initial_cash: float,
    start_symbol: Optional[str],
    position_fraction: float = 1.0,
) -> dict[str, Any]:
    if start_symbol is None:
        return {
            "initial_cash": float(initial_cash),
            "initial_inventory": 0.0,
            "initial_symbol": None,
            "initial_open_price": None,
            "initial_open_ts": None,
        }

    fraction = min(1.0, max(0.0, float(position_fraction)))
    symbol = str(start_symbol).upper()
    symbol_rows = merged.loc[merged["symbol"].astype(str).str.upper() == symbol].sort_values("timestamp")
    if symbol_rows.empty:
        raise ValueError(f"Cannot seed start state for {symbol}: symbol is missing from the evaluation window.")

    entry_price = float(symbol_rows["close"].iloc[0])
    if entry_price <= 0.0:
        raise ValueError(f"Cannot seed start state for {symbol}: invalid entry price {entry_price}.")
    entry_ts = pd.to_datetime(symbol_rows["timestamp"].iloc[0], utc=True)
    deployed_cash = float(initial_cash) * fraction
    qty = deployed_cash / entry_price

    return {
        "initial_cash": float(initial_cash) - deployed_cash,
        "initial_inventory": float(qty),
        "initial_symbol": symbol,
        "initial_open_price": float(entry_price),
        "initial_open_ts": entry_ts,
    }


def compute_selection_score(
    summary: dict[str, float],
    *,
    min_trade_count_mean: float,
    require_all_positive: bool,
) -> float:
    trade_shortfall = max(0.0, float(min_trade_count_mean) - float(summary.get("trade_count_mean", 0.0)))
    score = float(summary.get("robust_score", 0.0)) - 0.75 * trade_shortfall
    worst_return = float(summary.get("return_worst_pct", 0.0))
    if require_all_positive and worst_return <= 0.0:
        score -= 100.0 + 10.0 * abs(worst_return)
    return score


def build_scenario_row(
    *,
    config_name: str,
    period: str,
    start_state: str,
    metrics: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    return {
        "config_name": str(config_name),
        "period": str(period),
        "start_state": str(start_state),
        "return_pct": float(metrics.get("total_return", 0.0) * 100.0),
        "annualized_return_pct": float(metrics.get("annualized_return", 0.0) * 100.0),
        "sortino": float(metrics.get("sortino", 0.0)),
        "calmar": float(metrics.get("calmar", 0.0)),
        "max_drawdown_pct": float(abs(metrics.get("max_drawdown", 0.0)) * 100.0),
        "pnl_smoothness": float(metrics.get("pnl_smoothness", 0.0)),
        "trade_count": int(round(float(metrics.get("trade_count", len(getattr(result, "trades", [])))))),
        "work_steal_count": int(round(float(metrics.get("work_steal_count", 0.0)))),
        "open_symbol": str(getattr(result, "open_symbol", "") or ""),
        "final_cash": float(getattr(result, "final_cash", 0.0)),
        "final_inventory": float(getattr(result, "final_inventory", 0.0)),
    }


def build_config_label(cfg: SweepConfig) -> str:
    hold_label = "none" if cfg.max_hold_hours is None else str(int(cfg.max_hold_hours))
    volume_label = "none" if cfg.max_volume_fraction is None else f"{cfg.max_volume_fraction:.3f}".rstrip("0").rstrip(".")
    touch_label = f"{cfg.touch_fill_fraction:.2f}".rstrip("0").rstrip(".")
    return (
        f"i{cfg.default_intensity:.2f}_o{cfg.default_offset:.4f}_edge{cfg.min_edge:.4f}_rw{cfg.risk_weight:.2f}_"
        f"{cfg.edge_mode}_hold{hold_label}_lag{cfg.decision_lag_bars}_buf{cfg.fill_buffer_bps:.1f}_"
        f"vol{volume_label}_fill{cfg.limit_fill_model}_touch{touch_label}_slots{cfg.max_concurrent_positions}"
    )


def build_actions_for_combo(
    *,
    base_actions_by_symbol: dict[str, pd.DataFrame],
    symbols: Sequence[str],
    default_intensity: float,
    default_offset: float,
    intensity_map: dict[str, float],
    offset_map: dict[str, float],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol in symbols:
        actions = base_actions_by_symbol[symbol]
        intensity = float(intensity_map.get(symbol, default_intensity))
        offset = float(offset_map.get(symbol, default_offset))
        if intensity != 1.0 or offset != 0.0:
            actions = apply_action_overrides(actions, intensity_scale=float(intensity), price_offset_pct=float(offset))
        parts.append(actions)
    return pd.concat(parts, ignore_index=True)


def build_best_command(
    *,
    best_cfg: SweepConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> str:
    cmd = [
        "python",
        "-m",
        "binanceexp1.run_multiasset_selector",
        "--symbols",
        ",".join(args.symbols),
        "--checkpoints",
        args.checkpoints,
        "--horizon",
        str(int(args.horizon)),
        "--sequence-length",
        str(int(args.sequence_length)),
        "--forecast-horizons",
        args.forecast_horizons,
        "--data-root",
        str(args.data_root),
        "--forecast-cache-root",
        str(args.forecast_cache_root),
        "--action-cache-root",
        str(args.action_cache_root),
        "--validation-days",
        str(float(args.validation_days)),
        "--default-intensity",
        str(float(best_cfg.default_intensity)),
        "--default-offset",
        str(float(best_cfg.default_offset)),
        "--min-edge",
        str(float(best_cfg.min_edge)),
        "--risk-weight",
        str(float(best_cfg.risk_weight)),
        "--edge-mode",
        best_cfg.edge_mode,
        "--decision-lag-bars",
        str(int(best_cfg.decision_lag_bars)),
        "--fill-buffer-bps",
        str(float(best_cfg.fill_buffer_bps)),
        "--limit-fill-model",
        str(best_cfg.limit_fill_model),
        "--touch-fill-fraction",
        str(float(best_cfg.touch_fill_fraction)),
        "--max-concurrent-positions",
        str(int(best_cfg.max_concurrent_positions)),
        "--output-dir",
        str(output_dir),
    ]
    if args.cache_only:
        cmd.append("--cache-only")
    if args.intensity_map:
        cmd.extend(["--intensity-map", args.intensity_map])
    if args.offset_map:
        cmd.extend(["--offset-map", args.offset_map])
    if best_cfg.max_hold_hours is not None:
        cmd.extend(["--max-hold-hours", str(int(best_cfg.max_hold_hours))])
    if best_cfg.max_volume_fraction is not None:
        cmd.extend(["--max-volume-fraction", str(float(best_cfg.max_volume_fraction))])
    if args.realistic_selection:
        cmd.append("--realistic-selection")
    if args.work_steal:
        cmd.append("--work-steal")
        cmd.extend(["--work-steal-min-profit-pct", str(float(args.work_steal_min_profit_pct))])
        cmd.extend(["--work-steal-min-edge", str(float(args.work_steal_min_edge))])
        cmd.extend(["--work-steal-edge-margin", str(float(args.work_steal_edge_margin))])
    return " ".join(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust BTC/ETH/SOL selector sweep across windows and starting positions.")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="Comma-separated SYMBOL=PATH checkpoint mapping.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--data-root", default=str(DatasetConfig().data_root))
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--action-cache-root", default="experiments/binance_action_cache")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--validation-days", type=float, default=float(DatasetConfig().validation_days))
    parser.add_argument("--device", default=None)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--seed-position-fraction", type=float, default=1.0)
    parser.add_argument("--window-hours", default="168,336,720")
    parser.add_argument("--default-intensity-list", default="1.0")
    parser.add_argument("--default-offset-list", default="0.0")
    parser.add_argument("--intensity-map", default=None)
    parser.add_argument("--offset-map", default=None)
    parser.add_argument("--min-edge-list", default="0.0,0.001,0.0025")
    parser.add_argument("--risk-weight-list", default="0.0,0.25,0.5")
    parser.add_argument("--edge-modes", default="high_low")
    parser.add_argument("--max-hold-hours-list", default="4,6,8")
    parser.add_argument("--decision-lag-list", default="0,1,2")
    parser.add_argument("--fill-buffer-bps-list", default="0,5,10,15")
    parser.add_argument("--max-volume-fraction-list", default="0.05,0.10,0.20")
    parser.add_argument("--limit-fill-model-list", default="binary")
    parser.add_argument("--touch-fill-fraction-list", default="0.0")
    parser.add_argument("--max-concurrent-positions-list", default="1,2")
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
    args.symbols = parse_symbols(args.symbols)
    checkpoint_map = parse_kv_pairs(args.checkpoints)
    for symbol in args.symbols:
        if symbol not in checkpoint_map:
            raise ValueError(f"Missing checkpoint path for symbol {symbol}.")

    device = resolve_device(args.device)
    intensity_map = parse_float_map(args.intensity_map)
    offset_map = parse_float_map(args.offset_map)
    forecast_horizons = tuple(int(token) for token in parse_csv_list(args.forecast_horizons))
    if not forecast_horizons:
        raise ValueError("At least one forecast horizon is required.")
    action_cache_root = Path(args.action_cache_root)

    base_actions_by_symbol: dict[str, pd.DataFrame] = {}
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    fee_by_symbol: dict[str, float] = {}
    periods_by_symbol: dict[str, float] = {}

    logger.info("Loading {} symbols on device {}", len(args.symbols), device)
    for symbol in args.symbols:
        data = BinanceExp1DataModule(
            DatasetConfig(
                symbol=symbol,
                data_root=Path(args.data_root),
                forecast_cache_root=Path(args.forecast_cache_root),
                sequence_length=int(args.sequence_length),
                forecast_horizons=forecast_horizons,
                cache_only=bool(args.cache_only),
                validation_days=float(args.validation_days),
            )
        )
        frame = data.val_dataset.frame.copy()
        checkpoint_path = Path(checkpoint_map[symbol]).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        def _generate_actions() -> pd.DataFrame:
            model = load_model(checkpoint_path, len(data.feature_columns), int(args.sequence_length))
            return generate_actions_from_frame(
                model=model,
                frame=frame,
                feature_columns=data.feature_columns,
                normalizer=data.normalizer,
                sequence_length=int(args.sequence_length),
                horizon=int(args.horizon),
                device=device,
                require_gpu=device.type == "cuda",
            )

        actions, cache_hit = load_or_generate_action_frame(
            cache_root=action_cache_root,
            symbol=symbol,
            checkpoint_path=checkpoint_path,
            frame=frame,
            feature_columns=data.feature_columns,
            normalizer=data.normalizer,
            sequence_length=int(args.sequence_length),
            horizon=int(args.horizon),
            generator=_generate_actions,
        )
        logger.info("Action cache {} for {}", "hit" if cache_hit else "miss", symbol)
        bars_by_symbol[symbol] = frame
        base_actions_by_symbol[symbol] = actions
        fee_by_symbol[symbol] = float(get_fee_for_symbol(symbol))
        periods_by_symbol[symbol] = 24.0 * 365.0

    windows = parse_float_list(args.window_hours)
    default_intensities = parse_float_list(args.default_intensity_list)
    default_offsets = parse_float_list(args.default_offset_list)
    min_edges = parse_float_list(args.min_edge_list)
    risk_weights = parse_float_list(args.risk_weight_list)
    edge_modes = parse_csv_list(args.edge_modes)
    max_hold_hours_list = parse_optional_int_list(args.max_hold_hours_list)
    decision_lags = parse_int_list(args.decision_lag_list)
    fill_buffer_bps_values = parse_float_list(args.fill_buffer_bps_list)
    max_volume_fraction_values = parse_optional_float_list(args.max_volume_fraction_list)
    limit_fill_models = parse_fill_model_list(args.limit_fill_model_list)
    touch_fill_fraction_values = parse_float_list(args.touch_fill_fraction_list)
    max_concurrent_positions_values = parse_int_list(args.max_concurrent_positions_list)

    configs = [
        SweepConfig(*combo)
        for combo in itertools.product(
            default_intensities,
            default_offsets,
            min_edges,
            risk_weights,
            edge_modes,
            max_hold_hours_list,
            decision_lags,
            fill_buffer_bps_values,
            max_volume_fraction_values,
            limit_fill_models,
            touch_fill_fraction_values,
            max_concurrent_positions_values,
        )
    ]
    if not configs:
        raise ValueError("No sweep configurations were produced.")

    exp_name = datetime.now(UTC).strftime("binance_selector_robust_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("experiments") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running {} configs into {}", len(configs), output_dir)

    summary_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_cfg: SweepConfig | None = None
    bars = pd.concat([bars_by_symbol[symbol] for symbol in args.symbols], ignore_index=True)
    combo_cache: dict[tuple[float, float], tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}

    for index, cfg in enumerate(configs, start=1):
        combo_key = (float(cfg.default_intensity), float(cfg.default_offset))
        cached = combo_cache.get(combo_key)
        if cached is None:
            actions = build_actions_for_combo(
                base_actions_by_symbol=base_actions_by_symbol,
                symbols=args.symbols,
                default_intensity=float(cfg.default_intensity),
                default_offset=float(cfg.default_offset),
                intensity_map=intensity_map,
                offset_map=offset_map,
            )
            merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner")
            window_frames = {format_window_label(hours): slice_eval_window(merged, hours) for hours in windows}
            combo_cache[combo_key] = (merged, window_frames)
        else:
            _, window_frames = cached
        config_scenarios: list[dict[str, Any]] = []
        for window_label, window_df in window_frames.items():
            if window_df.empty:
                raise ValueError(f"Evaluation window {window_label} is empty for config {build_config_label(cfg)}.")
            for start_symbol in [None, *args.symbols]:
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
                    min_edge=float(cfg.min_edge),
                    risk_weight=float(cfg.risk_weight),
                    edge_mode=cfg.edge_mode,
                    max_hold_hours=cfg.max_hold_hours,
                    symbols=args.symbols,
                    allow_reentry_same_bar=False,
                    max_volume_fraction=cfg.max_volume_fraction,
                    select_fillable_only=not bool(args.realistic_selection),
                    fee_by_symbol=fee_by_symbol,
                    periods_per_year_by_symbol=periods_by_symbol,
                    limit_fill_model=str(cfg.limit_fill_model),
                    touch_fill_fraction=float(cfg.touch_fill_fraction),
                    max_concurrent_positions=int(cfg.max_concurrent_positions),
                    work_steal_enabled=bool(args.work_steal),
                    work_steal_min_profit_pct=float(args.work_steal_min_profit_pct),
                    work_steal_min_edge=float(args.work_steal_min_edge),
                    work_steal_edge_margin=float(args.work_steal_edge_margin),
                    decision_lag_bars=int(cfg.decision_lag_bars),
                    bar_margin=float(cfg.fill_buffer_bps) / 10_000.0,
                )
                result = run_best_trade_simulation_merged(window_df, sim_cfg, horizon=int(args.horizon))
                metrics = result.metrics
                scenario_row = build_scenario_row(
                    config_name=build_config_label(cfg),
                    period=window_label,
                    start_state=start_state,
                    metrics=metrics,
                    result=result,
                )
                config_scenarios.append(scenario_row)
                scenario_rows.append({**asdict(cfg), **scenario_row})

        summary = summarize_scenario_results(config_scenarios, sortino_clip=float(args.sortino_clip))
        selection_score = compute_selection_score(
            summary,
            min_trade_count_mean=float(args.min_trade_count_mean),
            require_all_positive=bool(args.require_all_positive),
        )
        row = {
            **asdict(cfg),
            "config_name": build_config_label(cfg),
            "selection_score": float(selection_score),
            "all_profitable": bool(float(summary.get("return_worst_pct", 0.0)) > 0.0),
            **summary,
        }
        summary_rows.append(row)
        if best_row is None or float(row["selection_score"]) > float(best_row["selection_score"]):
            best_row = row
            best_cfg = cfg

        logger.info(
            "[{}/{}] {} score={:.3f} worst_ret={:+.2f}% mean_ret={:+.2f}% worst_dd={:.2f}% trades={:.1f}",
            index,
            len(configs),
            row["config_name"],
            row["selection_score"],
            row["return_worst_pct"],
            row["return_mean_pct"],
            row["max_drawdown_worst_pct"],
            row["trade_count_mean"],
        )

    if best_row is None:
        raise RuntimeError("Sweep produced no results.")
    if best_cfg is None:
        raise RuntimeError("Sweep did not retain a best configuration.")

    ranking_df = pd.DataFrame(summary_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    scenarios_df = pd.DataFrame(scenario_rows)

    ranking_df.to_csv(output_dir / "ranking.csv", index=False)
    scenarios_df.to_csv(output_dir / "scenarios.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "symbols": args.symbols,
                "windows": [format_window_label(hours) for hours in windows],
                "best": best_row,
                "top": ranking_df.head(10).to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    (output_dir / "best_config.json").write_text(json.dumps(best_row, indent=2, default=str))
    best_command = build_best_command(best_cfg=best_cfg, args=args, output_dir=output_dir / "best_run")
    (output_dir / "best_command.txt").write_text(best_command + "\n")

    logger.success(
        "Best config {} | score={:.3f} | worst_ret={:+.2f}% | mean_ret={:+.2f}% | worst_dd={:.2f}%",
        best_row["config_name"],
        best_row["selection_score"],
        best_row["return_worst_pct"],
        best_row["return_mean_pct"],
        best_row["max_drawdown_worst_pct"],
    )
    logger.info("Saved ranking to {}", output_dir / "ranking.csv")
    logger.info("Saved scenarios to {}", output_dir / "scenarios.csv")
    logger.info("Best rerun command: {}", best_command)


if __name__ == "__main__":
    main()
