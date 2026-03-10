from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
from newnanoalpacahourlyexp.config import DatasetConfig, ExperimentConfig
from newnanoalpacahourlyexp.data import AlpacaHourlyDataModule
from newnanoalpacahourlyexp.inference import generate_actions_multi_context
from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)
from newnanoalpacahourlyexp.run_hourly_trader_sim import (
    _load_model,
    _parse_int_tuple,
    _parse_symbols,
    _resolve_data_root,
    _resolve_device,
    _slice_eval_window,
)


def _parse_float_csv(raw: str) -> list[float]:
    values = [float(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_checkpoint_map(raw: Optional[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    if not raw:
        return parsed
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Checkpoint map must use SYMBOL=path entries.")
        symbol, path_text = item.split("=", 1)
        parsed[symbol.strip().upper()] = Path(path_text.strip()).expanduser().resolve()
    return parsed


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    return float(drawdown.min())


def _attach_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()

    baseline_mask = (
        (summary["stop_loss_pct"] == 0.0)
        & (summary["stop_loss_slippage_pct"] == 0.0)
        & (summary["stop_loss_cooldown_bars"] == 0)
    )
    baseline_cols = [
        "symbols",
        "fill_buffer_bps",
        "intensity_scale",
        "price_offset_pct",
    ]
    baseline = summary.loc[
        baseline_mask,
        baseline_cols + ["total_return", "sortino", "max_drawdown", "fills_total"],
    ].rename(
        columns={
            "total_return": "baseline_total_return",
            "sortino": "baseline_sortino",
            "max_drawdown": "baseline_max_drawdown",
            "fills_total": "baseline_fills_total",
        }
    )
    merged = summary.merge(baseline, on=baseline_cols, how="left")
    merged["delta_total_return"] = merged["total_return"] - merged["baseline_total_return"]
    merged["delta_sortino"] = merged["sortino"] - merged["baseline_sortino"]
    merged["delta_max_drawdown"] = merged["max_drawdown"] - merged["baseline_max_drawdown"]
    merged["delta_fills_total"] = merged["fills_total"] - merged["baseline_fills_total"]
    return merged


def _generate_actions_for_frame(
    *,
    model: torch.nn.Module,
    frame: pd.DataFrame,
    data: AlpacaHourlyDataModule,
    sequence_length: int,
    horizon: int,
    context_lengths: Tuple[int, ...],
    experiment_cfg: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    if context_lengths:
        aggregated = generate_actions_multi_context(
            model=model,
            frame=frame,
            feature_columns=data.feature_columns,
            normalizer=data.normalizer,
            base_sequence_length=sequence_length,
            horizon=horizon,
            experiment=experiment_cfg,
            device=device,
        )
        return aggregated.aggregated
    return generate_actions_from_frame(
        model=model,
        frame=frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=sequence_length,
        horizon=horizon,
        device=device,
        require_gpu=True,
    )


def _load_bars_and_actions(
    *,
    symbols: Sequence[str],
    checkpoint_map: Dict[str, Path],
    default_checkpoint: Optional[Path],
    sequence_length: int,
    horizon: int,
    forecast_horizons: Tuple[int, ...],
    context_lengths: Tuple[int, ...],
    experiment_cfg: ExperimentConfig,
    cache_only: bool,
    forecast_cache_root: Path,
    crypto_data_root: Optional[Path],
    stock_data_root: Optional[Path],
    moving_average_windows: Tuple[int, ...],
    ema_windows: Tuple[int, ...],
    atr_windows: Tuple[int, ...],
    trend_windows: Tuple[int, ...],
    drawdown_windows: Tuple[int, ...],
    volume_z_window: int,
    volume_shock_window: int,
    vol_regime_short: int,
    vol_regime_long: int,
    min_history_hours: int,
    eval_days: Optional[float],
    eval_hours: Optional[float],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded: list[tuple[str, AlpacaHourlyDataModule, pd.DataFrame, Path]] = []
    model_cache: dict[tuple[Path, int, int], torch.nn.Module] = {}

    for symbol in symbols:
        checkpoint = checkpoint_map.get(symbol) or default_checkpoint
        if checkpoint is None:
            raise ValueError(f"Missing checkpoint for symbol {symbol}.")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {symbol}: {checkpoint}")

        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=_resolve_data_root(symbol, crypto_data_root, stock_data_root),
            forecast_cache_root=forecast_cache_root,
            sequence_length=int(sequence_length),
            forecast_horizons=forecast_horizons,
            cache_only=bool(cache_only),
            moving_average_windows=moving_average_windows,
            ema_windows=ema_windows,
            atr_windows=atr_windows,
            trend_windows=trend_windows,
            drawdown_windows=drawdown_windows,
            volume_z_window=volume_z_window,
            volume_shock_window=volume_shock_window,
            vol_regime_short=vol_regime_short,
            vol_regime_long=vol_regime_long,
            min_history_hours=int(min_history_hours),
        )
        data = AlpacaHourlyDataModule(data_cfg)
        loaded.append((symbol, data, data.frame.copy(), checkpoint))

    ts_end_common = None
    for _, _, frame, _ in loaded:
        ts = pd.to_datetime(frame["timestamp"], utc=True).max()
        if pd.isna(ts):
            continue
        ts_end_common = ts if ts_end_common is None else max(ts_end_common, ts)
    if ts_end_common is None:
        raise RuntimeError("Failed to infer a shared evaluation window end timestamp.")

    eval_window_hours = 0.0
    if eval_days:
        eval_window_hours = max(eval_window_hours, float(eval_days) * 24.0)
    if eval_hours:
        eval_window_hours = max(eval_window_hours, float(eval_hours))
    ts_start_common = None
    if eval_window_hours > 0.0:
        ts_start_common = ts_end_common - pd.Timedelta(hours=eval_window_hours)

    all_bars: list[pd.DataFrame] = []
    all_actions: list[pd.DataFrame] = []
    rows_needed = None
    if eval_window_hours > 0.0:
        rows_needed = int(eval_window_hours) + int(sequence_length) + 10

    for symbol, data, frame, checkpoint in loaded:
        if rows_needed is not None and rows_needed > 0 and len(frame) > rows_needed:
            frame = frame.tail(rows_needed).reset_index(drop=True)

        model_key = (checkpoint, len(data.feature_columns), int(sequence_length))
        model = model_cache.get(model_key)
        if model is None:
            model = _load_model(checkpoint, len(data.feature_columns), int(sequence_length))
            model_cache[model_key] = model

        actions = _generate_actions_for_frame(
            model=model,
            frame=frame,
            data=data,
            sequence_length=int(sequence_length),
            horizon=int(horizon),
            context_lengths=context_lengths,
            experiment_cfg=experiment_cfg,
            device=device,
        )
        bars = frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
        if ts_start_common is not None:
            actions, bars = _slice_eval_window(
                actions,
                bars,
                eval_days=None,
                eval_hours=None,
                ts_start=ts_start_common,
                ts_end=ts_end_common,
            )
        all_bars.append(bars)
        all_actions.append(actions)

    return pd.concat(all_bars, ignore_index=True), pd.concat(all_actions, ignore_index=True)


def evaluate_stoploss_grid(
    *,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    symbols: Sequence[str],
    initial_cash: float,
    allocation_usd: Optional[float],
    allocation_pct: Optional[float],
    allocation_mode: str,
    decision_lag_bars: int,
    cancel_ack_delay_bars: int,
    partial_fill_on_touch: bool,
    fill_buffers_bps: Sequence[float],
    intensity_scales: Sequence[float],
    price_offset_pcts: Sequence[float],
    stop_loss_pcts: Sequence[float],
    stop_loss_slippage_pcts: Sequence[float],
    stop_loss_cooldown_bars: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    symbols_key = ",".join(str(symbol).upper() for symbol in symbols)

    for fill_buffer_bps in fill_buffers_bps:
        for intensity_scale in intensity_scales:
            for price_offset_pct in price_offset_pcts:
                for stop_loss_pct in stop_loss_pcts:
                    slippage_grid = stop_loss_slippage_pcts if stop_loss_pct > 0.0 else [0.0]
                    cooldown_grid = stop_loss_cooldown_bars if stop_loss_pct > 0.0 else [0]
                    for stop_slippage in slippage_grid:
                        for cooldown in cooldown_grid:
                            sim = HourlyTraderMarketSimulator(
                                HourlyTraderSimulationConfig(
                                    initial_cash=float(initial_cash),
                                    allocation_usd=float(allocation_usd) if allocation_usd is not None else None,
                                    allocation_pct=float(allocation_pct) if allocation_pct is not None else None,
                                    allocation_mode=str(allocation_mode),
                                    decision_lag_bars=int(decision_lag_bars),
                                    cancel_ack_delay_bars=int(cancel_ack_delay_bars),
                                    partial_fill_on_touch=bool(partial_fill_on_touch),
                                    fill_buffer_bps=float(fill_buffer_bps),
                                    intensity_scale=float(intensity_scale),
                                    price_offset_pct=float(price_offset_pct),
                                    stop_loss_pct=float(stop_loss_pct),
                                    stop_loss_slippage_pct=float(stop_slippage),
                                    stop_loss_cooldown_bars=int(cooldown),
                                    symbols=[str(symbol).upper() for symbol in symbols],
                                )
                            )
                            result = sim.run(bars, actions)
                            rows.append(
                                {
                                    "symbols": symbols_key,
                                    "fill_buffer_bps": float(fill_buffer_bps),
                                    "intensity_scale": float(intensity_scale),
                                    "price_offset_pct": float(price_offset_pct),
                                    "stop_loss_pct": float(stop_loss_pct),
                                    "stop_loss_slippage_pct": float(stop_slippage),
                                    "stop_loss_cooldown_bars": int(cooldown),
                                    "total_return": float(result.metrics["total_return"]),
                                    "sortino": float(result.metrics["sortino"]),
                                    "mean_hourly_return": float(result.metrics["mean_hourly_return"]),
                                    "max_drawdown": float(_max_drawdown(result.equity_curve)),
                                    "fills_total": int(len(result.fills)),
                                    "stop_fills_total": int(sum(1 for fill in result.fills if fill.kind == "stop")),
                                    "buy_fills_total": int(sum(1 for fill in result.fills if fill.side == "buy")),
                                    "sell_fills_total": int(sum(1 for fill in result.fills if fill.side == "sell")),
                                    "final_cash": float(result.final_cash),
                                    "final_reserved_cash": float(result.final_reserved_cash),
                                }
                            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = _attach_baseline_deltas(summary)
    return summary.sort_values(
        ["delta_total_return", "delta_sortino", "max_drawdown"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep stop-loss overlays for Alpaca hourly crypto checkpoints.")
    parser.add_argument("--symbols", default="ETHUSD,BTCUSD")
    parser.add_argument("--checkpoints", default=None, help="Comma-separated SYMBOL=PATH checkpoint map.")
    parser.add_argument("--default-checkpoint", default=None, help="Fallback checkpoint path for any unmapped symbols.")
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--context-lengths", default="64,96,192")
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--moving-average-windows", default=None)
    parser.add_argument("--ema-windows", default=None)
    parser.add_argument("--atr-windows", default=None)
    parser.add_argument("--trend-windows", default=None)
    parser.add_argument("--drawdown-windows", default=None)
    parser.add_argument("--volume-z-window", type=int, default=None)
    parser.add_argument("--volume-shock-window", type=int, default=None)
    parser.add_argument("--vol-regime-short", type=int, default=None)
    parser.add_argument("--vol-regime-long", type=int, default=None)
    parser.add_argument("--min-history-hours", type=int, default=None)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--allocation-usd", type=float, default=None)
    parser.add_argument("--allocation-pct", type=float, default=0.05)
    parser.add_argument("--allocation-mode", choices=("per_symbol", "portfolio"), default="per_symbol")
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--cancel-ack-delay-bars", type=int, default=1)
    parser.add_argument("--partial-fill-on-touch", dest="partial_fill_on_touch", action="store_true")
    parser.add_argument("--no-partial-fill-on-touch", dest="partial_fill_on_touch", action="store_false")
    parser.set_defaults(partial_fill_on_touch=True)
    parser.add_argument("--fill-buffer-bps", default="0,5")
    parser.add_argument("--intensity-scales", default="1.0")
    parser.add_argument("--price-offset-pcts", default="0.0")
    parser.add_argument("--stop-loss-pcts", default="0,0.01,0.015,0.02,0.03")
    parser.add_argument("--stop-loss-slippage-pcts", default="0,0.0005")
    parser.add_argument("--stop-loss-cooldown-bars", default="0,1,2")
    parser.add_argument("--eval-days", type=float, default=10.0)
    parser.add_argument("--eval-hours", type=float, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    checkpoint_map = _parse_checkpoint_map(args.checkpoints)
    default_checkpoint = Path(args.default_checkpoint).expanduser().resolve() if args.default_checkpoint else None
    device = _resolve_device(args.device)

    forecast_horizons = tuple(int(token) for token in str(args.forecast_horizons).split(",") if token.strip())
    context_lengths = tuple(int(token) for token in str(args.context_lengths).split(",") if token.strip())
    experiment_cfg = ExperimentConfig(context_lengths=context_lengths, trim_ratio=float(args.trim_ratio))

    ma_windows = _parse_int_tuple(args.moving_average_windows) or DatasetConfig().moving_average_windows
    ema_windows = _parse_int_tuple(args.ema_windows) or DatasetConfig().ema_windows
    atr_windows = _parse_int_tuple(args.atr_windows) or DatasetConfig().atr_windows
    trend_windows = _parse_int_tuple(args.trend_windows) or DatasetConfig().trend_windows
    drawdown_windows = _parse_int_tuple(args.drawdown_windows) or DatasetConfig().drawdown_windows
    volume_z_window = args.volume_z_window if args.volume_z_window is not None else DatasetConfig().volume_z_window
    volume_shock_window = (
        args.volume_shock_window if args.volume_shock_window is not None else DatasetConfig().volume_shock_window
    )
    vol_regime_short = args.vol_regime_short if args.vol_regime_short is not None else DatasetConfig().vol_regime_short
    vol_regime_long = args.vol_regime_long if args.vol_regime_long is not None else DatasetConfig().vol_regime_long
    min_history_hours = args.min_history_hours if args.min_history_hours is not None else DatasetConfig().min_history_hours

    bars, actions = _load_bars_and_actions(
        symbols=symbols,
        checkpoint_map=checkpoint_map,
        default_checkpoint=default_checkpoint,
        sequence_length=int(args.sequence_length),
        horizon=int(args.horizon),
        forecast_horizons=forecast_horizons,
        context_lengths=context_lengths,
        experiment_cfg=experiment_cfg,
        cache_only=bool(args.cache_only),
        forecast_cache_root=Path(args.forecast_cache_root),
        crypto_data_root=Path(args.crypto_data_root) if args.crypto_data_root else None,
        stock_data_root=Path(args.stock_data_root) if args.stock_data_root else None,
        moving_average_windows=ma_windows,
        ema_windows=ema_windows,
        atr_windows=atr_windows,
        trend_windows=trend_windows,
        drawdown_windows=drawdown_windows,
        volume_z_window=int(volume_z_window),
        volume_shock_window=int(volume_shock_window),
        vol_regime_short=int(vol_regime_short),
        vol_regime_long=int(vol_regime_long),
        min_history_hours=int(min_history_hours),
        eval_days=args.eval_days,
        eval_hours=args.eval_hours,
        device=device,
    )

    summary = evaluate_stoploss_grid(
        bars=bars,
        actions=actions,
        symbols=symbols,
        initial_cash=float(args.initial_cash),
        allocation_usd=float(args.allocation_usd) if args.allocation_usd is not None else None,
        allocation_pct=float(args.allocation_pct) if args.allocation_pct is not None else None,
        allocation_mode=str(args.allocation_mode),
        decision_lag_bars=int(args.decision_lag_bars),
        cancel_ack_delay_bars=int(args.cancel_ack_delay_bars),
        partial_fill_on_touch=bool(args.partial_fill_on_touch),
        fill_buffers_bps=_parse_float_csv(args.fill_buffer_bps),
        intensity_scales=_parse_float_csv(args.intensity_scales),
        price_offset_pcts=_parse_float_csv(args.price_offset_pcts),
        stop_loss_pcts=_parse_float_csv(args.stop_loss_pcts),
        stop_loss_slippage_pcts=_parse_float_csv(args.stop_loss_slippage_pcts),
        stop_loss_cooldown_bars=_parse_int_csv(args.stop_loss_cooldown_bars),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bars.to_csv(args.output_dir / "bars.csv", index=False)
    actions.to_csv(args.output_dir / "actions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    best_row = summary.iloc[0].to_dict() if not summary.empty else {}
    (args.output_dir / "best_result.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    preview_cols = [
        "symbols",
        "fill_buffer_bps",
        "intensity_scale",
        "price_offset_pct",
        "stop_loss_pct",
        "stop_loss_slippage_pct",
        "stop_loss_cooldown_bars",
        "total_return",
        "sortino",
        "max_drawdown",
        "fills_total",
        "stop_fills_total",
        "delta_total_return",
        "delta_sortino",
        "delta_max_drawdown",
    ]
    print(summary[preview_cols].head(20).to_string(index=False))
    print(f"Wrote: {args.output_dir / 'summary.csv'}")
    print(f"Wrote: {args.output_dir / 'best_result.json'}")


if __name__ == "__main__":
    main()
