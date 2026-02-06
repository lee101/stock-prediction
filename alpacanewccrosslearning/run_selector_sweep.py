from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from binanceexp1.sweep import apply_action_overrides
from src.torch_device_utils import require_cuda as require_cuda_device

from newnanoalpacahourlyexp.config import DatasetConfig
from newnanoalpacahourlyexp.data import AlpacaHourlyDataModule
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]

def _parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _load_model(checkpoint_path: Path, input_dim: int, sequence_length: int) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


def _slice_eval_window(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    eval_days: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if eval_days <= 0:
        return frame, actions
    ts_end = pd.to_datetime(frame["timestamp"], utc=True).max()
    ts_start = ts_end - pd.Timedelta(days=eval_days)
    frame = frame[pd.to_datetime(frame["timestamp"], utc=True) >= ts_start]
    actions = actions[pd.to_datetime(actions["timestamp"], utc=True) >= ts_start]
    return frame.reset_index(drop=True), actions.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep selector hyperparams for global policy.")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-only", action="store_true")
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
    parser.add_argument("--intensity-list", default="1.0")
    parser.add_argument("--offset-list", default="0.0")
    parser.add_argument("--min-edge-list", default="0.0,0.0005,0.001")
    parser.add_argument("--risk-weight-list", default="0.25,0.5,0.75")
    parser.add_argument("--edge-modes", default="high_low")
    parser.add_argument("--dip-threshold-list", default="0.0,0.005,0.01")
    parser.add_argument(
        "--max-hold-hours-list",
        default="0",
        help="Comma-separated max-hold hours (0 disables). Example: 0,6,12,24",
    )
    parser.add_argument("--eval-days", type=float, default=10.0)
    parser.add_argument("--output-dir", default="alpacanewccrosslearning/outputs/selector_sweeps")
    parser.add_argument("--allow-short", action="store_true", help="Allow selector to open short positions (stocks only).")
    parser.add_argument("--long-only-symbols", default=None, help="Comma-separated symbols to restrict to long-only.")
    parser.add_argument("--short-only-symbols", default=None, help="Comma-separated symbols to restrict to short-only.")
    args = parser.parse_args()

    device = require_cuda_device("selector sweep", allow_fallback=False)

    symbols = _parse_symbols(args.symbols)
    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)
    ma_windows = (
        tuple(int(x) for x in args.moving_average_windows.split(",") if x.strip())
        if args.moving_average_windows
        else DatasetConfig().moving_average_windows
    )
    ema_windows = (
        tuple(int(x) for x in args.ema_windows.split(",") if x.strip())
        if args.ema_windows
        else DatasetConfig().ema_windows
    )
    atr_windows = (
        tuple(int(x) for x in args.atr_windows.split(",") if x.strip())
        if args.atr_windows
        else DatasetConfig().atr_windows
    )
    trend_windows = (
        tuple(int(x) for x in args.trend_windows.split(",") if x.strip())
        if args.trend_windows
        else DatasetConfig().trend_windows
    )
    drawdown_windows = (
        tuple(int(x) for x in args.drawdown_windows.split(",") if x.strip())
        if args.drawdown_windows
        else DatasetConfig().drawdown_windows
    )
    volume_z_window = args.volume_z_window if args.volume_z_window is not None else DatasetConfig().volume_z_window
    volume_shock_window = (
        args.volume_shock_window if args.volume_shock_window is not None else DatasetConfig().volume_shock_window
    )
    vol_regime_short = args.vol_regime_short if args.vol_regime_short is not None else DatasetConfig().vol_regime_short
    vol_regime_long = args.vol_regime_long if args.vol_regime_long is not None else DatasetConfig().vol_regime_long
    min_history = args.min_history_hours if args.min_history_hours is not None else DatasetConfig().min_history_hours

    bars_frames: List[pd.DataFrame] = []
    base_actions_frames: List[pd.DataFrame] = []
    fee_by_symbol: Dict[str, float] = {}
    periods_by_symbol: Dict[str, float] = {}

    for symbol in symbols:
        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=Path(args.data_root) if args.data_root else None,
            forecast_cache_root=Path(args.forecast_cache_root),
            sequence_length=args.sequence_length,
            forecast_horizons=forecast_horizons,
            cache_only=args.cache_only,
            moving_average_windows=ma_windows,
            ema_windows=ema_windows,
            atr_windows=atr_windows,
            trend_windows=trend_windows,
            drawdown_windows=drawdown_windows,
            volume_z_window=volume_z_window,
            volume_shock_window=volume_shock_window,
            vol_regime_short=vol_regime_short,
            vol_regime_long=vol_regime_long,
            min_history_hours=min_history,
        )
        data = AlpacaHourlyDataModule(data_cfg)
        model = _load_model(Path(args.checkpoint), len(data.feature_columns), args.sequence_length)
        frame = data.val_dataset.frame.copy()
        actions = generate_actions_from_frame(
            model=model,
            frame=frame,
            feature_columns=data.feature_columns,
            normalizer=data.normalizer,
            sequence_length=args.sequence_length,
            horizon=args.horizon,
            device=device,
            require_gpu=True,
        )
        if args.eval_days:
            frame, actions = _slice_eval_window(frame, actions, args.eval_days)
        bars_frames.append(frame)
        base_actions_frames.append(actions)
        fee_by_symbol[symbol] = float(data.asset_meta.maker_fee)
        periods_by_symbol[symbol] = float(data.asset_meta.periods_per_year)

    bars = pd.concat(bars_frames, ignore_index=True)
    base_actions = pd.concat(base_actions_frames, ignore_index=True)

    long_only_symbols = [token.strip().upper() for token in (args.long_only_symbols or "").split(",") if token.strip()]
    short_only_symbols = [token.strip().upper() for token in (args.short_only_symbols or "").split(",") if token.strip()]

    intensity_list = _parse_float_list(args.intensity_list)
    offset_list = _parse_float_list(args.offset_list)
    min_edge_list = _parse_float_list(args.min_edge_list)
    risk_weight_list = _parse_float_list(args.risk_weight_list)
    dip_list = _parse_float_list(args.dip_threshold_list)
    edge_modes = [mode.strip() for mode in args.edge_modes.split(",") if mode.strip()]
    max_hold_list = _parse_int_list(args.max_hold_hours_list)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "selector_sweep.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        rows_written = 0
        writer.writerow(
            [
                "intensity",
                "offset",
                "min_edge",
                "risk_weight",
                "edge_mode",
                "dip_threshold",
                "max_hold_hours",
                "total_return",
                "sortino",
                "final_cash",
                "open_symbol",
            ]
        )
        handle.flush()

        for intensity in intensity_list:
            for offset in offset_list:
                actions = apply_action_overrides(
                    base_actions,
                    intensity_scale=float(intensity),
                    price_offset_pct=float(offset),
                )
                for dip_threshold in dip_list:
                    if dip_threshold > 0:
                        merged = actions.merge(
                            bars[["timestamp", "symbol", "close", "low"]],
                            on=["timestamp", "symbol"],
                            how="left",
                        )
                        mask = merged["low"] <= merged["close"] * (1.0 - dip_threshold)
                        filtered = actions.copy()
                        filtered.loc[~mask.fillna(False), "buy_amount"] = 0.0
                        candidate_actions = filtered
                    else:
                        candidate_actions = actions

                    for max_hold_hours_raw in max_hold_list:
                        max_hold_hours = int(max_hold_hours_raw) if int(max_hold_hours_raw) > 0 else None
                        for min_edge in min_edge_list:
                            for risk_weight in risk_weight_list:
                                for edge_mode in edge_modes:
                                    cfg = SelectionConfig(
                                        initial_cash=10_000.0,
                                        min_edge=float(min_edge),
                                        risk_weight=float(risk_weight),
                                        edge_mode=edge_mode,
                                        max_hold_hours=max_hold_hours,
                                        allow_reentry_same_bar=False,
                                        enforce_market_hours=True,
                                        close_at_eod=True,
                                        fee_by_symbol=fee_by_symbol,
                                        periods_per_year_by_symbol=periods_by_symbol,
                                        symbols=symbols,
                                        allow_short=bool(args.allow_short),
                                        long_only_symbols=long_only_symbols,
                                        short_only_symbols=short_only_symbols,
                                    )
                                    result = run_best_trade_simulation(
                                        bars,
                                        candidate_actions,
                                        cfg,
                                        horizon=args.horizon,
                                    )
                                    metrics = result.metrics
                                    writer.writerow(
                                        [
                                            intensity,
                                            offset,
                                            min_edge,
                                            risk_weight,
                                            edge_mode,
                                            dip_threshold,
                                            max_hold_hours or 0,
                                            metrics.get("total_return", 0.0),
                                            metrics.get("sortino", 0.0),
                                            result.final_cash,
                                            result.open_symbol,
                                        ]
                                    )
                                    rows_written += 1
                                    if rows_written % 25 == 0:
                                        handle.flush()

    print(f"Saved sweep results to {csv_path}")


if __name__ == "__main__":
    main()
