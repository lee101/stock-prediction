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

from .marketsimulator.selector import SelectionConfig, run_best_trade_simulation
from .symbols import build_longable_symbols, build_shortable_symbols, normalize_symbols


def _parse_symbols(raw: str | None) -> List[str]:
    if raw is None:
        return []
    symbols = normalize_symbols([token for token in raw.split(",") if token.strip()])
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


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


def _slice_frame_for_eval(
    frame: pd.DataFrame,
    *,
    eval_days: Optional[float],
    sequence_length: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    if not eval_days or eval_days <= 0:
        return frame
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    ts_end = ts.max()
    if pd.isna(ts_end):
        return frame
    ts_start = ts_end - pd.Timedelta(days=float(eval_days))
    eval_mask = ts >= ts_start
    eval_count = int(eval_mask.sum())
    if eval_count <= 0:
        return frame
    needed = eval_count + max(0, int(sequence_length))
    if needed >= len(frame):
        return frame
    return frame.iloc[-needed:].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep selector hyperparams for constrained global policy.")
    parser.add_argument("--symbols", default=None)
    parser.add_argument(
        "--long-stocks",
        default=None,
        help="Comma-separated longable stock symbols (default: NVDA,GOOG,MSFT).",
    )
    parser.add_argument(
        "--short-stocks",
        default=None,
        help="Comma-separated shortable stock symbols (default: YELP,EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA,NYT).",
    )
    parser.add_argument(
        "--crypto",
        default=None,
        help="Comma-separated crypto symbols to include (default: BTCUSD,ETHUSD,SOLUSD).",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--forecast-cache-root", default="alpacaconstrainedexp/forecast_cache")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--moving-average-windows", default=None)
    parser.add_argument("--min-history-hours", type=int, default=None)
    parser.add_argument("--max-feature-lookback-hours", type=int, default=None)
    parser.add_argument("--intensity-list", default="1.0")
    parser.add_argument("--offset-list", default="0.0")
    parser.add_argument("--min-edge-list", default="0.0,0.0005,0.001")
    parser.add_argument("--risk-weight-list", default="0.25,0.5,0.75")
    parser.add_argument("--edge-modes", default="high_low")
    parser.add_argument("--dip-threshold-list", default="0.0,0.005,0.01")
    parser.add_argument("--eval-days", type=float, default=10.0)
    parser.add_argument("--output-dir", default="alpacanewccrosslearning/outputs/selector_sweeps")
    args = parser.parse_args()

    device = require_cuda_device("selector sweep", allow_fallback=False)

    long_symbols = build_longable_symbols(
        crypto_symbols=_parse_symbols(args.crypto) if args.crypto else None,
        stock_symbols=_parse_symbols(args.long_stocks) if args.long_stocks else None,
    )
    short_symbols = build_shortable_symbols(
        stock_symbols=_parse_symbols(args.short_stocks) if args.short_stocks else None,
    )

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        symbols = normalize_symbols(long_symbols + short_symbols)
        if not symbols:
            raise ValueError("At least one symbol is required.")
    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)
    ma_windows = (
        tuple(int(x) for x in args.moving_average_windows.split(",") if x.strip())
        if args.moving_average_windows
        else DatasetConfig().moving_average_windows
    )
    min_history = args.min_history_hours if args.min_history_hours is not None else DatasetConfig().min_history_hours
    max_lookback = (
        int(args.max_feature_lookback_hours)
        if args.max_feature_lookback_hours is not None
        else DatasetConfig().max_feature_lookback_hours
    )

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
            min_history_hours=min_history,
            max_feature_lookback_hours=max_lookback,
        )
        data = AlpacaHourlyDataModule(data_cfg)
        model = _load_model(Path(args.checkpoint), len(data.feature_columns), args.sequence_length)
        frame = data.val_dataset.frame.copy()
        frame = _slice_frame_for_eval(
            frame,
            eval_days=args.eval_days,
            sequence_length=args.sequence_length,
        )
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

    intensity_list = _parse_float_list(args.intensity_list)
    offset_list = _parse_float_list(args.offset_list)
    min_edge_list = _parse_float_list(args.min_edge_list)
    risk_weight_list = _parse_float_list(args.risk_weight_list)
    dip_list = _parse_float_list(args.dip_threshold_list)
    edge_modes = [mode.strip() for mode in args.edge_modes.split(",") if mode.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "selector_sweep.csv"

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "intensity",
                "offset",
                "min_edge",
                "risk_weight",
                "edge_mode",
                "dip_threshold",
                "total_return",
                "sortino",
                "final_cash",
                "open_symbol",
            ]
        )

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

                    for min_edge in min_edge_list:
                        for risk_weight in risk_weight_list:
                            for edge_mode in edge_modes:
                                cfg = SelectionConfig(
                                    initial_cash=10_000.0,
                                    min_edge=float(min_edge),
                                    risk_weight=float(risk_weight),
                                    edge_mode=edge_mode,
                                    allow_reentry_same_bar=False,
                                    enforce_market_hours=True,
                                    close_at_eod=True,
                                    fee_by_symbol=fee_by_symbol,
                                    periods_per_year_by_symbol=periods_by_symbol,
                                    symbols=symbols,
                                    long_symbols=long_symbols,
                                    short_symbols=short_symbols,
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
                                        metrics.get("total_return", 0.0),
                                        metrics.get("sortino", 0.0),
                                        result.final_cash,
                                        result.open_symbol,
                                    ]
                                )

    print(f"Saved sweep results to {csv_path}")


if __name__ == "__main__":
    main()
