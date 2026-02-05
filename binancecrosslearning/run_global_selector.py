from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from binanceexp1.sweep import apply_action_overrides
from src.torch_device_utils import require_cuda as require_cuda_device

from newnanoalpacahourlyexp.config import DatasetConfig
from newnanoalpacahourlyexp.data import AlpacaHourlyDataModule
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,LINKUSDT,ADAUSDT,APTUSDT,AVAXUSDT"


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


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


def _apply_dip_filter(actions: pd.DataFrame, bars: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    if threshold_pct <= 0:
        return actions
    merged = actions.merge(
        bars[["timestamp", "symbol", "close", "low"]],
        on=["timestamp", "symbol"],
        how="left",
    )
    trigger = merged["low"] <= merged["close"] * (1.0 - threshold_pct)
    filtered = actions.copy()
    filtered.loc[~trigger.fillna(False), "buy_amount"] = 0.0
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run global best-trade selector for a shared checkpoint.")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,4,24")
    parser.add_argument("--forecast-cache-root", default="binancecrosslearning/forecast_cache")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--moving-average-windows", default=None)
    parser.add_argument("--min-history-hours", type=int, default=None)
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--price-offset-pct", type=float, default=0.0)
    parser.add_argument("--dip-threshold-pct", type=float, default=0.0)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--risk-weight", type=float, default=0.5)
    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--max-hold-hours", type=int, default=None)
    parser.add_argument("--allow-reentry-same-bar", action="store_true")
    parser.add_argument("--no-enforce-market-hours", action="store_true")
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--eval-days", type=float, default=None)
    parser.add_argument("--eval-hours", type=float, default=None)
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = require_cuda_device("global selector inference", allow_fallback=False)

    symbols = _parse_symbols(args.symbols)
    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)

    bars_frames: List[pd.DataFrame] = []
    action_frames: List[pd.DataFrame] = []
    fee_by_symbol: Dict[str, float] = {}
    periods_by_symbol: Dict[str, float] = {}

    if args.moving_average_windows:
        ma_windows = tuple(int(x) for x in args.moving_average_windows.split(",") if x.strip())
    else:
        ma_windows = DatasetConfig().moving_average_windows
    min_history_hours = args.min_history_hours if args.min_history_hours is not None else DatasetConfig().min_history_hours

    fee_override = float(args.maker_fee) if args.maker_fee is not None else None
    periods_override = float(args.periods_per_year) if args.periods_per_year is not None else None

    for symbol in symbols:
        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=Path(args.data_root) if args.data_root else None,
            forecast_cache_root=Path(args.forecast_cache_root),
            sequence_length=args.sequence_length,
            forecast_horizons=forecast_horizons,
            cache_only=args.cache_only,
            moving_average_windows=ma_windows,
            min_history_hours=min_history_hours,
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

        if args.intensity_scale != 1.0 or args.price_offset_pct != 0.0:
            actions = apply_action_overrides(
                actions,
                intensity_scale=args.intensity_scale,
                price_offset_pct=args.price_offset_pct,
            )

        actions = _apply_dip_filter(actions, frame, args.dip_threshold_pct)

        if args.eval_days or args.eval_hours:
            hours = max(0.0, float(args.eval_days or 0) * 24.0, float(args.eval_hours or 0))
            if hours > 0:
                ts_end = pd.to_datetime(frame["timestamp"], utc=True).max()
                ts_start = ts_end - pd.Timedelta(hours=hours)
                frame = frame[pd.to_datetime(frame["timestamp"], utc=True) >= ts_start].reset_index(drop=True)
                actions = actions[pd.to_datetime(actions["timestamp"], utc=True) >= ts_start].reset_index(drop=True)

        bars_frames.append(frame)
        action_frames.append(actions)
        fee_by_symbol[symbol] = fee_override if fee_override is not None else float(data.asset_meta.maker_fee)
        periods_by_symbol[symbol] = (
            periods_override if periods_override is not None else float(data.asset_meta.periods_per_year)
        )

    bars = pd.concat(bars_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)

    cfg = SelectionConfig(
        initial_cash=args.initial_cash,
        min_edge=args.min_edge,
        risk_weight=args.risk_weight,
        edge_mode=args.edge_mode,
        max_hold_hours=args.max_hold_hours,
        allow_reentry_same_bar=args.allow_reentry_same_bar,
        enforce_market_hours=not args.no_enforce_market_hours,
        close_at_eod=not args.no_close_at_eod,
        fee_by_symbol=fee_by_symbol,
        periods_per_year_by_symbol=periods_by_symbol,
        symbols=symbols,
    )

    result = run_best_trade_simulation(bars, actions, cfg, horizon=args.horizon)
    metrics = result.metrics
    print(f"total_return: {metrics.get('total_return', 0.0):.4f}")
    print(f"sortino: {metrics.get('sortino', 0.0):.4f}")
    print(f"final_cash: {result.final_cash:.4f}")
    print(f"open_symbol: {result.open_symbol}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.per_hour.to_csv(output_dir / "selector_per_hour.csv", index=False)
        pd.DataFrame([t.__dict__ for t in result.trades]).to_csv(output_dir / "selector_trades.csv", index=False)
        (output_dir / "selector_metrics.json").write_text(pd.Series(metrics).to_json(indent=2))


if __name__ == "__main__":
    main()
