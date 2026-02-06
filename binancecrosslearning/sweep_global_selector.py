from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from binanceexp1.sweep import apply_action_overrides
from src.torch_device_utils import require_cuda as require_cuda_device
from src.torch_load_utils import torch_load_compat

from newnanoalpacahourlyexp.config import DatasetConfig
from newnanoalpacahourlyexp.data import AlpacaHourlyDataModule
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation


@dataclass(frozen=True)
class SweepRow:
    intensity_scale: float
    price_offset_pct: float
    min_edge: float
    risk_weight: float
    max_volume_fraction: Optional[float]
    total_return: float
    sortino: float
    final_cash: float
    open_symbol: Optional[str]


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in str(raw).split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _parse_csv_floats(raw: str) -> List[float]:
    values: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _parse_csv_optional_floats(raw: Optional[str]) -> List[Optional[float]]:
    if raw is None:
        return [None]
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        return [None]
    values: List[Optional[float]] = []
    for t in tokens:
        if t.lower() in ("none", "null"):
            values.append(None)
        else:
            values.append(float(t))
    return values


def _load_model(checkpoint_path: Path, input_dim: int, sequence_length: int) -> torch.nn.Module:
    payload = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", {})
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    policy_cfg.max_len = max(int(policy_cfg.max_len or 0), int(sequence_length))
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


def _slice_eval_window(
    actions: pd.DataFrame,
    bars: pd.DataFrame,
    eval_days: Optional[float],
    eval_hours: Optional[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if actions.empty or bars.empty:
        return actions, bars
    hours = 0.0
    if eval_days:
        hours = max(hours, float(eval_days) * 24.0)
    if eval_hours:
        hours = max(hours, float(eval_hours))
    if hours <= 0:
        return actions, bars
    ts_end = pd.to_datetime(bars["timestamp"], utc=True).max()
    if pd.isna(ts_end):
        return actions, bars
    ts_start = ts_end - pd.Timedelta(hours=hours)
    bars_slice = bars[pd.to_datetime(bars["timestamp"], utc=True) >= ts_start]
    actions_slice = actions[pd.to_datetime(actions["timestamp"], utc=True) >= ts_start]
    return actions_slice.reset_index(drop=True), bars_slice.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep global selector parameters (intensity/offset/min_edge/etc) with a single inference pass."
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. BTCU,ETHU,SOLU,BNBU).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--frame-split", default="val", choices=["val", "full"])
    parser.add_argument("--forecast-horizons", default="1,4,24")
    parser.add_argument("--forecast-cache-root", default="binancecrosslearning/forecast_cache")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--moving-average-windows", default=None)
    parser.add_argument("--min-history-hours", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--validation-days", type=int, default=None)
    parser.add_argument("--feature-max-window-hours", type=int, default=None)

    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--dip-threshold-pct", type=float, default=0.0)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--max-hold-hours", type=int, default=None)
    parser.add_argument("--allow-reentry-same-bar", action="store_true")
    parser.add_argument("--no-enforce-market-hours", action="store_true")
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--maker-fee", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=float, default=None)
    parser.add_argument("--eval-days", type=float, default=None)
    parser.add_argument("--eval-hours", type=float, default=None)

    parser.add_argument("--intensity-scales", default="1,2,5,10,20")
    parser.add_argument("--price-offset-pcts", default="0,0.0001,0.00025")
    parser.add_argument("--min-edges", default="0,0.001,0.002")
    parser.add_argument("--risk-weights", default="0.5")
    parser.add_argument(
        "--max-volume-fractions",
        default=None,
        help="Comma-separated fractions (e.g. 0.1,0.2) or 'none'. Default: no cap.",
    )

    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = require_cuda_device("global selector sweep inference", allow_fallback=False)
    symbols = _parse_symbols(args.symbols)
    forecast_horizons = tuple(int(x) for x in str(args.forecast_horizons).split(",") if x.strip())

    if args.moving_average_windows:
        ma_windows = tuple(int(x) for x in str(args.moving_average_windows).split(",") if x.strip())
    else:
        ma_windows = DatasetConfig().moving_average_windows
    min_history_hours = args.min_history_hours if args.min_history_hours is not None else DatasetConfig().min_history_hours
    val_fraction = float(args.val_fraction) if args.val_fraction is not None else DatasetConfig().val_fraction
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"--val-fraction must be in (0, 1), got {val_fraction}.")
    validation_days = int(args.validation_days) if args.validation_days is not None else DatasetConfig().validation_days
    if validation_days < 0:
        raise ValueError(f"--validation-days must be >= 0, got {validation_days}.")

    fee_override = float(args.maker_fee) if args.maker_fee is not None else None
    periods_override = float(args.periods_per_year) if args.periods_per_year is not None else None

    # Build frames + base actions once.
    bars_frames: List[pd.DataFrame] = []
    action_frames: List[pd.DataFrame] = []
    fee_by_symbol: Dict[str, float] = {}
    periods_by_symbol: Dict[str, float] = {}

    model: Optional[torch.nn.Module] = None
    model_input_dim: Optional[int] = None

    for idx, symbol in enumerate(symbols):
        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=Path(args.data_root) if args.data_root else None,
            forecast_cache_root=Path(args.forecast_cache_root),
            sequence_length=args.sequence_length,
            forecast_horizons=forecast_horizons,
            cache_only=bool(args.cache_only),
            moving_average_windows=ma_windows,
            min_history_hours=int(min_history_hours),
            val_fraction=val_fraction,
            validation_days=validation_days,
        )
        if args.feature_max_window_hours is not None:
            from src.hourly_feature_windowing import apply_feature_max_window_hours

            data_cfg = apply_feature_max_window_hours(data_cfg, max_window_hours=int(args.feature_max_window_hours))

        data = AlpacaHourlyDataModule(data_cfg)
        input_dim = len(data.feature_columns)
        if model is None:
            model = _load_model(Path(args.checkpoint), input_dim=input_dim, sequence_length=args.sequence_length).to(device)
            model_input_dim = input_dim
        elif model_input_dim != input_dim:
            raise ValueError(
                f"Feature dim mismatch across symbols: {symbols[0]} dim={model_input_dim}, {symbol} dim={input_dim}. "
                "Use --feature-max-window-hours (or explicit feature_columns) to force a shared feature set."
            )

        frame = data.frame.copy() if args.frame_split == "full" else data.val_dataset.frame.copy()
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

        # Apply fixed filters/window slicing once.
        actions = _apply_dip_filter(actions, frame, float(args.dip_threshold_pct or 0.0))
        if args.eval_days or args.eval_hours:
            actions, frame = _slice_eval_window(actions, frame, args.eval_days, args.eval_hours)

        bars_frames.append(frame)
        action_frames.append(actions)
        fee_by_symbol[symbol] = fee_override if fee_override is not None else float(data.asset_meta.maker_fee)
        periods_by_symbol[symbol] = (
            periods_override if periods_override is not None else float(data.asset_meta.periods_per_year)
        )

    assert model is not None

    bars = pd.concat(bars_frames, ignore_index=True)
    base_actions = pd.concat(action_frames, ignore_index=True)

    intensity_scales = _parse_csv_floats(args.intensity_scales)
    price_offsets = _parse_csv_floats(args.price_offset_pcts)
    min_edges = _parse_csv_floats(args.min_edges)
    risk_weights = _parse_csv_floats(args.risk_weights)
    max_volume_fracs = _parse_csv_optional_floats(args.max_volume_fractions)

    results: List[SweepRow] = []
    best: Optional[SweepRow] = None

    for intensity in intensity_scales:
        for offset in price_offsets:
            adjusted_actions = apply_action_overrides(
                base_actions,
                intensity_scale=float(intensity),
                price_offset_pct=float(offset),
            )
            for min_edge in min_edges:
                for risk_weight in risk_weights:
                    for max_vol in max_volume_fracs:
                        cfg = SelectionConfig(
                            initial_cash=float(args.initial_cash),
                            min_edge=float(min_edge),
                            risk_weight=float(risk_weight),
                            edge_mode=str(args.edge_mode),
                            max_volume_fraction=max_vol,
                            max_hold_hours=args.max_hold_hours,
                            allow_reentry_same_bar=bool(args.allow_reentry_same_bar),
                            enforce_market_hours=not bool(args.no_enforce_market_hours),
                            close_at_eod=not bool(args.no_close_at_eod),
                            fee_by_symbol=fee_by_symbol,
                            periods_per_year_by_symbol=periods_by_symbol,
                            symbols=symbols,
                        )
                        sim = run_best_trade_simulation(bars, adjusted_actions, cfg, horizon=args.horizon)
                        metrics = sim.metrics
                        row = SweepRow(
                            intensity_scale=float(intensity),
                            price_offset_pct=float(offset),
                            min_edge=float(min_edge),
                            risk_weight=float(risk_weight),
                            max_volume_fraction=max_vol,
                            total_return=float(metrics.get("total_return", 0.0)),
                            sortino=float(metrics.get("sortino", 0.0)),
                            final_cash=float(sim.final_cash),
                            open_symbol=sim.open_symbol,
                        )
                        results.append(row)
                        if best is None or row.total_return > best.total_return:
                            best = row

    results_sorted = sorted(results, key=lambda r: (r.total_return, r.sortino), reverse=True)
    top = results_sorted[:10]
    if best is not None:
        print(
            "best_total_return:",
            json.dumps(
                {
                    "total_return": best.total_return,
                    "sortino": best.sortino,
                    "intensity": best.intensity_scale,
                    "offset": best.price_offset_pct,
                    "min_edge": best.min_edge,
                    "risk_weight": best.risk_weight,
                    "max_volume_fraction": best.max_volume_fraction,
                    "final_cash": best.final_cash,
                    "open_symbol": best.open_symbol,
                }
            ),
        )
    print("top10:")
    for row in top:
        print(json.dumps(asdict(row)))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sweep_results.json").write_text(json.dumps([asdict(r) for r in results_sorted], indent=2) + "\n")
        if best is not None:
            (out_dir / "sweep_best.json").write_text(json.dumps(asdict(best), indent=2) + "\n")


if __name__ == "__main__":
    main()

