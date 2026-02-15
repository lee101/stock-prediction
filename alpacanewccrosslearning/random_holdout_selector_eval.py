from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from binanceexp1.sweep import apply_action_overrides
from newnanoalpacahourlyexp.config import DatasetConfig
from newnanoalpacahourlyexp.data import AlpacaHourlyDataModule
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation_merged
from src.metrics_utils import annualized_sortino
from src.tradinglib.metrics import max_drawdown, pnl_metrics
from src.torch_device_utils import require_cuda as require_cuda_device
from src.torch_load_utils import torch_load_compat


ACTION_COLS: Tuple[str, ...] = ("buy_price", "sell_price", "buy_amount", "sell_amount", "trade_amount")


@dataclass(frozen=True)
class WindowMetrics:
    start_ts: str
    end_ts: str
    num_timestamps: int
    total_return: float
    sortino_hourly: float
    max_drawdown: float
    pct_profitable_days: float
    sortino_daily_365: float
    final_equity: float
    num_trades: int


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    items = [int(x) for x in str(raw).split(",") if str(x).strip()]
    if not items:
        raise ValueError("Expected a comma-separated list of integers.")
    return tuple(items)


def _load_model(checkpoint_path: Path, *, input_dim: int, sequence_length: int) -> torch.nn.Module:
    payload = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", {})
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    policy_cfg.max_len = max(policy_cfg.max_len, int(sequence_length))
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


def _apply_decision_lag(
    merged: pd.DataFrame,
    *,
    horizon: int,
    decision_lag_bars: int,
) -> pd.DataFrame:
    """Shift decision-time inputs back by N bars per symbol.

    We pre-shift before window slicing so each window starts flat while still
    using lagged actions/forecasts (live-like: decision on bar t executes on bar t+N).
    """
    lag = int(decision_lag_bars)
    if lag <= 0:
        return merged
    merged = merged.copy()
    merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    forecast_cols = [
        f"predicted_high_p50_h{int(horizon)}",
        f"predicted_low_p50_h{int(horizon)}",
        f"predicted_close_p50_h{int(horizon)}",
    ]
    shift_cols = [c for c in (list(ACTION_COLS) + forecast_cols) if c in merged.columns]
    if not shift_cols:
        raise ValueError("No action/forecast columns found to shift for decision lag.")
    shifted = merged.groupby("symbol", sort=False)[shift_cols].shift(lag)
    for col in shift_cols:
        merged[col] = shifted[col]
    merged = merged.dropna(subset=shift_cols).reset_index(drop=True)
    return merged


def _window_summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _choose_random_windows(
    timestamps: pd.DatetimeIndex,
    *,
    window_hours: float,
    num_windows: int,
    seed: int,
    eligible_last_days: Optional[float] = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if timestamps.empty:
        raise ValueError("No timestamps available for sampling.")
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).dropna().sort_values()
    max_ts = timestamps.max()
    if pd.isna(max_ts):
        raise ValueError("Invalid timestamp range for sampling.")
    duration = pd.Timedelta(hours=float(window_hours))
    eligible = timestamps[timestamps <= (max_ts - duration)]
    if eligible_last_days is not None:
        days = float(eligible_last_days)
        if not np.isfinite(days) or days <= 0.0:
            raise ValueError(f"eligible_last_days must be > 0, got {eligible_last_days}.")
        earliest = max_ts - pd.Timedelta(days=days) - duration
        eligible = eligible[eligible >= earliest]
    if eligible.empty:
        raise ValueError(
            f"Window too long for available data: window_hours={window_hours}, "
            f"range_hours={(max_ts - timestamps.min()).total_seconds() / 3600.0:.1f}"
        )
    rng = np.random.default_rng(int(seed))
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for _ in range(int(num_windows)):
        start = eligible[int(rng.integers(0, len(eligible)))]
        end = start + duration
        windows.append((pd.Timestamp(start), pd.Timestamp(end)))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Random holdout evaluation for global selector (stocks + crypto).")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--use-full-frame", action="store_true", help="Sample windows from full lookback (not just val split).")
    parser.add_argument(
        "--max-feature-lookback-hours",
        type=int,
        default=None,
        help=(
            "Max lookback window for feature construction. Note: for Alpaca hourly datasets this is treated as a "
            "row cap (not literal hours), but it also controls the forecast start timestamp via end_ts - hours."
        ),
    )

    # Feature window overrides (must match training for apples-to-apples evaluation).
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

    # Action overrides / entry gating.
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--price-offset-pct", type=float, default=0.0)
    parser.add_argument("--dip-threshold-pct", type=float, default=0.0)

    # Selector realism + portfolio behaviour.
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--risk-weight", type=float, default=0.5)
    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--max-hold-hours", type=int, default=None)
    parser.add_argument("--allow-reentry-same-bar", action="store_true")
    parser.add_argument("--no-enforce-market-hours", action="store_true")
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument(
        "--select-fillable-only",
        action="store_true",
        help="Legacy optimistic selection: filter entry candidates by same-bar fillability at decision time.",
    )
    parser.add_argument("--max-volume-fraction", type=float, default=None)

    # Long/short + leverage realism.
    parser.add_argument("--allow-short", action="store_true", help="Allow selector to open stock shorts (crypto stays long-only).")
    parser.add_argument("--long-only-symbols", default=None)
    parser.add_argument("--short-only-symbols", default=None)
    parser.add_argument("--max-leverage-stock", type=float, default=1.0)
    parser.add_argument("--max-leverage-crypto", type=float, default=1.0)
    parser.add_argument("--margin-interest-annual", type=float, default=0.0)
    parser.add_argument("--short-borrow-cost-annual", type=float, default=0.0)
    parser.add_argument("--max-concurrent-positions", type=int, default=1)
    parser.add_argument("--work-steal-enabled", action="store_true")
    parser.add_argument("--work-steal-min-profit-pct", type=float, default=0.001)
    parser.add_argument("--work-steal-min-edge", type=float, default=0.005)
    parser.add_argument("--work-steal-edge-margin", type=float, default=0.0)

    # Random holdout settings.
    parser.add_argument("--window-hours", type=float, default=None)
    parser.add_argument("--window-days", type=float, default=30.0)
    parser.add_argument(
        "--eligible-last-days",
        type=float,
        default=None,
        help="If set, sample windows whose *end* falls within the last N days of available data.",
    )
    parser.add_argument("--num-windows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = require_cuda_device("random holdout selector eval", allow_fallback=False)
    symbols = _parse_symbols(args.symbols)
    forecast_horizons = _parse_int_tuple(args.forecast_horizons)

    if args.window_hours is not None:
        window_hours = float(args.window_hours)
    else:
        window_hours = float(args.window_days) * 24.0
    if window_hours <= 0:
        raise ValueError("window_hours must be > 0")

    # Dataset config overrides.
    if args.moving_average_windows:
        ma_windows = tuple(int(x) for x in args.moving_average_windows.split(",") if x.strip())
    else:
        ma_windows = DatasetConfig().moving_average_windows
    if args.ema_windows:
        ema_windows = tuple(int(x) for x in args.ema_windows.split(",") if x.strip())
    else:
        ema_windows = DatasetConfig().ema_windows
    if args.atr_windows:
        atr_windows = tuple(int(x) for x in args.atr_windows.split(",") if x.strip())
    else:
        atr_windows = DatasetConfig().atr_windows
    if args.trend_windows:
        trend_windows = tuple(int(x) for x in args.trend_windows.split(",") if x.strip())
    else:
        trend_windows = DatasetConfig().trend_windows
    if args.drawdown_windows:
        drawdown_windows = tuple(int(x) for x in args.drawdown_windows.split(",") if x.strip())
    else:
        drawdown_windows = DatasetConfig().drawdown_windows

    volume_z_window = args.volume_z_window if args.volume_z_window is not None else DatasetConfig().volume_z_window
    volume_shock_window = (
        args.volume_shock_window if args.volume_shock_window is not None else DatasetConfig().volume_shock_window
    )
    vol_regime_short = args.vol_regime_short if args.vol_regime_short is not None else DatasetConfig().vol_regime_short
    vol_regime_long = args.vol_regime_long if args.vol_regime_long is not None else DatasetConfig().vol_regime_long
    min_history_hours = args.min_history_hours if args.min_history_hours is not None else DatasetConfig().min_history_hours
    max_lookback = (
        int(args.max_feature_lookback_hours)
        if args.max_feature_lookback_hours is not None
        else DatasetConfig().max_feature_lookback_hours
    )

    bars_frames: List[pd.DataFrame] = []
    action_frames: List[pd.DataFrame] = []
    periods_by_symbol: Dict[str, float] = {}
    fee_by_symbol: Dict[str, float] = {}

    logger.info("Loading model: {}", args.checkpoint)
    model: Optional[torch.nn.Module] = None

    for idx, symbol in enumerate(symbols):
        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=Path(args.data_root) if args.data_root else None,
            forecast_cache_root=Path(args.forecast_cache_root),
            sequence_length=int(args.sequence_length),
            forecast_horizons=forecast_horizons,
            cache_only=bool(args.cache_only),
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
            max_feature_lookback_hours=int(max_lookback),
            allow_mixed_asset_class=True,  # evaluation can mix; direction constraints handled by selector config.
        )
        data = AlpacaHourlyDataModule(data_cfg)
        frame = data.frame.copy() if args.use_full_frame else data.val_dataset.frame.copy()
        frame = frame.reset_index(drop=True)
        periods_by_symbol[symbol] = float(data.asset_meta.periods_per_year)
        fee_by_symbol[symbol] = float(data.asset_meta.maker_fee)

        if model is None:
            model = _load_model(Path(args.checkpoint), input_dim=len(data.feature_columns), sequence_length=args.sequence_length)

        actions = generate_actions_from_frame(
            model=model,
            frame=frame,
            feature_columns=data.feature_columns,
            normalizer=data.normalizer,
            sequence_length=int(args.sequence_length),
            horizon=int(args.horizon),
            device=device,
            require_gpu=True,
        )

        if args.intensity_scale != 1.0 or args.price_offset_pct != 0.0:
            actions = apply_action_overrides(
                actions,
                intensity_scale=float(args.intensity_scale),
                price_offset_pct=float(args.price_offset_pct),
            )

        actions = _apply_dip_filter(actions, frame, float(args.dip_threshold_pct))

        bars_frames.append(frame)
        action_frames.append(actions)
        logger.info("Prepared {}: bars={} actions={}", symbol, len(frame), len(actions))

    if model is None:
        raise RuntimeError("Failed to load model.")

    bars = pd.concat(bars_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner")
    if merged.empty:
        raise ValueError("Merged dataframe is empty; actions do not cover bars.")

    merged = _apply_decision_lag(merged, horizon=int(args.horizon), decision_lag_bars=int(args.decision_lag_bars))

    # Sample windows from the merged+lagged timestamps.
    all_ts = pd.DatetimeIndex(pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")).dropna().unique()
    windows = _choose_random_windows(
        all_ts,
        window_hours=window_hours,
        num_windows=int(args.num_windows),
        seed=int(args.seed),
        eligible_last_days=float(args.eligible_last_days) if args.eligible_last_days is not None else None,
    )

    long_only_symbols = [token.strip().upper() for token in (args.long_only_symbols or "").split(",") if token.strip()]
    short_only_symbols = [token.strip().upper() for token in (args.short_only_symbols or "").split(",") if token.strip()]

    cfg = SelectionConfig(
        initial_cash=float(args.initial_cash),
        min_edge=float(args.min_edge),
        risk_weight=float(args.risk_weight),
        edge_mode=str(args.edge_mode),
        max_hold_hours=int(args.max_hold_hours) if args.max_hold_hours else None,
        allow_reentry_same_bar=bool(args.allow_reentry_same_bar),
        enforce_market_hours=not bool(args.no_enforce_market_hours),
        close_at_eod=not bool(args.no_close_at_eod),
        fee_by_symbol=fee_by_symbol,
        periods_per_year_by_symbol=periods_by_symbol,
        symbols=symbols,
        allow_short=bool(args.allow_short),
        long_only_symbols=long_only_symbols,
        short_only_symbols=short_only_symbols,
        max_leverage_stock=float(args.max_leverage_stock),
        max_leverage_crypto=float(args.max_leverage_crypto),
        margin_interest_annual=float(args.margin_interest_annual),
        short_borrow_cost_annual=float(args.short_borrow_cost_annual),
        max_concurrent_positions=int(args.max_concurrent_positions),
        work_steal_enabled=bool(args.work_steal_enabled),
        work_steal_min_profit_pct=float(args.work_steal_min_profit_pct),
        work_steal_min_edge=float(args.work_steal_min_edge),
        work_steal_edge_margin=float(args.work_steal_edge_margin),
        select_fillable_only=bool(args.select_fillable_only),
        max_volume_fraction=float(args.max_volume_fraction) if args.max_volume_fraction is not None else None,
        # We pre-shifted the frame for lagged execution; keep selector shift disabled per window.
        decision_lag_bars=0,
    )

    per_window: List[WindowMetrics] = []
    for start_ts, end_ts in windows:
        window_df = merged[
            (pd.to_datetime(merged["timestamp"], utc=True) >= start_ts) & (pd.to_datetime(merged["timestamp"], utc=True) <= end_ts)
        ].reset_index(drop=True)
        if window_df.empty:
            continue

        # Weighted periods-per-year for metrics (align with selector internal logic).
        counts = window_df["symbol"].astype(str).str.upper().value_counts()
        total = float(counts.sum()) if not counts.empty else 0.0
        weighted_periods = 24.0 * 365.0
        if total > 0:
            weighted_periods = float(
                sum(float(periods_by_symbol.get(sym, 24.0 * 365.0)) * float(cnt) for sym, cnt in counts.items()) / total
            )

        sim = run_best_trade_simulation_merged(window_df, cfg, horizon=int(args.horizon))
        eq = sim.equity_curve.astype(float)
        if eq.empty:
            continue

        hourly = pnl_metrics(equity_curve=eq.to_numpy(dtype=float), periods_per_year=weighted_periods)
        mdd = max_drawdown(eq.to_numpy(dtype=float))

        daily_eq = eq.resample("1D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
        pct_prof_days = float((daily_ret > 0).mean()) if len(daily_ret) else 0.0
        daily_sortino = float(annualized_sortino(daily_ret.to_numpy(dtype=float), periods_per_year=365.0))

        per_window.append(
            WindowMetrics(
                start_ts=str(pd.Timestamp(start_ts).isoformat()),
                end_ts=str(pd.Timestamp(end_ts).isoformat()),
                num_timestamps=int(eq.size),
                total_return=float(hourly.total_return),
                sortino_hourly=float(hourly.sortino),
                max_drawdown=float(mdd),
                pct_profitable_days=pct_prof_days,
                sortino_daily_365=daily_sortino,
                final_equity=float(eq.iloc[-1]),
                num_trades=int(len(sim.trades)),
            )
        )

    if not per_window:
        raise RuntimeError("No evaluation windows produced metrics (check window length and available data).")

    output_root = Path(args.output_dir) if args.output_dir else Path("experiments") / (
        f"holdout_selector_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    windows_csv = output_root / "windows.csv"
    pd.DataFrame([asdict(row) for row in per_window]).to_csv(windows_csv, index=False)

    summary = {
        "symbols": symbols,
        "checkpoint": str(Path(args.checkpoint)),
        "config": {
            "sequence_length": int(args.sequence_length),
            "horizon": int(args.horizon),
            "forecast_horizons": list(forecast_horizons),
            "window_hours": float(window_hours),
            "num_windows": int(args.num_windows),
            "seed": int(args.seed),
            "decision_lag_bars": int(args.decision_lag_bars),
            "select_fillable_only": bool(args.select_fillable_only),
            "work_steal_enabled": bool(args.work_steal_enabled),
            "allow_short": bool(args.allow_short),
            "long_only_symbols": long_only_symbols,
            "short_only_symbols": short_only_symbols,
        },
        "metrics": {
            "total_return": _window_summary([w.total_return for w in per_window]),
            "sortino_hourly": _window_summary([w.sortino_hourly for w in per_window]),
            "max_drawdown": _window_summary([w.max_drawdown for w in per_window]),
            "pct_profitable_days": _window_summary([w.pct_profitable_days for w in per_window]),
            "sortino_daily_365": _window_summary([w.sortino_daily_365 for w in per_window]),
        },
        "artifacts": {
            "windows_csv": str(windows_csv),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info("Saved windows: {}", windows_csv)
    logger.info("Saved summary: {}", summary_path)

    # Print condensed summary for quick copy/paste into progress logs.
    metrics = summary["metrics"]
    print("\n=== RANDOM HOLDOUT SUMMARY ===")
    print(f"output_dir: {output_root}")
    print(f"windows: {len(per_window)} | window_hours={window_hours:.1f} | seed={args.seed}")
    print(f"total_return median={metrics['total_return']['median']:+.4f} p10={metrics['total_return']['p10']:+.4f}")
    print(f"sortino_hourly median={metrics['sortino_hourly']['median']:+.2f} p10={metrics['sortino_hourly']['p10']:+.2f}")
    print(f"max_drawdown median={metrics['max_drawdown']['median']:+.4f} p90={metrics['max_drawdown']['p90']:+.4f}")
    print(f"pct_profitable_days median={metrics['pct_profitable_days']['median']:.3f}")
    print(f"sortino_daily_365 median={metrics['sortino_daily_365']['median']:+.2f}")


if __name__ == "__main__":
    main()
