from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.symbol_utils import is_crypto_symbol
from src.torch_device_utils import require_cuda as require_cuda_device
from src.torch_load_utils import torch_load_compat

from .config import DatasetConfig, ExperimentConfig
from .data import AlpacaHourlyDataModule
from .inference import generate_actions_multi_context
from .marketsimulator.hourly_trader import HourlyTraderMarketSimulator, HourlyTraderSimulationConfig, OpenOrder
from src.hourly_trader_utils import infer_working_order_kind


def _parse_symbols(raw: Optional[str]) -> list[str]:
    if not raw:
        return ["SOLUSD", "LINKUSD", "UNIUSD", "BTCUSD", "ETHUSD"]
    return [token.strip().upper() for token in raw.split(",") if token.strip()]


def _parse_symbol_list(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [token.strip().upper() for token in str(raw).split(",") if token.strip()]


def _parse_int_tuple(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        return None
    return tuple(int(v) for v in values)


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type != "cuda":
            raise RuntimeError(f"GPU required for hourly trader sim inference; received device={device_arg!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for hourly trader sim inference but CUDA is not available.")
        return device
    return require_cuda_device("hourly trader sim inference", allow_fallback=False)


def _load_model(checkpoint_path: Path, input_dim: int, sequence_length: int) -> torch.nn.Module:
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


def _slice_eval_window(
    actions: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    eval_days: Optional[float],
    eval_hours: Optional[float],
    ts_start: Optional[pd.Timestamp] = None,
    ts_end: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if actions.empty or bars.empty:
        return actions, bars

    if ts_start is None or ts_end is None:
        ts_end = pd.to_datetime(bars["timestamp"], utc=True).max()
        if pd.isna(ts_end):
            return actions, bars
        hours = 0.0
        if eval_days:
            hours = max(hours, float(eval_days) * 24.0)
        if eval_hours:
            hours = max(hours, float(eval_hours))
        if hours <= 0:
            return actions, bars
        ts_start = ts_end - pd.Timedelta(hours=hours)

    bars_ts = pd.to_datetime(bars["timestamp"], utc=True)
    actions_ts = pd.to_datetime(actions["timestamp"], utc=True)
    bars_slice = bars[(bars_ts >= ts_start) & (bars_ts <= ts_end)]
    actions_slice = actions[(actions_ts >= ts_start) & (actions_ts <= ts_end)]
    return actions_slice.reset_index(drop=True), bars_slice.reset_index(drop=True)


def _resolve_data_root(symbol: str, crypto_root: Optional[Path], stock_root: Optional[Path]) -> Optional[Path]:
    return crypto_root if is_crypto_symbol(symbol) else stock_root


def _normalize_symbol(value: object) -> str:
    return str(value or "").replace("/", "").replace("-", "").upper()


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return float(result)


def _coerce_ts(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    return pd.Timestamp(ts)


def _load_initial_state(path: Path) -> tuple[float, dict[str, float], list[OpenOrder]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"Initial state file must contain a JSON object: {path}")

    initial_cash = _coerce_float(
        payload.get("initial_cash", payload.get("cash", payload.get("starting_cash", 0.0))),
        default=0.0,
    )

    raw_positions = payload.get("positions", {})
    positions: dict[str, float] = {}
    if isinstance(raw_positions, Mapping):
        for raw_symbol, raw_qty in raw_positions.items():
            symbol = _normalize_symbol(raw_symbol)
            if not symbol:
                continue
            positions[symbol] = _coerce_float(raw_qty, default=0.0)
    elif isinstance(raw_positions, list):
        for row in raw_positions:
            if not isinstance(row, Mapping):
                continue
            symbol = _normalize_symbol(row.get("symbol", row.get("asset", "")))
            if not symbol:
                continue
            qty = _coerce_float(row.get("quantity", row.get("qty", row.get("position", 0.0))), default=0.0)
            positions[symbol] = qty
    else:
        raise ValueError(f"positions must be an object or list in {path}")

    open_orders: list[OpenOrder] = []
    raw_orders = payload.get("open_orders", payload.get("orders", []))
    if raw_orders is None:
        raw_orders = []
    if not isinstance(raw_orders, list):
        raise ValueError(f"open_orders must be a list in {path}")
    for row in raw_orders:
        if not isinstance(row, Mapping):
            continue
        symbol = _normalize_symbol(row.get("symbol", row.get("asset", "")))
        side = str(row.get("side", "") or "").strip().lower()
        qty = _coerce_float(row.get("quantity", row.get("qty", row.get("orig_qty", 0.0))), default=0.0)
        price = _coerce_float(row.get("limit_price", row.get("price", 0.0)), default=0.0)
        if not symbol or side not in {"buy", "sell"} or qty <= 0.0 or price <= 0.0:
            continue
        kind = str(row.get("kind", "") or "").strip().lower()
        if kind not in {"entry", "exit"}:
            kind = infer_working_order_kind(side=side, position_qty=float(positions.get(symbol, 0.0)))
        placed_at = _coerce_ts(row.get("placed_at", row.get("created_at", row.get("timestamp", row.get("submitted_at")))))
        cancel_requested_raw = row.get("cancel_requested_at")
        cancel_effective_raw = row.get("cancel_effective_at")
        cancel_requested_at = _coerce_ts(cancel_requested_raw) if cancel_requested_raw else None
        cancel_effective_at = _coerce_ts(cancel_effective_raw) if cancel_effective_raw else None
        open_orders.append(
            OpenOrder(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=price,
                kind=kind,
                placed_at=placed_at,
                reserved_cash=_coerce_float(row.get("reserved_cash", 0.0), default=0.0),
                cancel_requested_at=cancel_requested_at,
                cancel_effective_at=cancel_effective_at,
            )
        )

    return initial_cash, positions, open_orders


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the live hourly trader loop with shared-cash execution.")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
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
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--price-offset-pct", type=float, default=0.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.001)
    parser.add_argument(
        "--fill-buffer-bps",
        type=float,
        default=5.0,
        help="Require bar to trade through limit by this many bps before fill (realism control, default: 5).",
    )
    parser.add_argument(
        "--allow-position-adds",
        action="store_true",
        help="Allow same-side add orders while already in a position (legacy behavior).",
    )
    parser.add_argument(
        "--always-full-exit",
        dest="always_full_exit",
        action="store_true",
        help="Always quote full-position exits when a position is open (default).",
    )
    parser.add_argument(
        "--no-always-full-exit",
        dest="always_full_exit",
        action="store_false",
        help="Respect model sell_amount/buy_amount for partial exits when a position is open.",
    )
    parser.set_defaults(always_full_exit=True)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument(
        "--cancel-ack-delay-bars",
        type=int,
        default=1,
        help="Bars to wait for same-side cancel acknowledgement before replacement is allowed.",
    )
    parser.add_argument(
        "--partial-fill-on-touch",
        dest="partial_fill_on_touch",
        action="store_true",
        help="Allow partial fills when a limit is only lightly touched intrabar (default).",
    )
    parser.add_argument(
        "--no-partial-fill-on-touch",
        dest="partial_fill_on_touch",
        action="store_false",
        help="Use legacy all-or-nothing fills once a bar touches the limit trigger.",
    )
    parser.set_defaults(partial_fill_on_touch=True)
    parser.add_argument("--eval-days", type=float, default=None)
    parser.add_argument("--eval-hours", type=float, default=None)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--long-only-symbols", default=None, help="Comma-separated symbols to restrict to long-only.")
    parser.add_argument("--short-only-symbols", default=None, help="Comma-separated symbols to restrict to short-only.")
    parser.add_argument(
        "--drawdown-profit-early-exit",
        dest="drawdown_profit_early_exit",
        action="store_true",
        help="Enable generic drawdown-vs-profit early exit.",
    )
    parser.add_argument(
        "--no-drawdown-profit-early-exit",
        dest="drawdown_profit_early_exit",
        action="store_false",
        help="Disable generic drawdown-vs-profit early exit.",
    )
    parser.set_defaults(drawdown_profit_early_exit=True)
    parser.add_argument("--drawdown-profit-early-exit-min-steps", type=int, default=20)
    parser.add_argument("--drawdown-profit-early-exit-progress-fraction", type=float, default=0.5)
    parser.add_argument(
        "--baseline-comparability-early-exit",
        dest="baseline_comparability_early_exit",
        action="store_true",
        help="Enable staged early exit relative to supplied baseline metrics.",
    )
    parser.add_argument(
        "--no-baseline-comparability-early-exit",
        dest="baseline_comparability_early_exit",
        action="store_false",
        help="Disable staged early exit relative to supplied baseline metrics.",
    )
    parser.set_defaults(baseline_comparability_early_exit=False)
    parser.add_argument("--baseline-total-return", type=float, default=None)
    parser.add_argument("--baseline-sortino", type=float, default=None)
    parser.add_argument("--baseline-max-drawdown", type=float, default=None)
    parser.add_argument("--baseline-early-exit-min-steps", type=int, default=40)
    parser.add_argument("--baseline-stage1-progress", type=float, default=0.30)
    parser.add_argument("--baseline-stage2-progress", type=float, default=0.50)
    parser.add_argument("--baseline-stage3-progress", type=float, default=0.75)
    parser.add_argument("--baseline-return-tolerance", type=float, default=0.02)
    parser.add_argument("--baseline-sortino-tolerance", type=float, default=0.50)
    parser.add_argument("--baseline-max-drawdown-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--initial-state",
        default=None,
        help="Optional JSON file containing seeded initial cash, positions, and open orders for replay parity.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="Override device (cuda/cuda:0).")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    forecast_horizons = tuple(int(x) for x in str(args.forecast_horizons).split(",") if str(x).strip())
    context_lengths = tuple(int(x) for x in str(args.context_lengths).split(",") if str(x).strip())
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
    long_only_symbols = _parse_symbol_list(args.long_only_symbols)
    short_only_symbols = _parse_symbol_list(args.short_only_symbols)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    device = _resolve_device(args.device)

    all_bars = []
    all_actions = []
    model: Optional[torch.nn.Module] = None

    crypto_root = Path(args.crypto_data_root) if args.crypto_data_root else None
    stock_root = Path(args.stock_data_root) if args.stock_data_root else None
    forecast_cache_root = Path(args.forecast_cache_root)

    loaded: list[tuple[str, AlpacaHourlyDataModule, pd.DataFrame]] = []
    for symbol in symbols:
        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=_resolve_data_root(symbol, crypto_root, stock_root),
            forecast_cache_root=forecast_cache_root,
            sequence_length=int(args.sequence_length),
            forecast_horizons=forecast_horizons,
            cache_only=bool(args.cache_only),
            moving_average_windows=ma_windows,
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
        frame = data.frame.copy()
        loaded.append((symbol, data, frame))

    # Resolve a common evaluation window end timestamp for consistent slicing.
    ts_end_common = None
    for _, _, frame in loaded:
        ts = pd.to_datetime(frame["timestamp"], utc=True).max()
        if pd.isna(ts):
            continue
        ts_end_common = ts if ts_end_common is None else max(ts_end_common, ts)
    if ts_end_common is None:
        raise RuntimeError("Failed to infer a common ts_end from loaded frames.")

    eval_hours = 0.0
    if args.eval_days:
        eval_hours = max(eval_hours, float(args.eval_days) * 24.0)
    if args.eval_hours:
        eval_hours = max(eval_hours, float(args.eval_hours))
    ts_start_common = None
    if eval_hours > 0:
        ts_start_common = ts_end_common - pd.Timedelta(hours=eval_hours)

    # Generate actions on a trailing slice to keep inference cost bounded.
    rows_needed = None
    if eval_hours > 0:
        # Include enough warmup history so the first evaluation decision has context.
        rows_needed = int(math.ceil(eval_hours)) + int(args.sequence_length) + 10
    for symbol, data, frame in loaded:
        if rows_needed is not None and rows_needed > 0 and len(frame) > rows_needed:
            frame = frame.tail(rows_needed).reset_index(drop=True)
        bars = frame[["timestamp", "symbol", "high", "low", "close"]].copy()

        if model is None:
            model = _load_model(checkpoint, len(data.feature_columns), int(args.sequence_length))

        if context_lengths:
            agg = generate_actions_multi_context(
                model=model,
                frame=frame,
                feature_columns=data.feature_columns,
                normalizer=data.normalizer,
                base_sequence_length=int(args.sequence_length),
                horizon=int(args.horizon),
                experiment=experiment_cfg,
                device=device,
            )
            actions = agg.aggregated
        else:
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

    bars_all = pd.concat(all_bars, ignore_index=True)
    actions_all = pd.concat(all_actions, ignore_index=True)

    initial_cash = float(args.initial_cash)
    initial_positions: Optional[dict[str, float]] = None
    initial_open_orders: Optional[list[OpenOrder]] = None
    if args.initial_state:
        initial_cash, initial_positions, initial_open_orders = _load_initial_state(Path(args.initial_state))

    sim = HourlyTraderMarketSimulator(
        HourlyTraderSimulationConfig(
            initial_cash=float(initial_cash),
            initial_positions=initial_positions,
            initial_open_orders=initial_open_orders,
            allocation_usd=float(args.allocation_usd) if args.allocation_usd is not None else None,
            allocation_pct=float(args.allocation_pct) if args.allocation_pct is not None else None,
            allocation_mode=str(args.allocation_mode),
            intensity_scale=float(args.intensity_scale),
            price_offset_pct=float(args.price_offset_pct),
            min_gap_pct=float(args.min_gap_pct),
            fill_buffer_bps=float(args.fill_buffer_bps),
            allow_position_adds=bool(args.allow_position_adds),
            always_full_exit=bool(args.always_full_exit),
            decision_lag_bars=int(args.decision_lag_bars),
            cancel_ack_delay_bars=int(args.cancel_ack_delay_bars),
            partial_fill_on_touch=bool(args.partial_fill_on_touch),
            allow_short=bool(args.allow_short),
            long_only_symbols=long_only_symbols,
            short_only_symbols=short_only_symbols,
            enable_drawdown_profit_early_exit=bool(args.drawdown_profit_early_exit),
            drawdown_profit_early_exit_min_steps=int(args.drawdown_profit_early_exit_min_steps),
            drawdown_profit_early_exit_progress_fraction=float(args.drawdown_profit_early_exit_progress_fraction),
            enable_baseline_comparability_early_exit=bool(args.baseline_comparability_early_exit),
            baseline_total_return=args.baseline_total_return,
            baseline_sortino=args.baseline_sortino,
            baseline_max_drawdown=args.baseline_max_drawdown,
            baseline_early_exit_min_steps=int(args.baseline_early_exit_min_steps),
            baseline_stage1_progress=float(args.baseline_stage1_progress),
            baseline_stage2_progress=float(args.baseline_stage2_progress),
            baseline_stage3_progress=float(args.baseline_stage3_progress),
            baseline_return_tolerance=float(args.baseline_return_tolerance),
            baseline_sortino_tolerance=float(args.baseline_sortino_tolerance),
            baseline_max_drawdown_tolerance=float(args.baseline_max_drawdown_tolerance),
            symbols=[s.upper() for s in symbols],
        )
    )
    result = sim.run(bars_all, actions_all)
    print(json.dumps(result.metrics, indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bars.csv").write_text(bars_all.to_csv(index=False))
        (out_dir / "actions.csv").write_text(actions_all.to_csv(index=False))
        (out_dir / "per_hour.csv").write_text(result.per_hour.to_csv(index=False))
        fills_rows = [f.__dict__ for f in result.fills]
        (out_dir / "fills.csv").write_text(pd.DataFrame(fills_rows).to_csv(index=False))
        (out_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
