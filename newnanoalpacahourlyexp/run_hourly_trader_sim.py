from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.allocation_utils import allocation_usd_for_symbol
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


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    initial_cash: float
    initial_positions: dict[str, float]
    initial_open_orders: tuple[OpenOrder, ...] = ()


def _first_close_by_symbol(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {}
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.sort_values(["timestamp", "symbol"]).drop_duplicates(subset=["symbol"], keep="first")
    closes: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        close_price = _coerce_float(getattr(row, "close", 0.0), default=0.0)
        if close_price <= 0.0:
            continue
        closes[str(row.symbol).upper()] = close_price
    return closes


def _seed_notional_for_symbol(
    symbol: str,
    *,
    initial_cash: float,
    allocation_usd: Optional[float],
    allocation_pct: Optional[float],
    allocation_mode: str,
    symbols_count: int,
) -> float:
    account_view = SimpleNamespace(
        cash=float(initial_cash),
        equity=float(initial_cash),
        buying_power=float(initial_cash),
    )
    allocation = allocation_usd_for_symbol(
        account_view,
        symbol=symbol,
        allocation_usd=allocation_usd,
        allocation_pct=allocation_pct,
        allocation_mode=allocation_mode,
        symbols_count=max(1, int(symbols_count)),
        prefer_cash_for_crypto=True,
    )
    if allocation is not None and allocation > 0.0:
        return float(allocation)
    return max(0.0, float(initial_cash) / max(1, int(symbols_count)))


def _scenario_positions_for_symbols(
    symbols: Sequence[str],
    close_by_symbol: Mapping[str, float],
    *,
    initial_cash: float,
    allocation_usd: Optional[float],
    allocation_pct: Optional[float],
    allocation_mode: str,
    side_sign: float,
) -> dict[str, float]:
    positions: dict[str, float] = {}
    symbols_count = len(symbols)
    for symbol in symbols:
        close_price = _coerce_float(close_by_symbol.get(symbol), default=0.0)
        if close_price <= 0.0:
            continue
        seed_notional = _seed_notional_for_symbol(
            symbol,
            initial_cash=initial_cash,
            allocation_usd=allocation_usd,
            allocation_pct=allocation_pct,
            allocation_mode=allocation_mode,
            symbols_count=symbols_count,
        )
        if seed_notional <= 0.0:
            continue
        positions[symbol] = float(side_sign) * (seed_notional / close_price)
    return positions


def _add_unique_scenario(
    scenarios: list[SimulationScenario],
    seen: set[tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str, float, float, str], ...], float]],
    scenario: SimulationScenario,
) -> None:
    positions_key = tuple(sorted((str(symbol).upper(), float(qty)) for symbol, qty in scenario.initial_positions.items()))
    orders_key = tuple(
        sorted(
            (
                str(order.symbol).upper(),
                str(order.side).lower(),
                float(order.qty),
                float(order.limit_price),
                str(order.kind).lower(),
            )
            for order in scenario.initial_open_orders
        )
    )
    key = (positions_key, orders_key, float(scenario.initial_cash))
    if key in seen:
        return
    seen.add(key)
    scenarios.append(scenario)


def _build_starting_position_scenarios(
    *,
    bars: pd.DataFrame,
    symbols: Sequence[str],
    initial_cash: float,
    allocation_usd: Optional[float],
    allocation_pct: Optional[float],
    allocation_mode: str,
    allow_short: bool,
    initial_positions: Optional[Mapping[str, float]] = None,
    initial_open_orders: Optional[Sequence[OpenOrder]] = None,
    symbol_limit: int = 3,
) -> list[SimulationScenario]:
    normalized_symbols = [str(symbol).upper() for symbol in symbols]
    close_by_symbol = _first_close_by_symbol(bars)
    seed_symbols = [symbol for symbol in normalized_symbols if symbol in close_by_symbol]
    if symbol_limit > 0:
        seed_symbols = seed_symbols[: int(symbol_limit)]

    scenarios: list[SimulationScenario] = []
    seen: set[tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str, float, float, str], ...], float]] = set()

    _add_unique_scenario(
        scenarios,
        seen,
        SimulationScenario(
            name="flat",
            initial_cash=float(initial_cash),
            initial_positions={},
        ),
    )

    provided_positions = {
        _normalize_symbol(symbol): float(qty)
        for symbol, qty in (initial_positions or {}).items()
        if _normalize_symbol(symbol)
    }
    provided_orders = tuple(initial_open_orders or ())
    if provided_positions or provided_orders:
        _add_unique_scenario(
            scenarios,
            seen,
            SimulationScenario(
                name="provided_state",
                initial_cash=float(initial_cash),
                initial_positions=provided_positions,
                initial_open_orders=provided_orders,
            ),
        )

    if not seed_symbols:
        return scenarios

    basket_long = _scenario_positions_for_symbols(
        seed_symbols,
        close_by_symbol,
        initial_cash=initial_cash,
        allocation_usd=allocation_usd,
        allocation_pct=allocation_pct,
        allocation_mode=allocation_mode,
        side_sign=1.0,
    )
    if basket_long:
        _add_unique_scenario(
            scenarios,
            seen,
            SimulationScenario(
                name="basket_long",
                initial_cash=float(initial_cash),
                initial_positions=basket_long,
            ),
        )

    for symbol in seed_symbols:
        single_long = _scenario_positions_for_symbols(
            [symbol],
            close_by_symbol,
            initial_cash=initial_cash,
            allocation_usd=allocation_usd,
            allocation_pct=allocation_pct,
            allocation_mode=allocation_mode,
            side_sign=1.0,
        )
        if single_long:
            _add_unique_scenario(
                scenarios,
                seen,
                SimulationScenario(
                    name=f"long_{symbol}",
                    initial_cash=float(initial_cash),
                    initial_positions=single_long,
                ),
            )

    if not allow_short:
        return scenarios

    basket_short = _scenario_positions_for_symbols(
        seed_symbols,
        close_by_symbol,
        initial_cash=initial_cash,
        allocation_usd=allocation_usd,
        allocation_pct=allocation_pct,
        allocation_mode=allocation_mode,
        side_sign=-1.0,
    )
    if basket_short:
        _add_unique_scenario(
            scenarios,
            seen,
            SimulationScenario(
                name="basket_short",
                initial_cash=float(initial_cash),
                initial_positions=basket_short,
            ),
        )

    for symbol in seed_symbols:
        single_short = _scenario_positions_for_symbols(
            [symbol],
            close_by_symbol,
            initial_cash=initial_cash,
            allocation_usd=allocation_usd,
            allocation_pct=allocation_pct,
            allocation_mode=allocation_mode,
            side_sign=-1.0,
        )
        if single_short:
            _add_unique_scenario(
                scenarios,
                seen,
                SimulationScenario(
                    name=f"short_{symbol}",
                    initial_cash=float(initial_cash),
                    initial_positions=single_short,
                ),
            )

    return scenarios


def _summarize_scenario_results(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    serialized = [dict(result) for result in results]
    if not serialized:
        return {"scenario_count": 0, "best_scenario": None, "scenarios": []}

    def _score(item: Mapping[str, object]) -> tuple[float, float, float]:
        metrics = item.get("metrics", {})
        if not isinstance(metrics, Mapping):
            return (float("-inf"), float("-inf"), float("-inf"))
        sortino = _coerce_float(metrics.get("sortino"), default=float("-inf"))
        total_return = _coerce_float(metrics.get("total_return"), default=float("-inf"))
        max_drawdown = _coerce_float(metrics.get("max_drawdown"), default=float("-inf"))
        return (sortino, total_return, max_drawdown)

    ranked = sorted(serialized, key=_score, reverse=True)
    best = ranked[0]
    return {
        "scenario_count": len(ranked),
        "best_scenario": best.get("scenario"),
        "best_metrics": best.get("metrics", {}),
        "scenarios": ranked,
    }


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
        "--allow-short",
        action="store_true",
        help="Allow short entries in the simulator and include short seeded scenarios for multi-start runs.",
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
    parser.add_argument(
        "--initial-state",
        default=None,
        help="Optional JSON file containing seeded initial cash, positions, and open orders for replay parity.",
    )
    parser.add_argument(
        "--multi-start",
        action="store_true",
        help="Evaluate flat plus seeded starting-position scenarios instead of a single starting state.",
    )
    parser.add_argument(
        "--starting-position-symbol-limit",
        type=int,
        default=3,
        help="Maximum number of symbols to use when generating seeded starting-position scenarios (default: 3).",
    )
    parser.add_argument(
        "--entry-near-book-bps",
        type=float,
        default=25.0,
        help="Require entry limits to be within this many bps of the current bar close before placing them (default: 25).",
    )
    parser.add_argument(
        "--early-stop-min-periods",
        type=int,
        default=None,
        help="If set, stop a scenario once it has run this many periods and PnL still does not beat drawdown.",
    )
    parser.add_argument(
        "--early-stop-pnl-vs-drawdown-multiple",
        type=float,
        default=1.0,
        help="PnL must exceed this multiple of absolute max drawdown to continue once early-stop checks begin.",
    )
    parser.add_argument(
        "--robust-60d",
        action="store_true",
        help="Shortcut for a 60-day multi-start evaluation with the 30-period PnL-vs-drawdown early-stop enabled.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="Override device (cuda/cuda:0).")
    args = parser.parse_args()

    eval_days = args.eval_days
    eval_hours_arg = args.eval_hours
    multi_start = bool(args.multi_start)
    early_stop_min_periods = args.early_stop_min_periods
    if args.robust_60d:
        if eval_days is None and eval_hours_arg is None:
            eval_days = 60.0
        multi_start = True
        if early_stop_min_periods is None:
            early_stop_min_periods = 30

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
    if eval_days:
        eval_hours = max(eval_hours, float(eval_days) * 24.0)
    if eval_hours_arg:
        eval_hours = max(eval_hours, float(eval_hours_arg))
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
    initial_positions: dict[str, float] = {}
    initial_open_orders: list[OpenOrder] = []
    if args.initial_state:
        initial_cash, initial_positions, initial_open_orders = _load_initial_state(Path(args.initial_state))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bars.csv").write_text(bars_all.to_csv(index=False))
        (out_dir / "actions.csv").write_text(actions_all.to_csv(index=False))

    allocation_usd = float(args.allocation_usd) if args.allocation_usd is not None else None
    allocation_pct = float(args.allocation_pct) if args.allocation_pct is not None else None
    scenario_configs = (
        _build_starting_position_scenarios(
            bars=bars_all,
            symbols=symbols,
            initial_cash=float(initial_cash),
            allocation_usd=allocation_usd,
            allocation_pct=allocation_pct,
            allocation_mode=str(args.allocation_mode),
            allow_short=bool(args.allow_short),
            initial_positions=initial_positions,
            initial_open_orders=initial_open_orders,
            symbol_limit=max(0, int(args.starting_position_symbol_limit)),
        )
        if multi_start
        else [
            SimulationScenario(
                name="provided_state" if (initial_positions or initial_open_orders) else "flat",
                initial_cash=float(initial_cash),
                initial_positions=dict(initial_positions),
                initial_open_orders=tuple(initial_open_orders),
            )
        ]
    )

    scenario_results: list[dict[str, object]] = []
    single_result = None
    for scenario in scenario_configs:
        sim = HourlyTraderMarketSimulator(
            HourlyTraderSimulationConfig(
                initial_cash=float(scenario.initial_cash),
                initial_positions=dict(scenario.initial_positions),
                initial_open_orders=list(scenario.initial_open_orders),
                allocation_usd=allocation_usd,
                allocation_pct=allocation_pct,
                allocation_mode=str(args.allocation_mode),
                intensity_scale=float(args.intensity_scale),
                price_offset_pct=float(args.price_offset_pct),
                min_gap_pct=float(args.min_gap_pct),
                fill_buffer_bps=float(args.fill_buffer_bps),
                allow_short=bool(args.allow_short),
                allow_position_adds=bool(args.allow_position_adds),
                always_full_exit=bool(args.always_full_exit),
                decision_lag_bars=int(args.decision_lag_bars),
                cancel_ack_delay_bars=int(args.cancel_ack_delay_bars),
                partial_fill_on_touch=bool(args.partial_fill_on_touch),
                entry_near_book_bps=float(args.entry_near_book_bps) if args.entry_near_book_bps is not None else None,
                early_stop_min_periods=early_stop_min_periods,
                early_stop_pnl_vs_drawdown_multiple=float(args.early_stop_pnl_vs_drawdown_multiple),
                symbols=[s.upper() for s in symbols],
            )
        )
        result = sim.run(bars_all, actions_all)
        if len(scenario_configs) == 1:
            single_result = result
        scenario_results.append(
            {
                "scenario": scenario.name,
                "initial_cash": float(scenario.initial_cash),
                "initial_positions": {k: float(v) for k, v in scenario.initial_positions.items()},
                "initial_open_orders": int(len(scenario.initial_open_orders)),
                "metrics": dict(result.metrics),
                "terminated_early": bool(result.terminated_early),
                "termination_reason": result.termination_reason,
            }
        )

        if args.output_dir:
            scenario_dir = out_dir / scenario.name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            (scenario_dir / "per_hour.csv").write_text(result.per_hour.to_csv(index=False))
            fills_rows = [f.__dict__ for f in result.fills]
            (scenario_dir / "fills.csv").write_text(pd.DataFrame(fills_rows).to_csv(index=False))
            (scenario_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))

    if len(scenario_results) == 1 and single_result is not None:
        print(json.dumps(single_result.metrics, indent=2))
        if args.output_dir:
            (out_dir / "per_hour.csv").write_text(single_result.per_hour.to_csv(index=False))
            fills_rows = [f.__dict__ for f in single_result.fills]
            (out_dir / "fills.csv").write_text(pd.DataFrame(fills_rows).to_csv(index=False))
            (out_dir / "metrics.json").write_text(json.dumps(single_result.metrics, indent=2))
    else:
        summary = _summarize_scenario_results(scenario_results)
        print(json.dumps(summary, indent=2))
        if args.output_dir:
            (out_dir / "scenario_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
