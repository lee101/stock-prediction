from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

import alpaca_wrapper
from binanceneural.inference import generate_actions_from_frame, generate_latest_action
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.allocation_utils import allocation_usd_for_symbol
from src.hourly_trader_utils import (
    OrderIntent,
    TradingPlan,
    build_order_intents,
    build_plan_from_action,
    ensure_valid_levels,
)
from src.hourly_order_reconcile import orders_to_cancel_for_live_symbol
from src.hourly_data_refresh import HourlyDataRefresher
from src.hourly_data_utils import HourlyDataValidator
from src.symbol_utils import is_crypto_symbol
from src.trade_directions import resolve_trade_directions
from src.torch_device_utils import require_cuda as require_cuda_device
from src.torch_load_utils import torch_load_compat

from .config import DatasetConfig, ExperimentConfig
from .data import AlpacaHourlyDataModule
from .inference import aggregate_actions, generate_actions_multi_context


logger = logging.getLogger("alpaca_hourly_trader")


def _parse_symbols(raw: Optional[str]) -> list[str]:
    if not raw:
        return ["SOLUSD", "LINKUSD", "UNIUSD"]
    return [token.strip().upper() for token in raw.split(",") if token.strip()]


def _parse_checkpoint_map(raw: Optional[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not raw:
        return mapping
    for item in raw.split(","):
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Checkpoint map must be in SYMBOL=path format")
        symbol, path = item.split("=", 1)
        symbol = symbol.strip().upper()
        if not symbol or not path:
            continue
        mapping[symbol] = Path(path).expanduser().resolve()
    return mapping


def _parse_horizon_map(raw: Optional[str]) -> Dict[str, Tuple[int, ...]]:
    mapping: Dict[str, Tuple[int, ...]] = {}
    if not raw:
        return mapping
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Forecast horizons map must be SYMBOL=1,24;SYMBOL=1 format")
        symbol, horizons = item.split("=", 1)
        symbol = symbol.strip().upper()
        horizon_list = tuple(int(x) for x in horizons.split(",") if x.strip())
        if not symbol or not horizon_list:
            continue
        mapping[symbol] = horizon_list
    return mapping


def _parse_int_tuple(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        return None
    return tuple(int(v) for v in values)


def _seconds_until_next_hour(buffer_seconds: int = 30) -> float:
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    delta = (next_hour + timedelta(seconds=buffer_seconds) - now).total_seconds()
    return max(1.0, delta)


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type != "cuda":
            raise RuntimeError(f"GPU required for Alpaca trading; received device={device_arg!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for Alpaca trading but CUDA is not available.")
        return device
    return require_cuda_device("alpaca hourly trading", allow_fallback=False)


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


def _latest_action_aggregate(
    *,
    model: torch.nn.Module,
    data: AlpacaHourlyDataModule,
    horizon: int,
    experiment_cfg: ExperimentConfig,
    device: torch.device,
) -> Optional[dict]:
    agg = generate_actions_multi_context(
        model=model,
        frame=data.frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        base_sequence_length=data.config.sequence_length,
        horizon=horizon,
        experiment=experiment_cfg,
        device=device,
    )
    if agg.aggregated.empty:
        return None
    latest = agg.aggregated.iloc[-1].to_dict()
    latest["timestamp"] = data.frame["timestamp"].iloc[-1]
    latest["symbol"] = data.frame["symbol"].iloc[-1]
    return latest


def _latest_action_single(
    *,
    model: torch.nn.Module,
    data: AlpacaHourlyDataModule,
    horizon: int,
    device: torch.device,
) -> dict:
    return generate_latest_action(
        model=model,
        frame=data.frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=data.config.sequence_length,
        horizon=horizon,
        device=device,
        require_gpu=True,
    )


def _resolve_data_root(symbol: str, crypto_root: Optional[Path], stock_root: Optional[Path]) -> Optional[Path]:
    if is_crypto_symbol(symbol):
        return crypto_root
    return stock_root


def _allocation_usd(
    account,
    *,
    allocation_usd: Optional[float],
    allocation_pct: Optional[float],
) -> Optional[float]:
    if allocation_usd is not None:
        return float(allocation_usd)
    if allocation_pct is None:
        return None
    buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
    equity = float(getattr(account, "equity", 0.0) or 0.0)
    base = buying_power if buying_power > 0 else equity
    return max(0.0, float(allocation_pct) * base)


def _find_position(positions, symbol: str):
    for position in positions:
        if str(getattr(position, "symbol", "")).upper() == symbol:
            return position
    return None


def _normalize_live_symbol(symbol: object) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").upper()


def _order_identity(order: object) -> str:
    order_id = str(getattr(order, "id", "") or "")
    if order_id:
        return order_id
    return f"anon:{id(order)}"


def _reconcile_live_symbol_orders(
    symbol: str,
    *,
    position_qty: float,
    intents: Sequence[OrderIntent],
    open_orders: Sequence[object],
    dry_run: bool,
) -> list[object]:
    normalized_symbol = _normalize_live_symbol(symbol)
    symbol_orders = [
        order
        for order in open_orders
        if _normalize_live_symbol(getattr(order, "symbol", "")) == normalized_symbol
    ]
    if not symbol_orders:
        return []

    orders_with_reasons = orders_to_cancel_for_live_symbol(
        symbol_orders,
        position_qty=float(position_qty),
        intents=intents,
    )
    if not orders_with_reasons:
        return symbol_orders

    if dry_run:
        for order, reason in orders_with_reasons:
            logger.info(
                "Dry-run: would cancel stale %s order for %s id=%s side=%s qty=%s @ %s",
                reason,
                normalized_symbol,
                getattr(order, "id", ""),
                getattr(order, "side", ""),
                getattr(order, "qty", ""),
                getattr(order, "limit_price", ""),
            )
        return symbol_orders

    cancelled_ids: set[str] = set()
    for order, reason in orders_with_reasons:
        logger.info(
            "Cancelling stale %s order for %s id=%s side=%s qty=%s @ %s",
            reason,
            normalized_symbol,
            getattr(order, "id", ""),
            getattr(order, "side", ""),
            getattr(order, "qty", ""),
            getattr(order, "limit_price", ""),
        )
        try:
            alpaca_wrapper.cancel_order(order)
        except Exception as exc:
            logger.warning(
                "Failed to cancel stale order for %s id=%s: %s",
                normalized_symbol,
                getattr(order, "id", ""),
                exc,
            )
            continue
        cancelled_ids.add(_order_identity(order))

    return [order for order in symbol_orders if _order_identity(order) not in cancelled_ids]




def _run_cycle(
    symbols: Iterable[str],
    checkpoint_map: Dict[str, Path],
    default_checkpoint: Optional[Path],
    *,
    horizon: int,
    sequence_length: int,
    intensity_scale: float,
    price_offset_pct: float,
    min_gap_pct: float,
    allocation_usd: Optional[float],
    allocation_pct: Optional[float],
    allocation_mode: str,
    forecast_horizons_default: Tuple[int, ...],
    forecast_horizons_map: Dict[str, Tuple[int, ...]],
    context_lengths: Tuple[int, ...],
    trim_ratio: float,
    cache_only: bool,
    crypto_data_root: Optional[Path],
    stock_data_root: Optional[Path],
    forecast_cache_root: Path,
    allow_short: bool,
    long_only_symbols: Sequence[str],
    short_only_symbols: Sequence[str],
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
    refresher: HourlyDataRefresher,
    device: torch.device,
    exit_only_symbols: Sequence[str],
    allow_position_adds: bool,
    always_full_exit: bool,
    dry_run: bool,
) -> None:
    symbols_list = [str(sym).upper() for sym in symbols]
    refresher.refresh(list(symbols_list))
    account = alpaca_wrapper.get_account(use_cache=False)
    positions = alpaca_wrapper.get_all_positions()

    exit_only_set = {s.upper() for s in exit_only_symbols}
    experiment_cfg = ExperimentConfig(context_lengths=context_lengths, trim_ratio=trim_ratio)

    for symbol in symbols_list:
        symbol = symbol.upper()
        directions = resolve_trade_directions(
            symbol,
            allow_short=bool(allow_short),
            long_only_symbols=long_only_symbols,
            short_only_symbols=short_only_symbols,
            use_default_groups=True,
        )
        direction_conflict = not directions.can_long and not directions.can_short
        if direction_conflict:
            logger.warning("Conflicting direction constraints for %s; forcing exit-only.", symbol)
        if not is_crypto_symbol(symbol):
            clock = alpaca_wrapper.get_clock()
            if not getattr(clock, "is_open", True):
                logger.info("Market closed; skipping %s", symbol)
                continue

        checkpoint = checkpoint_map.get(symbol) or default_checkpoint
        if checkpoint is None:
            logger.warning("No checkpoint provided for %s; skipping", symbol)
            continue
        if not checkpoint.exists():
            logger.warning("Checkpoint missing for %s: %s", symbol, checkpoint)
            continue

        horizons = forecast_horizons_map.get(symbol, forecast_horizons_default)
        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=_resolve_data_root(symbol, crypto_data_root, stock_data_root),
            forecast_cache_root=forecast_cache_root,
            sequence_length=sequence_length,
            forecast_horizons=horizons,
            cache_only=cache_only,
            moving_average_windows=moving_average_windows,
            ema_windows=ema_windows,
            atr_windows=atr_windows,
            trend_windows=trend_windows,
            drawdown_windows=drawdown_windows,
            volume_z_window=volume_z_window,
            volume_shock_window=volume_shock_window,
            vol_regime_short=vol_regime_short,
            vol_regime_long=vol_regime_long,
            min_history_hours=min_history_hours,
        )
        data = AlpacaHourlyDataModule(data_cfg)
        model = _load_model(checkpoint, len(data.feature_columns), sequence_length)

        if context_lengths:
            action = _latest_action_aggregate(
                model=model,
                data=data,
                horizon=horizon,
                experiment_cfg=experiment_cfg,
                device=device,
            )
            if action is None:
                logger.warning("No aggregated actions for %s", symbol)
                continue
        else:
            action = _latest_action_single(model=model, data=data, horizon=horizon, device=device)

        plan = build_plan_from_action(action, intensity_scale=intensity_scale)
        buy_price = plan.buy_price * (1.0 - price_offset_pct)
        sell_price = plan.sell_price * (1.0 + price_offset_pct)
        adjusted = ensure_valid_levels(buy_price, sell_price, min_gap_pct=min_gap_pct)
        if adjusted is None:
            logger.warning("Invalid price levels for %s (buy=%.4f sell=%.4f)", symbol, buy_price, sell_price)
            continue
        buy_price, sell_price = adjusted

        position = _find_position(positions, symbol)
        position_qty = float(getattr(position, "qty", 0.0) or 0.0) if position else 0.0
        allocation = allocation_usd_for_symbol(
            account,
            symbol=symbol,
            allocation_usd=allocation_usd,
            allocation_pct=allocation_pct,
            allocation_mode=allocation_mode,
            symbols_count=len(symbols_list),
            prefer_cash_for_crypto=True,
        )
        exit_only = symbol in exit_only_set
        if position_qty > 0 and not directions.can_long:
            exit_only = True
        if position_qty < 0 and not directions.can_short:
            exit_only = True
        if direction_conflict:
            exit_only = True

        intents = build_order_intents(
            plan,
            position_qty=position_qty,
            allocation_usd=allocation,
            buy_price=buy_price,
            sell_price=sell_price,
            can_long=bool(directions.can_long),
            can_short=bool(directions.can_short),
            allow_short=bool(allow_short),
            exit_only=exit_only,
            allow_position_adds=bool(allow_position_adds),
            always_full_exit=bool(always_full_exit),
        )
        if exit_only:
            logger.info("Exit-only mode for %s: %d intent(s)", symbol, len(intents))

        if not dry_run:
            try:
                current_open_orders = list(alpaca_wrapper.get_orders())
            except Exception as exc:
                logger.warning("Failed to fetch open orders for %s before reconcile: %s", symbol, exc)
                current_open_orders = []
            _reconcile_live_symbol_orders(
                symbol,
                position_qty=position_qty,
                intents=intents,
                open_orders=current_open_orders,
                dry_run=dry_run,
            )

        logger.info(
            "%s action buy=%.4f sell=%.4f buy_amt=%.2f sell_amt=%.2f pos_qty=%.6f can_long=%s can_short=%s intents=%s",
            symbol,
            buy_price,
            sell_price,
            plan.buy_amount,
            plan.sell_amount,
            position_qty,
            directions.can_long,
            directions.can_short,
            [(i.kind, i.side, round(i.qty, 6)) for i in intents],
        )

        if dry_run:
            continue

        for intent in intents:
            if intent.qty <= 0:
                continue
            alpaca_wrapper.open_order_at_price_or_all(symbol, float(intent.qty), intent.side, float(intent.limit_price))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly Alpaca trading loop (Chronos2 + Binance neural policy).")
    parser.add_argument("--symbols", default="SOLUSD,LINKUSD,UNIUSD")
    parser.add_argument("--checkpoints", default=None, help="Comma-separated SYMBOL=path map.")
    parser.add_argument("--default-checkpoint", default=None)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument(
        "--forecast-horizons-map",
        default=None,
        help="Per-symbol horizons, e.g. SOLUSD=1,24;LINKUSD=1,24;UNIUSD=1",
    )
    parser.add_argument("--context-lengths", default="64,96,192")
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--price-offset-pct", type=float, default=0.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.001)
    parser.add_argument("--allocation-usd", type=float, default=None)
    parser.add_argument("--allocation-pct", type=float, default=0.05)
    parser.add_argument(
        "--allocation-mode",
        choices=("per_symbol", "portfolio"),
        default="per_symbol",
        help=(
            "How to interpret allocation_pct. 'per_symbol' applies allocation_pct to each symbol independently "
            "(legacy). 'portfolio' applies allocation_pct to the account base and splits evenly across symbols."
        ),
    )
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
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
    parser.add_argument("--exit-only-symbols", default="", help="Comma-separated symbols to only exit/close.")
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow opening short positions for stocks in live trading (crypto remains long-only).",
    )
    parser.add_argument("--long-only-symbols", default=None, help="Comma-separated symbols to restrict to long-only.")
    parser.add_argument("--short-only-symbols", default=None, help="Comma-separated symbols to restrict to short-only.")
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
    parser.add_argument("--device", default=None, help="Override device (cuda/cuda:0).")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--buffer-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = _parse_symbols(args.symbols)
    checkpoint_map = _parse_checkpoint_map(args.checkpoints)
    default_checkpoint = Path(args.default_checkpoint).expanduser().resolve() if args.default_checkpoint else None
    forecast_horizons_default = tuple(int(x) for x in args.forecast_horizons.split(",") if x.strip())
    forecast_horizons_map = _parse_horizon_map(args.forecast_horizons_map)
    context_lengths = tuple(int(x) for x in args.context_lengths.split(",") if x.strip())
    exit_only_symbols = [s.strip().upper() for s in args.exit_only_symbols.split(",") if s.strip()]
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
    long_only_symbols = [token.strip().upper() for token in (args.long_only_symbols or "").split(",") if token.strip()]
    short_only_symbols = [token.strip().upper() for token in (args.short_only_symbols or "").split(",") if token.strip()]

    device = _resolve_device(args.device)

    data_root = Path("trainingdatahourly")
    validator = HourlyDataValidator(data_root, max_staleness_hours=6)
    refresher = HourlyDataRefresher(
        data_root,
        validator,
        backfill_hours=48,
        overlap_hours=2,
        crypto_max_staleness_hours=1.5,
        sleep_seconds=0.0,
    )

    crypto_root = Path(args.crypto_data_root) if args.crypto_data_root else None
    stock_root = Path(args.stock_data_root) if args.stock_data_root else None
    forecast_cache_root = Path(args.forecast_cache_root)

    # Cancel stale orders from previous runs on startup
    if not args.dry_run:
        try:
            stale_orders = alpaca_wrapper.get_orders()
            cancelled = 0
            for order in stale_orders:
                order_symbol = str(getattr(order, "symbol", "")).replace("/", "").upper()
                if order_symbol in {s.replace("/", "").upper() for s in symbols}:
                    logger.info("Cancelling stale order on startup: %s %s qty=%s @ %s",
                                order_symbol, order.side, order.qty, order.limit_price)
                    alpaca_wrapper.cancel_order(order)
                    cancelled += 1
            if cancelled:
                logger.info("Cancelled %d stale order(s) from previous run", cancelled)
        except Exception as exc:
            logger.warning("Failed to clean up stale orders on startup: %s", exc)

    while True:
        _run_cycle(
            symbols,
            checkpoint_map,
            default_checkpoint,
            horizon=args.horizon,
            sequence_length=args.sequence_length,
            intensity_scale=args.intensity_scale,
            price_offset_pct=args.price_offset_pct,
            min_gap_pct=args.min_gap_pct,
            allocation_usd=args.allocation_usd,
            allocation_pct=args.allocation_pct,
            allocation_mode=args.allocation_mode,
            forecast_horizons_default=forecast_horizons_default,
            forecast_horizons_map=forecast_horizons_map,
            context_lengths=context_lengths,
            trim_ratio=args.trim_ratio,
            cache_only=args.cache_only,
            crypto_data_root=crypto_root,
            stock_data_root=stock_root,
            forecast_cache_root=forecast_cache_root,
            allow_short=bool(args.allow_short),
            long_only_symbols=long_only_symbols,
            short_only_symbols=short_only_symbols,
            moving_average_windows=ma_windows,
            ema_windows=ema_windows,
            atr_windows=atr_windows,
            trend_windows=trend_windows,
            drawdown_windows=drawdown_windows,
            volume_z_window=volume_z_window,
            volume_shock_window=volume_shock_window,
            vol_regime_short=vol_regime_short,
            vol_regime_long=vol_regime_long,
            min_history_hours=min_history_hours,
            refresher=refresher,
            device=device,
            exit_only_symbols=exit_only_symbols,
            allow_position_adds=bool(args.allow_position_adds),
            always_full_exit=bool(args.always_full_exit),
            dry_run=args.dry_run,
        )
        if args.once:
            break
        sleep_seconds = _seconds_until_next_hour(buffer_seconds=args.buffer_seconds)
        logger.info("Sleeping %.1fs until next hour", sleep_seconds)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
