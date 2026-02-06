from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

import alpaca_wrapper
from binanceneural.inference import generate_latest_action
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.hourly_data_refresh import HourlyDataRefresher
from src.hourly_data_utils import HourlyDataValidator
from src.symbol_utils import is_crypto_symbol
from src.torch_device_utils import require_cuda as require_cuda_device
from src.torch_load_utils import torch_load_compat

from .config import DatasetConfig
from .data import AlpacaHourlyDataModule


logger = logging.getLogger("alpaca_selector_trader")


@dataclass
class CandidatePlan:
    symbol: str
    timestamp: datetime
    buy_price: float
    sell_price: float
    buy_amount: float
    sell_amount: float
    buy_intensity: float
    sell_intensity: float
    edge_score: float
    close: float
    low: float
    high: float
    pred_high: float
    pred_low: float
    pred_close: float
    fee_rate: float


def _parse_symbols(raw: Optional[str]) -> list[str]:
    if not raw:
        return ["SOLUSD", "LINKUSD", "UNIUSD", "BTCUSD", "ETHUSD", "NVDA", "NFLX"]
    return [token.strip().upper() for token in raw.split(",") if token.strip()]


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
            raise RuntimeError(f"GPU required for Alpaca selector trading; received device={device_arg!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for Alpaca selector trading but CUDA is not available.")
        return device
    return require_cuda_device("alpaca selector trading", allow_fallback=False)


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


def _ensure_valid_levels(buy_price: float, sell_price: float, *, min_gap_pct: float) -> Optional[Tuple[float, float]]:
    if buy_price <= 0 or sell_price <= 0:
        return None
    if sell_price <= buy_price:
        sell_price = buy_price * (1.0 + min_gap_pct)
        if sell_price <= buy_price:
            return None
    return buy_price, sell_price


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


def _amount_to_intensity(amount: float) -> float:
    scale = 100.0 if abs(amount) > 1.0 else 1.0
    return max(0.0, min(1.0, float(amount) / scale))


def _edge_score(
    *,
    buy_price: float,
    pred_high: float,
    pred_low: float,
    pred_close: float,
    fee_rate: float,
    buy_intensity: float,
    risk_weight: float,
    edge_mode: str,
) -> Optional[float]:
    if buy_price <= 0:
        return None
    mode = edge_mode.lower()
    if mode == "close":
        upside = (pred_close - buy_price) / buy_price
        downside = 0.0
    elif mode == "high":
        upside = (pred_high - buy_price) / buy_price
        downside = 0.0
    elif mode == "high_low":
        upside = (pred_high - buy_price) / buy_price
        downside = max(0.0, (buy_price - pred_low) / buy_price)
    else:
        raise ValueError(f"Unsupported edge_mode '{edge_mode}'.")
    edge = upside - risk_weight * downside - (2.0 * fee_rate)
    if not math.isfinite(edge):
        return None
    return float(edge * buy_intensity)


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _coerce_dt(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _should_force_eod_close(symbol: str, clock, *, close_at_eod: bool, buffer_minutes: int = 60) -> bool:
    if not close_at_eod or is_crypto_symbol(symbol):
        return False
    if not getattr(clock, "is_open", True):
        return False
    ts = _coerce_dt(getattr(clock, "timestamp", None))
    next_close = _coerce_dt(getattr(clock, "next_close", None))
    if ts is None or next_close is None:
        return False
    return (next_close - ts) <= timedelta(minutes=buffer_minutes)


def _build_candidate(
    *,
    symbol: str,
    action: dict,
    bar_row: dict,
    horizon: int,
    intensity_scale: float,
    price_offset_pct: float,
    min_gap_pct: float,
    dip_threshold_pct: float,
    fee_rate: float,
    risk_weight: float,
    edge_mode: str,
    min_edge: float,
) -> Optional[CandidatePlan]:
    buy_price = float(action.get("buy_price", 0.0)) * (1.0 - price_offset_pct)
    sell_price = float(action.get("sell_price", 0.0)) * (1.0 + price_offset_pct)
    adjusted = _ensure_valid_levels(buy_price, sell_price, min_gap_pct=min_gap_pct)
    if adjusted is None:
        return None
    buy_price, sell_price = adjusted

    buy_amount = max(0.0, min(100.0, float(action.get("buy_amount", 0.0)) * intensity_scale))
    sell_amount = max(0.0, min(100.0, float(action.get("sell_amount", 0.0)) * intensity_scale))
    buy_intensity = _amount_to_intensity(buy_amount)
    sell_intensity = _amount_to_intensity(sell_amount)

    close_price = _safe_float(bar_row.get("close", 0.0)) or 0.0
    low_price = _safe_float(bar_row.get("low", 0.0)) or 0.0
    high_price = _safe_float(bar_row.get("high", 0.0)) or 0.0

    if dip_threshold_pct > 0 and close_price > 0:
        trigger = low_price <= close_price * (1.0 - dip_threshold_pct)
        if not trigger:
            buy_intensity = 0.0

    pred_high = _safe_float(bar_row.get(f"predicted_high_p50_h{int(horizon)}", 0.0)) or 0.0
    pred_low = _safe_float(bar_row.get(f"predicted_low_p50_h{int(horizon)}", 0.0)) or 0.0
    pred_close = _safe_float(bar_row.get(f"predicted_close_p50_h{int(horizon)}", 0.0)) or 0.0
    if pred_high <= 0 or pred_low <= 0 or pred_close <= 0:
        return None

    edge = _edge_score(
        buy_price=buy_price,
        pred_high=pred_high,
        pred_low=pred_low,
        pred_close=pred_close,
        fee_rate=fee_rate,
        buy_intensity=buy_intensity,
        risk_weight=risk_weight,
        edge_mode=edge_mode,
    )
    if edge is None or edge < min_edge:
        return None

    return CandidatePlan(
        symbol=symbol,
        timestamp=bar_row["timestamp"],
        buy_price=buy_price,
        sell_price=sell_price,
        buy_amount=buy_amount,
        sell_amount=sell_amount,
        buy_intensity=buy_intensity,
        sell_intensity=sell_intensity,
        edge_score=edge,
        close=close_price,
        low=low_price,
        high=high_price,
        pred_high=pred_high,
        pred_low=pred_low,
        pred_close=pred_close,
        fee_rate=fee_rate,
    )


def _latest_bar_row(data: AlpacaHourlyDataModule) -> dict:
    frame = data.frame
    row = frame.iloc[-1]
    return row.to_dict()


def _prepare_data(
    *,
    symbol: str,
    sequence_length: int,
    forecast_horizons: Tuple[int, ...],
    cache_only: bool,
    crypto_root: Optional[Path],
    stock_root: Optional[Path],
    forecast_cache_root: Path,
    ma_windows: Tuple[int, ...],
    ema_windows: Tuple[int, ...],
    atr_windows: Tuple[int, ...],
    trend_windows: Tuple[int, ...],
    drawdown_windows: Tuple[int, ...],
    volume_z_window: int,
    volume_shock_window: int,
    vol_regime_short: int,
    vol_regime_long: int,
    min_history_hours: int,
) -> AlpacaHourlyDataModule:
    data_root = crypto_root if is_crypto_symbol(symbol) else stock_root
    data_cfg = DatasetConfig(
        symbol=symbol,
        data_root=data_root,
        forecast_cache_root=forecast_cache_root,
        sequence_length=sequence_length,
        forecast_horizons=forecast_horizons,
        cache_only=cache_only,
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
    )
    return AlpacaHourlyDataModule(data_cfg)


def _run_cycle(
    symbols: Sequence[str],
    *,
    checkpoint: Path,
    sequence_length: int,
    horizon: int,
    forecast_horizons: Tuple[int, ...],
    intensity_scale: float,
    price_offset_pct: float,
    min_gap_pct: float,
    min_edge: float,
    risk_weight: float,
    edge_mode: str,
    dip_threshold_pct: float,
    allocation_usd: Optional[float],
    allocation_pct: Optional[float],
    cache_only: bool,
    crypto_data_root: Optional[Path],
    stock_data_root: Optional[Path],
    forecast_cache_root: Path,
    ma_windows: Tuple[int, ...],
    ema_windows: Tuple[int, ...],
    atr_windows: Tuple[int, ...],
    trend_windows: Tuple[int, ...],
    drawdown_windows: Tuple[int, ...],
    volume_z_window: int,
    volume_shock_window: int,
    vol_regime_short: int,
    vol_regime_long: int,
    min_history_hours: int,
    close_at_eod: bool,
    device: torch.device,
    refresher: HourlyDataRefresher,
    dry_run: bool,
) -> None:
    refresher.refresh(list(symbols))
    account = alpaca_wrapper.get_account(use_cache=False)
    positions = alpaca_wrapper.get_all_positions()
    open_orders = alpaca_wrapper.get_orders()

    tracked = {sym.upper() for sym in symbols}
    tracked_positions = [pos for pos in positions if str(getattr(pos, "symbol", "")).upper() in tracked]

    open_symbol = None
    open_position = None
    if tracked_positions:
        open_position = max(
            tracked_positions,
            key=lambda pos: abs(float(getattr(pos, "market_value", 0.0) or 0.0)),
        )
        open_symbol = str(getattr(open_position, "symbol", "")).upper()
        if len(tracked_positions) > 1:
            logger.warning(
                "Multiple open positions detected (%s). Managing %s only.",
                ",".join(str(getattr(pos, "symbol", "")) for pos in tracked_positions),
                open_symbol,
            )

    if open_orders:
        for order in open_orders:
            symbol = str(getattr(order, "symbol", "")).upper()
            if symbol in tracked:
                alpaca_wrapper.cancel_order(order)

    clock = alpaca_wrapper.get_clock()

    candidates: Dict[str, CandidatePlan] = {}
    model: Optional[torch.nn.Module] = None

    for symbol in symbols:
        symbol = symbol.upper()
        if not is_crypto_symbol(symbol) and not getattr(clock, "is_open", True):
            logger.info("Market closed; skipping %s", symbol)
            continue

        data = _prepare_data(
            symbol=symbol,
            sequence_length=sequence_length,
            forecast_horizons=forecast_horizons,
            cache_only=cache_only,
            crypto_root=crypto_data_root,
            stock_root=stock_data_root,
            forecast_cache_root=forecast_cache_root,
            ma_windows=ma_windows,
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

        if model is None:
            model = _load_model(checkpoint, len(data.feature_columns), sequence_length)

        action = generate_latest_action(
            model=model,
            frame=data.frame,
            feature_columns=data.feature_columns,
            normalizer=data.normalizer,
            sequence_length=sequence_length,
            horizon=horizon,
            device=device,
            require_gpu=True,
        )
        bar_row = _latest_bar_row(data)
        fee_rate = float(data.asset_meta.maker_fee)

        candidate = _build_candidate(
            symbol=symbol,
            action=action,
            bar_row=bar_row,
            horizon=horizon,
            intensity_scale=intensity_scale,
            price_offset_pct=price_offset_pct,
            min_gap_pct=min_gap_pct,
            dip_threshold_pct=dip_threshold_pct,
            fee_rate=fee_rate,
            risk_weight=risk_weight,
            edge_mode=edge_mode,
            min_edge=min_edge,
        )
        if candidate is None:
            continue
        candidates[symbol] = candidate

    if open_symbol and open_symbol in candidates and open_position is not None:
        candidate = candidates[open_symbol]
        position_qty = abs(float(getattr(open_position, "qty", 0.0) or 0.0))
        if position_qty <= 0:
            return
        force_close = _should_force_eod_close(open_symbol, clock, close_at_eod=close_at_eod)
        sell_qty = position_qty if force_close else position_qty * (candidate.sell_amount / 100.0)
        if sell_qty <= 0:
            logger.info("No sell qty for %s (amount %.2f)", open_symbol, candidate.sell_amount)
            return
        logger.info(
            "SELL %s qty=%.6f price=%.4f force_eod=%s edge=%.6f",
            open_symbol,
            sell_qty,
            candidate.sell_price,
            force_close,
            candidate.edge_score,
        )
        if dry_run:
            return
        if force_close:
            alpaca_wrapper.open_market_order_violently(open_symbol, sell_qty, "sell")
        else:
            alpaca_wrapper.open_order_at_price_or_all(open_symbol, sell_qty, "sell", candidate.sell_price)
        return

    if open_symbol:
        logger.info("Holding %s; no candidate for exit.", open_symbol)
        return

    if not candidates:
        logger.info("No trade candidates after filters.")
        return

    best = max(candidates.values(), key=lambda item: item.edge_score)
    allocation = _allocation_usd(account, allocation_usd=allocation_usd, allocation_pct=allocation_pct)
    if allocation is None or allocation <= 0:
        logger.info("No allocation available; skipping buy.")
        return
    buy_notional = allocation * (best.buy_amount / 100.0)
    buy_qty = buy_notional / best.buy_price if best.buy_price > 0 else 0.0
    if buy_qty <= 0:
        logger.info("Computed buy_qty <= 0 for %s", best.symbol)
        return

    if _should_force_eod_close(best.symbol, clock, close_at_eod=close_at_eod):
        logger.info("Skipping new %s buy; within EOD close window.", best.symbol)
        return

    logger.info(
        "BUY %s qty=%.6f price=%.4f edge=%.6f buy_amt=%.2f",
        best.symbol,
        buy_qty,
        best.buy_price,
        best.edge_score,
        best.buy_amount,
    )
    if dry_run:
        return
    alpaca_wrapper.open_order_at_price_or_all(best.symbol, buy_qty, "buy", best.buy_price)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly Alpaca best-trade selector (live).")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--price-offset-pct", type=float, default=0.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.001)
    parser.add_argument("--allocation-usd", type=float, default=None)
    parser.add_argument("--allocation-pct", type=float, default=1.0)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--risk-weight", type=float, default=0.5)
    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--dip-threshold-pct", type=float, default=0.0)
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
    parser.add_argument("--close-at-eod", action="store_true")
    parser.add_argument("--buffer-seconds", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = _parse_symbols(args.symbols)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x.strip())
    device = _resolve_device(args.device)

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

    while True:
        _run_cycle(
            symbols,
            checkpoint=checkpoint,
            sequence_length=args.sequence_length,
            horizon=args.horizon,
            forecast_horizons=forecast_horizons,
            intensity_scale=args.intensity_scale,
            price_offset_pct=args.price_offset_pct,
            min_gap_pct=args.min_gap_pct,
            min_edge=args.min_edge,
            risk_weight=args.risk_weight,
            edge_mode=args.edge_mode,
            dip_threshold_pct=args.dip_threshold_pct,
            allocation_usd=args.allocation_usd,
            allocation_pct=args.allocation_pct,
            cache_only=args.cache_only,
            crypto_data_root=crypto_root,
            stock_data_root=stock_root,
            forecast_cache_root=forecast_cache_root,
            ma_windows=ma_windows,
            ema_windows=ema_windows,
            atr_windows=atr_windows,
            trend_windows=trend_windows,
            drawdown_windows=drawdown_windows,
            volume_z_window=volume_z_window,
            volume_shock_window=volume_shock_window,
            vol_regime_short=vol_regime_short,
            vol_regime_long=vol_regime_long,
            min_history_hours=min_history_hours,
            close_at_eod=args.close_at_eod,
            device=device,
            refresher=refresher,
            dry_run=args.dry_run,
        )
        if args.once:
            break
        sleep_seconds = _seconds_until_next_hour(buffer_seconds=args.buffer_seconds)
        logger.info("Sleeping %.1fs until next hour", sleep_seconds)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
