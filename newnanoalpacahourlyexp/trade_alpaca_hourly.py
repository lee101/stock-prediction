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
from binanceneural.inference import generate_actions_from_frame, generate_latest_action
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.hourly_data_refresh import HourlyDataRefresher
from src.hourly_data_utils import HourlyDataValidator
from src.symbol_utils import is_crypto_symbol
from src.torch_device_utils import require_cuda as require_cuda_device

from .config import DatasetConfig, ExperimentConfig
from .data import AlpacaHourlyDataModule
from .inference import aggregate_actions, generate_actions_multi_context


logger = logging.getLogger("alpaca_hourly_trader")


@dataclass
class TradingPlan:
    symbol: str
    buy_price: float
    sell_price: float
    buy_amount: float
    sell_amount: float
    timestamp: datetime


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


def _seconds_until_next_hour(buffer_seconds: int = 30) -> float:
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    delta = (next_hour + timedelta(seconds=buffer_seconds) - now).total_seconds()
    return max(1.0, delta)


def _build_plan(action: dict, *, intensity_scale: float) -> TradingPlan:
    buy_amount = max(0.0, min(100.0, float(action["buy_amount"]) * intensity_scale))
    sell_amount = max(0.0, min(100.0, float(action["sell_amount"]) * intensity_scale))
    return TradingPlan(
        symbol=str(action["symbol"]).upper(),
        buy_price=float(action["buy_price"]),
        sell_price=float(action["sell_price"]),
        buy_amount=buy_amount,
        sell_amount=sell_amount,
        timestamp=action["timestamp"],
    )


def _ensure_valid_levels(buy_price: float, sell_price: float, *, min_gap_pct: float) -> Optional[Tuple[float, float]]:
    if buy_price <= 0 or sell_price <= 0:
        return None
    if sell_price <= buy_price:
        sell_price = buy_price * (1.0 + min_gap_pct)
        if sell_price <= buy_price:
            return None
    return buy_price, sell_price


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
    forecast_horizons_default: Tuple[int, ...],
    forecast_horizons_map: Dict[str, Tuple[int, ...]],
    context_lengths: Tuple[int, ...],
    trim_ratio: float,
    cache_only: bool,
    crypto_data_root: Optional[Path],
    stock_data_root: Optional[Path],
    forecast_cache_root: Path,
    refresher: HourlyDataRefresher,
    device: torch.device,
    exit_only_symbols: Sequence[str],
    dry_run: bool,
) -> None:
    refresher.refresh(list(symbols))
    account = alpaca_wrapper.get_account(use_cache=False)
    positions = alpaca_wrapper.get_all_positions()

    exit_only_set = {s.upper() for s in exit_only_symbols}
    experiment_cfg = ExperimentConfig(context_lengths=context_lengths, trim_ratio=trim_ratio)

    for symbol in symbols:
        symbol = symbol.upper()
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

        plan = _build_plan(action, intensity_scale=intensity_scale)
        buy_price = plan.buy_price * (1.0 - price_offset_pct)
        sell_price = plan.sell_price * (1.0 + price_offset_pct)
        adjusted = _ensure_valid_levels(buy_price, sell_price, min_gap_pct=min_gap_pct)
        if adjusted is None:
            logger.warning("Invalid price levels for %s (buy=%.4f sell=%.4f)", symbol, buy_price, sell_price)
            continue
        buy_price, sell_price = adjusted

        position = _find_position(positions, symbol)
        position_qty = abs(float(getattr(position, "qty", 0.0) or 0.0)) if position else 0.0
        allocation = _allocation_usd(account, allocation_usd=allocation_usd, allocation_pct=allocation_pct)
        buy_notional = 0.0 if allocation is None else allocation * (plan.buy_amount / 100.0)
        buy_qty = buy_notional / buy_price if buy_notional > 0 else 0.0
        sell_qty = position_qty * (plan.sell_amount / 100.0) if position_qty > 0 else 0.0

        if symbol in exit_only_set:
            buy_qty = 0.0
            if position_qty > 0:
                sell_qty = position_qty
            logger.info("Exit-only mode for %s: sell_qty=%.6f", symbol, sell_qty)

        logger.info(
            "%s action buy=%.4f sell=%.4f buy_amt=%.2f sell_amt=%.2f buy_qty=%.6f sell_qty=%.6f",
            symbol,
            buy_price,
            sell_price,
            plan.buy_amount,
            plan.sell_amount,
            buy_qty,
            sell_qty,
        )

        if dry_run:
            continue

        if sell_qty > 0:
            alpaca_wrapper.open_order_at_price_or_all(symbol, sell_qty, "sell", sell_price)
        if buy_qty > 0:
            alpaca_wrapper.open_order_at_price_or_all(symbol, buy_qty, "buy", buy_price)


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
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--exit-only-symbols", default="", help="Comma-separated symbols to only exit/close.")
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
            forecast_horizons_default=forecast_horizons_default,
            forecast_horizons_map=forecast_horizons_map,
            context_lengths=context_lengths,
            trim_ratio=args.trim_ratio,
            cache_only=args.cache_only,
            crypto_data_root=crypto_root,
            stock_data_root=stock_root,
            forecast_cache_root=forecast_cache_root,
            refresher=refresher,
            device=device,
            exit_only_symbols=exit_only_symbols,
            dry_run=args.dry_run,
        )
        if args.once:
            break
        sleep_seconds = _seconds_until_next_hour(buffer_seconds=args.buffer_seconds)
        logger.info("Sleeping %.1fs until next hour", sleep_seconds)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
