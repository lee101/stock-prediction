from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

from src.price_guard import enforce_gap
from src.process_utils import enforce_min_spread

from .binance_watchers import WatcherPlan, spawn_watcher
from .config import DatasetConfig, TrainingConfig
from .data import BinanceHourlyDataModule
from .execution import compute_order_quantities, get_free_balances, resolve_symbol_rules, quantize_price
from .inference import generate_latest_action
from .model import BinancePolicyBase, build_policy, policy_config_from_payload
from .pnl_state import get_probe_mode


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
        return ["BTCUSD", "SOLUSD"]
    tokens = []
    for part in raw.replace(" ", "").split(","):
        if part:
            tokens.append(part.upper())
    return tokens


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


def _load_model(checkpoint_path: Path, input_dim: int, default_cfg: TrainingConfig) -> BinancePolicyBase:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    cfg = payload.get("config", default_cfg)
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _build_plan(action: dict, *, intensity_scale: float) -> TradingPlan:
    buy_amount = max(0.0, min(100.0, action["buy_amount"] * intensity_scale))
    sell_amount = max(0.0, min(100.0, action["sell_amount"] * intensity_scale))
    return TradingPlan(
        symbol=str(action["symbol"]).upper(),
        buy_price=float(action["buy_price"]),
        sell_price=float(action["sell_price"]),
        buy_amount=buy_amount,
        sell_amount=sell_amount,
        timestamp=action["timestamp"],
    )


def _ensure_valid_levels(
    symbol: str,
    buy_price: float,
    sell_price: float,
    *,
    min_gap_pct: float,
    rules,
) -> Optional[tuple[float, float]]:
    if buy_price <= 0 or sell_price <= 0:
        return None

    buy_price = quantize_price(buy_price, tick_size=rules.tick_size, side="buy")
    sell_price = quantize_price(sell_price, tick_size=rules.tick_size, side="sell")

    if rules.min_price is not None:
        if buy_price < rules.min_price or sell_price < rules.min_price:
            return None

    if sell_price <= buy_price:
        adjusted_sell = buy_price * (1.0 + min_gap_pct)
        sell_price = quantize_price(adjusted_sell, tick_size=rules.tick_size, side="sell")
        if sell_price <= buy_price:
            return None

    return buy_price, sell_price


def _apply_probe_allocation(
    symbol: str,
    allocation_usdt: Optional[float],
    *,
    enable_probe: bool,
    probe_notional: float,
) -> Optional[float]:
    if not enable_probe or probe_notional <= 0:
        return allocation_usdt
    if not get_probe_mode(symbol):
        return allocation_usdt
    if allocation_usdt is None:
        return float(probe_notional)
    return min(float(allocation_usdt), float(probe_notional))


def _seconds_until_next_hour(buffer_seconds: int = 30) -> float:
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    delta = (next_hour + timedelta(seconds=buffer_seconds) - now).total_seconds()
    return max(1.0, delta)


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
    allocation_usdt: Optional[float],
    probe_after_loss: bool,
    probe_notional: float,
    data_root: Path,
    forecast_cache_root: Path,
    forecast_horizons: Tuple[int, ...],
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    cache_only: bool,
    dry_run: bool,
) -> None:
    for symbol in symbols:
        try:
            checkpoint_path = checkpoint_map.get(symbol) or default_checkpoint
            if checkpoint_path is None:
                print(f"No checkpoint provided for {symbol}; skipping.")
                continue
            data_cfg = DatasetConfig(
                symbol=symbol,
                sequence_length=sequence_length,
                data_root=data_root,
                forecast_cache_root=forecast_cache_root,
                forecast_horizons=forecast_horizons,
                cache_only=cache_only,
            )
            data = BinanceHourlyDataModule(data_cfg)
            model = _load_model(checkpoint_path, len(data.feature_columns), TrainingConfig(sequence_length=sequence_length))
            action = generate_latest_action(
                model=model,
                frame=data.frame,
                feature_columns=data.feature_columns,
                normalizer=data.normalizer,
                sequence_length=sequence_length,
                horizon=horizon,
            )
            plan = _build_plan(action, intensity_scale=intensity_scale)
            buy_price = plan.buy_price * (1.0 - price_offset_pct)
            sell_price = plan.sell_price * (1.0 + price_offset_pct)

            buy_price, sell_price = enforce_min_spread(buy_price, sell_price, min_spread_pct=min_gap_pct)
            buy_price, sell_price = enforce_gap(symbol, buy_price, sell_price, min_gap_pct=min_gap_pct)

            if buy_price <= 0 or sell_price <= 0 or buy_price >= sell_price:
                print(f"Invalid price levels for {symbol}: buy={buy_price:.4f} sell={sell_price:.4f}")
                continue

            try:
                quote_free, base_free = get_free_balances(symbol)
            except Exception as exc:
                print(f"Failed to fetch Binance balances for {symbol}: {exc}")
                continue
            allocation_usdt_eff = _apply_probe_allocation(
                symbol,
                allocation_usdt,
                enable_probe=probe_after_loss,
                probe_notional=probe_notional,
            )
            try:
                rules = resolve_symbol_rules(symbol)
            except Exception as exc:
                print(f"Failed to fetch Binance symbol rules for {symbol}: {exc}")
                continue
            validated = _ensure_valid_levels(
                symbol,
                buy_price,
                sell_price,
                min_gap_pct=min_gap_pct,
                rules=rules,
            )
            if validated is None:
                print(f"Invalid quantized price levels for {symbol}: buy={buy_price:.6f} sell={sell_price:.6f}")
                continue
            buy_price, sell_price = validated
            sizing = compute_order_quantities(
                symbol=symbol,
                buy_amount=plan.buy_amount,
                sell_amount=plan.sell_amount,
                buy_price=buy_price,
                sell_price=sell_price,
                quote_free=quote_free,
                base_free=base_free,
                allocation_usdt=allocation_usdt_eff,
                rules=rules,
            )

            if sizing.buy_qty > 0:
                spawn_watcher(
                    WatcherPlan(
                        symbol=symbol,
                        side="buy",
                        mode="entry",
                        limit_price=buy_price,
                        target_qty=sizing.buy_qty,
                        expiry_minutes=expiry_minutes,
                        poll_seconds=poll_seconds,
                        price_tolerance=price_tolerance,
                        dry_run=dry_run,
                    )
                )
            if sizing.sell_qty > 0:
                spawn_watcher(
                    WatcherPlan(
                        symbol=symbol,
                        side="sell",
                        mode="exit",
                        limit_price=sell_price,
                        target_qty=sizing.sell_qty,
                        expiry_minutes=expiry_minutes,
                        poll_seconds=poll_seconds,
                        price_tolerance=price_tolerance,
                        dry_run=dry_run,
                    )
                )

            print(
                f"{symbol} plan @ {plan.timestamp}: buy={buy_price:.4f}({sizing.buy_qty:.6f}) "
                f"sell={sell_price:.4f}({sizing.sell_qty:.6f}) buy_amt={plan.buy_amount:.2f} "
                f"sell_amt={plan.sell_amount:.2f} probe_mode={get_probe_mode(symbol)}"
            )
        except Exception as exc:  # pragma: no cover - defensive loop safety
            print(f"Error processing {symbol}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Binance hourly trading with binanceneural model.")
    parser.add_argument("--symbols", help="Comma-separated symbols (default BTCUSD,SOLUSD)")
    parser.add_argument("--checkpoint", help="Checkpoint path for all symbols")
    parser.add_argument("--checkpoints", help="Symbol-specific checkpoints, e.g. BTCUSD=path,SOLUSD=path")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--intensity-scale", type=float, default=0.8)
    parser.add_argument("--price-offset-pct", type=float, default=0.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.0003)
    parser.add_argument("--allocation-usdt", type=float, default=None)
    parser.add_argument("--probe-after-loss", action="store_true", help="Enable probe mode after a losing sell")
    parser.add_argument("--probe-notional", type=float, default=1.0, help="Probe trade notional in USDT")
    parser.add_argument("--data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--forecast-cache-root", default="binanceneural/forecast_cache")
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--expiry-minutes", type=int, default=90)
    parser.add_argument("--price-tolerance", type=float, default=0.0008)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    checkpoint_map = _parse_checkpoint_map(args.checkpoints)
    default_checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    horizons = tuple(int(h.strip()) for h in str(args.forecast_horizons).split(",") if h.strip())
    if not horizons:
        raise ValueError("At least one forecast horizon is required.")
    data_root = Path(args.data_root)
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
            allocation_usdt=args.allocation_usdt,
            probe_after_loss=args.probe_after_loss,
            probe_notional=args.probe_notional,
            data_root=data_root,
            forecast_cache_root=forecast_cache_root,
            forecast_horizons=horizons,
            poll_seconds=args.poll_seconds,
            expiry_minutes=args.expiry_minutes,
            price_tolerance=args.price_tolerance,
            cache_only=args.cache_only,
            dry_run=args.dry_run,
        )
        if args.once:
            break
        sleep_seconds = _seconds_until_next_hour()
        print(f"Sleeping {sleep_seconds:.1f}s until next hour...")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
