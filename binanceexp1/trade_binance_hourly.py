from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import torch

from src.price_guard import enforce_gap
from src.process_utils import enforce_min_spread
from src.metrics_utils import compute_step_returns, annualized_sortino
from src.binan import binance_wrapper
from stock.state import get_state_dir, resolve_state_suffix, get_paper_suffix

from binanceneural.binance_watchers import WatcherPlan, spawn_watcher
from binanceneural.config import TrainingConfig, PolicyConfig
from binanceneural.execution import compute_order_quantities, get_free_balances, resolve_symbol_rules
from binanceneural.inference import generate_latest_action
from binanceneural.model import build_policy, policy_config_from_payload
from binanceneural.pnl_state import get_probe_mode
from binanceneural.trade_binance_hourly import _ensure_valid_levels, _parse_checkpoint_map, _parse_symbols, _apply_probe_allocation

from .config import DatasetConfig
from .data import BinanceExp1DataModule, build_default_feature_columns


@dataclass
class TradingPlan:
    symbol: str
    buy_price: float
    sell_price: float
    buy_amount: float
    sell_amount: float
    timestamp: datetime


def _load_checkpoint_payload(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        payload = {"state_dict": payload}
    return payload


def _infer_input_dim(state_dict: object, fallback: int) -> int:
    if isinstance(state_dict, dict):
        embed_weight = state_dict.get("embed.weight")
        if isinstance(embed_weight, torch.Tensor) and embed_weight.ndim == 2:
            return int(embed_weight.shape[1])
    return int(fallback)


def _resolve_dataset_config(
    base_cfg: DatasetConfig, *, input_dim: int, horizon: int
) -> DatasetConfig:
    candidates = []
    if (horizon,) not in candidates:
        candidates.append((horizon,))
    if base_cfg.forecast_horizons not in candidates:
        candidates.append(base_cfg.forecast_horizons)
    for horizons in candidates:
        cfg = replace(base_cfg, forecast_horizons=tuple(int(h) for h in horizons))
        feature_cols = build_default_feature_columns(cfg)
        if len(feature_cols) == input_dim:
            return replace(cfg, feature_columns=feature_cols)
    raise RuntimeError(
        "Feature dimension mismatch: "
        f"checkpoint input_dim={input_dim} cannot be matched with forecast_horizons "
        f"{candidates} (computed dims: {[len(build_default_feature_columns(replace(base_cfg, forecast_horizons=h))) for h in candidates]}). "
        "Update DatasetConfig.feature_columns to match the checkpoint."
    )


def _load_model_from_payload(payload: dict, input_dim: int, default_cfg: TrainingConfig):
    state_dict = payload.get("state_dict", payload)
    if isinstance(state_dict, dict):
        embed_weight = state_dict.get("embed.weight")
        if isinstance(embed_weight, torch.Tensor) and embed_weight.ndim == 2:
            input_dim = int(embed_weight.shape[1])
    cfg = payload.get("config", default_cfg)
    payload_cfg = cfg if isinstance(cfg, dict) else getattr(cfg, "__dict__", {})
    policy_cfg = policy_config_from_payload(
        payload_cfg,
        input_dim=input_dim,
        state_dict=state_dict if isinstance(state_dict, dict) else None,
    )
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


def _seconds_until_next_hour(buffer_seconds: int = 30) -> float:
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    delta = (next_hour + timedelta(seconds=buffer_seconds) - now).total_seconds()
    return max(1.0, delta)


def _default_metrics_log_path() -> Path:
    suffix = f"{get_paper_suffix()}{resolve_state_suffix() or ''}"
    return get_state_dir() / f"binanceexp1_pnl_history{suffix}.csv"


def _log_account_metrics(
    symbols: Iterable[str],
    *,
    log_path: Path,
    periods_per_year: int = 24 * 365,
) -> None:
    try:
        account = binance_wrapper.get_account_value_usdt(include_locked=True)
        total_usdt = float(account.get("total_usdt", 0.0))
    except Exception as exc:
        print(f"Failed to fetch account value: {exc}")
        return

    now = datetime.now(timezone.utc)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path.exists():
        history = pd.read_csv(log_path)
    else:
        history = pd.DataFrame(columns=["timestamp", "total_usdt"])

    history = history.copy()
    if "timestamp" in history.columns:
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history = history.dropna(subset=["timestamp"]).reset_index(drop=True)

    history = pd.concat(
        [
            history,
            pd.DataFrame([{"timestamp": now, "total_usdt": total_usdt}]),
        ],
        ignore_index=True,
    )

    # Find first non-zero value as baseline
    nonzero_values = history["total_usdt"][history["total_usdt"] > 0]
    if len(nonzero_values) > 0:
        start_value = float(nonzero_values.iloc[0])
    else:
        start_value = total_usdt if total_usdt > 0 else 1.0
    
    pnl_usdt = total_usdt - start_value
    pnl_pct = pnl_usdt / start_value if start_value > 0 else 0.0
    
    # Calculate returns only from non-zero values
    valid_values = history["total_usdt"][history["total_usdt"] > 0].to_numpy()
    if len(valid_values) > 1:
        returns = compute_step_returns(valid_values)
        sortino = annualized_sortino(returns, periods_per_year=periods_per_year)
    else:
        sortino = 0.0

    history["pnl_usdt"] = pnl_usdt
    history["pnl_pct"] = pnl_pct
    history["sortino"] = sortino
    symbol_set = sorted({s.upper() for s in symbols})
    history["symbols"] = ",".join(symbol_set)
    history["probe_modes"] = ",".join(
        f"{sym}:{'probe' if get_probe_mode(sym) else 'normal'}" for sym in symbol_set
    )
    history.tail(1).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    print(
        f"Account total_usdt={total_usdt:.2f} pnl={pnl_usdt:.2f} pnl_pct={pnl_pct:.4f} "
        f"sortino={sortino:.4f} symbols={','.join(symbol_set)}"
    )


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
    poll_seconds: int,
    expiry_minutes: int,
    price_tolerance: float,
    data_root: Path,
    cache_only: bool,
    dry_run: bool,
) -> None:
    for symbol in symbols:
        try:
            checkpoint_path = checkpoint_map.get(symbol) or default_checkpoint
            if checkpoint_path is None:
                print(f"No checkpoint provided for {symbol}; skipping.")
                continue
            payload = _load_checkpoint_payload(checkpoint_path)
            state_dict = payload.get("state_dict", payload)
            base_cfg = DatasetConfig(
                symbol=symbol,
                data_root=data_root,
                sequence_length=sequence_length,
                cache_only=cache_only,
            )
            fallback_dim = len(build_default_feature_columns(base_cfg))
            input_dim = _infer_input_dim(state_dict, fallback=fallback_dim)
            data_cfg = _resolve_dataset_config(base_cfg, input_dim=input_dim, horizon=horizon)
            data = BinanceExp1DataModule(data_cfg)
            model = _load_model_from_payload(payload, input_dim, TrainingConfig(sequence_length=sequence_length))
            action = generate_latest_action(
                model=model,
                frame=data.frame,
                feature_columns=data.feature_columns,
                normalizer=data.normalizer,
                sequence_length=sequence_length,
                horizon=horizon,
                require_gpu=True,
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
    parser = argparse.ArgumentParser(description="Run Binance hourly trading with binanceexp1 model.")
    parser.add_argument("--symbols", help="Comma-separated symbols (default BTCUSD,SOLUSD)")
    parser.add_argument("--checkpoint", help="Checkpoint path for all symbols")
    parser.add_argument("--checkpoints", help="Symbol-specific checkpoints, e.g. BTCUSD=path,SOLUSD=path")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--price-offset-pct", type=float, default=0.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.0003)
    parser.add_argument("--allocation-usdt", type=float, default=None)
    parser.add_argument("--probe-after-loss", action="store_true", help="Enable probe mode after a losing sell")
    parser.add_argument("--probe-notional", type=float, default=1.0, help="Probe trade notional in USDT")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--expiry-minutes", type=int, default=90)
    parser.add_argument("--price-tolerance", type=float, default=0.0008)
    parser.add_argument("--log-metrics", action="store_true", help="Log account PnL/sortino each cycle")
    parser.add_argument("--metrics-log-path", help="CSV path for PnL history logs")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--data-root",
        default=str(DatasetConfig().data_root),
        help="Root directory for hourly data (e.g., trainingdatahourly/crypto or trainingdatahourly/stocks).",
    )
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cycle-minutes", type=int, default=5, help="Minutes between trading cycles (default 5 for high-frequency)")
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    checkpoint_map = _parse_checkpoint_map(args.checkpoints)
    default_checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None

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
            poll_seconds=args.poll_seconds,
            expiry_minutes=args.expiry_minutes,
            price_tolerance=args.price_tolerance,
            data_root=Path(args.data_root),
            cache_only=args.cache_only,
            dry_run=args.dry_run,
        )
        if args.log_metrics and not args.dry_run:
            log_path = Path(args.metrics_log_path).expanduser().resolve() if args.metrics_log_path else _default_metrics_log_path()
            _log_account_metrics(symbols, log_path=log_path)
        if args.once:
            break
        sleep_seconds = args.cycle_minutes * 60
        print(f"Sleeping {sleep_seconds:.1f}s until next cycle...")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
