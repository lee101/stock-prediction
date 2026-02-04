from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from binanceexp1.sweep import apply_action_overrides
from loss_utils import CRYPTO_TRADING_FEE
from src.symbol_utils import is_crypto_symbol
from src.torch_device_utils import require_cuda as require_cuda_device

from .config import DatasetConfig
from .data import AlpacaHourlyDataModule
from .marketsimulator.selector import SelectionConfig, run_best_trade_simulation


def _parse_kv_pairs(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    pairs = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected KEY=VALUE pair, got '{token}'.")
        key, value = token.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid KEY=VALUE pair '{token}'.")
        pairs[key] = value
    return pairs


def _parse_float_map(raw: Optional[str]) -> Dict[str, float]:
    pairs = _parse_kv_pairs(raw)
    parsed: Dict[str, float] = {}
    for key, value in pairs.items():
        try:
            parsed[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid float value for {key}: '{value}'") from exc
    return parsed


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


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type != "cuda":
            raise RuntimeError(f"GPU required for inference; received device={device_arg!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for inference but CUDA is not available.")
        return device
    return require_cuda_device("multi-asset inference", allow_fallback=False)


def _load_symbol_data(
    symbol: str,
    *,
    crypto_root: Optional[Path],
    stock_root: Optional[Path],
    forecast_cache_root: Path,
    sequence_length: int,
    forecast_horizons: Sequence[int],
    cache_only: bool,
) -> AlpacaHourlyDataModule:
    data_root = crypto_root if is_crypto_symbol(symbol) else stock_root
    config = DatasetConfig(
        symbol=symbol,
        data_root=data_root,
        forecast_cache_root=forecast_cache_root,
        sequence_length=sequence_length,
        forecast_horizons=tuple(int(h) for h in forecast_horizons),
        cache_only=cache_only,
    )
    return AlpacaHourlyDataModule(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-asset best-trade selector simulation (Alpaca hourly).")
    parser.add_argument(
        "--symbols",
        default="BTCUSD,ETHUSD,LINKUSD,UNIUSD,SOLUSD",
        help="Comma-separated symbols to include.",
    )
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="Comma-separated SYMBOL=PATH checkpoint mapping.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--crypto-data-root", default=None)
    parser.add_argument("--stock-data-root", default=None)
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--device", default=None, help="Override inference device (e.g., cuda, cuda:0).")
    parser.add_argument("--default-intensity", type=float, default=1.0)
    parser.add_argument("--default-offset", type=float, default=0.0)
    parser.add_argument("--intensity-map", help="Comma-separated SYMBOL=VALUE overrides for intensity.")
    parser.add_argument("--offset-map", help="Comma-separated SYMBOL=VALUE overrides for offset.")
    parser.add_argument(
        "--crypto-fee",
        type=float,
        default=CRYPTO_TRADING_FEE,
        help="Override per-side fee for crypto symbols (defaults to CRYPTO_TRADING_FEE).",
    )
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--risk-weight", type=float, default=0.5)
    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--max-hold-hours", type=int, default=None)
    parser.add_argument("--allow-reentry-same-bar", action="store_true")
    parser.add_argument("--no-enforce-market-hours", action="store_true")
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--eval-days", type=float, default=None, help="Limit evaluation to last N days")
    parser.add_argument("--eval-hours", type=float, default=None, help="Limit evaluation to last N hours")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    device = _resolve_device(args.device)

    symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")

    checkpoint_map = _parse_kv_pairs(args.checkpoints)
    for symbol in symbols:
        if symbol not in checkpoint_map:
            raise ValueError(f"Missing checkpoint path for symbol {symbol}.")

    intensity_map = _parse_float_map(args.intensity_map)
    offset_map = _parse_float_map(args.offset_map)

    forecast_horizons = [int(x) for x in args.forecast_horizons.split(",") if x]
    crypto_root = Path(args.crypto_data_root) if args.crypto_data_root else None
    stock_root = Path(args.stock_data_root) if args.stock_data_root else None
    forecast_cache_root = Path(args.forecast_cache_root)

    bars_frames: List[pd.DataFrame] = []
    actions_frames: List[pd.DataFrame] = []
    fee_by_symbol: Dict[str, float] = {}
    periods_by_symbol: Dict[str, float] = {}

    for symbol in symbols:
        data = _load_symbol_data(
            symbol,
            crypto_root=crypto_root,
            stock_root=stock_root,
            forecast_cache_root=forecast_cache_root,
            sequence_length=args.sequence_length,
            forecast_horizons=forecast_horizons,
            cache_only=args.cache_only,
        )
        frame = data.val_dataset.frame.copy()
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol

        checkpoint_path = Path(checkpoint_map[symbol]).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = _load_model(checkpoint_path, len(data.feature_columns), args.sequence_length)

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

        intensity = float(intensity_map.get(symbol, args.default_intensity))
        offset = float(offset_map.get(symbol, args.default_offset))
        if intensity != 1.0 or offset != 0.0:
            actions = apply_action_overrides(
                actions,
                intensity_scale=intensity,
                price_offset_pct=offset,
            )

        bars_frames.append(frame)
        actions_frames.append(actions)
        fee_by_symbol[symbol] = float(args.crypto_fee if is_crypto_symbol(symbol) else data.asset_meta.maker_fee)
        periods_by_symbol[symbol] = float(data.asset_meta.periods_per_year)

    bars = pd.concat(bars_frames, ignore_index=True)
    actions = pd.concat(actions_frames, ignore_index=True)

    if args.eval_days or args.eval_hours:
        actions, bars = _slice_eval_window(actions, bars, args.eval_days, args.eval_hours)

    sim_config = SelectionConfig(
        initial_cash=args.initial_cash,
        min_edge=args.min_edge,
        risk_weight=args.risk_weight,
        edge_mode=args.edge_mode,
        max_hold_hours=args.max_hold_hours,
        symbols=symbols,
        allow_reentry_same_bar=args.allow_reentry_same_bar,
        enforce_market_hours=not args.no_enforce_market_hours,
        close_at_eod=not args.no_close_at_eod,
        fee_by_symbol=fee_by_symbol,
        periods_per_year_by_symbol=periods_by_symbol,
    )
    result = run_best_trade_simulation(bars, actions, sim_config, horizon=args.horizon)

    metrics = result.metrics
    print(f"total_return: {metrics.get('total_return', 0.0):.4f}")
    print(f"sortino: {metrics.get('sortino', 0.0):.4f}")
    print(f"final_cash: {result.final_cash:.4f}")
    print(f"final_inventory: {result.final_inventory:.6f}")
    print(f"open_symbol: {result.open_symbol}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.per_hour.to_csv(output_dir / "selector_per_hour.csv", index=False)
        trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
        trades_df.to_csv(output_dir / "selector_trades.csv", index=False)


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


if __name__ == "__main__":
    main()
