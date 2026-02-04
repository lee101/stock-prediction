from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch

from binanceneural.inference import generate_actions_from_frame
try:  # Optional selector simulation (may be unavailable in older builds).
    from binanceneural.marketsimulator import SelectionConfig, run_best_trade_simulation
except ImportError:  # pragma: no cover - fallback path for older simulators
    SelectionConfig = None
    run_best_trade_simulation = None
    from binanceneural.marketsimulator import (
        BinanceMarketSimulator,
        SimulationConfig,
        run_shared_cash_simulation,
    )
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload

from .config import DatasetConfig
from .data import BinanceExp1DataModule
from .sweep import apply_action_overrides


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


def _load_symbol_data(
    symbol: str,
    *,
    data_root: Path,
    forecast_cache_root: Path,
    sequence_length: int,
    forecast_horizons: Sequence[int],
    cache_only: bool,
    validation_days: Optional[float] = None,
) -> BinanceExp1DataModule:
    config = DatasetConfig(
        symbol=symbol,
        data_root=data_root,
        forecast_cache_root=forecast_cache_root,
        sequence_length=sequence_length,
        forecast_horizons=tuple(int(h) for h in forecast_horizons),
        cache_only=cache_only,
        validation_days=validation_days if validation_days is not None else DatasetConfig().validation_days,
    )
    return BinanceExp1DataModule(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-asset best-trade selector simulation.")
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
    parser.add_argument("--data-root", default=str(DatasetConfig().data_root))
    parser.add_argument("--forecast-cache-root", default=str(DatasetConfig().forecast_cache_root))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--validation-days",
        type=float,
        default=None,
        help="Override validation window length in days (e.g., 10 for a 10-day sim).",
    )
    parser.add_argument("--default-intensity", type=float, default=1.0)
    parser.add_argument("--default-offset", type=float, default=0.0)
    parser.add_argument("--intensity-map", help="Comma-separated SYMBOL=VALUE overrides for intensity.")
    parser.add_argument("--offset-map", help="Comma-separated SYMBOL=VALUE overrides for offset.")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--risk-weight", type=float, default=0.5)
    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--max-hold-hours", type=int, default=None)
    parser.add_argument("--allow-reentry-same-bar", action="store_true")
    parser.add_argument(
        "--strategy",
        default="independent",
        choices=["independent", "shared_cash"],
        help="Simulation strategy when selector is unavailable.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

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
    data_root = Path(args.data_root)
    forecast_cache_root = Path(args.forecast_cache_root)

    bars_frames: List[pd.DataFrame] = []
    actions_frames: List[pd.DataFrame] = []

    for symbol in symbols:
        data_cfg_kwargs = dict(
            data_root=data_root,
            forecast_cache_root=forecast_cache_root,
            sequence_length=args.sequence_length,
            forecast_horizons=forecast_horizons,
            cache_only=args.cache_only,
        )
        if args.validation_days is not None:
            data_cfg_kwargs["validation_days"] = float(args.validation_days)
        data = _load_symbol_data(
            symbol,
            **data_cfg_kwargs,
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

    bars = pd.concat(bars_frames, ignore_index=True)
    actions = pd.concat(actions_frames, ignore_index=True)

    if run_best_trade_simulation is not None and SelectionConfig is not None:
        sim_config = SelectionConfig(
            initial_cash=args.initial_cash,
            min_edge=args.min_edge,
            risk_weight=args.risk_weight,
            edge_mode=args.edge_mode,
            max_hold_hours=args.max_hold_hours,
            symbols=symbols,
            allow_reentry_same_bar=args.allow_reentry_same_bar,
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
    else:
        print("Selector simulation unavailable; used fallback strategy instead.")
        if args.strategy == "shared_cash":
            result = run_shared_cash_simulation(
                bars,
                actions,
                SimulationConfig(initial_cash=args.initial_cash, max_hold_hours=args.max_hold_hours),
            )
        else:
            sim = BinanceMarketSimulator(
                SimulationConfig(initial_cash=args.initial_cash, max_hold_hours=args.max_hold_hours)
            )
            result = sim.run(bars, actions)
        metrics = result.metrics
        print(f"total_return: {metrics.get('total_return', 0.0):.4f}")
        print(f"sortino: {metrics.get('sortino', 0.0):.4f}")

        if result.per_symbol:
            negative = []
            any_metrics = False
            for symbol, sym_result in result.per_symbol.items():
                total_return = sym_result.metrics.get("total_return") if sym_result.metrics else None
                if total_return is not None:
                    any_metrics = True
                    print(f"{symbol} total_return: {total_return:.4f}")
                    if total_return < 0:
                        negative.append(symbol)
            if any_metrics:
                if negative:
                    print(f"Negative PnL symbols: {', '.join(sorted(negative))}")
                else:
                    print("Negative PnL symbols: none")
            else:
                print("Per-symbol metrics unavailable for this strategy.")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            result.combined_equity.to_frame(name="equity").to_csv(
                output_dir / "combined_equity.csv", index=True
            )


if __name__ == "__main__":
    main()
