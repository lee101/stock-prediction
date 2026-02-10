from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import pandas as pd
import torch

REPO_ROOT = None
for parent in Path(__file__).resolve().parents:
    if (parent / "pyproject.toml").exists():
        REPO_ROOT = parent
        break
if REPO_ROOT is None:
    raise RuntimeError("Failed to locate repo root (pyproject.toml).")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binanceexp1.sweep import apply_action_overrides
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.symbol_utils import is_crypto_symbol
from src.torch_device_utils import require_cuda as require_cuda_device
from src.torch_load_utils import torch_load_compat

from newnanoalpacahourlyexp.config import DatasetConfig
from newnanoalpacahourlyexp.data import AlpacaHourlyDataModule
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation


def _parse_int_tuple(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        return None
    return tuple(int(v) for v in values)


def _slice_eval_window(
    actions: pd.DataFrame,
    bars: pd.DataFrame,
    eval_days: float | None,
    eval_hours: float | None,
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


def _load_model(checkpoint_path: Path, input_dim: int, sequence_length: int) -> torch.nn.Module:
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


def _parse_float_list(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [float(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep selector thresholds with decision_lag_bars enabled.")
    parser.add_argument(
        "--symbols",
        default="SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX",
        help="Comma-separated symbols.",
    )
    parser.add_argument(
        "--checkpoint",
        default="binanceneural/checkpoints/alpaca_cross_global_mixed7_robust_short_seq128_20260205_043448/epoch_003.pt",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--eval-days", type=float, default=10.0)
    parser.add_argument("--eval-hours", type=float, default=None)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--intensities", default="1.0,2.0")
    parser.add_argument("--min-edges", default="0.0,0.0005,0.001,0.002")
    parser.add_argument("--risk-weights", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--edge-mode", default="high_low", choices=["high_low", "high", "close"])
    parser.add_argument("--no-enforce-market-hours", action="store_true")
    parser.add_argument("--close-at-eod", action="store_true")
    parser.add_argument("--crypto-data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--stock-data-root", default="trainingdatahourly/stocks")
    parser.add_argument("--forecast-cache-root", default="binanceneural/forecast_cache")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--moving-average-windows", default="24,72")
    parser.add_argument("--ema-windows", default="24,72")
    parser.add_argument("--atr-windows", default="24,72")
    parser.add_argument("--trend-windows", default="72")
    parser.add_argument("--drawdown-windows", default="72")
    parser.add_argument("--volume-z-window", type=int, default=72)
    parser.add_argument("--volume-shock-window", type=int, default=24)
    parser.add_argument("--vol-regime-short", type=int, default=24)
    parser.add_argument("--vol-regime-long", type=int, default=72)
    parser.add_argument("--min-history-hours", type=int, default=50)
    parser.add_argument("--output-csv", default=str(Path(__file__).resolve().parent / "sweep_results.csv"))
    parser.add_argument("--device", default=None, help="Override inference device (e.g. cuda, cuda:0)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.device:
        device = torch.device(args.device)
        if device.type != "cuda":
            raise RuntimeError(f"GPU required for inference; received device={args.device!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for inference but CUDA is not available.")
    else:
        device = require_cuda_device("selector sweep", allow_fallback=False)

    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x.strip())
    crypto_root = Path(args.crypto_data_root)
    stock_root = Path(args.stock_data_root)
    forecast_cache_root = Path(args.forecast_cache_root)

    ma_windows = _parse_int_tuple(args.moving_average_windows) or DatasetConfig().moving_average_windows
    ema_windows = _parse_int_tuple(args.ema_windows) or DatasetConfig().ema_windows
    atr_windows = _parse_int_tuple(args.atr_windows) or DatasetConfig().atr_windows
    trend_windows = _parse_int_tuple(args.trend_windows) or DatasetConfig().trend_windows
    drawdown_windows = _parse_int_tuple(args.drawdown_windows) or DatasetConfig().drawdown_windows

    bars_frames: List[pd.DataFrame] = []
    base_actions_frames: List[pd.DataFrame] = []
    fee_by_symbol: Dict[str, float] = {}
    periods_by_symbol: Dict[str, float] = {}

    model: torch.nn.Module | None = None

    for symbol in symbols:
        data_root = crypto_root if is_crypto_symbol(symbol) else stock_root
        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=data_root,
            forecast_cache_root=forecast_cache_root,
            sequence_length=int(args.sequence_length),
            forecast_horizons=forecast_horizons,
            cache_only=bool(args.cache_only),
            moving_average_windows=ma_windows,
            ema_windows=ema_windows,
            atr_windows=atr_windows,
            trend_windows=trend_windows,
            drawdown_windows=drawdown_windows,
            volume_z_window=int(args.volume_z_window),
            volume_shock_window=int(args.volume_shock_window),
            vol_regime_short=int(args.vol_regime_short),
            vol_regime_long=int(args.vol_regime_long),
            min_history_hours=int(args.min_history_hours),
        )
        data = AlpacaHourlyDataModule(data_cfg)
        frame = data.val_dataset.frame.copy()
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol

        if model is None:
            model = _load_model(checkpoint_path, len(data.feature_columns), int(args.sequence_length))

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

        bars_frames.append(frame)
        base_actions_frames.append(actions)
        fee_by_symbol[symbol] = float(data.asset_meta.maker_fee)
        periods_by_symbol[symbol] = float(data.asset_meta.periods_per_year)

    bars = pd.concat(bars_frames, ignore_index=True)
    base_actions = pd.concat(base_actions_frames, ignore_index=True)

    base_actions, bars = _slice_eval_window(base_actions, bars, args.eval_days, args.eval_hours)

    intensities = _parse_float_list(args.intensities)
    min_edges = _parse_float_list(args.min_edges)
    risk_weights = _parse_float_list(args.risk_weights)

    results: List[dict] = []

    for intensity in intensities:
        adjusted = apply_action_overrides(base_actions, intensity_scale=float(intensity), price_offset_pct=0.0)
        for risk_weight in risk_weights:
            for min_edge in min_edges:
                cfg = SelectionConfig(
                    initial_cash=10_000.0,
                    min_edge=float(min_edge),
                    risk_weight=float(risk_weight),
                    edge_mode=str(args.edge_mode),
                    symbols=symbols,
                    enforce_market_hours=not args.no_enforce_market_hours,
                    close_at_eod=bool(args.close_at_eod),
                    fee_by_symbol=fee_by_symbol,
                    periods_per_year_by_symbol=periods_by_symbol,
                    decision_lag_bars=int(args.decision_lag_bars),
                )
                sim = run_best_trade_simulation(bars, adjusted, cfg, horizon=int(args.horizon))
                metrics = sim.metrics
                results.append(
                    {
                        "intensity": float(intensity),
                        "min_edge": float(min_edge),
                        "risk_weight": float(risk_weight),
                        "total_return": float(metrics.get("total_return", 0.0)),
                        "sortino": float(metrics.get("sortino", 0.0)),
                        "final_cash": float(sim.final_cash),
                        "final_inventory": float(sim.final_inventory),
                        "open_symbol": sim.open_symbol or "",
                        "financing_cost_paid": float(metrics.get("financing_cost_paid", 0.0)),
                    }
                )

    out = pd.DataFrame(results)
    out.sort_values(["total_return", "sortino"], ascending=[False, False], inplace=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote: {output_path}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
