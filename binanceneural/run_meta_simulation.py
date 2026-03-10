"""Meta-switching strategy selector experiment for crypto hourly trading.

Evaluates multiple checkpoint strategies and selects the best one using:
  A) Trailing window return (simple: pick best past-24h performer)
  B) Chronos2 PnL forecasting (predict next 6h PnL for each strategy, pick best predicted)

Usage:
    python -m binanceneural.run_meta_simulation \
        --symbol BTCUSD \
        --strategies h1only=btcusd_h1only_ft30_20260208:epoch_029 \
                     seed42=seed42_btcusd_ft30_20260209_014309:epoch_029 \
                     robust=selector_robust_btcusd_..._20260308_023407:epoch_012 \
        --sequence-length 72 \
        --decision-lag-bars 1 \
        --fee-rate 0.001 \
        --meta-lookback-hours 24 \
        --chronos-forecast-hours 6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.torch_load_utils import torch_load_compat

from .config import DatasetConfig, TrainingConfig
from .data import BinanceHourlyDataModule
from .inference import generate_actions_from_frame
from .marketsimulator import BinanceMarketSimulator, SimulationConfig, SymbolResult
from .model import BinancePolicyBase, align_state_dict_input_dim, build_policy, policy_config_from_payload
from differentiable_loss_utils import DEFAULT_MAKER_FEE_RATE


@dataclass
class StrategySpec:
    name: str
    checkpoint_path: Path
    sequence_length: int = 72


def _load_model(checkpoint_path: Path, input_dim: int, default_cfg: TrainingConfig) -> BinancePolicyBase:
    payload = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", default_cfg)
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _generate_strategy_actions(
    spec: StrategySpec,
    data: BinanceHourlyDataModule,
    val_frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Generate per-bar actions for a single strategy."""
    default_cfg = TrainingConfig(sequence_length=spec.sequence_length)
    model = _load_model(spec.checkpoint_path, len(data.feature_columns), default_cfg)
    actions = generate_actions_from_frame(
        model=model,
        frame=val_frame,
        feature_columns=data.feature_columns,
        normalizer=data.normalizer,
        sequence_length=spec.sequence_length,
        horizon=horizon,
    )
    return actions


def _simulate_strategy(
    actions: pd.DataFrame,
    val_frame: pd.DataFrame,
    symbol: str,
    sim_config: SimulationConfig,
) -> SymbolResult:
    """Run market simulation for one strategy, return its result."""
    sim = BinanceMarketSimulator(sim_config)
    result = sim.run(val_frame, actions)
    if symbol in result.per_symbol:
        return result.per_symbol[symbol]
    # Fallback: use combined
    return SymbolResult(
        equity_curve=result.combined_equity,
        trades=[],
        per_hour=pd.DataFrame(),
        metrics=result.metrics,
    )


def _hourly_returns_from_equity(equity_curve: pd.Series) -> pd.Series:
    """Convert equity curve to hourly returns."""
    equity = equity_curve.astype(float)
    returns = equity.pct_change().fillna(0.0)
    return returns


def _trailing_return(hourly_returns: pd.Series, idx: int, lookback_hours: int) -> float:
    """Compute cumulative return over trailing window ending at idx."""
    start = max(0, idx - lookback_hours)
    window = hourly_returns.iloc[start:idx]
    if len(window) == 0:
        return 0.0
    return float((1.0 + window).prod() - 1.0)


def _chronos2_forecast_pnl(
    equity_series: pd.Series,
    idx: int,
    context_hours: int,
    forecast_hours: int,
) -> float:
    """Use Chronos2 to forecast the next forecast_hours of PnL from equity curve.

    Returns predicted cumulative return over the forecast window.
    """
    try:
        from chronos import ChronosPipeline
    except ImportError:
        # Fall back to simple extrapolation if Chronos2 not available
        return _simple_trend_forecast(equity_series, idx, context_hours, forecast_hours)

    start = max(0, idx - context_hours)
    context = equity_series.iloc[start:idx].values.astype(np.float32)
    if len(context) < 10:
        return 0.0

    # Normalize to returns for better Chronos2 performance
    context_returns = np.diff(context) / np.maximum(context[:-1], 1e-8)
    if len(context_returns) < 5:
        return 0.0

    context_tensor = torch.from_numpy(context_returns).unsqueeze(0).float()

    try:
        pipeline = _get_chronos_pipeline()
        forecast = pipeline.predict(context_tensor, forecast_hours)
        # forecast shape: (1, num_samples, forecast_hours) or (1, forecast_hours)
        if forecast.dim() == 3:
            median_forecast = forecast.median(dim=1).values[0]
        else:
            median_forecast = forecast[0]

        # Sum predicted returns = predicted cumulative return
        predicted_cum_return = float(median_forecast.sum().item())
        return predicted_cum_return
    except Exception:
        return _simple_trend_forecast(equity_series, idx, context_hours, forecast_hours)


_chronos_pipeline_cache = {}


def _get_chronos_pipeline():
    """Lazy-load and cache the Chronos2 pipeline."""
    if "pipeline" not in _chronos_pipeline_cache:
        from chronos import ChronosPipeline

        _chronos_pipeline_cache["pipeline"] = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    return _chronos_pipeline_cache["pipeline"]


def _simple_trend_forecast(
    equity_series: pd.Series,
    idx: int,
    context_hours: int,
    forecast_hours: int,
) -> float:
    """Simple linear trend extrapolation as fallback."""
    start = max(0, idx - context_hours)
    context = equity_series.iloc[start:idx].values.astype(np.float64)
    if len(context) < 5:
        return 0.0
    returns = np.diff(context) / np.maximum(context[:-1], 1e-8)
    avg_return = float(np.mean(returns[-min(24, len(returns)):]))
    return avg_return * forecast_hours


def run_meta_simulation(
    *,
    symbol: str,
    strategies: list[StrategySpec],
    sim_config: SimulationConfig,
    horizon: int = 1,
    meta_lookback_hours: int = 24,
    chronos_forecast_hours: int = 6,
    chronos_context_hours: int = 168,
    cache_only: bool = True,
) -> dict:
    """Run the full meta-simulation experiment.

    Returns dict with per-strategy results and meta-selection results.
    """
    # Load data once
    data_cfg = DatasetConfig(
        symbol=symbol,
        sequence_length=max(s.sequence_length for s in strategies),
        cache_only=cache_only,
    )
    data = BinanceHourlyDataModule(data_cfg)
    val_frame = data.val_dataset.frame.copy()
    if "symbol" not in val_frame.columns:
        val_frame["symbol"] = symbol

    print(f"Validation frame: {len(val_frame)} bars")

    # Generate actions and simulate each strategy
    strategy_results: dict[str, SymbolResult] = {}
    strategy_equity: dict[str, pd.Series] = {}
    strategy_returns: dict[str, pd.Series] = {}

    for spec in strategies:
        print(f"  Generating actions for strategy '{spec.name}'...")
        actions = _generate_strategy_actions(spec, data, val_frame, horizon)
        if "symbol" not in actions.columns:
            actions["symbol"] = symbol

        print(f"  Simulating strategy '{spec.name}'...")
        result = _simulate_strategy(actions, val_frame, symbol, sim_config)
        strategy_results[spec.name] = result
        strategy_equity[spec.name] = result.equity_curve
        strategy_returns[spec.name] = _hourly_returns_from_equity(result.equity_curve)

        final_return = float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1.0)
        print(f"    {spec.name}: total_return={final_return:.4f}")

    # Now run meta-selection simulation
    n_bars = min(len(eq) for eq in strategy_equity.values())
    initial_cash = sim_config.initial_cash
    names = [s.name for s in strategies]

    # Meta A: trailing return selector
    meta_a_equity = [initial_cash]
    meta_a_selections = []

    # Meta B: Chronos2 forecast selector
    meta_b_equity = [initial_cash]
    meta_b_selections = []

    # CASH option: just hold cash
    cash_equity = [initial_cash] * n_bars

    print(f"\nRunning meta-selection over {n_bars} bars...")

    for i in range(1, n_bars):
        # --- Meta A: trailing lookback ---
        best_a_name = "CASH"
        best_a_score = 0.0
        for name in names:
            score = _trailing_return(strategy_returns[name], i, meta_lookback_hours)
            if score > best_a_score:
                best_a_score = score
                best_a_name = name

        if best_a_name == "CASH":
            # No strategy has positive trailing return -> hold cash
            meta_a_equity.append(meta_a_equity[-1])
        else:
            # Follow the selected strategy's return for this bar
            bar_return = float(strategy_returns[best_a_name].iloc[i])
            meta_a_equity.append(meta_a_equity[-1] * (1.0 + bar_return))
        meta_a_selections.append(best_a_name)

        # --- Meta B: Chronos2 forecast ---
        best_b_name = "CASH"
        best_b_score = 0.0

        # Only run Chronos forecast every N hours to avoid being too expensive
        if i % max(1, chronos_forecast_hours) == 0 or i == 1:
            for name in names:
                score = _chronos2_forecast_pnl(
                    strategy_equity[name],
                    idx=i,
                    context_hours=chronos_context_hours,
                    forecast_hours=chronos_forecast_hours,
                )
                if score > best_b_score:
                    best_b_score = score
                    best_b_name = name
            _last_b_selection = best_b_name
        else:
            best_b_name = _last_b_selection if "_last_b_selection" in dir() else "CASH"

        if best_b_name == "CASH":
            meta_b_equity.append(meta_b_equity[-1])
        else:
            bar_return = float(strategy_returns[best_b_name].iloc[i])
            meta_b_equity.append(meta_b_equity[-1] * (1.0 + bar_return))
        meta_b_selections.append(best_b_name)

    # Compute metrics
    def _compute_metrics(equity_list):
        eq = np.array(equity_list, dtype=np.float64)
        total_return = float(eq[-1] / eq[0] - 1.0)
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
        neg_returns = returns[returns < 0]
        downside_std = float(np.std(neg_returns)) if len(neg_returns) > 0 else 1e-8
        mean_return = float(np.mean(returns))
        sortino = (mean_return / downside_std) * np.sqrt(8760) if downside_std > 0 else 0.0
        return {"total_return": total_return, "sortino": sortino}

    results = {
        "individual_strategies": {},
        "meta_a_trailing": {
            "equity": meta_a_equity,
            "metrics": _compute_metrics(meta_a_equity),
            "selections": meta_a_selections,
        },
        "meta_b_chronos": {
            "equity": meta_b_equity,
            "metrics": _compute_metrics(meta_b_equity),
            "selections": meta_b_selections,
        },
    }

    for name in names:
        eq = strategy_equity[name].values
        results["individual_strategies"][name] = {
            "metrics": _compute_metrics(eq),
        }

    # Print summary
    print("\n" + "=" * 70)
    print("META-SIMULATION RESULTS")
    print("=" * 70)
    print(f"\n{'Strategy':<30} {'Total Return':>14} {'Sortino':>10}")
    print("-" * 56)
    for name in names:
        m = results["individual_strategies"][name]["metrics"]
        print(f"  {name:<28} {m['total_return']:>+13.2%} {m['sortino']:>10.2f}")

    m_a = results["meta_a_trailing"]["metrics"]
    print(f"\n  {'META-A (trailing ' + str(meta_lookback_hours) + 'h)':<28} {m_a['total_return']:>+13.2%} {m_a['sortino']:>10.2f}")

    m_b = results["meta_b_chronos"]["metrics"]
    print(f"  {'META-B (Chronos2 ' + str(chronos_forecast_hours) + 'h)':<28} {m_b['total_return']:>+13.2%} {m_b['sortino']:>10.2f}")

    print(f"\n  {'CASH (hold)':<28} {'0.00%':>14} {'0.00':>10}")

    # Strategy selection frequency
    print(f"\nMeta-A selection frequency:")
    from collections import Counter
    for name, count in Counter(meta_a_selections).most_common():
        print(f"  {name}: {count} bars ({count/len(meta_a_selections)*100:.1f}%)")

    print(f"\nMeta-B selection frequency:")
    for name, count in Counter(meta_b_selections).most_common():
        print(f"  {name}: {count} bars ({count/len(meta_b_selections)*100:.1f}%)")

    return results


def parse_strategy_arg(arg: str, checkpoint_base: Path) -> StrategySpec:
    """Parse 'name=checkpoint_dir:epoch' format."""
    name, rest = arg.split("=", 1)
    if ":" in rest:
        ckpt_dir, epoch = rest.rsplit(":", 1)
    else:
        ckpt_dir = rest
        epoch = None

    ckpt_path = checkpoint_base / ckpt_dir
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_path}")

    if epoch:
        pt_file = ckpt_path / f"{epoch}.pt"
        if not pt_file.exists():
            pt_file = ckpt_path / epoch
            if not pt_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {pt_file}")
    else:
        # Find latest epoch
        pts = sorted(ckpt_path.glob("epoch_*.pt"))
        if not pts:
            raise FileNotFoundError(f"No epoch_*.pt files in {ckpt_path}")
        pt_file = pts[-1]

    return StrategySpec(name=name, checkpoint_path=pt_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-switching strategy experiment")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument(
        "--strategy", action="append", required=True,
        help="Strategy spec: name=checkpoint_dir[:epoch] (can specify multiple)",
    )
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--fill-buffer-bps", type=float, default=0.0)
    parser.add_argument("--fee-rate", type=float, default=None)
    parser.add_argument("--one-side-per-bar", action="store_true")
    parser.add_argument("--meta-lookback-hours", type=int, default=24)
    parser.add_argument("--chronos-forecast-hours", type=int, default=6)
    parser.add_argument("--chronos-context-hours", type=int, default=168)
    parser.add_argument(
        "--checkpoint-base",
        default="binanceneural/checkpoints",
        help="Base directory for checkpoint dirs",
    )
    args = parser.parse_args()

    checkpoint_base = Path(args.checkpoint_base)
    strats = []
    for s in args.strategy:
        spec = parse_strategy_arg(s, checkpoint_base)
        spec.sequence_length = args.sequence_length
        strats.append(spec)

    fee_rate = args.fee_rate if args.fee_rate is not None else DEFAULT_MAKER_FEE_RATE
    sim_config = SimulationConfig(
        maker_fee=fee_rate,
        initial_cash=args.initial_cash,
        decision_lag_bars=args.decision_lag_bars,
        fill_buffer_bps=args.fill_buffer_bps,
        one_side_per_bar=args.one_side_per_bar,
    )

    run_meta_simulation(
        symbol=args.symbol,
        strategies=strats,
        sim_config=sim_config,
        horizon=args.horizon,
        meta_lookback_hours=args.meta_lookback_hours,
        chronos_forecast_hours=args.chronos_forecast_hours,
        chronos_context_hours=args.chronos_context_hours,
        cache_only=args.cache_only,
    )


if __name__ == "__main__":
    main()
