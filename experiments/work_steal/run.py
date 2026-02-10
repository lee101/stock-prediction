"""Work-stealing experiment: sell profitable position to enter better opportunity.

Baseline: ft30 selector 2478x (70d val, BTC ep26 + ETH ep27 + SOL ep26)
Sweep work-steal params to find optimal settings for production.
Uses FDUSD 0-fee pairs matching live config.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from src.torch_load_utils import torch_load_compat
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation

from binanceexp1.config import DatasetConfig
from binanceexp1.data import BinanceExp1DataModule
from binanceexp1.sweep import apply_action_overrides

RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
DEFAULT_OFFSETS = {"BTCUSD": 0.0, "ETHUSD": 0.0003, "SOLUSD": 0.0005}
FDUSD_FEES = {"BTCUSD": 0.0, "ETHUSD": 0.0, "SOLUSD": 0.0}

MIN_PROFITS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.01]
MIN_EDGES = [0.0, 0.001, 0.003, 0.005, 0.008, 0.01, 0.015]
EDGE_MARGINS = [0.0, 0.001, 0.003, 0.005, 0.01]


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


def load_data(
    symbols: List[str],
    checkpoint_map: Dict[str, str],
    *,
    sequence_length: int = 96,
    horizon: int = 1,
    validation_days: float = 70,
    cache_only: bool = True,
    offset_map: Optional[Dict[str, float]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offset_map = offset_map or DEFAULT_OFFSETS
    bars_list, actions_list = [], []
    for symbol in symbols:
        ckpt_path = Path(checkpoint_map[symbol]).expanduser()
        data_cfg = DatasetConfig(
            symbol=symbol,
            data_root=Path("trainingdatahourly/crypto"),
            forecast_cache_root=Path("binanceneural/forecast_cache"),
            sequence_length=sequence_length,
            forecast_horizons=(horizon,),
            cache_only=cache_only,
            validation_days=validation_days,
        )
        data = BinanceExp1DataModule(data_cfg)
        frame = data.val_dataset.frame.copy()
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol
        model = _load_model(ckpt_path, len(data.feature_columns), sequence_length)
        actions = generate_actions_from_frame(
            model=model, frame=frame, feature_columns=data.feature_columns,
            normalizer=data.normalizer, sequence_length=sequence_length, horizon=horizon,
        )
        offset = offset_map.get(symbol, 0.0)
        if offset != 0.0:
            actions = apply_action_overrides(actions, intensity_scale=1.0, price_offset_pct=offset)
        bars_list.append(frame)
        actions_list.append(actions)
    return pd.concat(bars_list, ignore_index=True), pd.concat(actions_list, ignore_index=True)


def run_config(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    symbols: List[str],
    *,
    work_steal_enabled: bool,
    min_profit: float = 0.001,
    min_edge: float = 0.005,
    edge_margin: float = 0.0,
    horizon: int = 1,
    initial_cash: float = 5092.0,
    fee_by_symbol: Optional[Dict[str, float]] = None,
) -> dict:
    cfg = SelectionConfig(
        initial_cash=initial_cash,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        max_hold_hours=6,
        symbols=symbols,
        allow_reentry_same_bar=True,
        fee_by_symbol=fee_by_symbol or FDUSD_FEES,
        work_steal_enabled=work_steal_enabled,
        work_steal_min_profit_pct=min_profit,
        work_steal_min_edge=min_edge,
        work_steal_edge_margin=edge_margin,
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=horizon)
    n_steals = sum(1 for t in result.trades if t.reason == "work_steal_exit")
    n_steal_entries = sum(1 for t in result.trades if t.reason == "work_steal_entry")
    steal_syms = {}
    for t in result.trades:
        if t.reason == "work_steal_exit":
            steal_syms[t.symbol] = steal_syms.get(t.symbol, 0) + 1
    return {
        "work_steal_enabled": work_steal_enabled,
        "min_profit": min_profit,
        "min_edge": min_edge,
        "edge_margin": edge_margin,
        "total_return": result.metrics.get("total_return", 0.0),
        "sortino": result.metrics.get("sortino", 0.0),
        "n_trades": len(result.trades),
        "n_work_steals": n_steals,
        "n_work_steal_entries": n_steal_entries,
        "steal_by_symbol": steal_syms,
        "final_equity": float(result.equity_curve.iloc[-1]) if not result.equity_curve.empty else 0.0,
        "max_drawdown": _max_drawdown(result.equity_curve),
    }


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True, help="SYMBOL=PATH mapping")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--validation-days", type=float, default=70)
    parser.add_argument("--initial-cash", type=float, default=5092.0)
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--quick", action="store_true", help="Reduced grid for fast iteration")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    checkpoint_map = {}
    for token in args.checkpoints.split(","):
        k, v = token.strip().split("=", 1)
        checkpoint_map[k.strip().upper()] = v.strip()

    print(f"Loading data for {symbols}...")
    t0 = time.time()
    bars, actions = load_data(symbols, checkpoint_map, validation_days=args.validation_days, cache_only=args.cache_only)
    print(f"Data loaded in {time.time() - t0:.1f}s ({len(bars)} rows)")

    results = []

    print("\n=== Baseline (no work-steal, 0-fee FDUSD) ===")
    baseline = run_config(bars, actions, symbols, work_steal_enabled=False, initial_cash=args.initial_cash)
    results.append(baseline)
    print(f"  return={baseline['total_return']:.4f}x  sortino={baseline['sortino']:.2f}  trades={baseline['n_trades']}  dd={baseline['max_drawdown']:.4f}")

    print("\n=== Also baseline with default 10bps fees (pessimistic) ===")
    baseline_fees = run_config(bars, actions, symbols, work_steal_enabled=False, initial_cash=args.initial_cash, fee_by_symbol=None)
    baseline_fees["label"] = "baseline_10bps"
    results.append(baseline_fees)
    print(f"  return={baseline_fees['total_return']:.4f}x  sortino={baseline_fees['sortino']:.2f}  trades={baseline_fees['n_trades']}  dd={baseline_fees['max_drawdown']:.4f}")

    if args.quick:
        min_profits = [0.0, 0.002, 0.005]
        min_edges = [0.0, 0.003, 0.008]
        edge_margins = [0.0, 0.003]
    else:
        min_profits = MIN_PROFITS
        min_edges = MIN_EDGES
        edge_margins = EDGE_MARGINS

    grid = list(product(min_profits, min_edges, edge_margins))
    print(f"\n=== Work-steal grid: {len(grid)} configs (0-fee FDUSD) ===")

    for i, (mp, me, em) in enumerate(grid):
        r = run_config(
            bars, actions, symbols,
            work_steal_enabled=True,
            min_profit=mp, min_edge=me, edge_margin=em,
            initial_cash=args.initial_cash,
        )
        results.append(r)
        delta = r["total_return"] - baseline["total_return"]
        tag = "+" if delta > 0 else ""
        print(
            f"  [{i:3d}/{len(grid)}] mp={mp:.3f} me={me:.3f} em={em:.3f} | "
            f"ret={r['total_return']:.4f}x ({tag}{delta:.4f}) sort={r['sortino']:.2f} "
            f"trades={r['n_trades']} steals={r['n_work_steals']} dd={r['max_drawdown']:.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "work_steal_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    ws_results = [r for r in results if r.get("work_steal_enabled")]
    if ws_results:
        best = max(ws_results, key=lambda r: r["total_return"])
        best_sortino = max(ws_results, key=lambda r: r["sortino"])
        print(f"\n=== Best by return ===")
        print(f"  return={best['total_return']:.4f}x  sortino={best['sortino']:.2f}  steals={best['n_work_steals']}")
        print(f"  min_profit={best['min_profit']}  min_edge={best['min_edge']}  edge_margin={best['edge_margin']}")
        print(f"  dd={best['max_drawdown']:.4f}  trades={best['n_trades']}")
        delta = best["total_return"] - baseline["total_return"]
        print(f"  vs baseline: {delta:+.4f}x")

        print(f"\n=== Best by sortino ===")
        print(f"  return={best_sortino['total_return']:.4f}x  sortino={best_sortino['sortino']:.2f}  steals={best_sortino['n_work_steals']}")
        print(f"  min_profit={best_sortino['min_profit']}  min_edge={best_sortino['min_edge']}  edge_margin={best_sortino['edge_margin']}")

        improved = [r for r in ws_results if r["total_return"] > baseline["total_return"]]
        print(f"\n{len(improved)}/{len(ws_results)} configs beat baseline")


if __name__ == "__main__":
    main()
