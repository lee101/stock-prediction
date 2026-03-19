#!/usr/bin/env python3
"""Compare single-best vs multi-position selector strategies."""
from pathlib import Path
import pandas as pd
from binanceexp1.run_multiasset_selector import _load_symbol_data, _load_model, _parse_kv_pairs
from binanceneural.inference import generate_actions_from_frame
from newnanoalpacahourlyexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
CHECKPOINTS = {
    "BTCUSD": "/home/lee/code/stock/binanceneural/checkpoints/btcusd_h1only_ft30_20260208/epoch_026.pt",
    "ETHUSD": "/home/lee/code/stock/binanceneural/checkpoints/ethusd_h1only_ft30_20260208/epoch_027.pt",
    "SOLUSD": "/home/lee/code/stock/binanceneural/checkpoints/solusd_h1only_ft30_20260208/epoch_026.pt",
}
DATA_ROOT = Path("trainingdatahourlybinance")
FORECAST_ROOT = Path("binanceneural/forecast_cache")
SEQ_LEN = 96
VAL_DAYS = 7
HORIZON = 1

def load_data():
    bars_frames, actions_frames = [], []
    for symbol in SYMBOLS:
        data = _load_symbol_data(
            symbol,
            data_root=DATA_ROOT,
            forecast_cache_root=FORECAST_ROOT,
            sequence_length=SEQ_LEN,
            forecast_horizons=[1, 24],
            cache_only=True,
            validation_days=VAL_DAYS,
        )
        frame = data.val_dataset.frame.copy()
        frame["symbol"] = symbol

        model = _load_model(Path(CHECKPOINTS[symbol]), len(data.feature_columns), SEQ_LEN)
        actions = generate_actions_from_frame(
            model=model, frame=frame, feature_columns=data.feature_columns,
            normalizer=data.normalizer, sequence_length=SEQ_LEN, horizon=HORIZON,
        )
        bars_frames.append(frame)
        actions_frames.append(actions)
    return pd.concat(bars_frames, ignore_index=True), pd.concat(actions_frames, ignore_index=True)

def run_sim(bars, actions, max_pos=1, work_steal=False):
    cfg = SelectionConfig(
        initial_cash=5000.0,
        min_edge=0.0,
        risk_weight=0.0,
        max_hold_hours=6,
        symbols=SYMBOLS,
        max_concurrent_positions=max_pos,
        work_steal_enabled=work_steal,
        work_steal_min_profit_pct=0.0,
        work_steal_min_edge=0.0,
        work_steal_edge_margin=0.0,
    )
    return run_best_trade_simulation(bars, actions, cfg, horizon=HORIZON)

if __name__ == "__main__":
    print("Loading data...")
    bars, actions = load_data()
    print(f"Bars: {len(bars)}, Actions: {len(actions)}")

    print("\n=== Single-Best Entry (max_concurrent_positions=1) ===")
    r1 = run_sim(bars, actions, max_pos=1, work_steal=False)
    print(f"total_return: {r1.metrics['total_return']:.4f}")
    print(f"sortino: {r1.metrics['sortino']:.4f}")
    print(f"n_trades: {len(r1.trades)}")

    print("\n=== Multi-Position Entry (max_concurrent_positions=3) ===")
    r2 = run_sim(bars, actions, max_pos=3, work_steal=False)
    print(f"total_return: {r2.metrics['total_return']:.4f}")
    print(f"sortino: {r2.metrics['sortino']:.4f}")
    print(f"n_trades: {len(r2.trades)}")

    print("\n=== Single-Best + Work-Steal ===")
    r3 = run_sim(bars, actions, max_pos=1, work_steal=True)
    print(f"total_return: {r3.metrics['total_return']:.4f}")
    print(f"sortino: {r3.metrics['sortino']:.4f}")
    print(f"n_trades: {len(r3.trades)}")
    n_steals = sum(1 for t in r3.trades if t.reason == "work_steal_exit")
    print(f"n_work_steals: {n_steals}")

    print("\n=== Summary ===")
    print(f"Single-Best:     {r1.metrics['total_return']*100:.2f}% return")
    print(f"Multi-Position:  {r2.metrics['total_return']*100:.2f}% return")
    print(f"Single+WorkSteal:{r3.metrics['total_return']*100:.2f}% return")
