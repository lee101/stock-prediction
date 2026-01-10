#!/usr/bin/env python3
"""Continue per-symbol tuning for remaining symbols."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from marketsimlong.config import DataConfigLong
from marketsimlong.data import DailyDataLoader
from marketsimlong.per_symbol_tuner import PerSymbolChronos2Tuner, SymbolTuningConfig

# Remaining symbols to tune
REMAINING_SYMBOLS = [
    "CRM",
    "COST",
    "COIN",
    "SHOP",
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "UNIUSD",
]


def main():
    data_config = DataConfigLong(
        data_root=Path("trainingdata/train"),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 22),
    )

    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()

    # Filter to only remaining symbols that exist in data
    symbols_to_tune = [s for s in REMAINING_SYMBOLS if s in data_loader._data_cache]
    print(f"Tuning {len(symbols_to_tune)} remaining symbols: {symbols_to_tune}")

    tuning_config = SymbolTuningConfig()
    tuner = PerSymbolChronos2Tuner(data_loader, tuning_config)

    try:
        results = tuner.tune_all_symbols(symbols_to_tune)

        print("\n" + "=" * 70)
        print("REMAINING SYMBOLS TUNING RESULTS")
        print("=" * 70)
        print(f"{'Symbol':<12} {'MAE%':<10} {'DirAcc%':<10} {'Ctx':<6} {'MV':<6} {'Preaug':<15} {'Skip':<10}")
        print("-" * 70)

        for symbol, cfg in sorted(results.items(), key=lambda x: x[1].mae_pct):
            print(
                f"{symbol:<12} {cfg.mae_pct:<10.2f} {cfg.directional_accuracy:<10.1f} "
                f"{cfg.context_length:<6} {str(cfg.use_multivariate):<6} {cfg.preaug_strategy:<15} {str(cfg.skip_rates):<10}"
            )

        print("=" * 70)
        return results

    finally:
        tuner.unload()


if __name__ == "__main__":
    main()
