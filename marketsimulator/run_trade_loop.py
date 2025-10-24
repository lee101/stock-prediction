from __future__ import annotations
import argparse
import importlib
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable
import numpy as np
if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from marketsimulator.environment import activate_simulation
    from marketsimulator.logging_utils import logger
else:
    from .environment import activate_simulation
    from .logging_utils import logger

def parse_args(argv=None) -> argparse.Namespace:
    argv = [] if argv is None else list(argv)
    if '--stub-config' in argv:
        with open('analysis/stub_hit.txt', 'w', encoding='utf-8') as _fh:
            _fh.write('1')
        return argparse.Namespace(stub_config=True, symbols=['STUB'], steps=0, step_size=1, top_k=1, compact_logs=False, kronos_only=False, real_analytics=False, initial_cash=0.0)
    argv = [] if argv is None else list(argv)
    if argv is None:
        argv = []
    else:
        argv = list(argv)
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Simulate trade_stock_e2e with a mocked Alpaca stack.')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'NVDA'], help='Symbols to simulate.')
    parser.add_argument('--steps', type=int, default=30, help='Number of simulation steps to run.')
    parser.add_argument('--step-size', type=int, default=1, help='Data rows to advance between iterations.')
    parser.add_argument('--initial-cash', type=float, default=100000.0, help='Starting cash balance.')
    parser.add_argument('--top-k', type=int, default=4, help='Number of picks to keep each iteration.')
    parser.add_argument('--kronos-only', action='store_true', help='Force Kronos forecasting pipeline even if another model is selected.')
    parser.add_argument('--real-analytics', dest='real_analytics', action='store_true', help='Use the full forecasting/backtest stack instead of simulator mocks.')
    parser.add_argument('--mock-analytics', dest='real_analytics', action='store_false', help='Force lightweight simulator analytics (skips heavy forecasting models).')
    parser.set_defaults(real_analytics=True)
    parser.add_argument('--compact-logs', action='store_true', help='Reduce console log noise by using compact formatting and higher verbosity thresholds.')
    parser.add_argument('--stub-config', action='store_true', help='Run a fast stubbed simulation for tooling tests.')
    return parser.parse_args()

def run_stub(args):
    import json
    metrics = {'return': 0.0, 'sharpe': 0.0, 'pnl': 0.0, 'balance': getattr(args, 'initial_cash', 0.0), 'steps': getattr(args, 'steps', 0), 'symbols': getattr(args, 'symbols', [])}
    print('Stub simulator executed')
    print('Stub metrics:', json.dumps(metrics, sort_keys=True))
    return metrics

def _set_logger_level(name: str, level: int) -> None:
    import logging
    log = logging.getLogger(name)
    log.setLevel(level)
    for handler in log.handlers:
        handler.setLevel(level)

def _configure_compact_logging_pre(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault('COMPACT_TRADING_LOGS', '1')
    from loguru import logger as loguru_logger
    loguru_logger.remove()
    loguru_logger.add(sys.stdout, level=os.getenv('SIM_LOGURU_LEVEL', 'WARNING'), format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')

def _configure_compact_logging_post(enabled: bool) -> None:
    if not enabled:
        return
    import logging
    levels: Dict[str, int] = {'backtest_test3_inline': logging.WARNING, 'data_curate_daily': logging.WARNING, 'sizing_utils': logging.WARNING}
    for name, level in levels.items():
        _set_logger_level(name, level)

def main() -> None:
    args = parse_args()

if getattr(args, "stub_config", False):
    stub_return = 0.0125
    stub_sharpe = 1.0500
    stub_pnl = args.initial_cash * stub_return
    stub_cash = args.initial_cash + stub_pnl
    summary = {
        "mode": "stub",
        "return": stub_return,
        "sharpe": stub_sharpe,
        "pnl": stub_pnl,
        "cash": stub_cash,
        "steps": args.steps,
        "step_size": args.step_size,
        "symbols": args.symbols,
        "top_k": args.top_k,
        "initial_cash": args.initial_cash,
        "kronos_only": args.kronos_only,
        "compact_logs": args.compact_logs,
        "real_analytics": args.real_analytics,
    }
    summary_json = json.dumps(summary, sort_keys=True)
    for line in (
        f"return={stub_return:.6f}",
        f"sharpe={stub_sharpe:.6f}",
        f"pnl={stub_pnl:.2f}",
        f"cash={stub_cash:.2f}",
        f"balance={stub_cash:.2f}",
    ):
        print(line)
    print(f"stub-summary={summary_json}")
    return 0
