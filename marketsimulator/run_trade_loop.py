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

def parse_args() -> argparse.Namespace:
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
    if getattr(args, 'stub_config', False):
        run_stub(args)
        return
    _configure_compact_logging_pre(args.compact_logs)
    mode = 'real' if args.real_analytics else 'mock'
    logger.info(f'[sim] Analytics mode set to {mode.upper()} forecasting stack.')
    with activate_simulation(symbols=args.symbols, initial_cash=args.initial_cash, use_mock_analytics=not args.real_analytics, force_kronos=args.kronos_only) as controller:
        trade_module = importlib.import_module('trade_stock_e2e')
        _configure_compact_logging_post(args.compact_logs)
        previous_picks = {}
        step_size = args.step_size if args.step_size and args.step_size > 0 else 1
        start_timestamp = controller.current_time()
        initial_value = float(args.initial_cash)
        for step in range(args.steps):
            timestamp = controller.current_time()
            logger.info(f'[sim] Step {step + 1}/{args.steps} @ {timestamp}')
            analyzed = trade_module.analyze_symbols(args.symbols)
            current = {symbol: data for symbol, data in list(analyzed.items())[:args.top_k] if data['avg_return'] > 0}
            if current:
                trade_module.log_trading_plan(current, f'SIM-STEP-{step + 1}')
            trade_module.manage_positions(current, previous_picks, analyzed)
            previous_picks = current
            controller.advance_steps(step_size)
        end_timestamp = controller.current_time()
        summary = controller.summary()
        logger.info(f'[sim] Final summary: cash={summary['cash']:.2f}, equity={summary['equity']:.2f}')
        open_positions_detail: Dict[str, float] = {}
        if summary['positions']:
            logger.info(f'[sim] Open positions: {summary['positions']}')
            detail = summary.get('positions_detail', {})
            for symbol, meta in detail.items():
                side = meta.get('side', 'n/a')
                qty = float(meta.get('qty', 0.0) or 0.0)
                price = float(meta.get('price', 0.0) or 0.0)
                value = float(meta.get('market_value', 0.0) or 0.0)
                open_positions_detail[symbol] = open_positions_detail.get(symbol, 0.0) + value
                logger.info(f'[sim] Position detail: {symbol} side={side} qty={qty:.6f} price={price:.4f} value={value:.2f}')
        total_equity = float(summary.get('equity', 0.0))
        pnl = total_equity - initial_value
        simple_return = pnl / initial_value if initial_value else 0.0
        elapsed = end_timestamp - start_timestamp
        elapsed_days = elapsed.total_seconds() / 86400.0
        if elapsed_days <= 0:
            effective_steps = max(args.steps * step_size, 1)
            elapsed_days = max(effective_steps / 24.0, 1.0 / 24.0)
        start_date = start_timestamp.date()
        end_date = end_timestamp.date()
        trading_days = int(np.busday_count(start_date.isoformat(), (end_date + timedelta(days=1)).isoformat()))
        if trading_days <= 0:
            trading_days = max(1, int(round(elapsed_days * 252 / 365.0)))

        def _annualize(return_ratio: float, periods: float, periods_per_year: float) -> float:
            if periods <= 0:
                return float('nan')
            if return_ratio <= -1.0:
                return float('-1.0')
            try:
                return (1.0 + return_ratio) ** (periods_per_year / periods) - 1.0
            except OverflowError:
                return float('inf')
        ann_calendar = _annualize(simple_return, elapsed_days, 365.0)
        ann_trading = _annualize(simple_return, trading_days, 252.0)
        ann_trading_pct = ann_trading * 100.0 if not np.isnan(ann_trading) else float('nan')
        ann_calendar_pct = ann_calendar * 100.0 if not np.isnan(ann_calendar) else float('nan')
        logger.info(f'[sim] PnL summary: pnl={pnl:+.2f} ({simple_return * 100.0:+.2f}%), ann_252={ann_trading_pct:+.2f}% ({trading_days} trading days), ann_365={ann_calendar_pct:+.2f}% ({elapsed_days:.2f} calendar days)')
        if 'liquidation_value' in summary:
            logger.info(f'[sim] Liquidation value: {summary['liquidation_value']:.2f}')
        realised_by_symbol: Dict[str, float] = {}
        history_accessor = getattr(trade_module, '_get_trade_history_store', None)
        if callable(history_accessor):
            history_store = history_accessor()
            if history_store is not None:
                try:
                    history_store.load()
                except Exception as exc:
                    logger.warning(f'[sim] Unable to load trade history store: {exc}')
                else:
                    histories: Iterable = history_store.values()
                    for entries in histories:
                        if not isinstance(entries, list):
                            continue
                        for record in entries:
                            if not isinstance(record, dict):
                                continue
                            symbol = record.get('symbol')
                            if not symbol:
                                continue
                            try:
                                pnl_value = float(record.get('pnl', 0.0) or 0.0)
                            except (TypeError, ValueError):
                                pnl_value = 0.0
                            realised_by_symbol[symbol] = realised_by_symbol.get(symbol, 0.0) + pnl_value
        if realised_by_symbol or open_positions_detail:
            logger.info('[sim] Symbol PnL breakdown:')
            all_symbols = sorted(set(realised_by_symbol) | set(open_positions_detail))
            for symbol in all_symbols:
                realised = realised_by_symbol.get(symbol, 0.0)
                open_mv = open_positions_detail.get(symbol, 0.0)
                total = realised + open_mv
                logger.info(f'[sim]   {symbol}: realised={realised:+.2f}, open={open_mv:+.2f}, total={total:+.2f}')
if __name__ == '__main__':
    main()
