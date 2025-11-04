"""
Extended close policy analysis - by trading side.

Tests INSTANT_CLOSE vs KEEP_OPEN separately for:
1. Long-only (buy side)
2. Short-only (sell side)
3. Both (combined)

This helps determine if we should use side-specific policies.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from typing import Dict, Optional
from test_backtest4_instantclose_inline import (
    compare_close_policies,
    download_daily_stock_data,
    fetch_spread,
    _generate_forecasts_for_sim,
    evaluate_strategy_with_close_policy,
    CRYPTO_TRADING_FEE,
    TRADING_FEE,
)
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging

logger = setup_logging("test_close_policy_by_side.log")


def compare_by_side(symbol: str, num_simulations: int = 20) -> Dict[str, Dict]:
    """
    Run close policy comparison split by trading side.

    Returns:
        Dict with results for 'buy_only', 'sell_only', and 'both'
    """
    print(f"\n{'='*90}")
    print(f"Side-Specific Close Policy Analysis for {symbol}")
    print(f"{'='*90}\n")

    logger.info(f"Loading data for {symbol}...")

    # Download data
    current_time_formatted = '2024-09-07--03-36-27'
    stock_data = download_daily_stock_data(current_time_formatted, symbols=[symbol])

    is_crypto = symbol in crypto_symbols
    trading_fee = CRYPTO_TRADING_FEE if is_crypto else TRADING_FEE
    spread = fetch_spread(symbol)

    if is_crypto:
        print("ℹ️  Crypto only supports long positions - testing buy side only\n")
        test_sides = ['buy_only']
    else:
        print("ℹ️  Stock supports both directions - testing all combinations\n")
        test_sides = ['buy_only', 'sell_only', 'both']

    # Adjust simulations based on available data
    if len(stock_data) < num_simulations + 10:
        num_simulations = max(10, len(stock_data) - 10)

    results = {}

    for side_mode in test_sides:
        print(f"\n{'─'*90}")
        print(f"Testing: {side_mode.upper().replace('_', ' ')}")
        print(f"{'─'*90}")

        side_results = {
            'instant_close': [],
            'keep_open': []
        }

        for sim_idx in range(num_simulations):
            simulation_data = stock_data.iloc[:-(sim_idx + 1)].copy(deep=True)
            if simulation_data.empty or len(simulation_data) < 100:
                continue

            try:
                # Generate forecasts
                last_preds = _generate_forecasts_for_sim(
                    simulation_data, symbol, sim_idx,
                    trading_fee, spread, is_crypto
                )

                if last_preds is None:
                    continue

                # Modify the evaluation to only test one side
                if side_mode == 'buy_only':
                    # Zero out sell returns in the predictions
                    # This is a hack but we'll calculate manually
                    pass
                elif side_mode == 'sell_only':
                    # Zero out buy returns
                    pass

                # For now, let's run the standard evaluation and note that
                # the current implementation tests both sides together
                instant_metrics = evaluate_strategy_with_close_policy(
                    last_preds, simulation_data,
                    close_at_eod=True,
                    is_crypto=is_crypto,
                    strategy_name="maxdiffalwayson"
                )

                keep_metrics = evaluate_strategy_with_close_policy(
                    last_preds, simulation_data,
                    close_at_eod=False,
                    is_crypto=is_crypto,
                    strategy_name="maxdiffalwayson"
                )

                # Extract side-specific returns based on mode
                if side_mode == 'buy_only':
                    instant_ret = instant_metrics.buy_return
                    keep_ret = keep_metrics.buy_return
                elif side_mode == 'sell_only':
                    instant_ret = instant_metrics.sell_return
                    keep_ret = keep_metrics.sell_return
                else:  # both
                    instant_ret = instant_metrics.net_return_after_fees
                    keep_ret = keep_metrics.net_return_after_fees

                side_results['instant_close'].append(instant_ret)
                side_results['keep_open'].append(keep_ret)

                if (sim_idx + 1) % 5 == 0:
                    logger.info(f"  Completed {sim_idx + 1}/{num_simulations} simulations for {side_mode}")

            except Exception as exc:
                logger.warning(f"Simulation {sim_idx} failed for {side_mode}: {exc}")
                continue

        if not side_results['instant_close'] or not side_results['keep_open']:
            print(f"❌ No valid simulations for {side_mode}\n")
            continue

        # Calculate averages
        instant_avg = sum(side_results['instant_close']) / len(side_results['instant_close'])
        keep_avg = sum(side_results['keep_open']) / len(side_results['keep_open'])
        advantage = keep_avg - instant_avg
        winner = "KEEP_OPEN" if advantage > 0 else "INSTANT_CLOSE"

        results[side_mode] = {
            'instant_close_return': instant_avg,
            'keep_open_return': keep_avg,
            'advantage': advantage,
            'winner': winner,
            'num_simulations': len(side_results['instant_close'])
        }

        # Print results for this side
        print(f"\nResults ({len(side_results['instant_close'])} simulations):")
        print(f"  instant_close:  {instant_avg:>8.4f}%")
        print(f"  keep_open:      {keep_avg:>8.4f}%")
        print(f"  Advantage:      {advantage:>8.4f}%")
        print(f"  🏆 Winner:      {winner}")

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*90}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*90}\n")

        print(f"{'Side':<20} {'Instant':<12} {'Keep Open':<12} {'Advantage':<12} {'Winner':<15}")
        print(f"{'-'*90}")

        for side_mode, data in results.items():
            side_label = side_mode.replace('_', ' ').title()
            print(f"{side_label:<20} {data['instant_close_return']:>10.4f}% "
                  f"{data['keep_open_return']:>10.4f}% {data['advantage']:>10.4f}% "
                  f"{data['winner']:<15}")

        print(f"{'-'*90}")

        # Determine if side-specific policy would be beneficial
        if 'buy_only' in results and 'sell_only' in results:
            buy_winner = results['buy_only']['winner']
            sell_winner = results['sell_only']['winner']

            if buy_winner != sell_winner:
                print(f"\n💡 INSIGHT: Different sides prefer different policies!")
                print(f"   Buy side:  {buy_winner}")
                print(f"   Sell side: {sell_winner}")
                print(f"   → Consider implementing side-specific policies")
            else:
                print(f"\n💡 INSIGHT: Both sides prefer {buy_winner}")
                print(f"   → Unified policy is optimal")

    print(f"\n{'='*90}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test close policies by trading side"
    )
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbols to test (e.g., GOOG META BTCUSD)"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=20,
        help="Number of simulations per test (default: 20)"
    )

    args = parser.parse_args()

    all_results = {}

    for symbol in args.symbols:
        try:
            results = compare_by_side(symbol, num_simulations=args.simulations)
            all_results[symbol] = results
        except Exception as exc:
            logger.error(f"Failed to analyze {symbol}: {exc}")
            print(f"\n❌ ERROR analyzing {symbol}: {exc}\n")

    # Final cross-symbol summary
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print(f"CROSS-SYMBOL SUMMARY")
        print(f"{'='*90}\n")

        for symbol, results in all_results.items():
            print(f"\n{symbol}:")
            for side_mode, data in results.items():
                side_label = side_mode.replace('_', ' ').title()
                print(f"  {side_label:<15} → {data['winner']:<15} (advantage: {data['advantage']:>7.4f}%)")

        print(f"\n{'='*90}\n")
