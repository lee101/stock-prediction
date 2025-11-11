#!/usr/bin/env python3
"""
Comprehensive market simulation script with all optimizations enabled.

This script runs a 20-day simulation across all configured stocks with:
- Disk caching for predictions
- torch.compile for faster inference
- FAST_SIMULATE mode for optimized backtesting
- WandB logging to 'marketsimulator' project
- Real inference and analytics
- 1 day at a time simulation for realism
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Set environment variables BEFORE importing any modules
os.environ["WANDB_PROJECT"] = "marketsimulator"
os.environ["MARKETSIM_CACHE_DISK"] = "1"  # Enable disk caching for predictions
os.environ["MARKETSIM_CACHE_PREDICTIONS"] = "1"  # Enable prediction caching
os.environ["MARKETSIM_FAST_SIMULATE"] = "1"  # Enable FAST_SIMULATE mode
os.environ["FAST_TESTING"] = "1"  # Enable fast testing mode
os.environ["MARKETSIM_BACKTEST_SIMULATIONS"] = "24"  # Reduce simulations for speed
os.environ["MARKETSIM_SWEEP_POINTS"] = "101"  # Sweep points for optimization
os.environ["MARKETSIM_KRONOS_SAMPLE_COUNT"] = "64"  # Kronos samples

# Enable torch.compile for toto models
os.environ["TOTO_COMPILE"] = "1"
os.environ["MARKETSIM_TOTO_COMPILE"] = "1"

# Ensure we use real analytics (not mocks)
os.environ["MARKETSIM_USE_MOCK_ANALYTICS"] = "0"

# Add repo root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from marketsimulator.run_trade_loop import main as run_simulation


def load_symbols() -> list[str]:
    """Load symbols from symbolsofinterest.txt"""
    symbols_file = repo_root / "symbolsofinterest.txt"
    if not symbols_file.exists():
        print(f"Error: {symbols_file} not found!")
        return []

    # Read and parse the symbols file
    with open(symbols_file, 'r') as f:
        content = f.read()

    # Extract symbols from the Python list
    symbols = []
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('symbols'):
            # Remove quotes and commas
            symbol = line.strip("',\"").strip()
            if symbol and symbol not in ['[', ']']:
                symbols.append(symbol)

    # Filter out commented symbols and ensure we have valid ones
    valid_symbols = [s for s in symbols if s and not s.startswith('#')]

    # Default to common stocks if parsing failed
    if not valid_symbols:
        valid_symbols = [
            'COUR', 'GOOG', 'TSLA', 'NVDA', 'AAPL', 'U', 'ADSK', 'ADBE',
            'COIN', 'MSFT', 'NFLX', 'PYPL', 'SAP', 'SONY', 'BTCUSD', 'ETHUSD'
        ]

    return valid_symbols


def parse_results(output: str) -> dict:
    """Parse simulation output to extract key metrics"""
    metrics = {}

    for line in output.split('\n'):
        if 'sim-summary=' in line:
            try:
                json_str = line.split('sim-summary=')[1]
                metrics = json.loads(json_str)
            except Exception as e:
                print(f"Failed to parse sim-summary: {e}")
        elif '=' in line and any(key in line for key in ['return', 'sharpe', 'pnl', 'balance']):
            try:
                key, value = line.split('=', 1)
                metrics[key.strip()] = float(value.strip())
            except Exception:
                pass

    return metrics


def generate_pnl_report(metrics: dict, symbols: list[str], output_dir: Path):
    """Generate PnL report by stock pairs"""
    report_file = output_dir / f"pnl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(report_file, 'w') as f:
        f.write("# Market Simulation PnL Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Overall Performance\n\n")
        f.write(f"- **Total Return**: ${metrics.get('pnl', 0):.2f} ({metrics.get('return', 0) * 100:.2f}%)\n")
        f.write(f"- **Sharpe Ratio**: {metrics.get('sharpe', 0):.4f}\n")
        f.write(f"- **Final Balance**: ${metrics.get('balance', 0):.2f}\n")
        f.write(f"- **Max Drawdown**: {metrics.get('max_drawdown_pct', 0) * 100:.2f}%\n")
        f.write(f"- **Fees Paid**: ${metrics.get('fees_paid', 0):.2f}\n")
        f.write(f"- **Total Trades**: {metrics.get('trades_executed', 0)}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Symbols**: {', '.join(symbols)}\n")
        f.write(f"- **Days Simulated**: {metrics.get('steps', 20)}\n")
        f.write(f"- **Step Size**: {metrics.get('step_size', 1)}\n")
        f.write(f"- **Initial Cash**: ${metrics.get('initial_cash', 100000):.2f}\n")
        f.write(f"- **Top K Positions**: {metrics.get('top_k', 4)}\n\n")

        f.write("## Optimizations Enabled\n\n")
        f.write("- Disk caching for predictions\n")
        f.write("- torch.compile for inference acceleration\n")
        f.write("- FAST_SIMULATE mode\n")
        f.write("- Real inference and analytics\n")
        f.write("- Strategy selection by forecasted PnL\n\n")

        # Extract symbol-specific performance if available
        analysis_summary = metrics.get('analysis_summary', {})
        if analysis_summary:
            f.write("## Strategy Usage\n\n")
            strategy_counts = analysis_summary.get('strategy_counts', {})
            for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"- **{strategy}**: {count} instances\n")
            f.write("\n")

        f.write("## Notes\n\n")
        f.write("This simulation uses the latest strategy selection logic from trade_stock_e2e.py, ")
        f.write("which prioritizes strategies with positive forecasted PnL and includes maxdiff, ")
        f.write("maxdiffalwayson, takeprofit, highlow, all_signals, and simple strategies.\n")

    print(f"\nPnL report written to: {report_file}")
    return report_file


def main():
    print("=" * 80)
    print("COMPREHENSIVE MARKET SIMULATION")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  WandB Project: {os.environ.get('WANDB_PROJECT')}")
    print(f"  Disk Caching: {os.environ.get('MARKETSIM_CACHE_DISK')}")
    print(f"  FAST_SIMULATE: {os.environ.get('MARKETSIM_FAST_SIMULATE')}")
    print(f"  torch.compile: {os.environ.get('TOTO_COMPILE')}")
    print(f"  Real Analytics: {os.environ.get('MARKETSIM_USE_MOCK_ANALYTICS') == '0'}")

    # Load symbols
    symbols = load_symbols()
    print(f"\nSymbols to simulate ({len(symbols)}): {', '.join(symbols)}")

    # Prepare output directory
    output_dir = repo_root / "simulation_results"
    output_dir.mkdir(exist_ok=True)

    metrics_json = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    metrics_csv = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades_csv = output_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades_summary_json = output_dir / f"trades_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Build command line arguments
    args = [
        "--symbols", *symbols,
        "--steps", "20",  # 20 days
        "--step-size", "1",  # 1 day at a time for realism
        "--initial-cash", "100000.0",
        "--top-k", "4",
        "--real-analytics",  # Use real forecasting/backtest stack
        "--flatten-end",  # Close all positions at end
        "--fast-sim",  # Enable fast simulation mode
        "--metrics-json", str(metrics_json),
        "--metrics-csv", str(metrics_csv),
        "--trades-csv", str(trades_csv),
        "--trades-summary-json", str(trades_summary_json),
    ]

    print(f"\nStarting 20-day simulation...")
    print(f"  Output directory: {output_dir}")
    print(f"  Metrics will be logged to WandB project: marketsimulator")
    print("\nThis may take a while with real inference enabled...\n")

    # Run simulation
    try:
        exit_code = run_simulation(args)

        if exit_code == 0:
            print("\n" + "=" * 80)
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print("=" * 80)

            # Load and display results
            if metrics_json.exists():
                with open(metrics_json, 'r') as f:
                    metrics = json.load(f)

                # Generate PnL report
                report_file = generate_pnl_report(metrics, symbols, output_dir)

                print(f"\nResults:")
                print(f"  Metrics JSON: {metrics_json}")
                print(f"  Metrics CSV: {metrics_csv}")
                print(f"  Trades CSV: {trades_csv}")
                print(f"  Trades Summary: {trades_summary_json}")
                print(f"  PnL Report: {report_file}")
                print(f"\nCheck WandB project 'marketsimulator' for detailed metrics and visualizations.")

                return 0
            else:
                print(f"\nWarning: Metrics file not found at {metrics_json}")
                return 1
        else:
            print(f"\nSimulation failed with exit code {exit_code}")
            return exit_code

    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
