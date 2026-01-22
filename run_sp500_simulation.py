#!/usr/bin/env python3
"""
Run market simulation on all S&P 500 stocks to evaluate PnL.
Tests the marketsimlong daily trading strategy across the full S&P 500 universe.
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# S&P 500 symbols
SP500_SYMBOLS = tuple([
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB",
    "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN",
    "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI",
    "AON", "APA", "APO", "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T",
    "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX",
    "BBY", "TECH", "BIIB", "BLK", "BX", "BK", "BA", "BKNG", "BSX", "BMY", "AVGO", "BR",
    "BRO", "BLDR", "BG", "BXP", "CHRW", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", "KMX",
    "CCL", "CARR", "CAT", "CBOE", "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW",
    "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX",
    "CME", "CMS", "KO", "CTSH", "COIN", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG",
    "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX",
    "CMI", "CVS", "DHR", "DRI", "DDOG", "DVA", "DAY", "DECK", "DE", "DELL", "DAL", "DVN",
    "DXCM", "FANG", "DLR", "DG", "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE",
    "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "ELV", "EMR", "ENPH",
    "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG", "EVRG",
    "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT",
    "FDX", "FIS", "FITB", "FSLR", "FE", "FI", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN",
    "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM", "GPC",
    "GILD", "GPN", "GL", "GDDY", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY",
    "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN",
    "HII", "IBM", "IEX", "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP",
    "IPG", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ",
    "JCI", "JPM", "K", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC",
    "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN",
    "LYV", "LKQ", "LMT", "L", "LOW", "LULU", "LYB", "MTB", "MPC", "MKTX", "MAR", "MMC",
    "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD",
    "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR",
    "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA",
    "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA",
    "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR",
    "PKG", "PLTR", "PANW", "PSKY", "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE",
    "PCG", "PM", "PSX", "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD",
    "PRU", "PEG", "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O",
    "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI",
    "CRM", "SBAC", "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA",
    "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS",
    "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA",
    "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT", "TDG", "TRV", "TRMB",
    "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH",
    "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VTRS", "VICI", "V", "VST",
    "VMC", "WRB", "GWW", "WAB", "WBA", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC",
    "WELL", "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM",
    "ZBRA", "ZBH", "ZTS"
])

# Also include crypto for comparison
CRYPTO_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD")

# Curated subset of S&P 500 with known good data (50 diverse stocks)
SP500_CURATED = tuple([
    # Tech
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'AVGO',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'LLY', 'TMO', 'DHR', 'ABT', 'BMY',
    # Consumer
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS',
    # Energy
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
])


def check_data_availability(data_dir: Path, symbols: tuple) -> tuple:
    """Check which symbols have data available."""
    available = []
    missing = []
    for sym in symbols:
        csv_path = data_dir / f"{sym}.csv"
        if csv_path.exists():
            # Check if file has enough data (at least 60 days)
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if len(df) >= 60:
                    available.append(sym)
                else:
                    missing.append(sym)
            except Exception:
                missing.append(sym)
        else:
            missing.append(sym)
    return tuple(available), tuple(missing)


def run_sp500_simulation(
    symbols: tuple,
    start_date: date,
    end_date: date,
    top_n: int = 1,
    initial_cash: float = 100_000.0,
    output_dir: Path = Path("reports/sp500_simulation"),
) -> dict:
    """Run simulation on the given symbols."""
    from marketsimlong.config import DataConfigLong, ForecastConfigLong, SimulationConfigLong
    from marketsimlong.simulator import run_simulation

    # Create configs with expanded symbol set
    data_config = DataConfigLong(
        stock_symbols=symbols,
        crypto_symbols=(),  # Focus on stocks for this run
        data_root=Path("trainingdata/train"),
        start_date=start_date,
        end_date=end_date,
    )

    forecast_config = ForecastConfigLong(
        context_length=512,
        prediction_length=1,
        use_multivariate=True,
    )

    sim_config = SimulationConfigLong(
        top_n=top_n,
        initial_cash=initial_cash,
    )

    print(f"\nRunning simulation on {len(symbols)} symbols...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Top N: {top_n}, Initial Cash: ${initial_cash:,.0f}")
    print()

    def progress_callback(day_num: int, total_days: int, day_result):
        if day_num % 10 == 0:
            pnl = day_result.ending_portfolio_value - initial_cash
            pnl_pct = (pnl / initial_cash) * 100
            print(
                f"  Day {day_num:3d}/{total_days} | "
                f"Portfolio: ${day_result.ending_portfolio_value:,.0f} | "
                f"PnL: ${pnl:+,.0f} ({pnl_pct:+.2f}%)"
            )

    result = run_simulation(
        data_config,
        forecast_config,
        sim_config,
        progress_callback=progress_callback,
    )

    return result


def print_results(result, description: str):
    """Print simulation results."""
    print()
    print("=" * 70)
    print(f"RESULTS: {description}")
    print("=" * 70)
    print(f"Period: {result.start_date} to {result.end_date} ({result.total_days} days)")
    print(f"Total Trades: {result.total_trades}")
    print()
    print("Portfolio Performance:")
    print(f"  Initial: ${result.initial_cash:,.0f}")
    print(f"  Final:   ${result.final_portfolio_value:,.0f}")
    print(f"  Return:  {result.total_return * 100:+.2f}%")
    print(f"  Annual:  {result.annualized_return * 100:+.2f}%")
    print()
    print("Risk Metrics:")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown:  {result.max_drawdown * 100:.2f}%")
    print(f"  Win Rate:      {result.win_rate * 100:.1f}%")
    print()

    if result.symbol_returns:
        print("Top 10 Performing Symbols:")
        sorted_syms = sorted(result.symbol_returns.items(), key=lambda x: x[1], reverse=True)[:10]
        for sym, ret in sorted_syms:
            print(f"  {sym:6s}: {ret * 100:+.2f}%")
        print()

        print("Bottom 10 Performing Symbols:")
        sorted_syms = sorted(result.symbol_returns.items(), key=lambda x: x[1])[:10]
        for sym, ret in sorted_syms:
            print(f"  {sym:6s}: {ret * 100:+.2f}%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run market simulation on S&P 500 stocks")
    parser.add_argument("--top-n", type=int, default=1, help="Number of top stocks to buy daily")
    parser.add_argument("--initial-cash", type=float, default=100_000, help="Starting capital")
    parser.add_argument("--start", type=str, default="2025-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--compare-universes", action="store_true", help="Compare S&P 500 vs original 15 stocks")
    parser.add_argument("--curated", action="store_true", help="Use curated 50-stock subset instead of full S&P 500")
    parser.add_argument("--output-dir", type=str, default="reports/sp500_simulation")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check data availability
    data_dir = Path("trainingdata/train")

    # Use curated subset or full S&P 500
    symbols_to_check = SP500_CURATED if args.curated else SP500_SYMBOLS
    available_sp500, missing_sp500 = check_data_availability(data_dir, symbols_to_check)

    print(f"S&P 500 Data Check:")
    print(f"  Available: {len(available_sp500)} / {len(SP500_SYMBOLS)}")
    print(f"  Missing:   {len(missing_sp500)}")
    if missing_sp500:
        print(f"  Missing symbols: {', '.join(missing_sp500[:20])}{'...' if len(missing_sp500) > 20 else ''}")
    print()

    if args.compare_universes:
        # Compare S&P 500 vs original 15 stocks
        original_15 = (
            "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD",
            "NFLX", "AVGO", "ADBE", "CRM", "COST", "COIN", "SHOP",
        )

        print("Running comparison: Original 15 vs S&P 500")
        print()

        # Run on original 15
        print("=" * 70)
        print("SIMULATION 1: Original 15 Tech/Growth Stocks")
        print("=" * 70)
        result_15 = run_sp500_simulation(
            original_15,
            start_date,
            end_date,
            args.top_n,
            args.initial_cash,
            output_dir / "original_15",
        )
        print_results(result_15, "Original 15 Stocks")

        # Run on S&P 500
        print()
        print("=" * 70)
        print("SIMULATION 2: Full S&P 500 Universe")
        print("=" * 70)
        result_sp500 = run_sp500_simulation(
            available_sp500,
            start_date,
            end_date,
            args.top_n,
            args.initial_cash,
            output_dir / "sp500",
        )
        print_results(result_sp500, "S&P 500 Universe")

        # Summary comparison
        print()
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<25} {'Original 15':<20} {'S&P 500':<20}")
        print("-" * 65)
        print(f"{'Symbols':<25} {15:<20} {len(available_sp500):<20}")
        print(f"{'Total Return':<25} {result_15.total_return*100:+.2f}%{'':<13} {result_sp500.total_return*100:+.2f}%")
        print(f"{'Annualized Return':<25} {result_15.annualized_return*100:+.2f}%{'':<13} {result_sp500.annualized_return*100:+.2f}%")
        print(f"{'Sharpe Ratio':<25} {result_15.sharpe_ratio:.2f}{'':<17} {result_sp500.sharpe_ratio:.2f}")
        print(f"{'Max Drawdown':<25} {result_15.max_drawdown*100:.2f}%{'':<14} {result_sp500.max_drawdown*100:.2f}%")
        print(f"{'Win Rate':<25} {result_15.win_rate*100:.1f}%{'':<15} {result_sp500.win_rate*100:.1f}%")
        print("=" * 70)

        # Save comparison
        comparison = {
            "original_15": {
                "symbols": len(original_15),
                "total_return": result_15.total_return,
                "annualized_return": result_15.annualized_return,
                "sharpe_ratio": result_15.sharpe_ratio,
                "max_drawdown": result_15.max_drawdown,
                "win_rate": result_15.win_rate,
            },
            "sp500": {
                "symbols": len(available_sp500),
                "total_return": result_sp500.total_return,
                "annualized_return": result_sp500.annualized_return,
                "sharpe_ratio": result_sp500.sharpe_ratio,
                "max_drawdown": result_sp500.max_drawdown,
                "win_rate": result_sp500.win_rate,
            },
        }
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {output_dir / 'comparison.json'}")

    else:
        # Single run on S&P 500
        result = run_sp500_simulation(
            available_sp500,
            start_date,
            end_date,
            args.top_n,
            args.initial_cash,
            output_dir,
        )
        print_results(result, f"S&P 500 Simulation (top_n={args.top_n})")

        # Save results
        summary = {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "symbols_count": len(available_sp500),
            "top_n": args.top_n,
            "initial_cash": args.initial_cash,
            "final_value": result.final_portfolio_value,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "symbol_returns": result.symbol_returns,
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
