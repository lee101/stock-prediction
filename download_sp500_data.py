#!/usr/bin/env python3
"""
Download historical data for S&P 500 stocks for simulation testing.
Downloads both hourly (trainingdatahourly) and daily (trainingdata) data.
"""
import argparse
import sys
from pathlib import Path

# S&P 500 constituents as of January 2025
SP500_SYMBOLS = [
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
]

def get_existing_symbols(data_dir: Path) -> set:
    """Get set of symbols that already have data files."""
    if not data_dir.exists():
        return set()
    return {f.stem for f in data_dir.glob("*.csv") if not f.stem.startswith(".")}


def main():
    parser = argparse.ArgumentParser(description="Download S&P 500 historical data for simulation")
    parser.add_argument("--hourly-only", action="store_true", help="Download only hourly data")
    parser.add_argument("--daily-only", action="store_true", help="Download only daily data")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of new symbols to download (0=all)")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep between downloads to avoid rate limits")
    args = parser.parse_args()

    base_path = Path(__file__).parent

    # Check existing data
    hourly_dir = base_path / "trainingdatahourly" / "stocks"
    daily_train_dir = base_path / "trainingdata" / "train"

    existing_hourly = get_existing_symbols(hourly_dir) if not args.force else set()
    existing_daily = get_existing_symbols(daily_train_dir) if not args.force else set()

    # Find new symbols
    sp500_set = set(SP500_SYMBOLS)
    new_for_hourly = sp500_set - existing_hourly
    new_for_daily = sp500_set - existing_daily

    print(f"S&P 500 symbols: {len(SP500_SYMBOLS)}")
    print(f"Existing hourly data: {len(existing_hourly)}")
    print(f"Existing daily data: {len(existing_daily)}")
    print(f"New for hourly: {len(new_for_hourly)}")
    print(f"New for daily: {len(new_for_daily)}")
    print()

    # Import download functions
    sys.path.insert(0, str(base_path))

    if not args.daily_only:
        print("=" * 60)
        print("DOWNLOADING HOURLY DATA")
        print("=" * 60)

        from download_hourly_data import download_all_symbols

        symbols_to_download = sorted(new_for_hourly)
        if args.limit > 0:
            symbols_to_download = symbols_to_download[:args.limit]

        if symbols_to_download:
            print(f"Downloading hourly data for {len(symbols_to_download)} symbols...")
            print(f"Symbols: {', '.join(symbols_to_download[:20])}{'...' if len(symbols_to_download) > 20 else ''}")

            download_all_symbols(
                symbols=symbols_to_download,
                include_stocks=True,
                include_crypto=False,
                output_dir=base_path / "trainingdatahourly",
                stock_years=1,  # 1 year of hourly data for simulation
                sleep_seconds=args.sleep,
                skip_if_exists=not args.force,
            )
        else:
            print("No new hourly data to download.")
        print()

    if not args.hourly_only:
        print("=" * 60)
        print("DOWNLOADING DAILY DATA")
        print("=" * 60)

        from alpaca_wrapper import download_training_pairs

        symbols_to_download = sorted(new_for_daily)
        if args.limit > 0:
            symbols_to_download = symbols_to_download[:args.limit]

        if symbols_to_download:
            print(f"Downloading daily data for {len(symbols_to_download)} symbols...")
            print(f"Symbols: {', '.join(symbols_to_download[:20])}{'...' if len(symbols_to_download) > 20 else ''}")

            download_training_pairs(
                symbols=symbols_to_download,
                output_dir=base_path / "trainingdata",
                history_days=365 * 2,  # 2 years of daily data
                test_days=30,
                skip_if_recent_days=0 if args.force else 7,
                sleep_seconds=args.sleep,
            )
        else:
            print("No new daily data to download.")

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
