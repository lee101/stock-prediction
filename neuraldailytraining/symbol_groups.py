"""Symbol grouping for cross-attention learning.

Symbols in the same group can attend to each other during training,
allowing the model to learn cross-symbol patterns (e.g., "when tech rallies, buy NVDA").
"""

from pathlib import Path
from typing import Dict, List, Set

# Define symbol groups by sector/type
SYMBOL_GROUPS: Dict[str, List[str]] = {
    # Technology
    "tech_mega": ["AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA", "TSLA"],
    "tech_semi": ["AMD", "INTC", "AVGO", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "NXPI", "ON", "TXN", "ADI", "ASML"],
    "tech_software": ["CRM", "ADBE", "NOW", "ORCL", "SAP", "SNOW", "DDOG", "NET", "PANW", "ZS", "CRWD", "OKTA", "WDAY", "TEAM", "ESTC", "MDB"],
    "tech_internet": ["NFLX", "SHOP", "SPOT", "PYPL", "SQ", "COIN", "HOOD", "UBER", "LYFT", "DASH", "ABNB", "RBLX", "U"],

    # Finance
    "finance_banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW"],
    "finance_payments": ["V", "MA", "PYPL", "SQ", "AXP", "COF"],
    "finance_other": ["BLK", "CME", "ICE", "SPGI", "MCO", "NDAQ"],

    # Healthcare
    "healthcare_pharma": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "GILD", "BIIB", "REGN", "VRTX", "MRNA", "BNTX"],
    "healthcare_devices": ["ABT", "MDT", "ISRG", "SYK", "BSX", "EW", "HOLX", "ZBH"],
    "healthcare_other": ["UNH", "CVS", "TMO", "DHR"],

    # Consumer
    "consumer_staples": ["PG", "KO", "PEP", "COST", "WMT", "MNST", "MCD", "SBUX", "YUM", "CMG", "NKE"],
    "consumer_discretionary": ["HD", "LOW", "TGT", "TJX", "ORLY", "AZO", "BKNG", "MAR", "HLT", "RCL", "CCL", "WYNN", "MGM"],

    # Energy
    "energy_oil": ["XOM", "CVX", "COP", "EOG", "SLB", "HAL", "OXY", "MPC", "VLO", "PSX"],
    "energy_other": ["EPD", "ET", "KMI", "WMB", "ENB", "BKR"],

    # Industrials
    "industrials": ["BA", "GE", "CAT", "DE", "HON", "UPS", "FDX", "RTX", "LMT", "NOC", "GD", "EMR", "ETN", "ITW", "ROK", "CMI", "DOV", "PH", "MMM"],

    # REITs
    "reits": ["AMT", "PLD", "CCI", "EQIX", "PSA", "EQR", "AVB", "MAA", "UDR", "ESS", "CPT", "O", "WELL", "VTR", "HCP", "BXP", "AIV"],

    # Utilities
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "ED", "ES", "DTE", "ETR", "NI", "XEL", "WEC", "PPL", "CNP", "EIX"],

    # Materials
    "materials": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "CLF", "STLD", "RS", "MLM", "VMC", "RPM"],

    # Crypto (major)
    "crypto_major": ["BTCUSD", "BTC-USD", "ETHUSD", "ETH-USD"],
    "crypto_alt": [
        "SOLUSD", "SOL-USD", "DOGEUSD", "DOGE-USD", "XRPUSD", "XRP-USD",
        "LINKUSD", "LINK-USD", "UNIUSD", "UNI-USD", "AVAXUSD", "AVAX-USD",
        "DOTUSD", "DOT-USD", "MATICUSD", "MATIC-USD", "ALGOUSD", "ALGO-USD",
        "LTCUSD", "AAVEUSD", "ADA-USD", "ATOM-USD", "SHIBUSD", "SHIB-USD",
        "TRXUSD", "SKYUSD", "PAXGUSD", "XLM-USD"
    ],

    # ETFs - Market
    "etf_market": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VXUS", "EEM", "EFA"],
    # ETFs - Sector
    "etf_sector": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "GLD", "SLV", "USO", "UNG", "OIH", "ICLN", "DBA", "DBC"],
    # ETFs - ARK
    "etf_ark": ["ARKK", "ARKG", "ARKQ", "ARKW"],
}


def get_symbol_group(symbol: str) -> str:
    """Get the group name for a symbol.

    Args:
        symbol: Stock/crypto symbol

    Returns:
        Group name, or 'other' if not found
    """
    # Normalize symbol (handle different formats)
    normalized = symbol.upper().replace("-", "")

    for group_name, symbols in SYMBOL_GROUPS.items():
        normalized_group = [s.upper().replace("-", "") for s in symbols]
        if normalized in normalized_group or symbol in symbols:
            return group_name

    return "other"


def get_group_id(symbol: str) -> int:
    """Get numeric group ID for a symbol.

    Args:
        symbol: Stock/crypto symbol

    Returns:
        Integer group ID
    """
    group_name = get_symbol_group(symbol)
    group_names = list(SYMBOL_GROUPS.keys()) + ["other"]
    return group_names.index(group_name)


def get_all_symbols() -> Set[str]:
    """Get all symbols from all groups."""
    all_symbols = set()
    for symbols in SYMBOL_GROUPS.values():
        all_symbols.update(symbols)
    return all_symbols


def get_symbols_by_group(group_name: str) -> List[str]:
    """Get all symbols in a group."""
    return SYMBOL_GROUPS.get(group_name, [])


def assign_group_ids(symbols: List[str]) -> List[int]:
    """Assign group IDs to a list of symbols.

    Args:
        symbols: List of symbol names

    Returns:
        List of integer group IDs (same order as input)
    """
    return [get_group_id(sym) for sym in symbols]


# Number of groups
NUM_GROUPS = len(SYMBOL_GROUPS) + 1  # +1 for 'other'


def list_training_symbols(data_root: str = "trainingdata/train") -> List[str]:
    """List all available symbols from training data directory.

    Args:
        data_root: Path to training data directory

    Returns:
        Sorted list of symbol names
    """
    data_path = Path(data_root)
    if not data_path.exists():
        return []

    symbols = set()
    for csv_file in data_path.glob("*.csv"):
        # Extract symbol from filename (e.g., "AAPL.csv" -> "AAPL")
        symbol = csv_file.stem
        # Remove date suffixes (e.g., "AAPL-2025-11-17" -> "AAPL")
        if "-202" in symbol:
            symbol = symbol.split("-202")[0]
        symbols.add(symbol)

    return sorted(symbols)


def get_training_symbols_by_group(
    data_root: str = "trainingdata/train",
    include_groups: List[str] = None,
    exclude_groups: List[str] = None,
) -> List[str]:
    """Get training symbols filtered by group.

    Args:
        data_root: Path to training data directory
        include_groups: Only include symbols from these groups (None = all)
        exclude_groups: Exclude symbols from these groups

    Returns:
        Filtered list of symbols
    """
    available = set(list_training_symbols(data_root))
    result = []

    for symbol in available:
        group = get_symbol_group(symbol)

        if include_groups and group not in include_groups:
            continue
        if exclude_groups and group in exclude_groups:
            continue

        result.append(symbol)

    return sorted(result)


if __name__ == "__main__":
    # Test the grouping
    test_symbols = ["AAPL", "MSFT", "BTCUSD", "SPY", "GS", "UNKNOWN"]
    for sym in test_symbols:
        group = get_symbol_group(sym)
        group_id = get_group_id(sym)
        print(f"{sym}: {group} (id={group_id})")

    print(f"\nTotal groups: {NUM_GROUPS}")
    print(f"Total defined symbols: {len(get_all_symbols())}")
