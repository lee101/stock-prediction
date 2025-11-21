"""Symbol grouping for cross-attention learning.

Symbols in the same group can attend to each other during training,
allowing the model to learn cross-symbol patterns (e.g., "when tech rallies, buy NVDA").
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

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


def is_crypto_symbol(symbol: str) -> bool:
    """Heuristic to identify crypto tickers.

    Handles common dash/USDT variants to keep grouping consistent with
    asset-class flags used elsewhere in the pipeline.
    """

    sym = symbol.upper()
    return sym.endswith("-USD") or sym.endswith("USD") or sym.endswith("USDT")


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


def _returns_from_frame(frame: pd.DataFrame, window_days: int) -> pd.Series:
    if "close" not in frame or "date" not in frame:
        return pd.Series(dtype=float)
    ordered = frame.sort_values("date")
    returns = ordered["close"].pct_change().dropna()
    if window_days > 0:
        returns = returns.iloc[-window_days:]
    # Align by date so different symbols sync for correlation
    returns.index = pd.to_datetime(ordered.loc[returns.index, "date"], utc=True)
    returns = returns[~returns.index.duplicated(keep="last")]
    return returns


def _correlation_matrix(
    frames: Dict[str, pd.DataFrame],
    *,
    window_days: int,
    min_overlap: int,
) -> pd.DataFrame:
    returns: Dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        series = _returns_from_frame(frame, window_days)
        if len(series) >= max(5, min_overlap):
            series = series[~series.index.duplicated(keep="last")].sort_index()
            returns[symbol] = series

    if not returns:
        return pd.DataFrame()

    merged = pd.concat(returns.values(), axis=1, join="inner")
    merged.columns = list(returns.keys())
    if merged.empty:
        return pd.DataFrame()

    corr = merged.corr(min_periods=min_overlap).fillna(0.0)
    return corr


def _greedy_clusters(corr: pd.DataFrame, *, min_corr: float, max_group_size: int) -> List[List[str]]:
    if corr.empty:
        return []
    remaining = set(corr.index)
    clusters: List[List[str]] = []

    while remaining:
        # Seed with symbol most correlated on average to others still unassigned
        seed = max(remaining, key=lambda sym: float(corr.loc[sym, list(remaining)].mean()))
        eligible = [sym for sym in remaining if corr.loc[seed, sym] >= min_corr]
        eligible.sort(key=lambda sym: float(corr.loc[seed, sym]), reverse=True)

        group = eligible[:max_group_size] if eligible else [seed]
        clusters.append(group)
        remaining -= set(group)

    return clusters


def build_correlation_groups(
    frames: Dict[str, pd.DataFrame],
    *,
    min_corr: float = 0.6,
    max_group_size: int = 12,
    window_days: int = 252,
    min_overlap: int = 60,
    split_crypto: bool = True,
) -> Dict[str, List[str]]:
    """Create symbol groups from a correlation matrix.

    Groups equities and crypto separately (unless ``split_crypto`` is False),
    clusters highly correlated symbols together, and caps group size to
    ``max_group_size`` to keep attention blocks tractable.
    """

    if not frames:
        return {}

    universes: List[Tuple[str, Dict[str, pd.DataFrame]]]
    if split_crypto:
        equities = {sym: df for sym, df in frames.items() if not is_crypto_symbol(sym)}
        cryptos = {sym: df for sym, df in frames.items() if is_crypto_symbol(sym)}
        universes = [("equity", equities), ("crypto", cryptos)]
    else:
        universes = [("all", frames)]

    groups: Dict[str, List[str]] = {}
    for universe_name, subset in universes:
        if not subset:
            continue
        corr = _correlation_matrix(subset, window_days=window_days, min_overlap=min_overlap)

        if corr.empty:
            # Fallback to singleton groups to keep training working even with sparse data
            clusters: List[List[str]] = [[sym] for sym in sorted(subset.keys())]
        else:
            clusters = _greedy_clusters(corr, min_corr=min_corr, max_group_size=max_group_size)

        for idx, cluster in enumerate(clusters, start=1):
            group_name = f"corr_{universe_name}_{idx}"
            groups[group_name] = cluster

    return groups


def group_ids_from_clusters(clusters: Dict[str, List[str]], *, start_id: int = 0) -> Dict[str, int]:
    """Convert cluster mapping to a stable symbol->group_id lookup."""

    mapping: Dict[str, int] = {}
    for gid, group_name in enumerate(sorted(clusters.keys()), start=start_id):
        for symbol in clusters[group_name]:
            mapping[symbol] = gid
    return mapping


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
