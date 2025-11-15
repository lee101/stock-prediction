"""Strategy-specific price lookup utilities.

This module centralizes the mapping from strategy names to their corresponding
price fields in analysis data. Previously duplicated in trade_stock_e2e.py.
"""


from src.comparisons import is_buy_side


def get_entry_price(
    data: dict[str, float],
    strategy: str | None,
    side: str,
) -> float | None:
    """Look up the entry price for a given strategy and side.

    Args:
        data: Analysis data dictionary containing strategy price fields
        strategy: Strategy name (e.g., "maxdiff", "highlow", "pctdiff")
        side: Trading side ("buy" or "sell")

    Returns:
        Entry price for the strategy/side combination, or None if not found

    Examples:
        >>> data = {
        ...     "maxdiffprofit_low_price": 95.0,
        ...     "maxdiffprofit_high_price": 105.0,
        ... }
        >>> get_entry_price(data, "maxdiff", "buy")
        95.0
        >>> get_entry_price(data, "maxdiff", "sell")
        105.0
    """
    normalized = (strategy or "").strip().lower()
    is_buy = is_buy_side(side)

    if normalized == "maxdiff":
        neural_key = "neuralpricing_low_price" if is_buy else "neuralpricing_high_price"
        if neural_key in data and data[neural_key] is not None:
            return data.get(neural_key)
        return data.get("maxdiffprofit_low_price" if is_buy else "maxdiffprofit_high_price")

    if normalized == "maxdiffalwayson":
        return data.get("maxdiffalwayson_low_price" if is_buy else "maxdiffalwayson_high_price")

    if normalized == "pctdiff":
        return data.get("pctdiff_entry_low_price" if is_buy else "pctdiff_entry_high_price")

    if normalized == "highlow":
        return data.get("predicted_low" if is_buy else "predicted_high")

    return None


def get_takeprofit_price(
    data: dict[str, float],
    strategy: str | None,
    side: str,
) -> float | None:
    """Look up the take-profit price for a given strategy and side.

    Args:
        data: Analysis data dictionary containing strategy price fields
        strategy: Strategy name (e.g., "maxdiff", "highlow", "pctdiff")
        side: Trading side ("buy" or "sell")

    Returns:
        Take-profit price for the strategy/side combination, or None if not found

    Examples:
        >>> data = {
        ...     "maxdiffprofit_low_price": 95.0,
        ...     "maxdiffprofit_high_price": 105.0,
        ... }
        >>> get_takeprofit_price(data, "maxdiff", "buy")
        105.0
        >>> get_takeprofit_price(data, "maxdiff", "sell")
        95.0
    """
    normalized = (strategy or "").strip().lower()
    is_buy = is_buy_side(side)

    if normalized == "maxdiff":
        neural_key = "neuralpricing_high_price" if is_buy else "neuralpricing_low_price"
        if neural_key in data and data[neural_key] is not None:
            return data.get(neural_key)
        return data.get("maxdiffprofit_high_price" if is_buy else "maxdiffprofit_low_price")

    if normalized == "maxdiffalwayson":
        return data.get("maxdiffalwayson_high_price" if is_buy else "maxdiffalwayson_low_price")

    if normalized == "pctdiff":
        return data.get("pctdiff_takeprofit_high_price" if is_buy else "pctdiff_takeprofit_low_price")

    if normalized == "highlow":
        return data.get("predicted_high" if is_buy else "predicted_low")

    return None


def get_strategy_price_fields(strategy: str) -> dict[str, str]:
    """Get metadata about price field names for a strategy.

    Args:
        strategy: Strategy name

    Returns:
        Dictionary with keys:
        - 'buy_entry': Field name for buy entry price
        - 'buy_takeprofit': Field name for buy take-profit price
        - 'sell_entry': Field name for sell entry price
        - 'sell_takeprofit': Field name for sell take-profit price

    Examples:
        >>> fields = get_strategy_price_fields("maxdiff")
        >>> fields['buy_entry']
        'maxdiffprofit_low_price'
        >>> fields['buy_takeprofit']
        'maxdiffprofit_high_price'
    """
    normalized = strategy.strip().lower()

    if normalized == "maxdiff":
        return {
            "buy_entry": "maxdiffprofit_low_price",
            "buy_takeprofit": "maxdiffprofit_high_price",
            "sell_entry": "maxdiffprofit_high_price",
            "sell_takeprofit": "maxdiffprofit_low_price",
        }

    if normalized == "maxdiffalwayson":
        return {
            "buy_entry": "maxdiffalwayson_low_price",
            "buy_takeprofit": "maxdiffalwayson_high_price",
            "sell_entry": "maxdiffalwayson_high_price",
            "sell_takeprofit": "maxdiffalwayson_low_price",
        }

    if normalized == "pctdiff":
        return {
            "buy_entry": "pctdiff_entry_low_price",
            "buy_takeprofit": "pctdiff_takeprofit_high_price",
            "sell_entry": "pctdiff_entry_high_price",
            "sell_takeprofit": "pctdiff_takeprofit_low_price",
        }

    if normalized == "highlow":
        return {
            "buy_entry": "predicted_low",
            "buy_takeprofit": "predicted_high",
            "sell_entry": "predicted_high",
            "sell_takeprofit": "predicted_low",
        }

    return {}


def is_limit_order_strategy(strategy: str | None) -> bool:
    """Check if a strategy uses limit orders (entry at specific prices).

    Args:
        strategy: Strategy name

    Returns:
        True if strategy uses limit orders, False otherwise

    Examples:
        >>> is_limit_order_strategy("maxdiff")
        True
        >>> is_limit_order_strategy("market")
        False
    """
    if not strategy:
        return False

    normalized = strategy.strip().lower()
    # These strategies wait for specific entry prices
    limit_strategies = {"maxdiff", "maxdiffalwayson", "pctdiff", "highlow"}
    return normalized in limit_strategies
