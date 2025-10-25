from __future__ import annotations

import ast
import math
from typing import Iterable, List, Mapping, Optional, Tuple

LIQUID_CRYPTO_PREFIXES: Tuple[str, ...] = ("BTC", "ETH", "SOL", "UNI")
TIGHT_SPREAD_EQUITIES = {"AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOG"}
DEFAULT_SPREAD_BPS = 25


def coerce_optional_float(value: object) -> Optional[float]:
    """
    Attempt to coerce an arbitrary object to a finite float.

    Returns None when the value is missing, empty, or not convertible.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, int):
        return float(value)

    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        parsed = float(value_str)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def parse_float_list(raw: object) -> Optional[List[float]]:
    """
    Parse a variety of inputs into a list of floats, ignoring NaNs.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None

    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        text = str(raw)
        if not text:
            return None
        text = text.replace("np.float32", "float")
        try:
            values = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    if not isinstance(values, (list, tuple)):
        return None

    result: List[float] = []
    for item in values:
        coerced = coerce_optional_float(item)
        if coerced is not None:
            result.append(coerced)
    return result or None


def compute_spread_bps(bid: Optional[float], ask: Optional[float]) -> float:
    """
    Compute the bid/ask spread in basis points.

    Returns infinity when the inputs are missing or invalid.
    """
    if bid is None or ask is None:
        return float("inf")
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return float("inf")
    return (ask - bid) / mid * 1e4


def resolve_spread_cap(symbol: str) -> int:
    """
    Determine the maximum spread (in bps) allowed for the given symbol.
    """
    if symbol.endswith("USD") and symbol.startswith(LIQUID_CRYPTO_PREFIXES):
        return 35
    if symbol in TIGHT_SPREAD_EQUITIES:
        return 8
    return DEFAULT_SPREAD_BPS


def expected_cost_bps(symbol: str) -> float:
    base = 20.0 if symbol.endswith("USD") else 6.0
    if symbol in {"META", "AMD", "LCID", "QUBT"}:
        base += 25.0
    return base


def agree_direction(*pred_signs: int) -> bool:
    """
    Return True when all non-zero predictions agree on direction.
    """
    signs = {sign for sign in pred_signs if sign in (-1, 1)}
    return len(signs) == 1


def kelly_lite(edge_pct: float, sigma_pct: float, cap: float = 0.15) -> float:
    if sigma_pct <= 0:
        return 0.0
    raw = edge_pct / (sigma_pct**2)
    scaled = 0.2 * raw
    if scaled <= 0:
        return 0.0
    return float(min(cap, max(0.0, scaled)))


def should_rebalance(
    current_pos_side: Optional[str],
    new_side: str,
    current_size: float,
    target_size: float,
    eps: float = 0.25,
) -> bool:
    current_side = (current_pos_side or "").lower()
    new_side_norm = new_side.lower()
    if current_side not in {"buy", "sell"} or new_side_norm not in {"buy", "sell"}:
        return True
    if current_side != new_side_norm:
        return True
    current_abs = abs(current_size)
    target_abs = abs(target_size)
    if current_abs <= 1e-9:
        return True
    delta = abs(target_abs - current_abs) / max(current_abs, 1e-9)
    return delta > eps


def edge_threshold_bps(symbol: str) -> float:
    base_cost = expected_cost_bps(symbol) + 10.0
    hard_floor = 40.0 if symbol.endswith("USD") else 15.0
    return max(base_cost, hard_floor)


def evaluate_strategy_entry_gate(
    symbol: str,
    stats: Mapping[str, float] | Iterable[Tuple[str, float]],
    *,
    fallback_used: bool,
    sample_size: int,
) -> Tuple[bool, str]:
    """
    Evaluate whether strategy statistics clear the entry thresholds.

    Parameters
    ----------
    symbol:
        The trading instrument identifier.
    stats:
        Iterable of (metric_name, metric_value) pairs. Only the first occurrence
        of each expected metric is considered.
    fallback_used:
        When True, the caller has already resorted to fallback metrics; we fail fast.
    sample_size:
        Number of samples backing the metrics.
    """
    if fallback_used:
        return False, "fallback_metrics"

    if isinstance(stats, Mapping):
        stats_map = {str(name): float(value) for name, value in stats.items()}
    else:
        stats_map = {str(name): float(value) for name, value in stats}
    avg_return = float(stats_map.get("avg_return", 0.0))
    sharpe = float(stats_map.get("sharpe", 0.0))
    turnover = float(stats_map.get("turnover", 0.0))
    max_drawdown = float(stats_map.get("max_drawdown", 0.0))

    edge_bps = avg_return * 1e4
    needed_edge = edge_threshold_bps(symbol)
    if edge_bps < needed_edge:
        return False, f"edge {edge_bps:.1f}bps < need {needed_edge:.1f}bps"
    if sharpe < 0.5:
        return False, f"sharpe {sharpe:.2f} below 0.50 gate"
    if sample_size < 120:
        return False, f"insufficient samples {sample_size} < 120"
    if max_drawdown < -0.08:
        return False, f"max drawdown {max_drawdown:.2f} below -0.08 gate"
    if turnover > 2.0 and sharpe < 0.8:
        return False, f"turnover {turnover:.2f} with sharpe {sharpe:.2f}"
    return True, "ok"
