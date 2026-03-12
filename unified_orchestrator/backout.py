"""Backout near market open — exit crypto positions before NYSE opens.

When stock signals have higher expected edge than current crypto holdings,
place limit sells to free crypto capital for better opportunities.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from unified_orchestrator.state import UnifiedPortfolioSnapshot, Position


def compute_crypto_remaining_edge(pos: Position) -> float:
    """Estimate remaining edge for a crypto position.

    Returns expected % gain from current price. Simple heuristic:
    if position is profitable, remaining edge is reduced.
    """
    if pos.avg_price <= 0 or pos.current_price <= 0:
        return 0.0
    current_return = (pos.current_price - pos.avg_price) / pos.avg_price
    # Assume mean reversion: positions that have gained have less remaining edge
    # and positions near entry still have their original edge
    return max(0.0, 0.01 - current_return * 0.5)


def select_backout_candidates(
    snapshot: UnifiedPortfolioSnapshot,
    best_stock_edges: Optional[dict[str, float]] = None,
    min_edge_ratio: float = 2.0,
    only_profitable: bool = True,
) -> list[dict]:
    """Select crypto positions to exit before market open.

    Args:
        snapshot: Current portfolio state
        best_stock_edges: Expected edges for stock signals
        min_edge_ratio: Stock edge must be this many times crypto edge
        only_profitable: Only exit positions at breakeven or better

    Returns:
        List of backout plans: [{symbol, qty, limit_price, reason}]
    """
    if not snapshot.binance_positions:
        return []

    if not best_stock_edges:
        return []

    best_stock_edge = max(best_stock_edges.values()) if best_stock_edges else 0.0
    if best_stock_edge <= 0:
        return []

    candidates = []
    for asset, pos in snapshot.binance_positions.items():
        if pos.market_value < 15:  # Skip dust
            continue

        crypto_edge = compute_crypto_remaining_edge(pos)

        # Check if stock edge justifies exiting
        if best_stock_edge < crypto_edge * min_edge_ratio:
            logger.debug(f"  {asset}: stock edge {best_stock_edge:.4f} < {min_edge_ratio}x "
                         f"crypto edge {crypto_edge:.4f}, keeping")
            continue

        # Only exit profitable or breakeven positions
        if only_profitable and pos.avg_price > 0:
            if pos.current_price < pos.avg_price * 0.998:  # Allow 0.2% tolerance
                logger.debug(f"  {asset}: position at loss, skipping backout")
                continue

        # Set sell price just below current to fill quickly
        sell_price = pos.current_price * 0.9998  # 2 bps below current

        candidates.append({
            "symbol": pos.symbol,
            "base_asset": asset,
            "qty": pos.qty,
            "limit_price": sell_price,
            "current_price": pos.current_price,
            "market_value": pos.market_value,
            "crypto_edge": crypto_edge,
            "best_stock_edge": best_stock_edge,
            "reason": f"Stock edge {best_stock_edge:.4f} > {min_edge_ratio}x crypto edge {crypto_edge:.4f}",
        })

    return candidates


def execute_backout(
    candidates: list[dict],
    dry_run: bool = True,
    expiry_minutes: int = 25,
) -> list[dict]:
    """Execute backout sells for crypto positions.

    Uses the Binance watcher system for limit order management.
    """
    results = []
    for plan in candidates:
        logger.info(f"BACKOUT {plan['base_asset']}: sell {plan['qty']:.6f} @ "
                    f"${plan['limit_price']:.2f} (${plan['market_value']:.0f})")
        logger.info(f"  Reason: {plan['reason']}")

        if dry_run:
            logger.info(f"  [DRY RUN] Would place limit sell")
            results.append({**plan, "status": "dry_run"})
            continue

        try:
            from binanceneural.binance_watchers import spawn_watcher, WatcherPlan

            watcher_plan = WatcherPlan(
                symbol=plan["symbol"],
                side="sell",
                mode="exit",
                limit_price=plan["limit_price"],
                target_qty=plan["qty"],
                expiry_minutes=expiry_minutes,
                poll_seconds=15,  # Poll faster for time-sensitive backout
            )
            spawn_watcher(watcher_plan)
            results.append({**plan, "status": "watcher_spawned"})
            logger.info(f"  Watcher spawned, expires in {expiry_minutes}min")
        except Exception as e:
            logger.error(f"  Backout failed: {e}")
            results.append({**plan, "status": f"error: {e}"})

    return results
