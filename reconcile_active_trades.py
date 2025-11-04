#!/usr/bin/env python3
"""
Reconcile active_trades.json with actual Alpaca positions.

This script:
1. Fetches all positions from Alpaca
2. Compares with active_trades.json metadata
3. Removes stale metadata (no position exists)
4. Updates quantities to match reality
5. Adds missing metadata for untracked positions
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import alpaca_wrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

STATE_FILE = Path("strategy_state/active_trades.json")


def normalize_side(side: str) -> str:
    """Normalize position side to 'buy' or 'sell'."""
    side_lower = str(side).lower()
    if "short" in side_lower or "sell" in side_lower:
        return "sell"
    return "buy"


def load_state():
    """Load current active trades state."""
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state):
    """Save active trades state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def reconcile():
    """Reconcile active trades with Alpaca positions."""
    logger.info("Starting reconciliation...")

    # Load current state
    state = load_state()
    logger.info(f"Loaded {len(state)} entries from {STATE_FILE}")

    # Fetch all positions from Alpaca
    try:
        positions = alpaca_wrapper.get_all_positions()
        logger.info(f"Fetched {len(positions)} positions from Alpaca")
    except Exception as e:
        logger.error(f"Failed to fetch positions from Alpaca: {e}")
        return

    # Build map of actual positions
    position_map = {}
    for pos in positions:
        symbol = pos.symbol
        side = normalize_side(pos.side)
        key = f"{symbol}|{side}"
        position_map[key] = {
            "qty": float(pos.qty),
            "avg_entry_price": float(pos.avg_entry_price),
            "market_value": float(pos.market_value),
        }
        logger.info(f"  Position: {key} qty={pos.qty}")

    # Reconcile
    reconciled = {}
    stale_count = 0
    updated_count = 0
    missing_count = 0

    # Check existing metadata
    for key, metadata in state.items():
        if key in position_map:
            # Position exists - update qty
            old_qty = metadata.get("qty")
            new_qty = position_map[key]["qty"]

            if old_qty != new_qty:
                logger.warning(f"  {key}: qty mismatch {old_qty} -> {new_qty}")
                updated_count += 1

            # Keep metadata, update qty
            reconciled[key] = {
                "entry_strategy": metadata.get("entry_strategy", "unknown"),
                "mode": metadata.get("mode", "normal"),
                "opened_at": metadata.get("opened_at", datetime.now(timezone.utc).isoformat()),
                "qty": new_qty,  # From Alpaca
            }
            if "opened_at_sim" in metadata:
                reconciled[key]["opened_at_sim"] = metadata["opened_at_sim"]
        else:
            # No position exists - stale metadata
            logger.warning(f"  {key}: STALE metadata (no position exists)")
            stale_count += 1

    # Check for positions without metadata
    for key, pos_data in position_map.items():
        if key not in state:
            logger.warning(f"  {key}: Missing metadata (position exists, creating default)")
            missing_count += 1
            reconciled[key] = {
                "entry_strategy": "unknown",
                "mode": "normal",
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "qty": pos_data["qty"],
            }

    # Save reconciled state
    save_state(reconciled)

    logger.info(f"\nReconciliation complete:")
    logger.info(f"  - Removed {stale_count} stale entries")
    logger.info(f"  - Updated {updated_count} quantities")
    logger.info(f"  - Added {missing_count} missing entries")
    logger.info(f"  - Final state: {len(reconciled)} entries")


if __name__ == "__main__":
    reconcile()
