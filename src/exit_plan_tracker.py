"""Exit Plan Tracker - Tracks predicted exits and triggers time-based backouts.

Ensures positions get closed when their predicted hold time expires,
even if the take-profit price wasn't hit.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.process_utils import backout_near_market

logger = logging.getLogger(__name__)

EXIT_PLANS_FILE = Path("exit_plans.json")


@dataclass
class ExitPlan:
    """Tracks a position's predicted exit."""

    symbol: str
    exit_price: float  # predicted take-profit price
    exit_deadline: str  # ISO timestamp when to backout_near_market
    entry_qty: float  # quantity we opened
    entry_strategy: str
    created_at: str

    @classmethod
    def create(
        cls,
        symbol: str,
        exit_price: float,
        position_length_hours: int,
        entry_qty: float,
        entry_strategy: str,
    ) -> "ExitPlan":
        """Create a new exit plan."""
        now = datetime.now(timezone.utc)
        deadline = now.replace(
            minute=5,  # 5 minutes past the hour
            second=0,
            microsecond=0,
        )
        # Add position_length hours
        from datetime import timedelta
        deadline = deadline + timedelta(hours=position_length_hours)

        return cls(
            symbol=symbol,
            exit_price=exit_price,
            exit_deadline=deadline.isoformat(),
            entry_qty=entry_qty,
            entry_strategy=entry_strategy,
            created_at=now.isoformat(),
        )

    @property
    def deadline_dt(self) -> datetime:
        """Parse exit_deadline as datetime."""
        return datetime.fromisoformat(self.exit_deadline)

    def is_expired(self) -> bool:
        """Check if this exit plan has passed its deadline."""
        return datetime.now(timezone.utc) >= self.deadline_dt


class ExitPlanTracker:
    """Manages exit plans for all positions."""

    def __init__(self, plans_file: Path = EXIT_PLANS_FILE):
        self.plans_file = plans_file
        self._plans: Dict[str, ExitPlan] = {}
        self._load()

    def _load(self) -> None:
        """Load plans from disk."""
        if not self.plans_file.exists():
            self._plans = {}
            return

        try:
            with open(self.plans_file) as f:
                data = json.load(f)

            self._plans = {
                symbol: ExitPlan(**plan_data)
                for symbol, plan_data in data.items()
            }
            logger.info(f"Loaded {len(self._plans)} exit plans")
        except Exception as e:
            logger.error(f"Error loading exit plans: {e}")
            self._plans = {}

    def _save(self) -> None:
        """Save plans to disk."""
        try:
            data = {
                symbol: asdict(plan)
                for symbol, plan in self._plans.items()
            }
            with open(self.plans_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving exit plans: {e}")

    def set_exit_plan(
        self,
        symbol: str,
        exit_price: float,
        position_length_hours: int,
        entry_qty: float,
        entry_strategy: str,
    ) -> ExitPlan:
        """Set or replace exit plan for a symbol.

        This clears any existing plan for the symbol (prevents stale data).
        """
        # Clear existing plan
        if symbol in self._plans:
            logger.info(f"Replacing existing exit plan for {symbol}")

        plan = ExitPlan.create(
            symbol=symbol,
            exit_price=exit_price,
            position_length_hours=position_length_hours,
            entry_qty=entry_qty,
            entry_strategy=entry_strategy,
        )

        self._plans[symbol] = plan
        self._save()

        logger.info(
            f"Set exit plan for {symbol}: TP @ ${exit_price:.4f}, "
            f"deadline {plan.exit_deadline}"
        )

        return plan

    def clear_exit_plan(self, symbol: str) -> Optional[ExitPlan]:
        """Clear exit plan for a symbol (e.g., when position is closed)."""
        plan = self._plans.pop(symbol, None)
        if plan:
            self._save()
            logger.info(f"Cleared exit plan for {symbol}")
        return plan

    def get_exit_plan(self, symbol: str) -> Optional[ExitPlan]:
        """Get exit plan for a symbol."""
        return self._plans.get(symbol)

    def get_all_plans(self) -> List[ExitPlan]:
        """Get all exit plans."""
        return list(self._plans.values())

    def get_expired_plans(self) -> List[ExitPlan]:
        """Get all exit plans that have passed their deadline."""
        return [p for p in self._plans.values() if p.is_expired()]

    def check_and_execute_expired(self, dry_run: bool = False) -> List[str]:
        """Check for expired exit plans and trigger backout_near_market.

        Returns:
            List of symbols that were backed out.
        """
        expired = self.get_expired_plans()
        backed_out = []

        for plan in expired:
            logger.info(
                f"Exit plan for {plan.symbol} expired "
                f"(deadline was {plan.exit_deadline}). "
                f"{'[DRY RUN] Would trigger' if dry_run else 'Triggering'} backout_near_market."
            )

            if not dry_run:
                try:
                    # Use start_offset_minutes=0 to start immediately
                    # Use ramp_minutes=5 for quick ramp
                    backout_near_market(
                        plan.symbol,
                        start_offset_minutes=0,
                        ramp_minutes=5,
                        market_after_minutes=10,
                    )
                    backed_out.append(plan.symbol)
                except Exception as e:
                    logger.error(f"Error triggering backout for {plan.symbol}: {e}")
            else:
                backed_out.append(plan.symbol)

            # Clear the plan
            self.clear_exit_plan(plan.symbol)

        return backed_out


# Singleton instance
_tracker: Optional[ExitPlanTracker] = None


def get_exit_tracker() -> ExitPlanTracker:
    """Get the singleton ExitPlanTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ExitPlanTracker()
    return _tracker
